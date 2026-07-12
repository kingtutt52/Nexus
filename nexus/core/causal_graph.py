"""
DAG-constrained GNN for causal KPI discovery.

Implements NOTEARS-MLP (Zheng et al. 2018) with a GNN encoder instead of
a plain MLP — the idea being that letting nodes communicate during the
structural equation learning should help with the kinds of dense causal
graphs you see in healthcare ops data.

TODO: benchmark against DAG-GNN (Yu et al. 2019) — their variational
approach might handle the non-Gaussian KPI distributions better.
TODO: the augmented Lagrangian outer loop is slow. Look into the
DAGMA (Bello et al. 2022) log-det barrier as a drop-in replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CausalGraphConfig:
    n_nodes: int                      # number of KPI variables
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    lambda1: float = 0.01             # L1 sparsity on adjacency
    lambda2: float = 0.01             # L2 regularization
    h_tol: float = 1e-8               # DAG constraint tolerance
    rho_max: float = 1e16             # augmented Lagrangian max rho
    max_iter: int = 100               # outer AL iterations
    lr: float = 1e-3


class MLPEncoder(nn.Module):
    """Per-node MLP that maps node features → latent embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GNNLayer(nn.Module):
    """
    Graph attention layer where edges are weighted by the learned
    adjacency matrix W. Message passing: h_i <- sum_j W_ij * h_j.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: Tensor, W: Tensor) -> Tensor:
        """
        h : (B, N, D)
        W : (N, N) — soft adjacency, W[i,j] = edge i→j weight
        """
        B, N, D = h.shape
        q = self.q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention gated by learned adjacency
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale          # (B, H, N, N)
        W_mask = W.unsqueeze(0).unsqueeze(0)               # (1, 1, N, N)
        attn = attn * W_mask
        attn = F.softmax(attn + 1e-9, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.out(out))


class CausalKPIGraph(nn.Module):
    """
    End-to-end causal graph learner over enterprise KPIs.

    Architecture:
      1. MLPEncoder: node features → latent embeddings
      2. Stack of GNNLayers: refine embeddings using soft adjacency
      3. Learnable adjacency matrix W (N x N, constrained to DAG)
      4. MLPDecoder: reconstruct KPI values from causal embeddings
      5. NOTEARS h(W) constraint: h = tr(e^{W◦W}) - d = 0 enforces acyclicity

    Training uses augmented Lagrangian:
      L = reconstruction_loss + lambda1*||W||_1 + lambda2*||W||_F^2
          + alpha*h(W) + rho/2 * h(W)^2
    """

    def __init__(self, cfg: CausalGraphConfig, node_feature_dim: int):
        super().__init__()
        self.cfg = cfg
        self.n = cfg.n_nodes

        self.encoder = MLPEncoder(node_feature_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(cfg.hidden_dim) for _ in range(cfg.n_layers)
        ])
        self.decoder = MLPEncoder(cfg.hidden_dim, cfg.hidden_dim, node_feature_dim, cfg.dropout)

        # Raw (unconstrained) adjacency weights — softplus to ensure non-negative
        self._W_raw = nn.Parameter(torch.zeros(self.n, self.n))
        nn.init.xavier_uniform_(self._W_raw.unsqueeze(0)).squeeze(0)

        # Augmented Lagrangian multipliers (not optimized by Adam, updated manually)
        self.register_buffer("alpha", torch.zeros(1))
        self.register_buffer("rho", torch.ones(1))

    @property
    def W(self) -> Tensor:
        """Non-negative adjacency with zero diagonal (no self-loops)."""
        W = F.softplus(self._W_raw)
        return W * (1 - torch.eye(self.n, device=W.device))

    def h_acyclicity(self, W: Tensor) -> Tensor:
        """NOTEARS constraint: h(W) = tr(e^{W◦W}) - d = 0 iff W is a DAG."""
        d = W.shape[0]
        WW = W * W
        # Matrix exponential via eigendecomposition for stability
        M = torch.linalg.matrix_exp(WW)
        return M.trace() - d

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        X : (B, N, F) — batch of KPI observation matrices
        Returns reconstructed X and the adjacency matrix W.
        """
        W = self.W
        h = self.encoder(X)                        # (B, N, hidden)
        for layer in self.gnn_layers:
            h = layer(h, W)
        X_hat = self.decoder(h)                    # (B, N, F)
        return X_hat, W

    def loss(self, X: Tensor) -> dict[str, Tensor]:
        X_hat, W = self.forward(X)
        cfg = self.cfg

        recon = F.mse_loss(X_hat, X)
        sparse = cfg.lambda1 * W.abs().sum()
        ridge = cfg.lambda2 * W.pow(2).sum()
        h = self.h_acyclicity(W)
        dag_penalty = self.alpha * h + 0.5 * self.rho * h * h

        total = recon + sparse + ridge + dag_penalty
        return {"total": total, "recon": recon, "sparse": sparse, "h": h}

    @torch.no_grad()
    def update_lagrangian(self, h_val: float) -> None:
        """Update augmented Lagrangian multipliers (called after each outer iteration)."""
        self.alpha += self.rho * h_val
        self.rho = torch.clamp(self.rho * 10, max=self.cfg.rho_max)

    def get_causal_graph(self, threshold: float = 0.3) -> np.ndarray:
        """Return thresholded binary adjacency matrix (N x N)."""
        W = self.W.detach().cpu().numpy()
        return (W > threshold).astype(np.float32)

    def get_causal_order(self) -> list[int]:
        """Topological sort of the learned DAG. Kahn's algorithm."""
        W_bin = self.get_causal_graph()
        n = W_bin.shape[0]
        in_degree = W_bin.sum(axis=0).astype(int)
        order, queue = [], [i for i in range(n) if in_degree[i] == 0]
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if W_bin[node, j] > 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        return order

    def compute_intervention_effect(
        self,
        kpi_idx: int,
        delta: float,
        X_baseline: Tensor,
    ) -> Tensor:
        """
        Estimate downstream KPI changes from do(KPI_i += delta).
        Uses the learned structural equations (decoder) with modified input.
        Returns delta_X_hat for all nodes.
        """
        W = self.W.detach()
        X_int = X_baseline.clone()
        X_int[:, kpi_idx, :] += delta

        with torch.no_grad():
            X_hat_base, _ = self.forward(X_baseline)
            X_hat_int, _ = self.forward(X_int)

        return X_hat_int - X_hat_base


def train_causal_graph(
    model: CausalKPIGraph,
    X: Tensor,
    n_outer: int = 50,
    n_inner: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict]:
    """
    Augmented Lagrangian training loop for DAG-constrained GNN.
    Returns training history.
    """
    model = model.to(device)
    X = X.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    h_prev = float("inf")

    for outer in range(n_outer):
        for inner in range(n_inner):
            optimizer.zero_grad()
            losses = model.loss(X)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            losses = model.loss(X)
            h_val = losses["h"].item()

        history.append({
            "outer": outer,
            "recon": losses["recon"].item(),
            "h": h_val,
            "rho": model.rho.item(),
            "alpha": model.alpha.item(),
        })

        if abs(h_val) < model.cfg.h_tol:
            break

        model.update_lagrangian(h_val)
        h_prev = h_val

    return history
