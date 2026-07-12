"""
Temporal Fusion Transformer for multi-horizon KPI forecasting.

Based on Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" — adapted for enterprise KPI data
with explicit support for static covariates (hospital/site metadata) and
known future inputs (calendar features, scheduled events).

Key output: P10/P50/P90 quantile forecasts + variable importance scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TFTConfig:
    n_targets: int                   # number of KPI targets
    n_static: int                    # static covariate dims (e.g., hospital size)
    n_temporal_known: int            # known future inputs (calendar, events)
    n_temporal_observed: int         # observed-only past inputs
    hidden_dim: int = 128
    n_heads: int = 4
    n_lstm_layers: int = 2
    dropout: float = 0.1
    quantiles: tuple = (0.1, 0.5, 0.9)
    max_encoder_length: int = 52     # weeks of history
    max_prediction_length: int = 13  # quarters ahead


class GatedResidualNetwork(nn.Module):
    """
    GRN: core building block of TFT.
    Applies nonlinear transformation with gated skip connection.
    Optional context injection for static covariate conditioning.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.has_context = context_dim is not None
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_ctx = nn.Linear(context_dim, hidden_dim, bias=False) if self.has_context else None
        self.fc2 = nn.Linear(hidden_dim, out_dim * 2)  # *2 for GLU gate
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        if self.has_context and context is not None:
            h = h + self.fc_ctx(context)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        # GLU gating
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Learns per-variable importance weights (Granger-like selection).
    Critical for interpretability: tells you which KPIs drive the forecast.
    """

    def __init__(self, n_vars: int, var_dim: int, hidden_dim: int,
                 context_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(var_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_vars)
        ])
        self.softmax_grn = GatedResidualNetwork(
            n_vars * hidden_dim, hidden_dim, n_vars,
            context_dim=context_dim, dropout=dropout
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        x       : (B, T, n_vars * var_dim)  or  (B, n_vars * var_dim)
        Returns : processed features (B, T, hidden) and importance weights (B, T, n_vars)
        """
        has_time = x.dim() == 3
        if not has_time:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        var_dim = x.shape[-1] // self.n_vars
        x_split = x.view(B, T, self.n_vars, var_dim)

        processed = torch.stack([
            self.var_grns[i](x_split[:, :, i, :])
            for i in range(self.n_vars)
        ], dim=2)                                           # (B, T, n_vars, hidden)

        flat = x.view(B, T, -1)
        weights = self.softmax_grn(flat, context)          # (B, T, n_vars)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)

        out = (processed * weights).sum(dim=2)             # (B, T, hidden)
        if not has_time:
            out = out.squeeze(1)
            weights = weights.squeeze(1)
        return out, weights.squeeze(-1)


class InterpretableMultiheadAttention(nn.Module):
    """
    Multi-head attention where heads share value projections,
    enabling meaningful per-head attention pattern analysis.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim)  # shared across heads
        self.out_proj = nn.Linear(self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        B, T, D = q.shape
        Q = self.q_proj(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ V).mean(dim=1)                       # avg over heads
        out = self.out_proj(out)
        return out, attn.mean(dim=1)                       # return mean attn for viz


class TemporalFusionTransformer(nn.Module):
    """
    Full TFT model for enterprise KPI forecasting.

    Produces:
      - Quantile forecasts (P10/P50/P90) per target KPI per horizon
      - Variable importance scores (past + future + static)
      - Temporal attention patterns (interpretable)
    """

    def __init__(self, cfg: TFTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        q = cfg.n_targets * len(cfg.quantiles)

        # Static enrichment
        self.static_vsn = VariableSelectionNetwork(
            cfg.n_static, D, D, dropout=cfg.dropout
        ) if cfg.n_static > 0 else None
        self.static_grn = GatedResidualNetwork(D, D, D, dropout=cfg.dropout) if cfg.n_static > 0 else None

        # Encoder variable selection (past observed + past known)
        n_enc_vars = cfg.n_temporal_observed + cfg.n_temporal_known
        self.encoder_vsn = VariableSelectionNetwork(
            n_enc_vars, D, D,
            context_dim=D if cfg.n_static > 0 else None,
            dropout=cfg.dropout
        )

        # Decoder variable selection (future known only)
        self.decoder_vsn = VariableSelectionNetwork(
            cfg.n_temporal_known, D, D,
            context_dim=D if cfg.n_static > 0 else None,
            dropout=cfg.dropout
        )

        # Input projections (map each variable to D dims)
        self.enc_input_proj = nn.Linear(n_enc_vars, n_enc_vars * D)
        self.dec_input_proj = nn.Linear(cfg.n_temporal_known, cfg.n_temporal_known * D)

        # LSTM encoder/decoder
        self.encoder_lstm = nn.LSTM(D, D, cfg.n_lstm_layers, batch_first=True, dropout=cfg.dropout)
        self.decoder_lstm = nn.LSTM(D, D, cfg.n_lstm_layers, batch_first=True, dropout=cfg.dropout)

        # Gate after LSTM
        self.lstm_gate = GatedResidualNetwork(D, D, D, dropout=cfg.dropout)

        # Static enrichment layer
        self.static_enrich = GatedResidualNetwork(
            D, D, D, context_dim=D if cfg.n_static > 0 else None, dropout=cfg.dropout
        )

        # Temporal self-attention
        self.attn = InterpretableMultiheadAttention(D, cfg.n_heads, cfg.dropout)
        self.attn_gate = GatedResidualNetwork(D, D, D, dropout=cfg.dropout)

        # Position-wise feedforward
        self.ff_grn = GatedResidualNetwork(D, D * 4, D, dropout=cfg.dropout)

        # Output projection to quantiles × targets
        self.output_proj = nn.Linear(D, q)

    def _embed_inputs(self, x: Tensor, proj: nn.Linear, n_vars: int) -> Tensor:
        """Project raw features → per-variable embeddings (B, T, n_vars * D)."""
        B, T, _ = x.shape
        return proj(x).view(B, T, n_vars, self.cfg.hidden_dim).view(B, T, -1)

    def forward(
        self,
        x_enc: Tensor,          # (B, enc_len, n_temporal_observed + n_temporal_known)
        x_dec: Tensor,          # (B, pred_len, n_temporal_known)
        x_static: Optional[Tensor] = None,  # (B, n_static)
    ) -> dict[str, Tensor]:

        cfg = self.cfg
        D = cfg.hidden_dim

        # Static processing
        ctx = None
        if cfg.n_static > 0 and x_static is not None and self.static_vsn is not None:
            static_in = x_static.unsqueeze(1).expand(-1, 1, -1)
            static_in = self._embed_inputs(static_in, nn.Linear(cfg.n_static, cfg.n_static * D).to(x_static.device), cfg.n_static)
            s, s_weights = self.static_vsn(static_in)
            ctx = self.static_grn(s.squeeze(1))             # (B, D)

        # Encoder
        n_enc_vars = cfg.n_temporal_observed + cfg.n_temporal_known
        enc_emb = self._embed_inputs(x_enc, self.enc_input_proj, n_enc_vars)
        enc_vsn, enc_weights = self.encoder_vsn(enc_emb, ctx.unsqueeze(1).expand(-1, x_enc.shape[1], -1) if ctx is not None else None)
        enc_out, (h, c) = self.encoder_lstm(enc_vsn)

        # Decoder
        dec_emb = self._embed_inputs(x_dec, self.dec_input_proj, cfg.n_temporal_known)
        dec_vsn, dec_weights = self.decoder_vsn(dec_emb, ctx.unsqueeze(1).expand(-1, x_dec.shape[1], -1) if ctx is not None else None)
        dec_out, _ = self.decoder_lstm(dec_vsn, (h, c))

        # Combine encoder + decoder sequence
        seq = torch.cat([enc_out, dec_out], dim=1)         # (B, enc+pred, D)
        seq = self.lstm_gate(seq)

        # Static enrichment
        if ctx is not None:
            seq = self.static_enrich(seq, ctx.unsqueeze(1).expand_as(seq))
        else:
            seq = self.static_enrich(seq)

        # Self-attention over full sequence
        seq_attn, attn_weights = self.attn(seq, seq, seq)
        seq = self.attn_gate(seq_attn + seq)

        # Position-wise FF
        seq = self.ff_grn(seq)

        # Slice prediction window
        pred = seq[:, -cfg.max_prediction_length:, :]      # (B, pred_len, D)
        out = self.output_proj(pred)                        # (B, pred_len, n_targets * n_quantiles)

        B, T, _ = out.shape
        out = out.view(B, T, cfg.n_targets, len(cfg.quantiles))

        return {
            "quantiles": out,                              # (B, pred_len, n_targets, n_quantiles)
            "encoder_importance": enc_weights,             # (B, enc_len, n_enc_vars)
            "decoder_importance": dec_weights,             # (B, pred_len, n_dec_vars)
            "attention": attn_weights,                     # (B, seq_len, seq_len)
        }

    def predict_intervals(self, output: dict) -> dict[str, Tensor]:
        """Extract P10, P50, P90 forecast bands from output."""
        q = output["quantiles"]
        return {
            "p10": q[..., 0],
            "p50": q[..., 1],
            "p90": q[..., 2],
        }


def quantile_loss(y_true: Tensor, y_pred: Tensor, quantiles: tuple) -> Tensor:
    """Pinball loss for multi-quantile training."""
    losses = []
    for i, tau in enumerate(quantiles):
        err = y_true - y_pred[..., i]
        losses.append(torch.max(tau * err, (tau - 1) * err))
    return torch.stack(losses, dim=-1).mean()
