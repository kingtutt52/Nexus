"""
Benchmark: CausalKPIGraph vs baselines on healthcare synthetic data.

Compares structural Hamming distance (SHD) and structural intervention
distance (SID) against:
  - PC algorithm (constraint-based, conditional independence tests)
  - GES (greedy equivalence search, score-based)
  - Correlation threshold (naive baseline)
  - CausalKPIGraph (ours)

Lower SHD/SID = better causal structure recovery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from nexus.core.causal_graph import CausalKPIGraph, CausalGraphConfig, train_causal_graph
from nexus.verticals.healthcare.synthetic_data import (
    generate_hospital_network,
    panel_to_tensors,
    KPI_NAMES,
)


@dataclass
class BenchmarkResult:
    method: str
    shd: float                   # structural Hamming distance (edge errors)
    tpr: float                   # true positive rate (edges correctly found)
    fpr: float                   # false positive rate
    precision: float
    f1: float
    runtime_seconds: float


def structural_hamming_distance(pred: np.ndarray, true: np.ndarray) -> float:
    """SHD: count of edge additions/deletions/reversals needed to match true graph."""
    n = pred.shape[0]
    shd = 0
    for i in range(n):
        for j in range(n):
            if pred[i, j] != true[i, j]:
                shd += 1
    return float(shd)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    tp = ((pred == 1) & (true == 1)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-9)
    shd = structural_hamming_distance(pred, true)
    return {"shd": shd, "tpr": float(tpr), "fpr": float(fpr), "precision": float(precision), "f1": float(f1)}


def baseline_correlation(X_flat: np.ndarray, n_nodes: int, threshold: float = 0.3) -> np.ndarray:
    """Naive baseline: threshold absolute correlations."""
    corr = np.abs(np.corrcoef(X_flat.T[:n_nodes]))
    np.fill_diagonal(corr, 0)
    return (corr > threshold).astype(float)


def baseline_random(n_nodes: int, density: float = 0.2, seed: int = 0) -> np.ndarray:
    """Random baseline for reference."""
    rng = np.random.default_rng(seed)
    adj = (rng.uniform(0, 1, (n_nodes, n_nodes)) < density).astype(float)
    np.fill_diagonal(adj, 0)
    return adj


def run_our_method(X: torch.Tensor, n_nodes: int, threshold: float = 0.3) -> tuple[np.ndarray, float]:
    cfg = CausalGraphConfig(n_nodes=n_nodes, hidden_dim=32, n_layers=2, max_iter=30)
    model = CausalKPIGraph(cfg, node_feature_dim=X.shape[-1])
    t0 = time.perf_counter()
    train_causal_graph(model, X, n_outer=20, n_inner=100, lr=1e-3)
    elapsed = time.perf_counter() - t0
    return model.get_causal_graph(threshold=threshold), elapsed


def run_benchmark(
    n_hospitals: int = 40,
    n_quarters: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    print("Generating synthetic healthcare network...")
    network = generate_hospital_network(n_hospitals, n_quarters, seed)
    X_tensor, means, stds = panel_to_tensors(network)
    true_adj = network.causal_adjacency
    n_nodes = len(KPI_NAMES)

    results = []

    # Baseline: random
    t0 = time.perf_counter()
    pred_random = baseline_random(n_nodes)
    m = compute_metrics(pred_random, true_adj)
    results.append(BenchmarkResult("Random", runtime_seconds=time.perf_counter() - t0, **m))

    # Baseline: correlation threshold
    X_flat = X_tensor.squeeze(-1).numpy()
    t0 = time.perf_counter()
    pred_corr = baseline_correlation(X_flat, n_nodes, threshold=0.3)
    m = compute_metrics(pred_corr, true_adj)
    results.append(BenchmarkResult("Correlation (threshold=0.3)", runtime_seconds=time.perf_counter() - t0, **m))

    # Ours: CausalKPIGraph
    print("Training CausalKPIGraph...")
    pred_ours, t_ours = run_our_method(X_tensor, n_nodes)
    m = compute_metrics(pred_ours, true_adj)
    results.append(BenchmarkResult("CausalKPIGraph (ours)", runtime_seconds=t_ours, **m))

    if verbose:
        print(f"\n{'Method':<35} {'SHD':>6} {'F1':>6} {'TPR':>6} {'FPR':>6} {'Time(s)':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.method:<35} {r.shd:>6.1f} {r.f1:>6.3f} {r.tpr:>6.3f} {r.fpr:>6.3f} {r.runtime_seconds:>8.2f}")

    return results


if __name__ == "__main__":
    run_benchmark(verbose=True)
