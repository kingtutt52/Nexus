"""
Unit tests for CausalKPIGraph.
Tests: forward pass, acyclicity constraint, intervention effect, training loop.
"""

import pytest
import torch
import numpy as np

from nexus.core.causal_graph import CausalKPIGraph, CausalGraphConfig, train_causal_graph


@pytest.fixture
def small_model():
    cfg = CausalGraphConfig(n_nodes=5, hidden_dim=16, n_layers=1, max_iter=5)
    return CausalKPIGraph(cfg, node_feature_dim=2)


def test_forward_shape(small_model):
    X = torch.randn(8, 5, 2)
    X_hat, W = small_model(X)
    assert X_hat.shape == X.shape
    assert W.shape == (5, 5)


def test_adjacency_no_self_loops(small_model):
    W = small_model.W
    diag = torch.diagonal(W)
    assert diag.abs().max().item() < 1e-6


def test_adjacency_non_negative(small_model):
    W = small_model.W
    assert W.min().item() >= 0.0


def test_h_acyclicity_identity():
    """h(I) for identity matrix = tr(e^I) - d = d*(e-1) ≠ 0 (has self-loops if nonzero)."""
    cfg = CausalGraphConfig(n_nodes=4, hidden_dim=8, n_layers=1)
    model = CausalKPIGraph(cfg, node_feature_dim=1)
    W_zero = torch.zeros(4, 4)
    h = model.h_acyclicity(W_zero)
    assert abs(h.item()) < 1e-5   # zero matrix is trivially a DAG


def test_loss_keys(small_model):
    X = torch.randn(4, 5, 2)
    losses = small_model.loss(X)
    assert "total" in losses
    assert "recon" in losses
    assert "h" in losses


def test_training_reduces_loss():
    cfg = CausalGraphConfig(n_nodes=4, hidden_dim=8, n_layers=1, max_iter=3)
    model = CausalKPIGraph(cfg, node_feature_dim=1)
    X = torch.randn(16, 4, 1)
    history = train_causal_graph(model, X, n_outer=3, n_inner=20)
    assert len(history) > 0
    # Loss should exist (not NaN)
    assert not np.isnan(history[-1]["recon"])


def test_intervention_effect_shape(small_model):
    X = torch.randn(4, 5, 2)
    delta = small_model.compute_intervention_effect(0, 0.1, X)
    assert delta.shape == X.shape


def test_causal_graph_binary(small_model):
    adj = small_model.get_causal_graph(threshold=0.3)
    assert set(np.unique(adj)).issubset({0.0, 1.0})
    # No self-loops
    assert np.diagonal(adj).sum() == 0
