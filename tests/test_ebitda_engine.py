import pytest
import numpy as np
from nexus.core.ebitda_engine import EBITDAEngine, LIFEPOINT_BASELINE, HEALTHCARE_INTERVENTION_LIBRARY


@pytest.fixture
def engine():
    return EBITDAEngine(**LIFEPOINT_BASELINE)


def test_simulate_returns_valid_result(engine):
    impacts = HEALTHCARE_INTERVENTION_LIBRARY["or_utilization_5pct"]
    result = engine.simulate_intervention(impacts)
    assert result.p10 <= result.p50 <= result.p90
    assert 0 <= result.prob_positive <= 1
    assert result.std > 0


def test_compare_interventions_sorted(engine):
    results = engine.compare_interventions(HEALTHCARE_INTERVENTION_LIBRARY)
    p50s = [r.p50 for r in results.values()]
    assert p50s == sorted(p50s, reverse=True)


def test_portfolio_respects_budget(engine):
    results = engine.compare_interventions(HEALTHCARE_INTERVENTION_LIBRARY)
    costs = {k: 10_000_000 for k in HEALTHCARE_INTERVENTION_LIBRARY}
    portfolio = engine.portfolio_optimization(results, budget=25_000_000, costs=costs)
    total_cost = sum(i["cost"] for i in portfolio["selected"])
    assert total_cost <= 25_000_000


def test_mc_samples_coverage(engine):
    impacts = HEALTHCARE_INTERVENTION_LIBRARY["staffing_optimization"]
    result = engine.simulate_intervention(impacts, implementation_success_rate=1.0)
    assert result.prob_positive > 0.5   # should be positive with full success rate
