"""
Monte Carlo EBITDA simulation engine.

Translates model outputs (KPI delta distributions) into
financial impact distributions. Key for board-level communication:
"This initiative has 80% probability of adding $8-12M EBITDA."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class EBITDAScenario:
    name: str
    kpi_deltas: dict[str, float]          # KPI name → % change
    probability: float                     # scenario probability
    time_horizon_quarters: int = 4


@dataclass
class EBITDASimResult:
    mean: float
    std: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    var_95: float                          # Value at Risk (5th percentile)
    cvar_95: float                         # Conditional VaR
    prob_positive: float
    scenarios: list[EBITDAScenario] = field(default_factory=list)
    samples: Optional[np.ndarray] = None  # raw MC samples (for viz)


class EBITDAEngine:
    """
    Monte Carlo EBITDA impact simulator.

    For a given set of interventions (from the agent system),
    simulates the distribution of EBITDA impact accounting for:
      - Implementation uncertainty (will it actually achieve the forecast?)
      - Market uncertainty (external factors)
      - Correlation between KPI improvements
      - Time-to-value (initiatives take time to show EBITDA impact)
    """

    def __init__(
        self,
        base_revenue: float,
        base_ebitda_margin: float,
        base_cost_structure: dict[str, float],
        n_simulations: int = 50_000,
        seed: int = 42,
    ):
        self.base_revenue = base_revenue
        self.base_ebitda_margin = base_ebitda_margin
        self.base_cost_structure = base_cost_structure
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_intervention(
        self,
        kpi_impacts: dict[str, dict],
        correlation_matrix: Optional[np.ndarray] = None,
        implementation_success_rate: float = 0.75,
    ) -> EBITDASimResult:
        """
        kpi_impacts: {
            "OR_utilization": {
                "mean_delta": 0.05,     # +5% improvement
                "std": 0.02,            # uncertainty
                "revenue_coefficient": 0.8,   # how much revenue moves per % KPI
                "cost_coefficient": -0.3,     # negative = cost reduction
            },
            ...
        }
        """
        kpis = list(kpi_impacts.keys())
        n_kpis = len(kpis)
        means = np.array([kpi_impacts[k]["mean_delta"] for k in kpis])
        stds = np.array([kpi_impacts[k]["std"] for k in kpis])

        # Correlated KPI improvement samples
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_kpis) + 0.2 * (1 - np.eye(n_kpis))
        L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(n_kpis))
        Z = self.rng.standard_normal((self.n_simulations, n_kpis))
        delta_samples = means + (Z @ L.T) * stds

        # Implementation uncertainty: did the initiative succeed?
        success = self.rng.uniform(0, 1, self.n_simulations) < implementation_success_rate
        delta_samples *= success[:, None]

        # Translate KPI deltas → revenue and cost impacts
        revenue_impact = np.zeros(self.n_simulations)
        cost_impact = np.zeros(self.n_simulations)
        for i, k in enumerate(kpis):
            r_coeff = kpi_impacts[k].get("revenue_coefficient", 0.0)
            c_coeff = kpi_impacts[k].get("cost_coefficient", 0.0)
            revenue_impact += delta_samples[:, i] * r_coeff * self.base_revenue
            cost_impact += delta_samples[:, i] * c_coeff * sum(self.base_cost_structure.values())

        # EBITDA impact = revenue gain - cost increase (+ means cost reduction)
        ebitda_samples = (revenue_impact - cost_impact) * self.base_ebitda_margin

        # Add macro uncertainty (±10% noise from market conditions)
        macro_noise = self.rng.normal(1.0, 0.10, self.n_simulations)
        ebitda_samples *= macro_noise

        sorted_samples = np.sort(ebitda_samples)
        n = len(sorted_samples)
        var_95 = float(np.percentile(sorted_samples, 5))
        cvar_95 = float(sorted_samples[:int(n * 0.05)].mean())

        return EBITDASimResult(
            mean=float(ebitda_samples.mean()),
            std=float(ebitda_samples.std()),
            p10=float(np.percentile(ebitda_samples, 10)),
            p25=float(np.percentile(ebitda_samples, 25)),
            p50=float(np.percentile(ebitda_samples, 50)),
            p75=float(np.percentile(ebitda_samples, 75)),
            p90=float(np.percentile(ebitda_samples, 90)),
            var_95=var_95,
            cvar_95=cvar_95,
            prob_positive=float((ebitda_samples > 0).mean()),
            samples=ebitda_samples,
        )

    def compare_interventions(
        self,
        interventions: dict[str, dict[str, dict]],
    ) -> dict[str, EBITDASimResult]:
        """Rank multiple proposed interventions by expected EBITDA impact."""
        results = {}
        for name, impacts in interventions.items():
            results[name] = self.simulate_intervention(impacts)
        return dict(sorted(results.items(), key=lambda x: -x[1].p50))

    def portfolio_optimization(
        self,
        interventions: dict[str, EBITDASimResult],
        budget: float,
        costs: dict[str, float],
        max_initiatives: int = 5,
    ) -> dict[str, list]:
        """
        Knapsack-style portfolio optimization:
        maximize expected EBITDA subject to budget and initiative count.
        Returns ranked selection with Sharpe-like ratio (EBITDA/risk).
        """
        selected = []
        remaining_budget = budget

        scored = {
            name: (res.p50 / (res.std + 1e-6), res.p50, costs.get(name, 0))
            for name, res in interventions.items()
        }
        ranked = sorted(scored.items(), key=lambda x: -x[1][0])

        for name, (sharpe, ebitda, cost) in ranked:
            if len(selected) >= max_initiatives:
                break
            if cost <= remaining_budget:
                selected.append({
                    "initiative": name,
                    "expected_ebitda": ebitda,
                    "cost": cost,
                    "ebitda_roi": ebitda / max(cost, 1),
                    "sharpe_ratio": sharpe,
                })
                remaining_budget -= cost

        return {"selected": selected, "remaining_budget": remaining_budget}


# ── Healthcare-specific financial parameters ──────────────────────────────────

LIFEPOINT_BASELINE = {
    "base_revenue": 9_500_000_000,         # ~$9.5B (40-hospital network)
    "base_ebitda_margin": 0.08,            # 8% EBITDA margin typical for health systems
    "base_cost_structure": {
        "labor": 5_500_000_000,
        "supplies": 1_200_000_000,
        "overhead": 800_000_000,
        "clinical_operations": 1_400_000_000,
    },
}

HEALTHCARE_INTERVENTION_LIBRARY = {
    "or_utilization_5pct": {
        "OR_utilization": {
            "mean_delta": 0.05,
            "std": 0.02,
            "revenue_coefficient": 0.90,   # OR drives ~90% of facility revenue
            "cost_coefficient": 0.05,      # marginal cost increase
        }
    },
    "readmission_reduction_15pct": {
        "readmission_rate": {
            "mean_delta": -0.15,
            "std": 0.04,
            "revenue_coefficient": 0.02,   # slight revenue dip: delta(-) * coeff(+) = negative impact
            "cost_coefficient": 0.12,      # cost reduction: delta(-) * coeff(+) = negative cost_impact
        }
    },
    "supply_chain_waste_20pct": {
        "supply_waste": {
            "mean_delta": -0.20,
            "std": 0.05,
            "revenue_coefficient": 0.0,
            "cost_coefficient": 0.20,      # delta(-) * coeff(+) = negative cost_impact (costs down)
        }
    },
    "staffing_optimization": {
        "nurse_overtime_pct": {
            "mean_delta": -0.25,
            "std": 0.07,
            "revenue_coefficient": 0.0,
            "cost_coefficient": 0.15,      # delta(-) * coeff(+) = cost reduction
        },
        "agency_staff_pct": {
            "mean_delta": -0.30,
            "std": 0.08,
            "revenue_coefficient": 0.0,
            "cost_coefficient": 0.08,      # delta(-) * coeff(+) = cost reduction
        },
    },
}
