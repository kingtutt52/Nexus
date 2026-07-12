"""
Realistic synthetic healthcare KPI data generator.

Generates statistically plausible data for a 40-hospital network
based on published CMS benchmarks and HFMA financial data.
Includes realistic seasonal patterns, facility-level heterogeneity,
and causal structure between KPIs.

True causal graph (ground truth for benchmarking):
  staffing_ratio → nurse_overtime → patient_satisfaction → readmission_rate
  staffing_ratio → agency_cost
  or_utilization → revenue_per_bed → ebitda_margin
  supply_waste → supply_cost → ebitda_margin
  readmission_rate → penalty_cost → ebitda_margin
  case_mix_index → revenue_per_bed
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Ground truth causal adjacency (for benchmarking causal discovery)
TRUE_CAUSAL_GRAPH = {
    "staffing_ratio": ["nurse_overtime", "agency_cost", "patient_satisfaction"],
    "nurse_overtime": ["patient_satisfaction"],
    "patient_satisfaction": ["readmission_rate"],
    "readmission_rate": ["penalty_cost"],
    "or_utilization": ["revenue_per_bed"],
    "case_mix_index": ["revenue_per_bed"],
    "revenue_per_bed": ["ebitda_margin"],
    "supply_waste": ["supply_cost"],
    "supply_cost": ["ebitda_margin"],
    "penalty_cost": ["ebitda_margin"],
    "agency_cost": ["ebitda_margin"],
}

KPI_NAMES = [
    "staffing_ratio",       # nurses per occupied bed
    "nurse_overtime",       # % overtime hours
    "agency_cost",          # % of labor cost from agency/travel nurses
    "patient_satisfaction", # HCAHPS score 0-100
    "readmission_rate",     # 30-day all-cause readmission %
    "penalty_cost",         # CMS readmission penalty ($M)
    "or_utilization",       # % OR time used (vs scheduled)
    "case_mix_index",       # DRG case mix index
    "revenue_per_bed",      # annual revenue per staffed bed ($K)
    "supply_waste",         # % supply budget wasted
    "supply_cost",          # supply cost as % of revenue
    "ebitda_margin",        # EBITDA / Revenue
]

# Realistic baseline ranges (from HFMA/CMS benchmarks)
BASELINES = {
    "staffing_ratio": (4.0, 0.5),          # nurses per bed, mean ± std
    "nurse_overtime": (0.12, 0.04),        # 12% overtime typical
    "agency_cost": (0.18, 0.06),           # 18% agency labor (post-COVID)
    "patient_satisfaction": (72.0, 8.0),   # HCAHPS percentile
    "readmission_rate": (0.155, 0.025),    # 15.5% national average
    "penalty_cost": (1.2, 0.8),            # $1.2M CMS penalties avg
    "or_utilization": (0.68, 0.08),        # 68% typical
    "case_mix_index": (1.65, 0.25),        # 1.65 typical community hospital
    "revenue_per_bed": (1850.0, 350.0),    # $1.85M per staffed bed/year
    "supply_waste": (0.22, 0.06),          # 22% supply waste typical
    "supply_cost": (0.18, 0.03),           # 18% of revenue
    "ebitda_margin": (0.06, 0.03),         # 6% EBITDA margin (tight)
}


@dataclass
class HospitalNetwork:
    n_hospitals: int
    n_quarters: int
    facilities: pd.DataFrame
    kpi_panel: pd.DataFrame          # (hospitals × quarters) panel
    causal_adjacency: np.ndarray     # ground truth for benchmarking


def generate_hospital_network(
    n_hospitals: int = 40,
    n_quarters: int = 20,
    seed: int = 42,
) -> HospitalNetwork:
    """
    Generate a realistic panel dataset for a hospital network.

    Structure:
      - 40 hospitals, each with fixed effects (size, region, type)
      - 20 quarters of KPI history (5 years)
      - Causal relationships encoded in data generating process
      - Seasonal patterns (Q4 higher volume, summer lower)
      - Trend: staffing crisis 2021-2022, recovery 2023+
    """
    rng = np.random.default_rng(seed)
    n_kpis = len(KPI_NAMES)

    # ── Facility metadata ─────────────────────────────────────────────────────
    facilities = pd.DataFrame({
        "hospital_id": range(n_hospitals),
        "beds": rng.integers(100, 600, n_hospitals),
        "region": rng.choice(["Southeast", "Midwest", "Southwest", "Mid-Atlantic"], n_hospitals),
        "type": rng.choice(["community", "regional", "critical_access"], n_hospitals, p=[0.6, 0.3, 0.1]),
        "teaching": rng.choice([True, False], n_hospitals, p=[0.15, 0.85]),
    })

    # ── Hospital fixed effects ─────────────────────────────────────────────────
    hospital_effects = {}
    for kpi in KPI_NAMES:
        mean, std = BASELINES[kpi]
        hospital_effects[kpi] = rng.normal(0, std * 0.3, n_hospitals)  # facility heterogeneity

    # ── Seasonal factors ───────────────────────────────────────────────────────
    quarters = np.arange(n_quarters)
    season = np.sin(2 * np.pi * quarters / 4)  # annual cycle
    trend = np.linspace(0, 0.05, n_quarters)   # slight upward trend in most KPIs

    # ── Staffing crisis shock (quarters 4-8 ~ 2021-2022) ──────────────────────
    crisis = np.zeros(n_quarters)
    crisis[4:9] = 1.0   # staffing crisis period

    # ── Generate panel with causal structure ──────────────────────────────────
    panel_records = []
    for h in range(n_hospitals):
        size_factor = facilities.loc[h, "beds"] / 300.0  # relative size
        teaching_bonus = 0.15 if facilities.loc[h, "teaching"] else 0.0

        for q in range(n_quarters):
            row = {"hospital_id": h, "quarter": q}

            # Layer 1: exogenous KPIs
            row["staffing_ratio"] = max(2.5, rng.normal(
                BASELINES["staffing_ratio"][0] + hospital_effects["staffing_ratio"][h]
                - 0.3 * crisis[q]  # crisis reduces staffing ratio
                + 0.05 * season[q],
                BASELINES["staffing_ratio"][1] * 0.2
            ))
            row["case_mix_index"] = max(0.8, rng.normal(
                BASELINES["case_mix_index"][0] + teaching_bonus + hospital_effects["case_mix_index"][h],
                BASELINES["case_mix_index"][1] * 0.15
            ))
            row["or_utilization"] = np.clip(rng.normal(
                BASELINES["or_utilization"][0] + hospital_effects["or_utilization"][h]
                + 0.03 * season[q] - 0.05 * crisis[q],
                BASELINES["or_utilization"][1] * 0.2
            ), 0.3, 0.95)

            # Layer 2: staffing → overtime + agency
            staffing_deficit = max(0, 4.0 - row["staffing_ratio"])
            row["nurse_overtime"] = np.clip(rng.normal(
                BASELINES["nurse_overtime"][0]
                + 0.08 * staffing_deficit
                + 0.04 * crisis[q]
                + hospital_effects["nurse_overtime"][h],
                0.02
            ), 0.0, 0.5)
            row["agency_cost"] = np.clip(rng.normal(
                BASELINES["agency_cost"][0]
                + 0.10 * staffing_deficit
                + 0.06 * crisis[q]
                + hospital_effects["agency_cost"][h],
                0.02
            ), 0.0, 0.6)

            # Layer 3: staffing/overtime → patient satisfaction
            row["patient_satisfaction"] = np.clip(rng.normal(
                BASELINES["patient_satisfaction"][0]
                - 15 * row["nurse_overtime"]
                + 5.0 * (row["staffing_ratio"] - 4.0)
                + hospital_effects["patient_satisfaction"][h],
                3.0
            ), 20, 99)

            # Layer 4: satisfaction → readmission
            row["readmission_rate"] = np.clip(rng.normal(
                BASELINES["readmission_rate"][0]
                - 0.001 * (row["patient_satisfaction"] - 72)
                + hospital_effects["readmission_rate"][h],
                0.005
            ), 0.05, 0.35)

            # Layer 5: readmission → CMS penalty
            excess_readmit = max(0, row["readmission_rate"] - 0.15)
            row["penalty_cost"] = max(0, rng.normal(
                excess_readmit * 20 * (facilities.loc[h, "beds"] / 300),
                0.2
            ))

            # Supply chain
            row["supply_waste"] = np.clip(rng.normal(
                BASELINES["supply_waste"][0] + hospital_effects["supply_waste"][h],
                0.02
            ), 0.05, 0.5)
            row["supply_cost"] = np.clip(
                BASELINES["supply_cost"][0] * (1 + 0.3 * row["supply_waste"]),
                0.10, 0.35
            )

            # Revenue per bed (driven by OR utilization + case mix)
            row["revenue_per_bed"] = max(500, rng.normal(
                BASELINES["revenue_per_bed"][0]
                * (0.5 + 0.5 * row["or_utilization"])
                * (row["case_mix_index"] / 1.65)
                * size_factor,
                100
            ))

            # EBITDA margin (final downstream KPI)
            revenue = row["revenue_per_bed"] * facilities.loc[h, "beds"]
            labor_cost_pct = 0.52 + 0.15 * row["agency_cost"] + 0.08 * row["nurse_overtime"]
            row["ebitda_margin"] = np.clip(
                1.0 - labor_cost_pct - row["supply_cost"] - 0.12 - row["penalty_cost"] / max(revenue / 1e6, 1),
                -0.15, 0.25
            )

            panel_records.append(row)

    panel = pd.DataFrame(panel_records)

    # ── Build ground truth adjacency matrix ───────────────────────────────────
    adj = np.zeros((n_kpis, n_kpis))
    kpi_idx = {k: i for i, k in enumerate(KPI_NAMES)}
    for parent, children in TRUE_CAUSAL_GRAPH.items():
        for child in children:
            adj[kpi_idx[parent], kpi_idx[child]] = 1.0

    return HospitalNetwork(
        n_hospitals=n_hospitals,
        n_quarters=n_quarters,
        facilities=facilities,
        kpi_panel=panel,
        causal_adjacency=adj,
    )


def panel_to_tensors(network: HospitalNetwork):
    """Convert panel DataFrame → tensors suitable for CausalKPIGraph."""
    import torch
    panel = network.kpi_panel
    hospitals = sorted(panel["hospital_id"].unique())
    quarters = sorted(panel["quarter"].unique())

    # Shape: (n_hospitals * n_quarters, n_kpis, 1) for node features
    kpi_data = []
    for h in hospitals:
        h_data = panel[panel["hospital_id"] == h].sort_values("quarter")
        kpi_data.append(h_data[KPI_NAMES].values)

    arr = np.stack(kpi_data)  # (n_hospitals, n_quarters, n_kpis)

    # Normalize per-KPI
    means = arr.mean(axis=(0, 1), keepdims=True)
    stds = arr.std(axis=(0, 1), keepdims=True) + 1e-8
    arr_norm = (arr - means) / stds

    # Return as (batch, n_kpis, 1) — each time step is a sample
    B = arr_norm.shape[0] * arr_norm.shape[1]
    X = arr_norm.reshape(B, len(KPI_NAMES), 1)
    return torch.FloatTensor(X), means.squeeze(), stds.squeeze()
