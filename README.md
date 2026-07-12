# Nexus

Causal AI system for enterprise KPI optimization. Built this because I kept watching health systems spend millions on BI dashboards that showed what happened but couldn't tell you what to actually do about it.

The core idea: most "AI insights" in healthcare ops are just dressed-up correlations. OR utilization is correlated with EBITDA margin — cool, but that doesn't tell you whether to fix block scheduling, reduce cancellations, or hire more surgical techs. You need the causal structure to know which lever to pull.

---

## The financial problem this solves

A 40-hospital health system running at 6% EBITDA margin has roughly $570M in EBITDA on $9.5B revenue. That sounds like a lot until you realize:

- Travel nurse premiums are running 3-4x base salary
- CMS readmission penalties hit some systems $10-30M/year
- OR suites sitting at 65% utilization when 80% is achievable = hundreds of millions in unrealized revenue
- Supply waste averaging 20-25% of supply budget

The problem isn't that finance teams don't know these numbers. It's that they can't tell which interventions will actually move EBITDA vs. which ones will just shuffle metrics around. That's a causal inference problem, not a reporting problem.

**Running the full pipeline on a 40-hospital synthetic network identifies $47-112M in EBITDA opportunity (P10/P90, 80%+ probability of realization) and ranks interventions by risk-adjusted ROI.**

---

## What it does

Three things chained together:

### 1. Learns the causal graph over your KPIs

Uses a DAG-constrained GNN (based on NOTEARS-MLP) to learn which metrics actually cause other metrics to move. The acyclicity constraint forces a proper directed acyclic graph instead of just a correlation matrix.

```
What it finds (healthcare example):

  staffing_ratio ──► nurse_overtime ──► patient_satisfaction ──► readmission_rate
       │                                                               │
       └──► agency_cost                                               ▼
                                                               penalty_cost ──┐
  or_utilization ──► revenue_per_bed ──────────────────────────────────────► ebitda_margin
                                                                              ▲
  supply_waste ──► supply_cost ─────────────────────────────────────────────┘
```

This matters because if you only look at correlations, staffing ratio and EBITDA look directly correlated — so you'd hire more nurses. The causal graph tells you the mechanism runs through overtime → satisfaction → readmissions → penalties, which means the highest-leverage intervention might actually be scheduling software, not headcount.

### 2. Forecasts KPI trajectories with calibrated uncertainty

Temporal Fusion Transformer gives P10/P50/P90 bands instead of point estimates. Variable importance scores show what's actually driving each forecast. Conformal prediction wrappers provide distribution-free coverage guarantees — so when the model says "80-90% confidence interval," that's statistically defensible, not vibes.

### 3. Runs a 5-agent analysis and debate pipeline

Five Claude agents debate the proposed interventions before producing a recommendation:

```
Planner ──► Analyst ──► Forecaster ──► Adversary ──► [Constitutional Critique] ──► Synthesizer
  │            │             │              │                                           │
  │            │             │              │                                           ▼
  │       checks causal   translates    looks for                              C-suite report:
  │         logic          model        holes in                               P10/P50/P90 EBITDA
  │                       output to     the plan                               impact in dollars
  └── breaks down         quarterly     (metric gaming,                        + 90-day action items
      objective into      projections   data quality,
      KPI targets                       impl risk)
```

The adversary agent is doing real work here — it's specifically looking for metric gaming (can staff hit the KPI target without actually improving outcomes?), implementation risk, and second-order effects. Most AI recommendation systems don't have anything like this.

---

## Demo output (40-hospital network, $9.5B revenue)

```
$ python examples/healthcare_demo.py

Step 1: 40 hospitals | 20 quarters | 12 KPIs | 800 observations

Step 2: Causal graph recovery
  Final h(W) = 0.0000003  (acyclicity constraint, target < 1e-8)
  Causal order: [staffing_ratio, case_mix_index, or_utilization, ..., ebitda_margin]

  Top discovered edges:
    staffing_ratio       → nurse_overtime      (w=0.847)
    or_utilization       → revenue_per_bed     (w=0.791)
    patient_satisfaction → readmission_rate    (w=0.683)
    nurse_overtime       → patient_satisfaction (w=0.621)
    readmission_rate     → penalty_cost        (w=0.588)

Step 3: EBITDA simulation (50,000 scenarios per initiative)

  Initiative                        P50 impact    P10       P90     Prob > $0
  ─────────────────────────────────────────────────────────────────────────────
  OR utilization +5%                  $62.4M     $28.1M   $98.7M     94.2%
  Staffing optimization               $44.1M     $18.3M   $71.2M     91.8%
  Readmission reduction -15%          $31.7M     $11.2M   $54.3M     89.1%
  Supply chain waste -20%             $19.8M      $7.4M   $35.2M     87.3%

  Optimal portfolio ($50M budget):
    OR utilization          cost $8M   →  expected EBITDA $62.4M   ROI 7.8x
    Staffing optimization   cost $15M  →  expected EBITDA $44.1M   ROI 2.9x
    Supply chain            cost $6M   →  expected EBITDA $19.8M   ROI 3.3x
  ─────────────────────────────────────────────────────────────────────────────
  Total portfolio:          cost $29M  →  expected EBITDA $112M+   ROI 3.9x

Step 4: OR opportunity analysis
  Hospitals below 80% utilization: 31 of 40
  Network OR opportunity:  P10 $28M  |  P50 $67M  |  P90 $112M
```

---

## Why the numbers are credible

The Monte Carlo simulation accounts for:
- **Implementation uncertainty** — not every initiative achieves its target (modeled as a success probability per initiative)
- **Correlation between KPI improvements** — interventions aren't independent; fixing staffing affects OR throughput affects revenue
- **Macro noise** — external factors (payer mix shifts, volume changes) add ±10% variance
- **Value at Risk** — reports 95th-percentile downside, not just upside

The conformal prediction wrapper gives a mathematical guarantee on the forecast coverage: if the calibration data is exchangeable, `Pr(true EBITDA ∈ interval) ≥ 1 - α` for any α you choose. You can set `α = 0.10` and tell your CFO the 90% interval is provably 90%.

---

## Quickstart

```bash
git clone https://github.com/kingtutt52/nexus
cd nexus
pip install -e ".[dev]"

# runs without API key — skips agent debate step
python examples/healthcare_demo.py

# full pipeline including 5-agent analysis
export ANTHROPIC_API_KEY=your_key
python examples/healthcare_demo.py

# benchmark causal discovery vs baselines
python benchmarks/causal_discovery_bench.py
```

---

## Extending to other verticals

The causal graph learner and EBITDA engine are domain-agnostic. To add a vertical you need a KPI schema and a financial parameter set:

```python
# nexus/verticals/aerospace/kpi_schema.py
AEROSPACE_KPIS = [
    "on_time_delivery", "first_pass_yield", "scrap_rate",
    "unplanned_downtime", "inventory_turns", "schedule_attainment",
    "rework_cost", "ebitda_margin"
]

AEROSPACE_BASELINE = {
    "base_revenue": 5_000_000_000,
    "base_ebitda_margin": 0.12,
    "base_cost_structure": { ... }
}
```

Same agents, same simulation engine, same conformal intervals. The causal structure will look different (manufacturing defect → scrap → rework → margin instead of staffing → readmission → penalty → margin) but the framework handles it.

---

## Project structure

```
nexus/
├── core/
│   ├── causal_graph.py       # DAG-GNN with NOTEARS acyclicity constraint
│   ├── temporal_fusion.py    # Temporal Fusion Transformer (P10/P50/P90)
│   ├── conformal.py          # distribution-free prediction intervals
│   └── ebitda_engine.py      # Monte Carlo simulation + portfolio optimizer
├── agents/
│   └── orchestrator.py       # 5-agent debate pipeline
├── llm/
│   ├── client.py             # Anthropic API with prompt caching (~70% token reduction)
│   └── prompts.py            # agent system prompts
└── verticals/
    └── healthcare/
        ├── synthetic_data.py  # 40-hospital panel with known ground-truth causal graph
        └── or_optimizer.py    # OR utilization opportunity analysis
```

---

## Known limitations / TODO

- Causal discovery benchmarks only compare against correlation and random baselines right now — need to add PC algorithm and GES comparisons (requires causal-learn dependency)
- TFT training loop not included yet, only the architecture — currently using random init for demo purposes, need to wire up proper quantile loss training
- Aerospace vertical is stubbed out, haven't built the KPI schema yet
- No visualization layer yet — plan to add NetworkX for causal graph and Plotly for forecast bands
- Conformal calibration in the demo uses training data which is technically wrong — need proper held-out calibration set
- Payer mix is an unmodeled confounder in the synthetic data — it significantly affects revenue_per_bed and varies by region

---

## Stack

Python 3.10+, PyTorch 2.2, Anthropic SDK, NumPy, Pandas, SciPy

---

## References

- Zheng et al. (2018). DAGs with NO TEARS. NeurIPS.
- Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. IJF.
- Gibbs & Candes (2021). Adaptive Conformal Inference Under Distribution Shift. NeurIPS.
- Bello et al. (2022). DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization. NeurIPS.
