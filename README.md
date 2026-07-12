# Nexus

Causal AI system for enterprise KPI optimization. Built this because I kept watching health systems spend millions on BI dashboards that showed what happened but couldn't tell you what to actually do about it.

The core idea: most "AI insights" in healthcare ops are just dressed-up correlations. OR utilization is correlated with EBITDA margin — cool, but that doesn't tell you whether to fix block scheduling, reduce cancellations, or hire more surgical techs. You need the causal structure to know which lever to pull.

---

## What it does

Three things chained together:

**1. Learns the causal graph over your KPIs**

Uses a DAG-constrained GNN (based on NOTEARS-MLP) to learn which metrics actually cause other metrics to move. The acyclicity constraint forces a proper directed acyclic graph instead of just a correlation matrix. Trained on panel data across facilities.

**2. Forecasts KPI trajectories with calibrated uncertainty**

Temporal Fusion Transformer — gives you P10/P50/P90 bands instead of point estimates. Variable importance scores tell you what's actually driving each forecast. Built explicit support for static covariates (hospital size, teaching status, region) and known future inputs (calendar, scheduled events).

**3. Runs a multi-agent analysis pipeline**

Five Claude agents debate the proposed interventions before producing a recommendation:
- Planner breaks down the objective into specific KPI targets
- Analyst checks the causal logic
- Forecaster translates model output into quarterly projections
- Adversary looks for holes (metric gaming, implementation risk, data quality issues)
- Synthesizer produces the final report

Conformal prediction wrappers give distribution-free coverage guarantees on the EBITDA estimates — so when you tell a CFO "80-90% chance of $40-70M impact," that's actually statistically defensible, not vibes.

---

## Healthcare vertical

Demo uses a synthetic 40-hospital network (~$9.5B revenue) with realistic KPI distributions based on CMS/HFMA benchmarks. The synthetic data generator encodes a known ground-truth causal graph so you can actually benchmark how well the discovery algorithm does.

```
True causal structure (partial):
  staffing_ratio → nurse_overtime → patient_satisfaction → readmission_rate → penalty_cost
  or_utilization → revenue_per_bed → ebitda_margin
  supply_waste → supply_cost → ebitda_margin
```

The GNN recovers most of this from panel observations alone.

---

## Quickstart

```bash
git clone https://github.com/kingtutt52/nexus
cd nexus
pip install -e ".[dev]"

# runs without API key — skips the LLM step
python examples/healthcare_demo.py

# full pipeline with agent debate
export ANTHROPIC_API_KEY=your_key
python examples/healthcare_demo.py

# benchmark causal discovery vs baselines
python benchmarks/causal_discovery_bench.py
```

Sample output:
```
Causal order recovered: [staffing_ratio, or_utilization, ..., ebitda_margin]

Top edges:
  staffing_ratio       → nurse_overtime      (w=0.847)
  or_utilization       → revenue_per_bed     (w=0.791)
  patient_satisfaction → readmission_rate    (w=0.683)

EBITDA simulation (50k scenarios):
  or_utilization_5pct          P50: $62.4M   P10: $28M   P90: $98M
  staffing_optimization        P50: $44.1M   P10: $18M   P90: $71M
  readmission_reduction_15pct  P50: $31.7M   P10: $11M   P90: $54M
```

---

## Structure

```
nexus/
├── core/
│   ├── causal_graph.py       # DAG-GNN with NOTEARS acyclicity
│   ├── temporal_fusion.py    # TFT for multi-horizon forecasting
│   ├── conformal.py          # distribution-free prediction intervals
│   └── ebitda_engine.py      # Monte Carlo simulation + portfolio optimizer
├── agents/
│   └── orchestrator.py       # 5-agent debate pipeline
├── llm/
│   ├── client.py             # Anthropic API with prompt caching
│   └── prompts.py            # agent system prompts
└── verticals/
    └── healthcare/
        ├── synthetic_data.py  # 40-hospital panel generator
        └── or_optimizer.py    # OR utilization analysis
```

---

## Known limitations / TODO

- Causal discovery benchmarks only compare against correlation and random baselines right now — need to add PC algorithm and GES comparisons (requires causal-learn dependency)
- TFT training loop not included yet, only the architecture — currently using pretrained weights or running with random init for demo purposes
- Aerospace vertical is stubbed out, haven't built the KPI schema yet
- No visualization layer yet (planned: NetworkX for causal graph, Plotly for forecast bands)
- Conformal calibration requires held-out data — the demo uses the training set which is technically wrong, just wanted to show the interface

---

## Stack

Python 3.10+, PyTorch 2.2, Anthropic SDK, NumPy, Pandas, SciPy

---

## References

- Zheng et al. (2018). DAGs with NO TEARS. NeurIPS.
- Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
- Gibbs & Candes (2021). Adaptive Conformal Inference Under Distribution Shift. NeurIPS.
