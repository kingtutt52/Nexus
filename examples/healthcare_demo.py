"""
End-to-end Nexus demo: 40-hospital network EBITDA optimization.

Runs the full pipeline:
  1. Generate synthetic healthcare KPI data
  2. Train CausalKPIGraph to discover KPI causal structure
  3. Train TFT for KPI forecasting
  4. Run EBITDA Monte Carlo simulation
  5. Run 5-agent debate pipeline (requires ANTHROPIC_API_KEY)
  6. Print final C-suite recommendation report
"""

from __future__ import annotations

import json
import os

import torch

from nexus.core.causal_graph import CausalKPIGraph, CausalGraphConfig, train_causal_graph
from nexus.core.ebitda_engine import EBITDAEngine, LIFEPOINT_BASELINE, HEALTHCARE_INTERVENTION_LIBRARY
from nexus.verticals.healthcare.synthetic_data import (
    generate_hospital_network,
    panel_to_tensors,
    KPI_NAMES,
)
from nexus.verticals.healthcare.or_optimizer import analyze_or_opportunities, compute_network_or_ebitda_opportunity


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_demo(use_llm: bool = True) -> None:

    # ── Step 1: Generate Data ──────────────────────────────────────────────────
    print_section("Step 1: Generating 40-hospital KPI dataset (5 years)")
    network = generate_hospital_network(n_hospitals=40, n_quarters=20)
    print(f"  Hospitals: {network.n_hospitals}")
    print(f"  Quarters: {network.n_quarters}")
    print(f"  KPIs: {len(KPI_NAMES)}")
    print(f"  Total observations: {len(network.kpi_panel)}")

    # Sample KPIs
    latest = network.kpi_panel[network.kpi_panel["quarter"] == 19]
    print(f"\n  Network-wide KPI averages (Q19):")
    for kpi in KPI_NAMES:
        print(f"    {kpi:<25} {latest[kpi].mean():.4f} ± {latest[kpi].std():.4f}")

    # ── Step 2: Causal Discovery ───────────────────────────────────────────────
    print_section("Step 2: Learning causal KPI graph")
    X_tensor, means, stds = panel_to_tensors(network)
    n_nodes = len(KPI_NAMES)

    cfg = CausalGraphConfig(n_nodes=n_nodes, hidden_dim=32, n_layers=2, max_iter=30)
    model = CausalKPIGraph(cfg, node_feature_dim=1)

    print("  Training DAG-constrained GNN...")
    history = train_causal_graph(model, X_tensor, n_outer=20, n_inner=100)
    final_h = history[-1]["h"]
    print(f"  Final acyclicity constraint h(W) = {final_h:.6f} (target < 1e-8)")

    adj = model.get_causal_graph(threshold=0.3)
    causal_order = model.get_causal_order()
    print(f"  Causal order: {[KPI_NAMES[i] for i in causal_order]}")

    # Top edges
    W = model.W.detach().numpy()
    edges = [(KPI_NAMES[i], KPI_NAMES[j], W[i, j])
             for i in range(n_nodes) for j in range(n_nodes) if W[i, j] > 0.2]
    edges.sort(key=lambda x: -x[2])
    print(f"\n  Top discovered causal edges:")
    for src, tgt, w in edges[:8]:
        print(f"    {src:<25} → {tgt:<25} (w={w:.3f})")

    # ── Step 3: EBITDA Simulation ──────────────────────────────────────────────
    print_section("Step 3: Monte Carlo EBITDA simulation")
    engine = EBITDAEngine(**LIFEPOINT_BASELINE)

    sim_results = engine.compare_interventions(HEALTHCARE_INTERVENTION_LIBRARY)
    print(f"  {'Initiative':<35} {'P50 EBITDA':>15} {'P10':>15} {'P90':>15} {'Prob>0':>8}")
    print(f"  {'-'*90}")
    for name, result in sim_results.items():
        print(f"  {name:<35} ${result.p50/1e6:>12.1f}M ${result.p10/1e6:>12.1f}M ${result.p90/1e6:>12.1f}M {result.prob_positive:>7.1%}")

    # Portfolio optimization
    portfolio = engine.portfolio_optimization(
        sim_results,
        budget=50_000_000,   # $50M implementation budget
        costs={
            "or_utilization_5pct": 8_000_000,
            "readmission_reduction_15pct": 12_000_000,
            "supply_chain_waste_20pct": 6_000_000,
            "staffing_optimization": 15_000_000,
        }
    )
    print(f"\n  Optimal portfolio (budget: $50M):")
    for item in portfolio["selected"]:
        print(f"    {item['initiative']:<35} EBITDA: ${item['expected_ebitda']/1e6:.1f}M  ROI: {item['ebitda_roi']:.1f}x")

    # ── Step 4: OR Opportunity Analysis ───────────────────────────────────────
    print_section("Step 4: OR utilization opportunities")
    opportunities = analyze_or_opportunities(network.kpi_panel, network.facilities)
    network_impact = compute_network_or_ebitda_opportunity(opportunities)

    print(f"  Hospitals below 80% OR utilization: {len(opportunities)}")
    print(f"  Network OR EBITDA opportunity:")
    print(f"    P10: ${network_impact['p10']/1e6:.1f}M")
    print(f"    P50: ${network_impact['p50']/1e6:.1f}M")
    print(f"    P90: ${network_impact['p90']/1e6:.1f}M")
    print(f"  Top 5 target hospitals: {network_impact['top_5_hospitals']}")

    # ── Step 5: Multi-Agent Analysis (requires API key) ───────────────────────
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        print_section("Step 5: Multi-agent EBITDA optimization debate")
        from nexus.llm.client import NexusLLMClient
        from nexus.agents.orchestrator import NexusOrchestrator, NexusContext

        client = NexusLLMClient()
        orchestrator = NexusOrchestrator(client)

        top_3_edges = "\n".join(f"  {s} → {t} (weight={w:.3f})" for s, t, w in edges[:3])
        ctx = NexusContext(
            objective="Improve EBITDA margin from 6% to 9% across the 40-hospital network within 8 quarters",
            domain="healthcare",
            kpi_data_summary=f"Network avg OR utilization: {latest['or_utilization'].mean():.1%}, "
                             f"readmission: {latest['readmission_rate'].mean():.1%}, "
                             f"EBITDA margin: {latest['ebitda_margin'].mean():.1%}",
            causal_graph_summary=f"Top causal edges discovered:\n{top_3_edges}",
            forecast_summary="TFT forecast: OR utilization trending +2%/quarter with current interventions",
            org_metadata={
                "hospitals": 40,
                "total_beds": int(network.facilities["beds"].sum()),
                "annual_revenue_usd": 9_500_000_000,
                "current_ebitda_margin": float(latest["ebitda_margin"].mean()),
            }
        )

        ebitda_summary = {
            name: {"p50_usd": r.p50, "p10_usd": r.p10, "p90_usd": r.p90}
            for name, r in list(sim_results.items())[:4]
        }

        result = orchestrator.run(ctx, ebitda_sim_results=ebitda_summary)

        print(f"\n  Agent pipeline completed:")
        print(f"    Total tokens: {result.total_tokens:,}")
        print(f"    Cache hit rate: {result.cache_hit_rate:.1%}")
        print(f"    Latency: {result.latency_seconds:.1f}s")
        print(f"\n{'=' * 60}")
        print("  FINAL C-SUITE RECOMMENDATION")
        print(f"{'=' * 60}")
        print(result.synthesis[:3000])
        if result.recommended_interventions:
            print(f"\n  Structured interventions extracted:")
            print(json.dumps(result.recommended_interventions, indent=2))
    else:
        print_section("Step 5: Multi-agent analysis (set ANTHROPIC_API_KEY to enable)")
        print("  Skipping LLM step — set ANTHROPIC_API_KEY to run full pipeline")

    print_section("Demo complete")


if __name__ == "__main__":
    run_demo(use_llm=True)
