# Entrypoints

Pinned run list for reviewers and maintainers.

| Script | What it does | Primary output(s) |
|---|---|---|
| `scripts/run_encounterplane_replicates.py` | Runs encounter-plane MC/IS replicate benchmark cases | `results/tables/encounterplane_replicates.csv` |
| `scripts/plot_encounterplane_paper_figures.py` | Builds paper-ready encounter-plane figures | `results/plots/encounterplane_paper_baseline.png`, `results/plots/encounterplane_paper_nearthreshold.png` |
| `scripts/run_splitting_suite.py` | Runs SPLIT baseline with replicate CIs | `results/tables/splitting_suite.csv` |
| `scripts/plot_ciwidth_vs_compute.py` | Compares CI width against compute proxy | `results/plots/ciwidth_vs_compute.png` |
| `scripts/generate_resource_estimate.py` | Creates oracle/resource viability artifacts | `results/oracle_spec.md`, `results/tables/resource_scaling_table.csv`, `results/plots/ae_vs_mc_calls.png` |
| `scripts/run_qdk_resource_estimate.py --bits 16,24,32 --error_budgets 1e-2,1e-3,1e-4` | Runs QDK resource-estimator sweep for arithmetic skeleton | `results/tables/qdk_resource_estimate_summary.csv`, `results/plots/qdk_resource_estimate.png`, `results/qdk_resource_estimate.md` |
| `scripts/run_gating_simulation.py` | Simulates SAFE/DANGER/GRAY routing policy | `results/tables/gating_simulation.csv` |
| `scripts/plot_gating_results.py` | Produces gating figures + Phase II markdown block | `results/plots/gating_*.png`, `results/gating_summary.md`, `results/phase2_plan_evidence.md` |
