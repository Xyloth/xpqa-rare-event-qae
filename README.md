[![DOI](https://zenodo.org/badge/1168537131.svg)](https://doi.org/10.5281/zenodo.18816742)
Paper (Zenodo report): https://zenodo.org/records/18816815

# XPRIZE Quantum Applications Benchmark Harness

This repository is the technical support repo for an XPRIZE Quantum Applications Phase I Wild Card submission focused on rare-event conjunction-risk estimation (`Pc`) under uncertainty. It contains reproducible classical baselines (MC / IS / SPLIT), realistic stress regimes (mixtures + near-threshold boundary layers), an oracle-oriented encounter-plane model, and Phase II evidence artifacts (precision calibration + telescoping/gating + QDK resource-estimator outputs).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate && python -m pip install -U pip && pip install -r requirements.txt
python scripts/run_encounterplane_replicates.py && python scripts/plot_encounterplane_paper_figures.py && python scripts/run_splitting_suite.py && python scripts/plot_ciwidth_vs_compute.py && python scripts/generate_resource_estimate.py && python scripts/run_qdk_resource_estimate.py --bits 16,24,32 --error_budgets 1e-2,1e-3,1e-4 && python scripts/run_gating_simulation.py && python scripts/plot_gating_results.py && python scripts/run_precision_sweep.py && python scripts/plot_precision_sweep.py
```

Windows activation equivalent: `.venv\\Scripts\\activate`

## Repo Map

- `src/xq/`: core models and estimators (encounter model, encounter-plane model, distributions/scenarios, IS, splitting, quantization).
- `scripts/`: runnable experiment/plot/summary entrypoints.
- `results/`: generated outputs (tables, plots, markdown summaries); generally treated as generated artifacts.
- `paper/`: tracked, judge-facing figure pack used directly in narrative/PDF assembly.

## Reproduce Key Results

### Encounter-plane suite + paper figures

```bash
python scripts/run_encounterplane_replicates.py && python scripts/plot_encounterplane_paper_figures.py
```

### Rare-event baselines (SPLIT + compute-normalized comparison)

```bash
python scripts/run_splitting_suite.py && python scripts/plot_ciwidth_vs_compute.py
```

### Viability resources (oracle decomposition + scaling tables/plots)

```bash
python scripts/generate_resource_estimate.py
```

### QDK resource estimator sweep

```bash
python scripts/run_qdk_resource_estimate.py --bits 16,24,32 --error_budgets 1e-2,1e-3,1e-4
```

### Phase II plan evidence (telescoping + precision calibration)

```bash
python scripts/run_gating_simulation.py && python scripts/plot_gating_results.py
python scripts/run_precision_sweep.py && python scripts/plot_precision_sweep.py
```

## How This Maps to the Narrative

- **Impact:** gating/telescoping evidence quantifies that only a minority of cases need expensive refinement.
- **Benchmarking:** MC, IS, SPLIT are compared on single, mixture, and near-threshold regimes.
- **Viability:** oracle sketch, scaling table, and QDK estimates provide transparent resource assumptions.
- **Phase II plan:** calibrated precision targets and gray-zone routing define a concrete hybrid insertion path for QAE.

## Script Index

### Benchmarks

| Script | Purpose | Key outputs |
|---|---|---|
| `scripts/run_bench_suite.py` | Tiered MC/IS benchmark summary | `results/tables/bench_suite_summary.csv` |
| `scripts/plot_ciwidth_across_tiers.py` | MC/IS CI-width across tiers | `results/plots/ciwidth_across_tiers.png` |
| `scripts/run_splitting_suite.py` | SPLIT replicate benchmark | `results/tables/splitting_suite.csv` |
| `scripts/plot_ciwidth_vs_compute.py` | MC/IS/SPLIT CI vs compute | `results/plots/ciwidth_vs_compute.png` |

### Encounter-plane

| Script | Purpose | Key outputs |
|---|---|---|
| `scripts/run_encounterplane_replicates.py` | Replicate CIs for MC/IS on encounter-plane scenarios | `results/tables/encounterplane_replicates.csv` |
| `scripts/plot_encounterplane_paper_figures.py` | Judge/paper figure set (baseline + near-threshold) | `results/plots/encounterplane_paper_baseline.png`, `results/plots/encounterplane_paper_nearthreshold.png` |
| `scripts/generate_encounterplane_summary.py` | Encounter-plane narrative summary | `results/encounterplane_summary.md` |

### Viability + QDK

| Script | Purpose | Key outputs |
|---|---|---|
| `scripts/generate_resource_estimate.py` | Parametric FT viability tables/plots + oracle note | `results/tables/resource_scaling_table.csv`, `results/plots/ae_vs_mc_calls.png`, `results/oracle_spec.md` |
| `scripts/run_qdk_resource_estimate.py` | QDK-based arithmetic skeleton resource sweep | `results/tables/qdk_resource_estimate_summary.csv`, `results/plots/qdk_resource_estimate.png`, `results/qdk_resource_estimate.md` |

### Phase II evidence

| Script | Purpose | Key outputs |
|---|---|---|
| `scripts/run_gating_simulation.py` | Telescoping policy simulation over scenario population | `results/tables/gating_simulation.csv` |
| `scripts/plot_gating_results.py` | Gating policy figures + summary markdown | `results/plots/gating_*.png`, `results/gating_summary.md`, `results/phase2_plan_evidence.md` |
| `scripts/run_precision_sweep.py` | Quantization precision sweep with replicate MC | `results/tables/precision_sweep.csv` |
| `scripts/plot_precision_sweep.py` | Precision error/CI plots + bit recommendation summary | `results/plots/precision_*_vs_bits.png`, `results/precision_sweep_summary.md` |

## Cite This Work

- Paper DOI: `10.5281/zenodo.18816815`
- Software/archive DOI: `10.5281/zenodo.18816743`
- Technical report PDF: `paper/technical_report_v0.1.pdf`
- Release tag: `v0.1.0`

## License

Source code is released under the MIT License. Research narrative, figures,
reports, and generated result artifacts are included for
competition/reproducibility context; cite the Zenodo records above when
referencing the submission.
