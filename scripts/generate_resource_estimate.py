from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out:
        raise ValueError("list must contain at least one integer")
    return out


def parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = [float(p) for p in parts]
    if not out:
        raise ValueError("list must contain at least one float")
    return out


def ae_calls(eps: float, c_ae: float) -> int:
    if eps <= 0:
        raise ValueError("eps must be positive")
    return int(math.ceil(c_ae / eps))


def mc_samples(eps: float) -> int:
    if eps <= 0:
        raise ValueError("eps must be positive")
    return int(math.ceil(1.0 / (eps**2)))


def oracle_toffoli(b: int, alpha: float, beta: float) -> float:
    return alpha * (b**2) + beta * b


def stateprep_toffoli(b: int, log2_m: float, gamma: float, eta: float) -> float:
    # Conservative QROM-style proxy: address/data loading scales with log2(M).
    return gamma * b * log2_m + eta * log2_m


def q_logical_estimate(b: int, kappa: float) -> float:
    return kappa * b


def build_resource_table(
    b_list: list[int],
    eps_list: list[float],
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
    kappa: float,
    m_grid: int,
    c_ae: float,
) -> pd.DataFrame:
    if m_grid <= 1:
        raise ValueError("m_grid must be > 1")
    log2_m = float(np.log2(m_grid))

    rows: list[dict[str, float | int]] = []
    for b in b_list:
        o_t = oracle_toffoli(b, alpha, beta)
        s_t = stateprep_toffoli(b, log2_m, gamma, eta)
        total_per_call = o_t + s_t
        q_log = q_logical_estimate(b, kappa)

        for eps in eps_list:
            m_ae = ae_calls(eps, c_ae)
            n_mc = mc_samples(eps)
            rows.append(
                {
                    "b": int(b),
                    "eps": float(eps),
                    "ae_oracle_calls": int(m_ae),
                    "mc_samples_for_eps": int(n_mc),
                    "oracle_toffoli_est": float(o_t),
                    "stateprep_toffoli_est": float(s_t),
                    "total_toffoli_per_call": float(total_per_call),
                    "total_toffoli_ae": float(total_per_call * m_ae),
                    "q_logical_est": float(q_log),
                }
            )

    return pd.DataFrame(rows).sort_values(["b", "eps"]).reset_index(drop=True)


def plot_ae_vs_mc_calls(
    out_path: Path,
    c_ae: float,
    eps_min: float,
    eps_max: float,
) -> None:
    eps = np.logspace(np.log10(eps_min), np.log10(eps_max), 250)
    y_ae = c_ae / eps
    y_mc = 1.0 / (eps**2)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.loglog(eps, y_ae, label="AE oracle calls ~ 1/eps", linewidth=2.0)
    ax.loglog(eps, y_mc, label="MC samples ~ 1/eps^2", linewidth=2.0)

    ax.set_xlabel("Target additive error eps")
    ax.set_ylabel("Calls / samples")
    ax.set_title("Asymptotic call scaling: AE vs MC")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)


def plot_total_toffoli_vs_eps(
    table: pd.DataFrame,
    out_path: Path,
    b_list: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    for b in b_list:
        sub = table[table["b"] == b].sort_values("eps")
        ax.loglog(
            sub["eps"],
            sub["total_toffoli_ae"],
            marker="o",
            linewidth=1.8,
            label=f"b={b}",
        )

    ax.set_xlabel("Target additive error eps")
    ax.set_ylabel("Total AE Toffoli estimate")
    ax.set_title("Estimated total AE Toffoli vs eps")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)


def write_oracle_spec(
    out_path: Path,
    m_grid: int,
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
) -> None:
    text = f"""# Oracle Spec (Encounter-Plane Predicate)

## Oracle Definition
- Input random variables: encounter-plane coordinates `(x, y)` sampled from either:
  - single 2D Gaussian, or
  - two-component Gaussian mixture `(1-w)N(mu,Sigma) + w N(mu+delta, k Sigma)`.
- Event predicate: `d(x,y) < d0`, with `d = sqrt(x^2 + y^2)`.
- Implemented oracle predicate (sqrt-free):
  - `event = (x^2 + y^2) < d0^2`.

## Reversible Circuit Sketch
1. Load fixed-point registers for `x` and `y` (`b` signed bits each).
2. Compute `x^2` and `y^2` using reversible fixed-point multiply.
3. Sum into accumulator: `s = x^2 + y^2`.
4. Reversible comparator: set flag/phase control if `s < d0^2`.
5. Apply phase flip (or write event flag) conditioned on comparator result.
6. Uncompute ancillas by reversing steps 2-4.

## Why Squared Distance Avoids Sqrt
- Reversible square root is depth-heavy and ancilla-heavy.
- Comparing `x^2 + y^2` to constant `d0^2` preserves event logic exactly while replacing sqrt with add/compare arithmetic.
- This reduces oracle depth and simplifies uncomputation.

## State-Prep Interface
- `U_prep` prepares amplitudes consistent with the uncertainty model in encounter-plane coordinates.
- Single Gaussian mode: analytic distribution parameters `(mu, Sigma)`.
- Mixture mode: `(w, inflation_k, mean_shift)` controls non-Gaussian contamination.
- For resource modeling we use a conservative QROM-style proxy with grid size `M={m_grid}`.

## Bottleneck Note
- Dominant FT cost is not the comparator itself; it is:
  1. state preparation (`U_prep`), and
  2. reversible fixed-point arithmetic depth/ancillas.
- Parametric per-call model used here:
  - `C_oracle(b) = alpha*b^2 + beta*b`, with `alpha={alpha:g}`, `beta={beta:g}`.
  - `C_stateprep(b,M) = gamma*b*log2(M) + eta*log2(M)`, with `gamma={gamma:g}`, `eta={eta:g}`.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def _pick_classical_datapoint(
    rep_csv: Path,
    suite_csv: Path,
    ci_target: float,
) -> dict[str, float | int | str]:
    rep = pd.read_csv(rep_csv) if rep_csv.exists() else pd.DataFrame()
    suite = pd.read_csv(suite_csv)

    # Prefer proposal-agnostic SPLIT baseline for crossover narrative.
    split = suite[
        (suite["tier"] == 3)
        & (suite["variant"] == "baseline_single")
        & (suite["method"] == "SPLIT")
        & (suite["ci_width"] <= ci_target)
        & (suite["hits"] > 0)
    ].sort_values("eval_count")

    if not split.empty:
        row = split.iloc[0]
        return {
            "method": "SPLIT",
            "tier": int(row["tier"]),
            "variant": str(row["variant"]),
            "N": int(row["N"]),
            "ci_width": float(row["ci_width"]),
            "eval_count": float(row["eval_count"]),
            "pc_hat": float(row["pc_hat"]),
        }

    # Fallback to replicate MC/IS if split target not met.
    if not rep.empty:
        cand = rep[
            (rep["tier"] == 3)
            & (rep["variant"] == "baseline_single")
            & (rep["ci_width"] <= ci_target)
            & (rep["hits_mean"] > 0)
        ].sort_values("eval_count_mean")
        if not cand.empty:
            row = cand.iloc[0]
            return {
                "method": str(row["method"]),
                "tier": int(row["tier"]),
                "variant": str(row["variant"]),
                "N": int(row["N"]),
                "ci_width": float(row["ci_width"]),
                "eval_count": float(row["eval_count_mean"]),
                "pc_hat": float(row["pc_hat_mean"]),
            }

    raise ValueError("No classical datapoint found that meets CI target")


def write_crossover_note(
    out_path: Path,
    rep_csv: Path,
    suite_csv: Path,
    ci_target: float,
    b_ref: int,
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
    m_grid: int,
    c_ae: float,
) -> dict[str, float | int | str]:
    dp = _pick_classical_datapoint(rep_csv, suite_csv, ci_target)

    eps_ref = ci_target / 2.0
    log2_m = float(np.log2(m_grid))
    per_call = oracle_toffoli(b_ref, alpha, beta) + stateprep_toffoli(
        b_ref, log2_m, gamma, eta
    )
    m_ae = ae_calls(eps_ref, c_ae)
    total_toffoli = per_call * m_ae

    text = f"""# Crossover Note (Empirical + Parametric)

- Classical datapoint (tier 3, baseline single):
  - method: {dp['method']}
  - target CI width: {ci_target:.1e}
  - achieved at `N={int(dp['N'])}` with approx `eval_count={dp['eval_count']:.0f}`
  - observed CI width: {dp['ci_width']:.3e}

- Quantum reference target:
  - map CI width target to additive error proxy `eps = CI_width/2 = {eps_ref:.1e}`.
  - canonical AE oracle calls: `M_AE ~= (pi/2)/eps ~= {m_ae:,}`.
  - with `b={b_ref}` and conservative per-call FT cost,
    `Cq_call ~= {per_call:,.0f}` Toffoli,
    so total AE Toffoli `~ {total_toffoli:,.3e}`.

- Crossover inequality (symbolic):
  - `(Cq_call * M_AE) < (Cc_eval * N_eval)`.
  - Here `N_eval` is classical event-oracle evaluations and `Cc_eval` is tiny on classical hardware,
    while `Cq_call` is large under FT assumptions.

Interpretation:
Quantum advantage is not guaranteed by `1/eps` vs `1/eps^2` alone; it depends critically on cheap state preparation + oracle arithmetic and sufficiently hard tail regimes where classical refinement costs remain high.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")

    return {
        "classical_method": str(dp["method"]),
        "classical_eval_count": float(dp["eval_count"]),
        "classical_n": int(dp["N"]),
        "classical_ci_width": float(dp["ci_width"]),
        "eps_ref": float(eps_ref),
        "ae_calls_ref": int(m_ae),
        "toq_per_call_ref": float(per_call),
        "total_toffoli_ref": float(total_toffoli),
    }


def write_viability_block(
    out_path: Path,
    ci_target: float,
    b_ref: int,
    summary: dict[str, float | int | str],
) -> None:
    text = f"""# Viability Block (Phase I Paste-Ready)

Our event oracle for conjunction-like encounter risk uses the encounter-plane predicate `d(x,y) < d0`, implemented as `(x^2 + y^2) < d0^2` in reversible fixed-point arithmetic. This squared-distance formulation avoids reversible square root and keeps the oracle structure compact: load `(x,y)`, compute two squares, sum, compare to constant `d0^2`, phase-mark, and uncompute ancillas. The same interface supports both single-Gaussian and Gaussian-mixture state preparation, matching the uncertainty models used in our classical benchmarks.

For feasibility, we report a transparent parametric FT model rather than full compilation. Per-call logical cost is modeled as arithmetic `O(b^2)` plus state-prep overhead, with logical qubits scaling roughly linearly in precision (`Q_logical ~ kappa*b`). Canonical amplitude estimation gives oracle calls `M_AE ~ (pi/2)/eps`, compared with naive Monte Carlo sample scaling `~1/eps^2`; this is the asymptotic lever motivating quantum insertion. However, constants dominate in Phase I: for `b={b_ref}` and target CI width `{ci_target:.1e}` (`eps~{summary['eps_ref']:.1e}`), AE requires about `{summary['ae_calls_ref']:,}` oracle calls, each with substantial FT cost.

Empirically, our tier-3 baseline-single classical benchmark reaches the CI target with about `{summary['classical_eval_count']:.0f}` event evaluations ({summary['classical_method']} at `N={summary['classical_n']}`). This yields the practical crossover condition `(Cq_call * M_AE) < (Cc_eval * N_eval)`: advantage depends on making oracle/state-prep very cheap and targeting regimes where classical tail-refinement remains expensive. Phase II work therefore focuses on oracle-aware state preparation, tighter arithmetic synthesis, and calibrated hardware-level cost models.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase I viability/resource artifacts"
    )
    parser.add_argument(
        "--b-list", type=parse_int_list, default=parse_int_list("16,24,32")
    )
    parser.add_argument(
        "--eps-list",
        type=parse_float_list,
        default=parse_float_list("1e-1,1e-2,1e-3,1e-4"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=8.0,
        help="Quadratic coefficient for oracle arithmetic Toffoli proxy",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=20.0,
        help="Linear coefficient for add/compare Toffoli proxy",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=4.0,
        help="State-prep coefficient on b*log2(M)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=20.0,
        help="State-prep coefficient on log2(M)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=8.0,
        help="Logical qubit slope Q_logical ~= kappa*b",
    )
    parser.add_argument(
        "--m-grid",
        type=int,
        default=2**20,
        help="Proxy grid size M for conservative QROM-like state prep",
    )
    parser.add_argument(
        "--ae-const",
        type=float,
        default=np.pi / 2,
        help="AE call constant c in M_AE ~= c/eps",
    )
    parser.add_argument(
        "--ci-target",
        type=float,
        default=2e-6,
        help="Tier-3 CI width target used for practical crossover note",
    )
    parser.add_argument(
        "--b-ref",
        type=int,
        default=32,
        help="Reference precision for crossover FT cost example",
    )
    parser.add_argument(
        "--replicate-csv",
        type=Path,
        default=TABLES_DIR / "encounterplane_replicates.csv",
    )
    parser.add_argument(
        "--suite-csv",
        type=Path,
        default=TABLES_DIR / "encounterplane_suite.csv",
    )
    args = parser.parse_args()

    table = build_resource_table(
        b_list=args.b_list,
        eps_list=args.eps_list,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        kappa=args.kappa,
        m_grid=args.m_grid,
        c_ae=args.ae_const,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "resource_scaling_table.csv"
    table.to_csv(table_path, index=False)

    ae_plot = PLOTS_DIR / "ae_vs_mc_calls.png"
    plot_ae_vs_mc_calls(
        out_path=ae_plot,
        c_ae=args.ae_const,
        eps_min=min(args.eps_list),
        eps_max=max(args.eps_list),
    )

    toff_plot = PLOTS_DIR / "total_toffoli_vs_eps.png"
    plot_total_toffoli_vs_eps(table=table, out_path=toff_plot, b_list=args.b_list)

    oracle_spec_path = RESULTS_DIR / "oracle_spec.md"
    write_oracle_spec(
        out_path=oracle_spec_path,
        m_grid=args.m_grid,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
    )

    crossover_path = RESULTS_DIR / "crossover_note.md"
    crossover_summary = write_crossover_note(
        out_path=crossover_path,
        rep_csv=args.replicate_csv,
        suite_csv=args.suite_csv,
        ci_target=args.ci_target,
        b_ref=args.b_ref,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
        m_grid=args.m_grid,
        c_ae=args.ae_const,
    )

    viability_path = RESULTS_DIR / "viability_block.md"
    write_viability_block(
        out_path=viability_path,
        ci_target=args.ci_target,
        b_ref=args.b_ref,
        summary=crossover_summary,
    )

    print(f"Saved: {oracle_spec_path}")
    print(f"Saved: {table_path}")
    print(f"Saved: {ae_plot}")
    print(f"Saved: {toff_plot}")
    print(f"Saved: {crossover_path}")
    print(f"Saved: {viability_path}")
    print(
        "Summary: "
        f"classical={crossover_summary['classical_method']} N={crossover_summary['classical_n']} "
        f"eval~{crossover_summary['classical_eval_count']:.0f}; "
        f"AE calls~{crossover_summary['ae_calls_ref']:,} at eps={crossover_summary['eps_ref']:.1e}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
