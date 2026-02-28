from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if not vals:
        raise ValueError("list must contain at least one value")
    return vals


def classify_vectorized(
    coarse_p: np.ndarray,
    coarse_ciw: np.ndarray,
    t_low: float,
    t_high: float,
    margin_mult: float,
    margin_floor: float,
) -> np.ndarray:
    margin = np.maximum(margin_mult * coarse_ciw, margin_floor)
    out = np.full(coarse_p.shape, "GRAY", dtype=object)
    out[coarse_p + margin < t_low] = "SAFE"
    out[coarse_p - margin > t_high] = "DANGER"
    return out


def expected_compute_for_policy(
    df: pd.DataFrame,
    classes: np.ndarray,
) -> float:
    coarse_eval = df["coarse_eval_count"].to_numpy(dtype=float)
    refine_eval = df["refine_eval_count"].to_numpy(dtype=float)
    refine_n_max = float(df["refine_n_max"].iloc[0])
    refine_proxy = np.where(
        np.isfinite(refine_eval) & (refine_eval > 0), refine_eval, refine_n_max
    )
    total = coarse_eval + np.where(classes == "GRAY", refine_proxy, 0.0)
    return float(np.mean(total))


def recommended_bits(df: pd.DataFrame, rule_ratio: float = 0.25) -> dict[str, str]:
    rec: dict[str, str] = {}
    for scenario_name, group in df.groupby("scenario_name"):
        g = group.sort_values("bits")
        ok = g[g["abs_error"] <= rule_ratio * g["ci_width"]]
        if ok.empty:
            rec[scenario_name] = f">{int(g['bits'].max())}"
        else:
            rec[scenario_name] = str(int(ok.iloc[0]["bits"]))
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot gating diagnostics and write summaries"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "gating_simulation.csv",
    )
    parser.add_argument(
        "--policy-margins",
        type=parse_float_list,
        default=parse_float_list("0.00,0.02,0.05,0.10,0.20,0.40"),
        help="Comma-separated margin_mult values for policy sweep",
    )
    parser.add_argument(
        "--fraction-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "gating_fraction_vs_policy.png",
    )
    parser.add_argument(
        "--hist-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "gating_compute_hist.png",
    )
    parser.add_argument(
        "--compute-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "gating_expected_compute.png",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "results" / "gating_summary.md",
    )
    parser.add_argument(
        "--phase2-out",
        type=Path,
        default=REPO_ROOT / "results" / "phase2_plan_evidence.md",
    )
    parser.add_argument(
        "--precision-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "precision_sweep.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("gating csv is empty")

    t_low = float(df["t_low"].iloc[0])
    t_high = float(df["t_high"].iloc[0])
    margin_floor = float(df["margin_floor"].iloc[0])
    baseline_margin = float(df["margin_mult"].iloc[0])

    coarse_p = df["coarse_Pc"].to_numpy(dtype=float)
    coarse_ciw = df["coarse_ci_width"].to_numpy(dtype=float)

    margins = sorted(set(args.policy_margins + [baseline_margin]))
    frac_rows: list[dict[str, float]] = []
    compute_rows: list[dict[str, float]] = []

    for mm in margins:
        cls = classify_vectorized(coarse_p, coarse_ciw, t_low, t_high, mm, margin_floor)
        safe = float(np.mean(cls == "SAFE"))
        danger = float(np.mean(cls == "DANGER"))
        gray = float(np.mean(cls == "GRAY"))
        frac_rows.append(
            {
                "margin_mult": mm,
                "safe_frac": safe,
                "danger_frac": danger,
                "gray_frac": gray,
            }
        )

        exp_compute = expected_compute_for_policy(df, cls)
        compute_rows.append({"margin_mult": mm, "expected_eval_count": exp_compute})

    frac_df = pd.DataFrame(frac_rows).sort_values("margin_mult")
    compute_df = pd.DataFrame(compute_rows).sort_values("margin_mult")

    # Plot 1: SAFE / DANGER / GRAY fractions vs policy margin.
    fig1, ax1 = plt.subplots(figsize=(7.4, 4.5))
    ax1.plot(frac_df["margin_mult"], frac_df["safe_frac"], marker="o", label="SAFE")
    ax1.plot(frac_df["margin_mult"], frac_df["danger_frac"], marker="s", label="DANGER")
    ax1.plot(frac_df["margin_mult"], frac_df["gray_frac"], marker="^", label="GRAY")
    ax1.axvline(
        baseline_margin, linestyle="--", color="black", alpha=0.45, linewidth=1.0
    )
    ax1.set_xlabel("Screen margin multiplier")
    ax1.set_ylabel("Scenario fraction")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Gating policy: SAFE / DANGER / GRAY fractions")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")
    fig1.tight_layout()
    args.fraction_out.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.fraction_out, dpi=170)

    # Plot 2: histogram of refine compute for GRAY cases under baseline policy.
    gray_mask = df["classification"] == "GRAY"
    gray_eval = df.loc[gray_mask, "refine_eval_count"].to_numpy(dtype=float)
    gray_eval = gray_eval[np.isfinite(gray_eval) & (gray_eval > 0)]
    fig2, ax2 = plt.subplots(figsize=(7.4, 4.5))
    if gray_eval.size > 0:
        bins = np.array([1000, 3000, 5000, 10000, 20000, 30000], dtype=float)
        ax2.hist(gray_eval, bins=bins, color="#2a6f97", alpha=0.82, edgecolor="black")
    ax2.set_xscale("log")
    ax2.set_xlabel("Refinement eval_count (GRAY cases, baseline policy)")
    ax2.set_ylabel("Count")
    ax2.set_title("Gating refinement load distribution")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    fig2.tight_layout()
    args.hist_out.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.hist_out, dpi=170)

    # Plot 3: expected compute vs policy margin, with always-refine baseline.
    refine_eval = df["refine_eval_count"].to_numpy(dtype=float)
    refine_n_max = float(df["refine_n_max"].iloc[0])
    coarse_eval = df["coarse_eval_count"].to_numpy(dtype=float)
    refine_proxy_all = np.where(
        np.isfinite(refine_eval) & (refine_eval > 0), refine_eval, refine_n_max
    )
    expected_always = float(np.mean(coarse_eval + refine_proxy_all))

    fig3, ax3 = plt.subplots(figsize=(7.4, 4.5))
    ax3.plot(
        compute_df["margin_mult"],
        compute_df["expected_eval_count"],
        marker="o",
        linewidth=1.8,
        label="Gated policy",
    )
    ax3.axhline(
        expected_always,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Always refine (proxy)",
    )
    ax3.axvline(baseline_margin, linestyle=":", color="gray", alpha=0.5, linewidth=1.0)
    ax3.set_xlabel("Screen margin multiplier")
    ax3.set_ylabel("Expected eval_count per scenario")
    ax3.set_title("Expected compute: gated vs always-refine")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="best")
    fig3.tight_layout()
    args.compute_out.parent.mkdir(parents=True, exist_ok=True)
    fig3.savefig(args.compute_out, dpi=170)

    baseline_classes = classify_vectorized(
        coarse_p=coarse_p,
        coarse_ciw=coarse_ciw,
        t_low=t_low,
        t_high=t_high,
        margin_mult=baseline_margin,
        margin_floor=margin_floor,
    )
    baseline_safe = float(np.mean(baseline_classes == "SAFE"))
    baseline_danger = float(np.mean(baseline_classes == "DANGER"))
    baseline_gray = float(np.mean(baseline_classes == "GRAY"))
    baseline_expected = expected_compute_for_policy(df, baseline_classes)
    baseline_reduction = 1.0 - (baseline_expected / expected_always)

    summary_lines = [
        "# Gating Summary",
        "",
        "This experiment quantifies a telescoping policy: a cheap conservative screen classifies",
        "scenarios as SAFE/DANGER/GRAY, and only GRAY cases receive expensive refinement.",
        "",
        f"- Policy parameters: `t_low={t_low:.2e}`, `t_high={t_high:.2e}`, `margin_mult={baseline_margin:.3f}`, `margin_floor={margin_floor:.1e}`.",
        f"- Scenario count: `K={len(df)}` with mixture fraction target `~{float(df['mixture_flag'].mean()):.1%}`.",
        f"- Baseline fractions: SAFE={baseline_safe:.1%}, DANGER={baseline_danger:.1%}, GRAY={baseline_gray:.1%}.",
        f"- Expected compute (eval_count/scenario): gated={baseline_expected:.1f}, always-refine proxy={expected_always:.1f}.",
        f"- Estimated compute reduction from gating: **{baseline_reduction:.1%}**.",
        "",
        "Interpretation: only GRAY scenarios require refinement, so a hybrid insertion strategy can",
        "target this expensive slice rather than all conjunction-like cases.",
    ]
    args.summary_out.write_text(
        "\n".join(summary_lines).strip() + "\n", encoding="utf-8"
    )

    phase2_lines = [
        "# Phase II Plan Evidence",
        "",
        "## Precision Calibration",
    ]

    if args.precision_csv.exists():
        p_df = pd.read_csv(args.precision_csv)
        rec = recommended_bits(p_df, rule_ratio=0.25)
        b_single_t3 = rec.get("ep_tier3_baseline_single", "n/a")
        b_mix_t3 = rec.get("ep_tier3_baseline_mixture", "n/a")
        phase2_lines.extend(
            [
                "- Recommended fixed-point bits (rule: `abs_error <= 0.25 * CI_width`):",
                f"  - tier3 baseline single: **b={b_single_t3}**",
                f"  - tier3 baseline mixture: **b={b_mix_t3}**",
                "- Takeaway: use calibrated bit precision in FT resource estimates instead of defaulting to 32 bits.",
            ]
        )
    else:
        phase2_lines.append(
            "- Precision CSV not found; run `python scripts/run_precision_sweep.py` first."
        )

    phase2_lines.extend(
        [
            "",
            "## Telescoping / Gating",
            f"- Baseline gating split: SAFE={baseline_safe:.1%}, DANGER={baseline_danger:.1%}, GRAY={baseline_gray:.1%}.",
            f"- Expected compute reduction with gating vs always-refine proxy: **{baseline_reduction:.1%}**.",
            "- Quantum insertion target: GRAY cases where refinement dominates classical compute.",
            "",
            "## Phase II Execution Path",
            "We will tune oracle/state-prep implementations at the calibrated bit precision, re-run",
            "resource estimates, and evaluate hybrid policies where quantum amplitude estimation is",
            "applied only to GRAY scenarios that dominate refinement cost.",
        ]
    )
    args.phase2_out.write_text("\n".join(phase2_lines).strip() + "\n", encoding="utf-8")

    print(f"Saved: {args.fraction_out}")
    print(f"Saved: {args.hist_out}")
    print(f"Saved: {args.compute_out}")
    print(f"Saved: {args.summary_out}")
    print(f"Saved: {args.phase2_out}")
    print(
        f"Baseline policy fractions: SAFE={baseline_safe:.2%} "
        f"DANGER={baseline_danger:.2%} GRAY={baseline_gray:.2%}"
    )
    print(
        f"Expected compute reduction vs always-refine proxy: {baseline_reduction:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
