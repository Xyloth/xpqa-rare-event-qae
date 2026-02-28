from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def recommended_bits_for_scenario(
    rows: pd.DataFrame,
    rule_ratio: float,
) -> str | int:
    rows = rows.sort_values("bits")
    ok = rows[rows["abs_error"] <= rule_ratio * rows["ci_width"]]
    if ok.empty:
        return f">{int(rows['bits'].max())}"
    return int(ok.iloc[0]["bits"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot precision sweep diagnostics")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "precision_sweep.csv",
    )
    parser.add_argument(
        "--rule-ratio",
        type=float,
        default=0.25,
        help="Recommend smallest b where abs_error <= rule_ratio * CI_width",
    )
    parser.add_argument(
        "--error-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "precision_error_vs_bits.png",
    )
    parser.add_argument(
        "--ci-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "precision_ciwidth_vs_bits.png",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "results" / "precision_sweep_summary.md",
    )
    args = parser.parse_args()

    if args.rule_ratio <= 0:
        raise ValueError("rule-ratio must be positive")

    df = pd.read_csv(args.csv)

    scenario_order = sorted(df["scenario_name"].unique())

    # Plot 1: abs error vs bits
    fig1, ax1 = plt.subplots(figsize=(7.4, 4.5))
    for name in scenario_order:
        sub = df[df["scenario_name"] == name].sort_values("bits")
        ax1.plot(sub["bits"], sub["abs_error"], marker="o", label=name)

    ax1.set_yscale("log")
    ax1.set_xlabel("Fixed-point bits")
    ax1.set_ylabel("Absolute error |Pc_b - Pc_float|")
    ax1.set_title("Precision sweep: quantization error vs bits")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=8)
    fig1.tight_layout()
    args.error_out.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.error_out, dpi=170)

    # Plot 2: CI width vs bits
    fig2, ax2 = plt.subplots(figsize=(7.4, 4.5))
    for name in scenario_order:
        sub = df[df["scenario_name"] == name].sort_values("bits")
        ax2.plot(sub["bits"], sub["ci_width"], marker="s", label=name)

    ax2.set_yscale("log")
    ax2.set_xlabel("Fixed-point bits")
    ax2.set_ylabel("Replicate CI width")
    ax2.set_title("Precision sweep: CI width vs bits")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    args.ci_out.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.ci_out, dpi=170)

    # Summary markdown + recommendations.
    rec_rows: list[dict[str, str | int | float]] = []
    lines: list[str] = []
    lines.append("# Precision Sweep Summary")
    lines.append("")
    lines.append(
        f"Recommendation rule: smallest `bits` with `abs_error <= {args.rule_ratio:.2f} * CI_width`."
    )
    lines.append("")

    for name in scenario_order:
        sub = df[df["scenario_name"] == name].sort_values("bits")
        rec = recommended_bits_for_scenario(sub, args.rule_ratio)

        tail = sub.iloc[-1]
        lines.append(
            f"- {name}: recommended bits = **{rec}**; "
            f"at b={int(tail['bits'])}, abs_error={float(tail['abs_error']):.3e}, "
            f"CI_width={float(tail['ci_width']):.3e}."
        )

        rec_rows.append(
            {
                "scenario_name": name,
                "recommended_bits": rec,
                "Pc_float_mean": float(sub.iloc[0]["Pc_float_mean"]),
                "rule_ratio": float(args.rule_ratio),
            }
        )

    lines.append("")
    lines.append(
        "Interpretation: when recommended bits are used, quantization error is below statistical uncertainty in this benchmark setting."
    )
    lines.append(
        "For FT resource forecasting, use scenario-specific recommended bits rather than defaulting to 32 bits."
    )

    args.summary_out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(f"Saved: {args.error_out}")
    print(f"Saved: {args.ci_out}")
    print(f"Saved: {args.summary_out}")
    for row in rec_rows:
        print(
            f"  {row['scenario_name']:<26s} recommended_bits={row['recommended_bits']} "
            f"Pc_float={row['Pc_float_mean']:.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
