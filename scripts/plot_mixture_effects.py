from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plot_ess_diagnostics(df: pd.DataFrame, out_path: Path) -> None:
    is_df = df[df["method"] == "IS"].copy()
    tiers = sorted(is_df["tier"].unique())

    fig, axes = plt.subplots(
        1, len(tiers), figsize=(4.8 * len(tiers), 3.6), sharey=True
    )
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        panel = is_df[(is_df["tier"] == tier) & (is_df["family"] == "baseline")]
        panel = panel.sort_values("N")

        single = panel[panel["distribution"] == "single_gaussian"]
        mixture = panel[panel["distribution"] == "mixture"]

        if not single.empty:
            ax.plot(single["N"], single["ess_over_n"], "o-", label="Baseline single")
        if not mixture.empty:
            ax.plot(
                mixture["N"], mixture["ess_over_n"], "s--", label="Baseline mixture"
            )

        ax.set_xscale("log")
        ax.set_xlabel("Samples N")
        ax.set_title(f"Tier {tier}")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("IS ESS/N")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("IS ESS diagnostics (baseline single vs baseline mixture)", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot mixture scenario effects")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "mixture_suite.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "mixture_effects.png",
    )
    parser.add_argument(
        "--ess-out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "is_ess_diagnostics.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = ["tier", "method", "N", "ci_width", "family", "distribution"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tiers = sorted(df["tier"].unique())
    methods = [m for m in ["MC", "IS"] if m in df["method"].unique()]

    fig, axes = plt.subplots(
        len(tiers),
        len(methods),
        figsize=(4.8 * len(methods), 3.4 * len(tiers)),
        sharex=True,
        sharey=True,
    )

    if len(tiers) == 1 and len(methods) == 1:
        axes = [[axes]]
    elif len(tiers) == 1:
        axes = [axes]
    elif len(methods) == 1:
        axes = [[ax] for ax in axes]

    lines = [
        ("baseline", "single_gaussian", "baseline single", "#1f77b4", "-"),
        ("baseline", "mixture", "baseline mixture", "#1f77b4", "--"),
        ("near_threshold", "single_gaussian", "near-threshold single", "#d62728", "-"),
        ("near_threshold", "mixture", "near-threshold mixture", "#d62728", "--"),
    ]

    for i, tier in enumerate(tiers):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            panel = df[(df["tier"] == tier) & (df["method"] == method)].copy()
            panel = panel.sort_values("N")

            for family, dist, label, color, ls in lines:
                sub = panel[
                    (panel["family"] == family) & (panel["distribution"] == dist)
                ]
                if sub.empty:
                    continue
                ax.plot(
                    sub["N"],
                    sub["ci_width"],
                    marker="o",
                    linestyle=ls,
                    color=color,
                    label=label,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
            ax.set_title(f"Tier {tier} - {method}")
            if j == 0:
                ax.set_ylabel("CI width")
            if i == len(tiers) - 1:
                ax.set_xlabel("Samples N")

    # Show scenario knobs in a compact anchored box.
    mix_row = df[df["distribution"] == "mixture"].iloc[0]
    mix_weight = float(mix_row.get("mixture_weight", float("nan")))
    inflation_k = float(mix_row.get("inflation_k", float("nan")))
    mean_shift_flag = bool(mix_row.get("mean_shift_flag", False))
    near_margin = float(mix_row.get("near_margin", float("nan")))
    knob_text = (
        f"mixture: w={mix_weight:.3f}, inflation_k={inflation_k:.1f}, mean_shift={mean_shift_flag}\n"
        f"near-threshold margin_frac={near_margin:.3f}"
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Mixture and near-threshold effects on CI width", y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.text(
        0.015,
        0.885,
        knob_text,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.1,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    fig.tight_layout(rect=[0, 0, 1, 0.83])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    _plot_ess_diagnostics(df, args.ess_out)

    print(f"Saved: {args.out}")
    print(f"Saved: {args.ess_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
