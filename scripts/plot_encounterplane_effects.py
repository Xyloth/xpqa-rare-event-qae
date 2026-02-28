from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_METHOD_STYLE = {
    "MC": {"color": "tab:blue", "marker": "o"},
    "IS": {"color": "tab:orange", "marker": "s"},
    "SPLIT": {"color": "tab:green", "marker": "d"},
}

_DIST_LABEL = {
    "single_gaussian": "single",
    "mixture": "mixture",
}


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot encounter-plane CI width effects across scenarios"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_suite.csv",
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "encounterplane_effects.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    families = ["baseline", "near_threshold"]

    fig, axes = plt.subplots(
        len(args.tiers),
        len(families),
        figsize=(12.0, 3.8 * len(args.tiers)),
        sharex=True,
    )
    if len(args.tiers) == 1:
        axes = [axes]

    legend_handles = {}

    for r, tier in enumerate(args.tiers):
        for c, family in enumerate(families):
            ax = axes[r][c]
            sub = df[(df["tier"] == tier) & (df["family"] == family)]

            for method in ["MC", "IS", "SPLIT"]:
                for dist in ["single_gaussian", "mixture"]:
                    rows = sub[
                        (sub["method"] == method) & (sub["distribution"] == dist)
                    ]
                    if rows.empty:
                        continue
                    rows = rows.sort_values("N")
                    style = _METHOD_STYLE[method]
                    line_style = "-" if dist == "single_gaussian" else "--"
                    label = f"{method} {_DIST_LABEL[dist]}"
                    line = ax.plot(
                        rows["N"],
                        rows["ci_width"],
                        linestyle=line_style,
                        marker=style["marker"],
                        color=style["color"],
                        linewidth=1.5,
                        markersize=4,
                        label=label,
                    )[0]
                    legend_handles[label] = line

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
            fam_label = "Baseline" if family == "baseline" else "Near-threshold"
            ax.set_title(f"Tier {tier} - {fam_label}")
            if c == 0:
                ax.set_ylabel("CI width")

    for ax in axes[-1]:
        ax.set_xlabel("Samples N")

    # Scenario knob text for readability in reviewer exports.
    knob = df.iloc[0]
    knob_text = (
        f"mixture: w={knob['mixture_weight']:.3f}, inflation_k={knob['inflation_k']:.1f}, "
        f"mean_shift={bool(knob['mean_shift_flag'])}; near-threshold margin={knob['near_margin_frac']:.3f}"
    )

    fig.suptitle("Encounter-plane model: mixture + near-threshold effects", y=0.99)
    fig.text(
        0.5,
        0.955,
        knob_text,
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0.02, 0.08, 1, 0.93])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
