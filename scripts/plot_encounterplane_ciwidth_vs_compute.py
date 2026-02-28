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


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot encounter-plane CI width vs compute (eval_count)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_suite.csv",
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument(
        "--family",
        type=str,
        default="baseline",
        choices=["baseline", "near_threshold"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "plots"
        / "encounterplane_ciwidth_vs_compute.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig, axes = plt.subplots(
        len(args.tiers), 1, figsize=(7.5, 3.4 * len(args.tiers)), sharex=False
    )
    if len(args.tiers) == 1:
        axes = [axes]

    legend_handles = {}
    for ax, tier in zip(axes, args.tiers):
        sub = df[(df["tier"] == tier) & (df["family"] == args.family)]

        for method in ["MC", "IS", "SPLIT"]:
            for dist in ["single_gaussian", "mixture"]:
                rows = sub[(sub["method"] == method) & (sub["distribution"] == dist)]
                if rows.empty:
                    continue

                rows = rows.sort_values("eval_count")
                style = _METHOD_STYLE[method]
                line_style = "-" if dist == "single_gaussian" else "--"
                label = (
                    f"{method} {'single' if dist == 'single_gaussian' else 'mixture'}"
                )
                line = ax.plot(
                    rows["eval_count"],
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
        ax.set_ylabel("CI width")
        ax.set_title(f"Tier {tier} ({args.family.replace('_', '-')})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Compute proxy: event-oracle evaluations")

    fig.suptitle("Encounter-plane CI width vs compute", y=0.995)
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
