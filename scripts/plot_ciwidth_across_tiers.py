from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CI width across tiers")
    parser.add_argument(
        "--tiers",
        type=parse_int_list,
        default=parse_int_list("1,2,3"),
        help="Comma-separated list of tiers",
    )
    parser.add_argument(
        "--mc-template",
        type=str,
        default=str(REPO_ROOT / "results" / "tables" / "mc_tier{tier}.csv"),
        help="MC CSV template with {tier}",
    )
    parser.add_argument(
        "--is-template",
        type=str,
        default=str(REPO_ROOT / "results" / "tables" / "is_tier{tier}.csv"),
        help="IS CSV template with {tier}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "ciwidth_across_tiers.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    tiers = args.tiers
    fig, axes = plt.subplots(len(tiers), 1, figsize=(7.2, 3.2 * len(tiers)), sharex=True)
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers):
        mc_path = Path(args.mc_template.format(tier=tier))
        is_path = Path(args.is_template.format(tier=tier))
        mc = pd.read_csv(mc_path).sort_values("n")
        is_df = pd.read_csv(is_path).sort_values("n")

        mc_width = mc["ci_high"] - mc["ci_low"]
        is_width = is_df["ci_high"] - is_df["ci_low"]

        ax.plot(mc["n"], mc_width, "o-", label="MC")
        ax.plot(is_df["n"], is_width, "s-", label="IS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("CI width")
        ax.set_title(f"Tier {tier}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Samples N")
    axes[0].legend(loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
