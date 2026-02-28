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


def _load_mc_or_is(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("n")
    out = pd.DataFrame(
        {
            "eval_count": df["n"].astype(float),
            "ci_width": (df["ci_high"] - df["ci_low"]).astype(float),
        }
    )
    return out


def _load_split(split_df: pd.DataFrame, tier: int, scenario_name: str) -> pd.DataFrame:
    rows = split_df[
        (split_df["tier"] == tier) & (split_df["scenario_name"] == scenario_name)
    ].sort_values("N")
    if rows.empty:
        return rows
    return pd.DataFrame(
        {
            "eval_count": rows["eval_count_mean"].astype(float),
            "ci_width": rows["ci_width"].astype(float),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot CI width vs compute for MC/IS/SPLIT"
    )
    parser.add_argument(
        "--tiers",
        type=parse_int_list,
        default=parse_int_list("2,3"),
        help="Comma-separated tier list",
    )
    parser.add_argument(
        "--mc-template",
        type=str,
        default=str(REPO_ROOT / "results" / "tables" / "mc_tier{tier}.csv"),
    )
    parser.add_argument(
        "--is-template",
        type=str,
        default=str(REPO_ROOT / "results" / "tables" / "is_tier{tier}.csv"),
    )
    parser.add_argument(
        "--splitting-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "splitting_suite.csv",
    )
    parser.add_argument(
        "--split-scenario-template",
        type=str,
        default="tier{tier}_single",
        help="Scenario used for SPLIT baseline comparison",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "ciwidth_vs_compute.png",
    )
    args = parser.parse_args()

    split_df = pd.read_csv(args.splitting_csv)

    fig, axes = plt.subplots(
        len(args.tiers), 1, figsize=(7.4, 3.2 * len(args.tiers)), sharex=False
    )
    if len(args.tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, args.tiers):
        mc_path = Path(args.mc_template.format(tier=tier))
        is_path = Path(args.is_template.format(tier=tier))

        mc = _load_mc_or_is(mc_path)
        is_df = _load_mc_or_is(is_path)
        split_name = args.split_scenario_template.format(tier=tier)
        split_rows = _load_split(split_df, tier=tier, scenario_name=split_name)

        ax.plot(mc["eval_count"], mc["ci_width"], "o-", label="MC")
        ax.plot(is_df["eval_count"], is_df["ci_width"], "s-", label="IS")
        if not split_rows.empty:
            ax.plot(
                split_rows["eval_count"], split_rows["ci_width"], "d-", label="SPLIT"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("CI width")
        ax.set_title(f"Tier {tier} (baseline single)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Compute proxy: d_min evaluations")
    axes[0].legend(loc="best")

    fig.suptitle("CI Width vs Compute (MC / IS / SPLIT)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
