from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MC vs IS convergence")
    parser.add_argument(
        "--mc",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "mc_tier1.csv",
        help="Input MC CSV path",
    )
    parser.add_argument(
        "--is",
        dest="is_csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "is_tier1.csv",
        help="Input IS CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "compare_mc_vs_is.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    mc = pd.read_csv(args.mc).sort_values("n")
    is_df = pd.read_csv(args.is_csv).sort_values("n")

    for col in ("n", "pc_hat", "ci_low", "ci_high"):
        if col not in mc.columns or col not in is_df.columns:
            raise ValueError(f"Missing required column '{col}' in input CSVs")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)

    mc_yerr = [mc["pc_hat"] - mc["ci_low"], mc["ci_high"] - mc["pc_hat"]]
    is_yerr = [
        is_df["pc_hat"] - is_df["ci_low"],
        is_df["ci_high"] - is_df["pc_hat"],
    ]

    ax1.errorbar(mc["n"], mc["pc_hat"], yerr=mc_yerr, fmt="o-", capsize=3)
    ax1.errorbar(is_df["n"], is_df["pc_hat"], yerr=is_yerr, fmt="s-", capsize=3)
    ax1.set_xscale("log")
    ax1.set_ylabel("Estimated Pc")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(["MC", "IS"], loc="best")

    mc_width = mc["ci_high"] - mc["ci_low"]
    is_width = is_df["ci_high"] - is_df["ci_low"]
    ax2.plot(mc["n"], mc_width, "o-", label="MC CI width")
    ax2.plot(is_df["n"], is_width, "s-", label="IS CI width")
    ax2.set_xscale("log")
    ax2.set_xlabel("Samples N")
    ax2.set_ylabel("CI width")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="best")

    tier = "?"
    if "tier" in mc.columns:
        tier = str(mc["tier"].iloc[0])
    elif "tier" in is_df.columns:
        tier = str(is_df["tier"].iloc[0])

    fig.suptitle(f"Tier-1 model, scenario tier={tier}: MC vs IS")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
