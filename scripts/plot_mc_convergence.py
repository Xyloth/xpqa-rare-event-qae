from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MC convergence")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "mc_tier1.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "mc_convergence.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "n" not in df.columns:
        raise ValueError("CSV missing required column 'n'")

    df = df.sort_values("n")
    yerr_low = df["pc_hat"] - df["ci_low"]
    yerr_high = df["ci_high"] - df["pc_hat"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        df["n"],
        df["pc_hat"],
        yerr=[yerr_low, yerr_high],
        fmt="o-",
        capsize=3,
        linewidth=1.5,
        markersize=4,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Samples N")
    ax.set_ylabel("Estimated Pc")

    tier = df["tier"].iloc[0] if "tier" in df.columns else "?"
    ax.set_title(f"Tier-1 model, scenario tier={tier}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
