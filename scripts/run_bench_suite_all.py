from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def min_n_for_target(df: pd.DataFrame, ci_target: float, n_max: int) -> str | int:
    meets = df[df["ci_width"] <= ci_target]
    if meets.empty:
        return f">{n_max}"
    return int(meets.iloc[0]["N"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge MC/IS bench summary with SPLIT baseline summary"
    )
    parser.add_argument(
        "--bench-summary",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "bench_suite_summary.csv",
    )
    parser.add_argument(
        "--splitting-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "splitting_suite.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "bench_suite_all_summary.csv",
    )
    args = parser.parse_args()

    bench_df = pd.read_csv(args.bench_summary)
    split_df = pd.read_csv(args.splitting_csv)

    out_rows: list[dict[str, float | int | str]] = []
    for _, row in bench_df.iterrows():
        out_rows.append(row.to_dict())

    tiers = sorted(bench_df["tier"].unique())
    for tier in tiers:
        ci_target = float(bench_df[(bench_df["tier"] == tier)]["ci_target"].iloc[0])
        split_rows = split_df[
            (split_df["tier"] == tier)
            & (split_df["scenario_name"] == f"tier{tier}_single")
        ].sort_values("N")
        if split_rows.empty:
            continue

        n_max = int(split_rows["N"].max())
        n_achieved = min_n_for_target(split_rows, ci_target, n_max)
        final = split_rows.iloc[-1]

        out_rows.append(
            {
                "tier": int(tier),
                "method": "SPLIT",
                "ci_target": ci_target,
                "n_achieved": n_achieved,
                "pc_hat_nmax": float(final["pc_hat"]),
                "ci_width_nmax": float(final["ci_width"]),
                "seed": int(final["seed_base"]),
                "n_max": n_max,
                "eval_count_nmax": float(final["eval_count_mean"]),
                "accept_rate_nmax": float(final["accept_rate_mean"]),
                "n_reps": int(final["n_reps"]),
            }
        )

    out_df = (
        pd.DataFrame(out_rows).sort_values(["tier", "method"]).reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
