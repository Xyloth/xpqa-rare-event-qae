from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "results" / "tables"


def main() -> int:
    summary_path = TABLES_DIR / "bench_suite_summary.csv"
    summary = pd.read_csv(summary_path)

    rows = []
    for tier in sorted(summary["tier"].unique()):
        tier_df = summary[summary["tier"] == tier]
        mc_row = tier_df[tier_df["method"] == "MC"].iloc[0]

        ci_target = float(mc_row["ci_target"])
        epsilon_proxy = ci_target / 2.0
        n_max = int(mc_row["n_max"]) if "n_max" in mc_row else None

        n_achieved = mc_row["n_achieved"]
        mc_is_lower_bound = False
        if isinstance(n_achieved, str) and n_achieved.startswith(">"):
            mc_samples = n_max
            mc_is_lower_bound = True
        else:
            mc_samples = int(n_achieved)

        ae_calls_proxy = int(math.ceil(1.0 / epsilon_proxy))
        mc_over_ae_ratio = mc_samples / ae_calls_proxy if ae_calls_proxy > 0 else float("nan")

        rows.append(
            {
                "tier": int(tier),
                "ci_target": ci_target,
                "epsilon_proxy": epsilon_proxy,
                "mc_samples": mc_samples,
                "mc_is_lower_bound": mc_is_lower_bound,
                "ae_calls_proxy": ae_calls_proxy,
                "mc_over_ae_ratio": mc_over_ae_ratio,
            }
        )

    out_path = TABLES_DIR / "crossover_table.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
