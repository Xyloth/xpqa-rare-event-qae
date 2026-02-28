from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "results" / "tables"
PLOTS_DIR = REPO_ROOT / "results" / "plots"


def ci_width_ratio(tier: int, n_max: int) -> float:
    mc = pd.read_csv(TABLES_DIR / f"mc_tier{tier}.csv")
    is_df = pd.read_csv(TABLES_DIR / f"is_tier{tier}.csv")
    mc_row = mc[mc["n"] == n_max].iloc[0]
    is_row = is_df[is_df["n"] == n_max].iloc[0]
    mc_w = float(mc_row["ci_high"] - mc_row["ci_low"])
    is_w = float(is_row["ci_high"] - is_row["ci_low"])
    return mc_w / is_w if is_w > 0 else float("nan")


def main() -> int:
    summary = pd.read_csv(TABLES_DIR / "bench_suite_summary.csv")
    tiers = sorted(summary["tier"].unique())
    n_max = int(summary["n_max"].iloc[0])

    mc_failed = []
    is_met = []
    for tier in tiers:
        tier_df = summary[summary["tier"] == tier]
        mc_row = tier_df[tier_df["method"] == "MC"].iloc[0]
        is_row = tier_df[tier_df["method"] == "IS"].iloc[0]

        mc_n = mc_row["n_achieved"]
        if isinstance(mc_n, str) and mc_n.startswith(">"):
            mc_failed.append(str(tier))

        is_n = is_row["n_achieved"]
        if not (isinstance(is_n, str) and is_n.startswith(">")):
            is_met.append((int(tier), is_n))

    ratio_t2 = ci_width_ratio(2, n_max) if 2 in tiers else None
    ratio_t3 = ci_width_ratio(3, n_max) if 3 in tiers else None

    lines = []
    lines.append("# Benchmark Card")
    lines.append("")
    lines.append("Pc estimation under uncertainty for conjunction-like encounters.")
    lines.append("We benchmark rare-event estimation difficulty under a tiered analytic model.")
    lines.append("")
    lines.append("## Implemented")
    lines.append(
        "- Tier-1 analytic closest-approach model with Gaussian initial uncertainty."
    )
    lines.append("- Baselines: Monte Carlo and importance sampling (IS).")
    lines.append("- Tiered difficulty settings (tiers 1/2/3).")
    lines.append("")
    lines.append("## Headline Results")
    if ratio_t2 is not None:
        lines.append(f"- Tier 2: MC/IS CI width ratio at N={n_max} is {ratio_t2:.2f}x.")
    if ratio_t3 is not None:
        lines.append(f"- Tier 3: MC/IS CI width ratio at N={n_max} is {ratio_t3:.2f}x.")
    if mc_failed:
        lines.append(f"- MC fails CI target by Nmax for tiers: {', '.join(mc_failed)}.")
    if is_met:
        entries = [f"tier {tier} at N≈{n}" for tier, n in is_met]
        lines.append("- IS meets CI target for " + ", ".join(entries) + ".")
    lines.append("")
    lines.append("## Plots")
    lines.append(f"- {PLOTS_DIR / 'ciwidth_across_tiers.png'}")
    lines.append("")

    out_path = REPO_ROOT / "results" / "benchmark_card.md"
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
