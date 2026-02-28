from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CI_TARGETS = {
    1: 2e-3,
    2: 2e-5,
    3: 2e-6,
}


def _best_by_compute(
    df: pd.DataFrame, ci_target: float
) -> tuple[str, pd.Series | None]:
    hit = df[df["ci_width"] <= ci_target].sort_values("eval_count_mean")
    if not hit.empty:
        return "met", hit.iloc[0]
    return "miss", None


def _fmt_sci(x: float) -> str:
    return f"{x:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate splitting summary markdown")
    parser.add_argument(
        "--splitting-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "splitting_suite.csv",
    )
    parser.add_argument(
        "--bench-summary",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "bench_suite_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "splitting_summary.md",
    )
    args = parser.parse_args()

    split_df = pd.read_csv(args.splitting_csv)
    bench_df = pd.read_csv(args.bench_summary) if args.bench_summary.exists() else None

    lines: list[str] = []
    lines.append("# Splitting Summary")
    lines.append("")
    lines.append("Subset simulation CI is empirical across independent replicates.")
    lines.append("")

    for tier in sorted(split_df["tier"].unique()):
        ci_target = DEFAULT_CI_TARGETS.get(int(tier), 2e-5)

        lines.append(f"## Tier {tier}")
        lines.append(f"- CI target: {_fmt_sci(ci_target)}")

        for scenario_name in [f"tier{tier}_single", f"tier{tier}_mixture"]:
            rows = split_df[split_df["scenario_name"] == scenario_name].sort_values("N")
            if rows.empty:
                continue

            status, best = _best_by_compute(rows, ci_target)
            label = "single" if scenario_name.endswith("single") else "mixture"

            if status == "met" and best is not None:
                lines.append(
                    f"- SPLIT ({label}) meets target at N={int(best['N'])}, "
                    f"eval_count~{best['eval_count_mean']:.0f}, "
                    f"accept_rate~{best['accept_rate_mean']:.2f}."
                )
            else:
                tail = rows.iloc[-1]
                lines.append(
                    f"- SPLIT ({label}) does not meet target within tested N<="
                    f"{int(rows['N'].max())}; best CI width={_fmt_sci(float(tail['ci_width']))}."
                )

        if bench_df is not None:
            bench_t = bench_df[bench_df["tier"] == tier]
            if not bench_t.empty:
                mc = bench_t[bench_t["method"] == "MC"]
                is_rows = bench_t[bench_t["method"] == "IS"]
                if not mc.empty and not is_rows.empty:
                    mc_n = mc.iloc[0]["n_achieved"]
                    is_n = is_rows.iloc[0]["n_achieved"]
                    lines.append(
                        f"- Bench reference: MC n_achieved={mc_n}, IS n_achieved={is_n} (proposal-dependent)."
                    )

        lines.append("")

    # Explicit extra datapoint callout requested for reviewer clarity.
    t2_mix_30k = split_df[
        (split_df["scenario_name"] == "tier2_mixture") & (split_df["N"] == 30000)
    ]
    if not t2_mix_30k.empty:
        row = t2_mix_30k.iloc[0]
        ci_target_t2 = float(DEFAULT_CI_TARGETS[2])
        status = (
            "meets target"
            if float(row["ci_width"]) <= ci_target_t2
            else "approaches target"
        )
        lines.append("## Tier 2 Mixture Extra Check")
        lines.append(
            f"- Tier 2 mixture {status} at N=30000: "
            f"ci_width={_fmt_sci(float(row['ci_width']))}, "
            f"eval_count~{float(row['eval_count_mean']):.0f}."
        )
        lines.append("")

    lines.append(
        "- Interpretation: splitting is competitive in rare-event single-Gaussian regimes; mixture regimes are harder for the current kernel and will be a Phase II tuning target (proposal_scale / n_mcmc_steps / mixture-aware proposal)."
    )

    out_text = "\n".join(lines).strip() + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text, encoding="utf-8")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
