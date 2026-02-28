from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CI_TARGETS = {
    2: 2e-5,
    3: 2e-6,
}

VARIANT_LABEL = {
    "baseline_single": "baseline single",
    "baseline_mixture": "baseline mixture",
    "near_single": "near-threshold single",
    "near_mixture": "near-threshold mixture",
}


def fmt_sci(x: float) -> str:
    return f"{x:.3e}"


def n_to_target(df: pd.DataFrame, ci_target: float, n_max: int) -> str | int:
    hit = df[df["ci_width"] <= ci_target].copy()
    if "hit_proxy" in hit.columns:
        hit = hit[hit["hit_proxy"] > 0]
    hit = hit.sort_values("N")
    if hit.empty:
        return f">{n_max}"
    return int(hit.iloc[0]["N"])


def to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _load_combined(
    replicate_csv: Path,
    suite_csv: Path,
) -> tuple[pd.DataFrame, list[str]]:
    suite = pd.read_csv(suite_csv)
    notes: list[str] = []

    frames: list[pd.DataFrame] = []
    if replicate_csv.exists():
        rep = pd.read_csv(replicate_csv)
        rep = rep.rename(
            columns={"pc_hat_mean": "pc_hat", "eval_count_mean": "eval_count"}
        )
        rep = rep[
            [
                "scenario_name",
                "tier",
                "variant",
                "family",
                "distribution",
                "method",
                "N",
                "pc_hat",
                "ci_low",
                "ci_high",
                "ci_width",
                "eval_count",
                "hits_mean",
            ]
        ]
        rep = rep.rename(columns={"hits_mean": "hit_proxy"})
        rep = rep[rep["method"].isin(["MC", "IS"])].copy()
        frames.append(rep)
        notes.append(
            "MC/IS CI widths: empirical across independent replicates (normal approximation on replicate means)."
        )
    else:
        approx = suite[suite["method"].isin(["MC", "IS"])].copy()
        approx = approx[
            [
                "scenario_name",
                "tier",
                "variant",
                "family",
                "distribution",
                "method",
                "N",
                "pc_hat",
                "ci_low",
                "ci_high",
                "ci_width",
                "eval_count",
                "hits",
            ]
        ]
        approx = approx.rename(columns={"hits": "hit_proxy"})
        frames.append(approx)
        notes.append(
            "MC/IS CI widths: single-run approximations from encounterplane_suite.csv."
        )

    split = suite[suite["method"] == "SPLIT"].copy()
    split = split[
        [
            "scenario_name",
            "tier",
            "variant",
            "family",
            "distribution",
            "method",
            "N",
            "pc_hat",
            "ci_low",
            "ci_high",
            "ci_width",
            "eval_count",
            "hits",
        ]
    ]
    split = split.rename(columns={"hits": "hit_proxy"})
    frames.append(split)
    notes.append(
        "SPLIT CI widths: empirical across replicate runs within splitting estimator outputs."
    )

    combined = pd.concat(frames, ignore_index=True)
    return combined, notes


def _tail_row(df: pd.DataFrame, tier: int, variant: str, method: str) -> pd.Series:
    rows = df[
        (df["tier"] == tier) & (df["variant"] == variant) & (df["method"] == method)
    ]
    return rows.sort_values("N").iloc[-1]


def _common_n(df: pd.DataFrame, tier: int, variant: str, methods: list[str]) -> int:
    maxima = []
    for method in methods:
        rows = df[
            (df["tier"] == tier) & (df["variant"] == variant) & (df["method"] == method)
        ]
        maxima.append(int(rows["N"].max()))
    return min(maxima)


def _row_at_n(
    df: pd.DataFrame, tier: int, variant: str, method: str, n: int
) -> pd.Series:
    rows = df[
        (df["tier"] == tier)
        & (df["variant"] == variant)
        & (df["method"] == method)
        & (df["N"] == n)
    ]
    return rows.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate encounter-plane summary")
    parser.add_argument(
        "--replicate-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_replicates.csv",
    )
    parser.add_argument(
        "--suite-csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_suite.csv",
    )
    parser.add_argument(
        "--tier1-summary",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "bench_suite_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "encounterplane_summary.md",
    )
    args = parser.parse_args()

    df, ci_method_notes = _load_combined(args.replicate_csv, args.suite_csv)
    tiers = sorted(df["tier"].unique())
    n_max = int(df["N"].max())
    n_list = sorted(df["N"].unique().tolist())

    table_rows: list[dict[str, str | int]] = []
    for tier in tiers:
        ci_target = CI_TARGETS.get(int(tier), 2e-5)
        for variant in [
            "baseline_single",
            "baseline_mixture",
            "near_single",
            "near_mixture",
        ]:
            sub = df[(df["tier"] == tier) & (df["variant"] == variant)]
            if sub.empty:
                continue
            row = {"Tier": int(tier), "Scenario": VARIANT_LABEL[variant]}
            for method in ["MC", "IS", "SPLIT"]:
                row[method] = n_to_target(
                    sub[sub["method"] == method].sort_values("N"),
                    ci_target,
                    n_max,
                )
            table_rows.append(row)
    target_table = pd.DataFrame(table_rows)

    n_common_t3 = _common_n(
        df, tier=3, variant="baseline_single", methods=["MC", "IS", "SPLIT"]
    )
    t3_mc = _row_at_n(df, tier=3, variant="baseline_single", method="MC", n=n_common_t3)
    t3_is = _row_at_n(df, tier=3, variant="baseline_single", method="IS", n=n_common_t3)
    t3_split = _row_at_n(
        df,
        tier=3,
        variant="baseline_single",
        method="SPLIT",
        n=n_common_t3,
    )

    ratio_t3_mc_is = float(t3_mc["ci_width"]) / float(t3_is["ci_width"])
    ratio_t3_mc_split = float(t3_mc["ci_width"]) / float(t3_split["ci_width"])

    t3_mix_mc = _row_at_n(
        df, tier=3, variant="baseline_mixture", method="MC", n=n_common_t3
    )
    t3_mix_is = _row_at_n(
        df, tier=3, variant="baseline_mixture", method="IS", n=n_common_t3
    )
    t3_mix_split = _row_at_n(
        df,
        tier=3,
        variant="baseline_mixture",
        method="SPLIT",
        n=n_common_t3,
    )

    mix_ratio_mc = float(t3_mix_mc["ci_width"]) / float(t3_mc["ci_width"])
    mix_ratio_is = float(t3_mix_is["ci_width"]) / float(t3_is["ci_width"])
    mix_ratio_split = float(t3_mix_split["ci_width"]) / float(t3_split["ci_width"])

    tier1_note = ""
    if args.tier1_summary.exists():
        tier1 = pd.read_csv(args.tier1_summary)
        t3_mc_t1 = tier1[(tier1["tier"] == 3) & (tier1["method"] == "MC")]
        t3_is_t1 = tier1[(tier1["tier"] == 3) & (tier1["method"] == "IS")]
        if not t3_mc_t1.empty and not t3_is_t1.empty:
            ratio_t1 = float(t3_mc_t1.iloc[0]["ci_width_nmax"]) / float(
                t3_is_t1.iloc[0]["ci_width_nmax"]
            )
            tier1_note = (
                f"Tier-1 reference (6D model) had MC/IS CI-width ratio ~{ratio_t1:.1f}x "
                f"at N={int(t3_mc_t1.iloc[0]['n_max'])}."
            )

    lines: list[str] = []
    lines.append("# Encounter-Plane Summary")
    lines.append("")
    lines.append(
        "This section uses a conjunction-recognizable encounter-plane uncertainty model: "
        "relative position at close approach is represented in 2D, and the event predicate is `d < d0`."
    )
    lines.append(
        "Single-Gaussian and Gaussian-mixture uncertainty are both supported, alongside near-threshold boundary-layer regimes."
    )
    lines.append("")
    lines.append("## Config")
    lines.append(f"- n_max: {n_max}")
    lines.append(f"- n_list: {n_list}")
    lines.append(
        f"- CI targets: tier2={fmt_sci(CI_TARGETS[2])}, tier3={fmt_sci(CI_TARGETS[3])}"
    )
    lines.append("")
    lines.append("## N Needed To Hit CI Target")
    lines.append(to_md_table(target_table))
    lines.append("")
    lines.append("## Headline Results")
    lines.append(
        f"- Tier 3 baseline single @ N={n_common_t3}: MC CI width={fmt_sci(float(t3_mc['ci_width']))}, "
        f"IS={fmt_sci(float(t3_is['ci_width']))}, SPLIT={fmt_sci(float(t3_split['ci_width']))}."
    )
    lines.append(
        f"- Tier 3 baseline single improvement: MC/IS={ratio_t3_mc_is:.1f}x, MC/SPLIT={ratio_t3_mc_split:.1f}x by CI width."
    )
    lines.append(
        f"- Tier 3 mixture effect at N={n_common_t3}: CI-width inflation vs baseline is "
        f"MC={mix_ratio_mc:.1f}x, IS={mix_ratio_is:.1f}x, SPLIT={mix_ratio_split:.1f}x."
    )
    if tier1_note:
        lines.append(f"- {tier1_note}")
    lines.append("")
    lines.append("## Method Notes")
    lines.append("- All CI widths shown are computed as:")
    for note in ci_method_notes:
        lines.append(f"  {note}")
    lines.append(
        "- Near-threshold scenarios are deliberate stress tests to emulate decision-boundary refinement, not nominal operating distributions."
    )
    lines.append(
        "- Mixture scenarios represent non-Gaussian uncertainty contamination; mixture-aware proposals are a Phase II tuning lever (for IS and SPLIT kernels)."
    )
    lines.append("")
    lines.append("## Plots")
    lines.append(
        f"- {REPO_ROOT / 'results' / 'plots' / 'encounterplane_paper_baseline.png'}"
    )
    lines.append(
        f"- {REPO_ROOT / 'results' / 'plots' / 'encounterplane_paper_nearthreshold.png'}"
    )
    lines.append(f"- {REPO_ROOT / 'results' / 'plots' / 'encounterplane_effects.png'}")

    out_text = "\n".join(lines).strip() + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text, encoding="utf-8")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
