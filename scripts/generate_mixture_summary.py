from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = REPO_ROOT / "results" / "tables" / "mixture_suite.csv"
OUT_PATH = REPO_ROOT / "results" / "mixture_summary.md"
PLOT_PATH = REPO_ROOT / "results" / "plots" / "mixture_effects.png"
ESS_PLOT_PATH = REPO_ROOT / "results" / "plots" / "is_ess_diagnostics.png"

CI_TARGETS = {
    1: 2e-3,
    2: 2e-5,
    3: 2e-6,
}


def fmt_ratio(x: float) -> str:
    if np.isnan(x) or np.isinf(x):
        return "n/a"
    return f"{x:.2f}x"


def fmt_sci(x: float) -> str:
    return f"{x:.3e}"


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return np.nan
    return a / b


def min_n_for_target(df: pd.DataFrame, ci_target: float, n_max: int) -> str | int:
    sub = df.sort_values("N")
    hit = sub[sub["ci_width"] <= ci_target]
    if hit.empty:
        return f">{n_max}"
    return int(hit.iloc[0]["N"])


def target_status_text(method: str, n_result: str | int, n_max: int) -> str:
    if isinstance(n_result, str) and n_result.startswith(">"):
        return f"{method} did NOT meet CI target within n_max={n_max}"

    n_val = int(n_result)
    if n_val == n_max:
        return f"{method} meets CI target at N={n_max} (only at n_max)"
    return f"{method} meets CI target at N={n_val}"


def fetch_row(df: pd.DataFrame, scenario: str, method: str, n: int) -> pd.Series:
    row = df[
        (df["scenario_name"] == scenario) & (df["method"] == method) & (df["N"] == n)
    ]
    if row.empty:
        raise ValueError(f"Missing row for {scenario}, {method}, N={n}")
    return row.iloc[0]


def main() -> int:
    df = pd.read_csv(TABLE_PATH)
    n_max = int(df["n_max"].iloc[0])
    n_list = df["n_list"].iloc[0]
    tiers = sorted(df["tier"].unique())

    mix_row = df[df["distribution"] == "mixture"].iloc[0]
    mix_weight = float(mix_row["mixture_weight"])
    inflation_k = float(mix_row["inflation_k"])
    mean_shift_flag = bool(mix_row["mean_shift_flag"])
    near_margin = float(mix_row.get("near_margin", np.nan))

    lines: list[str] = []
    lines.append("# Mixture Suite Summary")
    lines.append("")
    lines.append("## Scenario")
    lines.append(f"- Tiers: {tiers}, n_max={n_max}, n_list={n_list}")
    lines.append(
        f"- Mixture wide-mode config: weight={mix_weight:.3f}, inflation_k={inflation_k:.1f}, mean_shift={mean_shift_flag}"
    )
    lines.append(f"- Near-threshold margin_frac={near_margin:.3f}")
    lines.append(
        "- Near-threshold regimes are deliberate decision-boundary stress tests (mean miss distance near d0); not intended as typical operating distributions."
    )
    lines.append("")

    key_bullets: list[str] = []

    for tier in tiers:
        target = CI_TARGETS.get(int(tier), 2e-5)

        base_single = f"tier{tier}_single"
        base_mix = f"tier{tier}_mixture"
        near_mix = f"tier{tier}_near_mixture"

        mc_single_nmax = fetch_row(df, base_single, "MC", n_max)
        mc_mix_nmax = fetch_row(df, base_mix, "MC", n_max)
        is_mix_nmax = fetch_row(df, base_mix, "IS", n_max)

        mc_single = df[(df["scenario_name"] == base_single) & (df["method"] == "MC")]
        mc_mix = df[(df["scenario_name"] == base_mix) & (df["method"] == "MC")]
        is_mix = df[(df["scenario_name"] == base_mix) & (df["method"] == "IS")]
        mc_single_req = min_n_for_target(mc_single, target, n_max)
        mc_mix_req = min_n_for_target(mc_mix, target, n_max)
        is_mix_req = min_n_for_target(is_mix, target, n_max)

        mc_single_status = target_status_text("MC (single)", mc_single_req, n_max)
        mc_mix_status = target_status_text("MC (mixture)", mc_mix_req, n_max)
        is_mix_status = target_status_text("IS (mixture)", is_mix_req, n_max)

        mc_single_hits = int(mc_single_nmax["hits"])
        mc_mix_hits = int(mc_mix_nmax["hits"])

        if mc_single_hits == 0:
            key_bullets.append(
                f"Tier {tier}: baseline MC has 0 hits at N={n_max}; mixture MC has {mc_mix_hits} hits, "
                f"Pc_hat={fmt_sci(float(mc_mix_nmax['pc_hat']))}, CI_width={fmt_sci(float(mc_mix_nmax['ci_width']))}."
            )
        else:
            pc_mult = safe_ratio(
                float(mc_mix_nmax["pc_hat"]), float(mc_single_nmax["pc_hat"])
            )
            key_bullets.append(
                f"Tier {tier}: baseline->mixture increases MC Pc by {fmt_ratio(pc_mult)} at N={n_max}."
            )

        key_bullets.append(
            f"Tier {tier}: {mc_single_status}; {mc_mix_status}; {is_mix_status}."
        )

        ci_mitig = safe_ratio(
            float(mc_mix_nmax["ci_width"]), float(is_mix_nmax["ci_width"])
        )
        key_bullets.append(
            f"Tier {tier}: for mixture at N={n_max}, IS CI width is {fmt_ratio(ci_mitig)} tighter than MC."
        )

        # Near-threshold amplification diagnostic.
        mc_base_mix = fetch_row(df, base_mix, "MC", n_max)
        mc_near_mix = fetch_row(df, near_mix, "MC", n_max)
        near_pc_mult = safe_ratio(
            float(mc_near_mix["pc_hat"]), float(mc_base_mix["pc_hat"])
        )
        key_bullets.append(
            f"Tier {tier}: near-threshold mixture amplifies MC Pc by {fmt_ratio(near_pc_mult)} vs baseline mixture at N={n_max}."
        )

        # IS diagnostics: baseline single vs baseline mixture (+ near-threshold mixture).
        is_base_single = fetch_row(df, base_single, "IS", n_max)
        is_base_mix = fetch_row(df, base_mix, "IS", n_max)
        is_near_mix = fetch_row(df, near_mix, "IS", n_max)
        key_bullets.append(
            f"Tier {tier} IS ESS/N at N={n_max}: baseline single={float(is_base_single['ess_over_n']):.4f}, "
            f"baseline mixture={float(is_base_mix['ess_over_n']):.4f}, near-threshold mixture={float(is_near_mix['ess_over_n']):.4f}."
        )

    lines.append("## Headline Findings")
    for bullet in key_bullets:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Table: {TABLE_PATH}")
    lines.append(f"- Plot: {PLOT_PATH}")
    lines.append(f"- IS diagnostics plot: {ESS_PLOT_PATH}")

    text = "\n".join(lines).strip() + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
