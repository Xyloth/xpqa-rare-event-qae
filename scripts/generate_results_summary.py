from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def load_ci_width(path: Path, n: int) -> float:
    df = pd.read_csv(path)
    row = df[df["n"] == n]
    if row.empty:
        raise ValueError(f"No row for n={n} in {path}")
    row = row.iloc[0]
    return float(row["ci_high"] - row["ci_low"])


def find_n_list(tiers: list[int]) -> list[int] | None:
    for tier in tiers:
        path = TABLES_DIR / f"mc_tier{tier}.csv"
        if path.exists():
            df = pd.read_csv(path)
            return sorted(df["n"].unique().tolist())
    return None


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    summary_path = TABLES_DIR / "bench_suite_summary.csv"
    summary = pd.read_csv(summary_path)

    tiers = sorted(summary["tier"].unique())
    ci_targets = (
        summary.groupby("tier")["ci_target"].first().to_dict()  # type: ignore[call-arg]
    )
    n_max = int(summary["n_max"].iloc[0]) if "n_max" in summary.columns else None
    n_list = find_n_list(tiers)

    # Build per-tier table
    table_rows = []
    for tier in tiers:
        tier_df = summary[summary["tier"] == tier]
        for _, row in tier_df.iterrows():
            table_rows.append(
                {
                    "Tier": int(row["tier"]),
                    "Method": row["method"],
                    "n_achieved": row["n_achieved"],
                    "ci_width_nmax": fmt_sci(float(row["ci_width_nmax"])),
                    "pc_hat_nmax": fmt_sci(float(row["pc_hat_nmax"])),
                }
            )

    table_df = pd.DataFrame(table_rows)

    # Ratios at n_max (tier 2 and 3)
    ratios = {}
    for tier in [2, 3]:
        tier_rows = summary[summary["tier"] == tier]
        if tier_rows.empty:
            ratios[tier] = None
            continue
        n_max = int(tier_rows["n_max"].iloc[0])
        try:
            mc_width = load_ci_width(TABLES_DIR / f"mc_tier{tier}.csv", n_max)
            is_width = load_ci_width(TABLES_DIR / f"is_tier{tier}.csv", n_max)
            ratios[tier] = mc_width / is_width if is_width > 0 else None
        except Exception:
            ratios[tier] = None

    # Interpretation bullets
    def tier_row(method: str, tier: int) -> pd.Series:
        return summary[(summary["tier"] == tier) & (summary["method"] == method)].iloc[0]

    interp = []
    for tier in [2, 3]:
        mc = tier_row("MC", tier)
        is_row = tier_row("IS", tier)
        mc_n = mc["n_achieved"]
        is_n = is_row["n_achieved"]
        shift_frac = is_row.get("shift_frac", None)
        if isinstance(mc_n, str) and mc_n.startswith(">"):
            mc_text = f"MC fails CI target by Nmax ({mc_n})."
        else:
            mc_text = f"MC meets CI target by N≈{mc_n}."

        if isinstance(is_n, str) and is_n.startswith(">"):
            is_text = "IS also fails to meet CI target by Nmax."
        else:
            if pd.notna(shift_frac):
                is_text = f"IS meets CI target by N≈{is_n} with shift_frac={shift_frac:.2f}."
            else:
                is_text = f"IS meets CI target by N≈{is_n}."
        interp.append(f"Tier {tier}: {mc_text} {is_text}")

    ratio_t3 = ratios.get(3)
    ratio_t2 = ratios.get(2)
    if ratio_t3 is not None:
        interp.append(
            f"Tier 3: IS reduces CI width by ~{ratio_t3:.1f}x vs MC at N=100k."
        )
    if ratio_t2 is not None:
        interp.append(
            f"Tier 2: IS reduces CI width by ~{ratio_t2:.1f}x vs MC at N=100k."
        )

    plot_path = PLOTS_DIR / "ciwidth_across_tiers.png"

    lines = []
    lines.append("# Results Summary")
    lines.append("")
    lines.append("## CI Targets")
    for tier in tiers:
        lines.append(f"- Tier {tier}: {fmt_sci(float(ci_targets[tier]))}")
    lines.append("")
    lines.append("## Config")
    if n_max is not None:
        lines.append(f"- n_max: {n_max}")
    if n_list is not None:
        lines.append(f"- n_list: {n_list}")
    lines.append("")
    lines.append("## Per-Tier Results")
    lines.append(to_markdown_table(table_df))
    lines.append("")
    lines.append("## CI Width Ratios at N=100000")
    for tier in [2, 3]:
        ratio = ratios.get(tier)
        if ratio is None:
            lines.append(f"- Tier {tier}: ratio unavailable")
        else:
            lines.append(f"- Tier {tier}: MC/IS = {ratio:.2f}")
    lines.append("")
    lines.append("## Interpretation")
    for bullet in interp[:5]:
        lines.append(f"- {bullet}")
    lines.append(
        "- Note: shift_frac may differ between the sweep script and bench suite due to deterministic tier seed offsets."
    )
    lines.append("")
    lines.append("## Plots")
    lines.append(f"- {plot_path}")

    summary_md = "\n".join(lines).strip() + "\n"

    out_path = RESULTS_DIR / "summary.md"
    out_path.write_text(summary_md, encoding="utf-8")
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
