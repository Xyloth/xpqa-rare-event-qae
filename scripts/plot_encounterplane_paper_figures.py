from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_METHOD_STYLE = {
    "MC": {"color": "tab:blue", "marker": "o"},
    "IS": {"color": "tab:orange", "marker": "s"},
    "SPLIT": {"color": "tab:green", "marker": "d"},
}

_VARIANT_LABEL = {
    "baseline_single": "single",
    "baseline_mixture": "mixture",
    "near_single": "single",
    "near_mixture": "mixture",
}


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def _prepare_data(
    replicate_csv: Path,
    suite_csv: Path,
) -> pd.DataFrame:
    rep = pd.read_csv(replicate_csv)
    rep = rep.rename(columns={"pc_hat_mean": "pc_hat", "eval_count_mean": "eval_count"})
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
        ]
    ]

    suite = pd.read_csv(suite_csv)
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
        ]
    ]

    return pd.concat([rep, split], ignore_index=True)


def _plot_family(
    df: pd.DataFrame,
    tiers: list[int],
    variants: list[str],
    title: str,
    subtitle: str | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(tiers), figsize=(11.5, 4.2), sharey=True)
    if len(tiers) == 1:
        axes = [axes]

    legend_handles = {}

    for ax, tier in zip(axes, tiers):
        sub_tier = df[(df["tier"] == tier) & (df["variant"].isin(variants))]

        for method in ["MC", "IS", "SPLIT"]:
            for variant in variants:
                rows = sub_tier[
                    (sub_tier["method"] == method) & (sub_tier["variant"] == variant)
                ]
                if rows.empty:
                    continue
                rows = rows.sort_values("N")
                style = _METHOD_STYLE[method]
                dist_label = _VARIANT_LABEL[variant]
                line_style = "-" if dist_label == "single" else "--"
                label = f"{method} {dist_label}"
                line = ax.plot(
                    rows["N"],
                    rows["ci_width"],
                    linestyle=line_style,
                    marker=style["marker"],
                    color=style["color"],
                    linewidth=1.6,
                    markersize=4,
                    label=label,
                )[0]
                legend_handles[label] = line

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Samples N")
        ax.set_title(f"Tier {tier}")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("CI width")

    fig.suptitle(title, y=0.98)
    if subtitle:
        fig.text(
            0.5,
            0.94,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
        )

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality encounter-plane figures"
    )
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
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument(
        "--out-baseline",
        type=Path,
        default=REPO_ROOT / "results" / "plots" / "encounterplane_paper_baseline.png",
    )
    parser.add_argument(
        "--out-near",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "plots"
        / "encounterplane_paper_nearthreshold.png",
    )
    args = parser.parse_args()

    df = _prepare_data(args.replicate_csv, args.suite_csv)

    _plot_family(
        df=df,
        tiers=args.tiers,
        variants=["baseline_single", "baseline_mixture"],
        title="Encounter-plane baseline scenarios: CI width vs N",
        subtitle="MC/IS CI from replicate runs; SPLIT CI from replicate suite",
        out_path=args.out_baseline,
    )

    _plot_family(
        df=df,
        tiers=args.tiers,
        variants=["near_single", "near_mixture"],
        title="Encounter-plane near-threshold scenarios: CI width vs N",
        subtitle="Decision-boundary stress test (mean miss distance near d0)",
        out_path=args.out_near,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
