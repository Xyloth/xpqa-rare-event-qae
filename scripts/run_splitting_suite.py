from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xq.benchmark import normalized_n_list  # noqa: E402
from xq.distributions import (  # noqa: E402
    mixture_scenario_params,
    near_threshold_mean,
    tier_params,
)
from xq.splitting import estimate_pc_splitting  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def parse_n_list(text: str) -> list[int]:
    return parse_int_list(text)


def ci_from_replicates(
    values: np.ndarray, z: float = 1.96
) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half = z * std / np.sqrt(values.size)
    return max(0.0, mean - half), min(1.0, mean + half), std


def scenario_definitions() -> list[dict[str, object]]:
    return [
        {
            "family": "baseline",
            "distribution": "single_gaussian",
            "near_threshold": False,
            "is_mixture": False,
            "name_suffix": "single",
        },
        {
            "family": "baseline",
            "distribution": "mixture",
            "near_threshold": False,
            "is_mixture": True,
            "name_suffix": "mixture",
        },
        {
            "family": "near_threshold",
            "distribution": "single_gaussian",
            "near_threshold": True,
            "is_mixture": False,
            "name_suffix": "near_single",
        },
        {
            "family": "near_threshold",
            "distribution": "mixture",
            "near_threshold": True,
            "is_mixture": True,
            "name_suffix": "near_mixture",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run splitting/subset-simulation suite"
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=3600.0)
    parser.add_argument("--nmax", type=int, default=10000)
    parser.add_argument("--n-list", type=parse_n_list, default=None)
    parser.add_argument("--n-reps", type=int, default=30)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--n-levels-max", type=int, default=20)
    parser.add_argument("--proposal-scale", type=float, default=0.8)
    parser.add_argument("--n-mcmc-steps", type=int, default=2)
    parser.add_argument("--r-sigma", type=float, default=None)
    parser.add_argument("--v-sigma", type=float, default=None)
    parser.add_argument("--mix-weight", type=float, default=0.05)
    parser.add_argument("--mix-inflation", type=float, default=10.0)
    parser.add_argument("--mix-shift-frac", type=float, default=0.2)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "splitting_suite.csv",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.T < 0:
        raise ValueError("T must be non-negative")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")
    if args.n_reps <= 0:
        raise ValueError("n-reps must be positive")

    n_list = normalized_n_list(args.n_list, args.nmax)

    rows: list[dict[str, float | int | bool | str]] = []
    n_list_str = ",".join(str(n) for n in n_list)

    print("Splitting suite run:")
    print(
        f"  tiers={args.tiers} d0={args.d0} T={args.T} n_list={n_list} "
        f"n_reps={args.n_reps} p0={args.p0}"
    )

    for tier in args.tiers:
        r_sigma = args.r_sigma if args.r_sigma is not None else args.d0
        if args.v_sigma is None:
            v_sigma = 0.0 if args.T == 0 else 0.1 * args.d0 / args.T
        else:
            v_sigma = args.v_sigma

        mean_base, cov6 = tier_params(args.d0, tier, r_sigma=r_sigma, v_sigma=v_sigma)

        for sc_idx, scenario in enumerate(scenario_definitions()):
            near = bool(scenario["near_threshold"])
            is_mixture = bool(scenario["is_mixture"])
            scenario_name = f"tier{tier}_{scenario['name_suffix']}"

            mean6 = (
                near_threshold_mean(mean_base, d0=args.d0, margin_frac=args.near_margin)
                if near
                else mean_base.copy()
            )

            if is_mixture:
                mix_cfg = mixture_scenario_params(
                    mean6=mean6,
                    d0=args.d0,
                    mixture_weight=args.mix_weight,
                    inflation_k=args.mix_inflation,
                    mean_shift6=None,
                    mean_shift_frac=args.mix_shift_frac,
                )
                target = "mixture"
                mix_w = float(mix_cfg["w"])
                mix_k = float(mix_cfg["inflation_k"])
                mean_shift6 = np.asarray(mix_cfg["mean_shift6"], dtype=float)
                mix_params = {
                    "w": mix_w,
                    "inflation_k": mix_k,
                    "mean_shift6": mean_shift6,
                }
            else:
                target = "single"
                mix_w = 0.0
                mix_k = 1.0
                mean_shift6 = np.zeros(6, dtype=float)
                mix_params = None

            for n in n_list:
                pc_hats: list[float] = []
                eval_counts: list[float] = []
                accept_rates: list[float] = []
                n_levels: list[float] = []
                final_hits: list[float] = []
                elapsed_s: list[float] = []

                seed_base = args.seed + tier * 100_000 + sc_idx * 10_000 + int(n) * 13

                for rep in range(args.n_reps):
                    rep_seed = seed_base + rep * 1000
                    stats = estimate_pc_splitting(
                        n=n,
                        mean6=mean6,
                        cov6=cov6,
                        T=args.T,
                        d0=args.d0,
                        p0=args.p0,
                        n_levels_max=args.n_levels_max,
                        proposal_scale=args.proposal_scale,
                        n_mcmc_steps=args.n_mcmc_steps,
                        seed=rep_seed,
                        target=target,
                        mixture_params=mix_params,
                    )

                    diag = stats["diagnostics"]
                    hits_level = diag["hits_per_level"]

                    pc_hats.append(float(stats["pc_hat"]))
                    eval_counts.append(float(stats["eval_count"]))
                    accept_rates.append(float(stats["acceptance_rate"]))
                    n_levels.append(float(stats["n_levels"]))
                    final_hits.append(float(hits_level[-1] if hits_level else 0.0))
                    elapsed_s.append(float(stats["elapsed_s"]))

                pc_arr = np.asarray(pc_hats, dtype=float)
                eval_arr = np.asarray(eval_counts, dtype=float)
                acc_arr = np.asarray(accept_rates, dtype=float)
                lvl_arr = np.asarray(n_levels, dtype=float)
                hit_arr = np.asarray(final_hits, dtype=float)
                elapsed_arr = np.asarray(elapsed_s, dtype=float)

                ci_low, ci_high, pc_std = ci_from_replicates(pc_arr)
                ci_width = ci_high - ci_low

                row = {
                    "scenario_name": scenario_name,
                    "tier": tier,
                    "method": "SPLIT",
                    "N": int(n),
                    "pc_hat": float(np.mean(pc_arr)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "ci_width": float(ci_width),
                    "eval_count_mean": float(np.mean(eval_arr)),
                    "eval_count_std": (
                        float(np.std(eval_arr, ddof=1)) if args.n_reps > 1 else 0.0
                    ),
                    "n_reps": int(args.n_reps),
                    "p0": float(args.p0),
                    "n_level_mean": float(np.mean(lvl_arr)),
                    "accept_rate_mean": float(np.mean(acc_arr)),
                    "hits": float(np.mean(hit_arr)),
                    "pc_hat_std": float(pc_std),
                    "elapsed_s_mean": float(np.mean(elapsed_arr)),
                    "seed_base": int(seed_base),
                    "n_max": int(n_list[-1]),
                    "n_list": n_list_str,
                    "d0": float(args.d0),
                    "T": float(args.T),
                    "r_sigma": float(r_sigma),
                    "v_sigma": float(v_sigma),
                    "family": str(scenario["family"]),
                    "distribution": str(scenario["distribution"]),
                    "near_threshold": near,
                    "mixture_weight": mix_w,
                    "inflation_k": mix_k,
                    "mean_shift_flag": bool(np.linalg.norm(mean_shift6) > 0.0),
                    "near_margin": float(args.near_margin),
                    "mix_shift_frac": float(args.mix_shift_frac),
                    "proposal_scale": float(args.proposal_scale),
                    "n_mcmc_steps": int(args.n_mcmc_steps),
                    "n_levels_max": int(args.n_levels_max),
                }
                rows.append(row)

                print(
                    f"  {scenario_name:<20s} N={n:<6d} "
                    f"Pc={row['pc_hat']:.3e} CIw={row['ci_width']:.3e} "
                    f"eval={row['eval_count_mean']:.0f} "
                    f"levels={row['n_level_mean']:.2f} acc={row['accept_rate_mean']:.2f}"
                )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["tier", "scenario_name", "N"])
        .reset_index(drop=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
