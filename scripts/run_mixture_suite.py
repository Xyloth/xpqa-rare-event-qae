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

from xq.benchmark import (  # noqa: E402
    normalized_n_list,
    run_is_prefix_gaussian,
    run_is_prefix_mixture_target,
    run_mc_prefix_gaussian,
    run_mc_prefix_mixture,
)
from xq.distributions import (  # noqa: E402
    mixture_scenario_params,
    near_threshold_mean,
    proposal_mean_toward_boundary,
    tier_params,
)


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def parse_n_list(text: str) -> list[int]:
    return parse_int_list(text)


def _rows_with_metadata(
    rows: list[dict[str, float | int]],
    scenario_name: str,
    tier: int,
    method: str,
    family: str,
    distribution: str,
    near_threshold: bool,
    shift_frac: float,
    d0: float,
    T: float,
    n_list: list[int],
    mixture_weight: float,
    inflation_k: float,
    mean_shift6: np.ndarray,
    near_margin: float,
    mix_shift_frac: float,
) -> list[dict[str, float | int | bool | str]]:
    out: list[dict[str, float | int | bool | str]] = []
    mean_shift_flag = bool(np.linalg.norm(mean_shift6) > 0)
    n_list_str = ",".join(str(n) for n in n_list)

    for row in rows:
        ess = float(row["ess"]) if "ess" in row else np.nan
        ess_over_n = float(row["ess_over_n"]) if "ess_over_n" in row else np.nan
        out.append(
            {
                "scenario_name": scenario_name,
                "tier": tier,
                "method": method,
                "N": int(row["n"]),
                "pc_hat": float(row["pc_hat"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "ci_width": float(row.get("ci_width", row["ci_high"] - row["ci_low"])),
                "hits": int(row["hits"]),
                "ess": ess,
                "ess_over_n": ess_over_n,
                "elapsed_s": float(row["elapsed_s"]),
                "seed": int(row["seed"]),
                "n_max": int(row["n_max"]),
                "n_list": n_list_str,
                "family": family,
                "distribution": distribution,
                "near_threshold": near_threshold,
                "proposal_shift_frac": shift_frac,
                "d0": d0,
                "T": T,
                "mixture_weight": mixture_weight,
                "inflation_k": inflation_k,
                "mean_shift_flag": mean_shift_flag,
                "near_margin": near_margin,
                "mix_shift_frac": mix_shift_frac,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mixture and near-threshold benchmark suite"
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=3600.0)
    parser.add_argument("--nmax", type=int, default=100000)
    parser.add_argument("--n-list", type=parse_n_list, default=None)
    parser.add_argument("--r-sigma", type=float, default=None)
    parser.add_argument("--v-sigma", type=float, default=None)
    parser.add_argument("--shift-frac", type=float, default=0.6)
    parser.add_argument("--mix-weight", type=float, default=0.05)
    parser.add_argument("--mix-inflation", type=float, default=10.0)
    parser.add_argument("--mix-shift-frac", type=float, default=0.2)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "mixture_suite.csv",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.T < 0:
        raise ValueError("T must be non-negative")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")

    n_list = normalized_n_list(args.n_list, args.nmax)

    all_rows: list[dict[str, float | int | bool | str]] = []

    print("Mixture suite run:")
    print(
        f"  tiers={args.tiers} d0={args.d0} T={args.T} n_list={n_list} "
        f"mix_weight={args.mix_weight} mix_inflation={args.mix_inflation}"
    )

    for tier in args.tiers:
        r_sigma = args.r_sigma if args.r_sigma is not None else args.d0
        if args.v_sigma is None:
            v_sigma = 0.0 if args.T == 0 else 0.1 * args.d0 / args.T
        else:
            v_sigma = args.v_sigma

        mean_base, cov6 = tier_params(args.d0, tier, r_sigma=r_sigma, v_sigma=v_sigma)

        for near in [False, True]:
            family = "near_threshold" if near else "baseline"
            mean6 = (
                near_threshold_mean(mean_base, d0=args.d0, margin_frac=args.near_margin)
                if near
                else mean_base.copy()
            )

            for is_mixture in [False, True]:
                dist_name = "mixture" if is_mixture else "single_gaussian"
                scenario_name = f"tier{tier}_{'near_' if near else ''}{'mixture' if is_mixture else 'single'}"
                scenario_seed = (
                    args.seed
                    + tier * 1000
                    + (100 if near else 0)
                    + (10 if is_mixture else 0)
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
                    w = float(mix_cfg["w"])
                    inflation_k = float(mix_cfg["inflation_k"])
                    mean_shift6 = np.asarray(mix_cfg["mean_shift6"], dtype=float)
                else:
                    w = 0.0
                    inflation_k = 1.0
                    mean_shift6 = np.zeros(6, dtype=float)

                mean_prop = proposal_mean_toward_boundary(
                    mean6, d0=args.d0, shift_frac=args.shift_frac
                )

                if is_mixture:
                    mc_rows = run_mc_prefix_mixture(
                        n_list=n_list,
                        mean6=mean6,
                        cov6=cov6,
                        w=w,
                        inflation_k=inflation_k,
                        mean_shift6=mean_shift6,
                        seed=scenario_seed,
                        T=args.T,
                        d0=args.d0,
                    )
                    is_rows = run_is_prefix_mixture_target(
                        n_list=n_list,
                        mean6=mean6,
                        cov6=cov6,
                        w=w,
                        inflation_k=inflation_k,
                        mean_shift6=mean_shift6,
                        mean_proposal=mean_prop,
                        cov_proposal=cov6,
                        seed=scenario_seed,
                        T=args.T,
                        d0=args.d0,
                    )
                else:
                    mc_rows = run_mc_prefix_gaussian(
                        n_list=n_list,
                        mean6=mean6,
                        cov6=cov6,
                        seed=scenario_seed,
                        T=args.T,
                        d0=args.d0,
                    )
                    is_rows = run_is_prefix_gaussian(
                        n_list=n_list,
                        mean6=mean6,
                        cov6=cov6,
                        mean_proposal=mean_prop,
                        seed=scenario_seed,
                        T=args.T,
                        d0=args.d0,
                    )

                all_rows.extend(
                    _rows_with_metadata(
                        rows=mc_rows,
                        scenario_name=scenario_name,
                        tier=tier,
                        method="MC",
                        family=family,
                        distribution=dist_name,
                        near_threshold=near,
                        shift_frac=args.shift_frac,
                        d0=args.d0,
                        T=args.T,
                        n_list=n_list,
                        mixture_weight=w,
                        inflation_k=inflation_k,
                        mean_shift6=mean_shift6,
                        near_margin=args.near_margin,
                        mix_shift_frac=args.mix_shift_frac,
                    )
                )
                all_rows.extend(
                    _rows_with_metadata(
                        rows=is_rows,
                        scenario_name=scenario_name,
                        tier=tier,
                        method="IS",
                        family=family,
                        distribution=dist_name,
                        near_threshold=near,
                        shift_frac=args.shift_frac,
                        d0=args.d0,
                        T=args.T,
                        n_list=n_list,
                        mixture_weight=w,
                        inflation_k=inflation_k,
                        mean_shift6=mean_shift6,
                        near_margin=args.near_margin,
                        mix_shift_frac=args.mix_shift_frac,
                    )
                )

                mc_final = [r for r in mc_rows if int(r["n"]) == n_list[-1]][0]
                is_final = [r for r in is_rows if int(r["n"]) == n_list[-1]][0]
                print(
                    f"  {scenario_name:<24s} MC ciw={mc_final['ci_width']:.3e} "
                    f"IS ciw={is_final['ci_width']:.3e}"
                )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["tier", "family", "distribution", "method", "N"]).reset_index(
        drop=True
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
