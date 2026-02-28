from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xq.benchmark import normalized_n_list  # noqa: E402
from xq.encounter_plane import (  # noqa: E402
    estimate_pc_splitting,
    importance_sampling_core,
    miss_distance,
    prefix_is_estimates,
    proposal_mean_toward_boundary,
    sample_positions_mixture,
    sample_positions_single,
)
from xq.scenarios import build_scenario, default_variants  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def parse_str_list(text: str) -> list[str]:
    out = [p.strip() for p in text.split(",") if p.strip()]
    if not out:
        raise ValueError("list must contain at least one entry")
    return out


def parse_n_list(text: str) -> list[int]:
    return parse_int_list(text)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def ci_from_replicates(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean
    std = float(np.std(values, ddof=1))
    half = z * std / np.sqrt(values.size)
    return max(0.0, mean - half), min(1.0, mean + half)


def _sample_target(
    n: int,
    scenario: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)
    distribution = str(scenario["distribution"])
    if distribution == "single_gaussian":
        return sample_positions_single(n, mu, Sigma, rng)
    if distribution == "mixture":
        samples, _ = sample_positions_mixture(
            n=n,
            mu=mu,
            Sigma=Sigma,
            w=float(scenario["mixture_weight"]),
            inflation_k=float(scenario["inflation_k"]),
            mean_shift=np.asarray(scenario["mean_shift"], dtype=float),
            rng=rng,
        )
        return samples
    raise ValueError(f"unknown distribution {distribution}")


def mc_prefix_rows(
    n_list: list[int],
    scenario: dict[str, object],
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]
    seed = int(scenario["seed"])

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    _ = miss_distance(_sample_target(warmup_n, scenario, warmup_rng))

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    samples = _sample_target(n_max, scenario, rng)
    d = miss_distance(samples)
    indicators = d < float(scenario["d0"])
    elapsed_core = time.perf_counter() - start

    cum_hits = np.cumsum(indicators)
    per_sample_time = elapsed_core / n_max

    rows: list[dict[str, float | int]] = []
    for n in n_list:
        hits = int(cum_hits[n - 1])
        pc_hat = hits / n
        ci_low, ci_high = wilson_ci(hits, n)
        rows.append(
            {
                "N": n,
                "pc_hat": float(pc_hat),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_width": float(ci_high - ci_low),
                "hits": hits,
                "ess": np.nan,
                "ess_over_n": np.nan,
                "eval_count": float(n),
                "eval_count_std": 0.0,
                "accept_rate": np.nan,
                "elapsed_s": float(per_sample_time * n),
                "n_reps": 1,
                "seed": seed,
            }
        )
    return rows


def is_prefix_rows(
    n_list: list[int],
    scenario: dict[str, object],
    shift_frac: float,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]
    seed = int(scenario["seed"])

    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)
    d0 = float(scenario["d0"])

    mu_prop = proposal_mean_toward_boundary(mu, d0=d0, shift_frac=shift_frac)
    target_kind = "mixture" if str(scenario["distribution"]) == "mixture" else "single"
    mixture_params = None
    if target_kind == "mixture":
        mixture_params = {
            "w": float(scenario["mixture_weight"]),
            "inflation_k": float(scenario["inflation_k"]),
            "mean_shift": np.asarray(scenario["mean_shift"], dtype=float),
        }

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    _ = importance_sampling_core(
        n=warmup_n,
        mu=mu,
        Sigma=Sigma,
        d0=d0,
        mu_proposal=mu_prop,
        rng=warmup_rng,
        target=target_kind,
        mixture_params=mixture_params,
    )

    rng = np.random.default_rng(seed)
    indicators, logw, elapsed_core = importance_sampling_core(
        n=n_max,
        mu=mu,
        Sigma=Sigma,
        d0=d0,
        mu_proposal=mu_prop,
        rng=rng,
        target=target_kind,
        mixture_params=mixture_params,
    )
    rows = prefix_is_estimates(indicators, logw, n_list, elapsed_core)

    out: list[dict[str, float | int]] = []
    for row in rows:
        n = int(row["n"])
        ess = float(row["ess"])
        out.append(
            {
                "N": n,
                "pc_hat": float(row["pc_hat"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "ci_width": float(row["ci_width"]),
                "hits": int(row["hits"]),
                "ess": ess,
                "ess_over_n": ess / n,
                "eval_count": float(n),
                "eval_count_std": 0.0,
                "accept_rate": np.nan,
                "elapsed_s": float(row["elapsed_s"]),
                "n_reps": 1,
                "seed": seed,
            }
        )
    return out


def split_rows(
    n_list: list[int],
    scenario: dict[str, object],
    p0: float,
    n_levels_max: int,
    proposal_scale: float,
    n_mcmc_steps: int,
    n_reps: int,
) -> list[dict[str, float | int]]:
    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)
    d0 = float(scenario["d0"])
    scenario_seed = int(scenario["seed"])

    target_kind = "mixture" if str(scenario["distribution"]) == "mixture" else "single"
    mixture_params = None
    if target_kind == "mixture":
        mixture_params = {
            "w": float(scenario["mixture_weight"]),
            "inflation_k": float(scenario["inflation_k"]),
            "mean_shift": np.asarray(scenario["mean_shift"], dtype=float),
        }

    rows: list[dict[str, float | int]] = []
    for n in n_list:
        pc_vals: list[float] = []
        eval_vals: list[float] = []
        acc_vals: list[float] = []
        hit_vals: list[float] = []
        elapsed_vals: list[float] = []

        seed_base = scenario_seed + 500_000 + n * 13

        for rep in range(n_reps):
            rep_seed = seed_base + rep * 1000
            stats = estimate_pc_splitting(
                n=n,
                mu=mu,
                Sigma=Sigma,
                d0=d0,
                p0=p0,
                n_levels_max=n_levels_max,
                proposal_scale=proposal_scale,
                n_mcmc_steps=n_mcmc_steps,
                seed=rep_seed,
                target=target_kind,
                mixture_params=mixture_params,
            )

            diag = stats["diagnostics"]
            hits_per_level = diag["hits_per_level"]
            final_hits = float(hits_per_level[-1] if hits_per_level else 0.0)

            pc_vals.append(float(stats["pc_hat"]))
            eval_vals.append(float(stats["eval_count"]))
            acc_vals.append(float(stats["acceptance_rate"]))
            hit_vals.append(final_hits)
            elapsed_vals.append(float(stats["elapsed_s"]))

        pc_arr = np.asarray(pc_vals, dtype=float)
        eval_arr = np.asarray(eval_vals, dtype=float)
        acc_arr = np.asarray(acc_vals, dtype=float)
        hit_arr = np.asarray(hit_vals, dtype=float)
        elapsed_arr = np.asarray(elapsed_vals, dtype=float)

        ci_low, ci_high = ci_from_replicates(pc_arr)
        row = {
            "N": n,
            "pc_hat": float(np.mean(pc_arr)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_width": float(ci_high - ci_low),
            "hits": float(np.mean(hit_arr)),
            "ess": np.nan,
            "ess_over_n": np.nan,
            "eval_count": float(np.mean(eval_arr)),
            "eval_count_std": float(np.std(eval_arr, ddof=1)) if n_reps > 1 else 0.0,
            "accept_rate": float(np.mean(acc_arr)),
            "elapsed_s": float(np.mean(elapsed_arr)),
            "n_reps": int(n_reps),
            "seed": int(seed_base),
        }
        rows.append(row)

    return rows


def with_metadata(
    rows: list[dict[str, float | int]],
    scenario: dict[str, object],
    method: str,
    n_list: list[int],
    shift_frac: float,
    split_p0: float,
    split_levels: int,
    split_scale: float,
    split_steps: int,
) -> list[dict[str, float | int | bool | str]]:
    out: list[dict[str, float | int | bool | str]] = []
    n_list_str = ",".join(str(n) for n in n_list)
    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)

    for row in rows:
        out.append(
            {
                "scenario_name": str(scenario["scenario_name"]),
                "tier": int(scenario["tier"]),
                "variant": str(scenario["variant"]),
                "family": str(scenario["family"]),
                "distribution": str(scenario["distribution"]),
                "method": method,
                "N": int(row["N"]),
                "pc_hat": float(row["pc_hat"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "ci_width": float(row["ci_width"]),
                "hits": float(row["hits"]),
                "ess": float(row["ess"]) if not pd.isna(row["ess"]) else np.nan,
                "ess_over_n": (
                    float(row["ess_over_n"])
                    if not pd.isna(row["ess_over_n"])
                    else np.nan
                ),
                "eval_count": float(row["eval_count"]),
                "eval_count_std": float(row["eval_count_std"]),
                "accept_rate": (
                    float(row["accept_rate"])
                    if not pd.isna(row["accept_rate"])
                    else np.nan
                ),
                "elapsed_s": float(row["elapsed_s"]),
                "n_reps": int(row["n_reps"]),
                "seed": int(row["seed"]),
                "n_max": int(n_list[-1]),
                "n_list": n_list_str,
                "d0": float(scenario["d0"]),
                "mu_x": float(mu[0]),
                "mu_y": float(mu[1]),
                "sigma_x": float(np.sqrt(Sigma[0, 0])),
                "sigma_y": float(np.sqrt(Sigma[1, 1])),
                "cov_xy": float(Sigma[0, 1]),
                "mixture_weight": float(scenario["mixture_weight"]),
                "inflation_k": float(scenario["inflation_k"]),
                "mean_shift_flag": bool(scenario["mean_shift_flag"]),
                "near_margin_frac": float(scenario["near_margin_frac"]),
                "sigma_scale": float(scenario["sigma_scale"]),
                "anisotropy": float(scenario["anisotropy"]),
                "corr": float(scenario["corr"]),
                "is_shift_frac": float(shift_frac),
                "split_p0": float(split_p0),
                "split_levels_max": int(split_levels),
                "split_proposal_scale": float(split_scale),
                "split_mcmc_steps": int(split_steps),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MC/IS/SPLIT benchmark suite for encounter-plane model"
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument("--variants", type=parse_str_list, default=default_variants())
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--nmax", type=int, default=30000)
    parser.add_argument("--n-list", type=parse_n_list, default=None)
    parser.add_argument("--sigma-scale", type=float, default=0.7)
    parser.add_argument("--anisotropy", type=float, default=0.7)
    parser.add_argument("--corr", type=float, default=0.2)
    parser.add_argument("--mix-weight", type=float, default=0.05)
    parser.add_argument("--mix-inflation", type=float, default=10.0)
    parser.add_argument("--mix-shift-frac", type=float, default=0.2)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--is-shift-frac", type=float, default=0.6)
    parser.add_argument("--split-p0", type=float, default=0.1)
    parser.add_argument("--split-levels-max", type=int, default=20)
    parser.add_argument("--split-proposal-scale", type=float, default=0.8)
    parser.add_argument("--split-mcmc-steps", type=int, default=2)
    parser.add_argument("--split-reps", type=int, default=15)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_suite.csv",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")
    if args.split_reps <= 0:
        raise ValueError("split-reps must be positive")

    n_list = normalized_n_list(args.n_list, args.nmax)
    rows: list[dict[str, float | int | bool | str]] = []

    print("Encounter-plane suite run:")
    print(
        f"  tiers={args.tiers} variants={args.variants} d0={args.d0} n_list={n_list} "
        f"split_reps={args.split_reps}"
    )

    for tier in args.tiers:
        for variant in args.variants:
            scenario = build_scenario(
                tier=tier,
                variant=variant,
                d0=args.d0,
                sigma_scale=args.sigma_scale,
                anisotropy=args.anisotropy,
                corr=args.corr,
                mixture_weight=args.mix_weight,
                inflation_k=args.mix_inflation,
                near_margin_frac=args.near_margin,
                mean_shift_frac=args.mix_shift_frac,
                seed_base=args.seed,
            )

            mc = mc_prefix_rows(n_list, scenario)
            is_rows = is_prefix_rows(
                n_list=n_list,
                scenario=scenario,
                shift_frac=args.is_shift_frac,
            )
            split = split_rows(
                n_list=n_list,
                scenario=scenario,
                p0=args.split_p0,
                n_levels_max=args.split_levels_max,
                proposal_scale=args.split_proposal_scale,
                n_mcmc_steps=args.split_mcmc_steps,
                n_reps=args.split_reps,
            )

            rows.extend(
                with_metadata(
                    rows=mc,
                    scenario=scenario,
                    method="MC",
                    n_list=n_list,
                    shift_frac=args.is_shift_frac,
                    split_p0=args.split_p0,
                    split_levels=args.split_levels_max,
                    split_scale=args.split_proposal_scale,
                    split_steps=args.split_mcmc_steps,
                )
            )
            rows.extend(
                with_metadata(
                    rows=is_rows,
                    scenario=scenario,
                    method="IS",
                    n_list=n_list,
                    shift_frac=args.is_shift_frac,
                    split_p0=args.split_p0,
                    split_levels=args.split_levels_max,
                    split_scale=args.split_proposal_scale,
                    split_steps=args.split_mcmc_steps,
                )
            )
            rows.extend(
                with_metadata(
                    rows=split,
                    scenario=scenario,
                    method="SPLIT",
                    n_list=n_list,
                    shift_frac=args.is_shift_frac,
                    split_p0=args.split_p0,
                    split_levels=args.split_levels_max,
                    split_scale=args.split_proposal_scale,
                    split_steps=args.split_mcmc_steps,
                )
            )

            mc_tail = mc[-1]
            is_tail = is_rows[-1]
            split_tail = split[-1]
            print(
                f"  {scenario['scenario_name']:<24s} "
                f"MC ciw={mc_tail['ci_width']:.3e} "
                f"IS ciw={is_tail['ci_width']:.3e} "
                f"SPLIT ciw={split_tail['ci_width']:.3e}"
            )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["tier", "variant", "method", "N"])
        .reset_index(drop=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
