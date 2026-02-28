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
    importance_sampling_core,
    miss_distance,
    proposal_mean_toward_boundary,
    sample_positions_mixture,
    sample_positions_single,
)
from xq.scenarios import build_scenario, default_variants  # noqa: E402

_METHOD_OFFSETS = {
    "MC": 0,
    "IS": 100_000,
}


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
    raise ValueError(f"unknown distribution: {distribution}")


def _mc_point_estimate(
    n: int,
    scenario: dict[str, object],
    seed: int,
) -> tuple[float, int, int, float]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    samples = _sample_target(n, scenario, rng)
    d = miss_distance(samples)
    indicators = d < float(scenario["d0"])
    elapsed = time.perf_counter() - start
    hits = int(np.sum(indicators))
    return hits / n, hits, n, elapsed


def _is_point_estimate(
    n: int,
    scenario: dict[str, object],
    seed: int,
    shift_frac: float,
) -> tuple[float, int, int, float, float]:
    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)
    d0 = float(scenario["d0"])

    mu_prop = proposal_mean_toward_boundary(mu, d0=d0, shift_frac=shift_frac)
    target_kind = "mixture" if str(scenario["distribution"]) == "mixture" else "single"

    mix_params = None
    if target_kind == "mixture":
        mix_params = {
            "w": float(scenario["mixture_weight"]),
            "inflation_k": float(scenario["inflation_k"]),
            "mean_shift": np.asarray(scenario["mean_shift"], dtype=float),
        }

    rng = np.random.default_rng(seed)
    indicators, logw, elapsed = importance_sampling_core(
        n=n,
        mu=mu,
        Sigma=Sigma,
        d0=d0,
        mu_proposal=mu_prop,
        rng=rng,
        target=target_kind,
        mixture_params=mix_params,
    )

    # Stable weighting.
    logw = np.asarray(logw, dtype=float)
    indicators = np.asarray(indicators, dtype=float)
    logw = logw - np.max(logw)
    w = np.exp(logw)

    pc_hat = float(np.sum(w * indicators) / np.sum(w))
    hits = int(np.sum(indicators))
    w_norm = w / np.sum(w)
    ess = float(1.0 / np.sum(w_norm**2))
    return pc_hat, hits, n, elapsed, ess


def _rows_with_metadata(
    rows: list[dict[str, float | int]],
    scenario: dict[str, object],
    n_list: list[int],
    is_shift_frac: float,
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
                "method": str(row["method"]),
                "N": int(row["N"]),
                "pc_hat_mean": float(row["pc_hat_mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "ci_width": float(row["ci_width"]),
                "hits_mean": float(row["hits_mean"]),
                "ess_mean": float(row["ess_mean"]) if "ess_mean" in row else np.nan,
                "ess_over_n": (
                    float(row["ess_over_n"]) if "ess_over_n" in row else np.nan
                ),
                "eval_count_mean": float(row["eval_count_mean"]),
                "elapsed_s_mean": float(row["elapsed_s_mean"]),
                "n_reps": int(row["n_reps"]),
                "seed_base": int(row["seed_base"]),
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
                "is_shift_frac": float(is_shift_frac),
            }
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run encounter-plane MC/IS with replicate-based empirical CIs"
    )
    parser.add_argument("--tiers", type=parse_int_list, default=parse_int_list("2,3"))
    parser.add_argument("--variants", type=parse_str_list, default=default_variants())
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--nmax", type=int, default=30000)
    parser.add_argument("--n-list", type=parse_n_list, default=None)
    parser.add_argument("--n-reps", type=int, default=25)
    parser.add_argument("--sigma-scale", type=float, default=0.7)
    parser.add_argument("--anisotropy", type=float, default=0.7)
    parser.add_argument("--corr", type=float, default=0.2)
    parser.add_argument("--mix-weight", type=float, default=0.05)
    parser.add_argument("--mix-inflation", type=float, default=10.0)
    parser.add_argument("--mix-shift-frac", type=float, default=0.2)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--is-shift-frac", type=float, default=0.6)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "encounterplane_replicates.csv",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")
    if args.n_reps <= 0:
        raise ValueError("n-reps must be positive")

    n_list = normalized_n_list(args.n_list, args.nmax)

    rows: list[dict[str, float | int | bool | str]] = []
    print("Encounter-plane replicate run (MC/IS):")
    print(
        f"  tiers={args.tiers} variants={args.variants} n_list={n_list} n_reps={args.n_reps}"
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

            scenario_rows: list[dict[str, float | int]] = []
            scenario_seed = int(scenario["seed"])

            for method in ["MC", "IS"]:
                for n in n_list:
                    seed_base = scenario_seed + _METHOD_OFFSETS[method] + n * 17

                    pc_vals: list[float] = []
                    hit_vals: list[float] = []
                    eval_vals: list[float] = []
                    elapsed_vals: list[float] = []
                    ess_vals: list[float] = []

                    for rep in range(args.n_reps):
                        rep_seed = seed_base + rep * 1000

                        if method == "MC":
                            pc_hat, hits, eval_count, elapsed = _mc_point_estimate(
                                n=n,
                                scenario=scenario,
                                seed=rep_seed,
                            )
                            ess = np.nan
                        else:
                            pc_hat, hits, eval_count, elapsed, ess = _is_point_estimate(
                                n=n,
                                scenario=scenario,
                                seed=rep_seed,
                                shift_frac=args.is_shift_frac,
                            )

                        pc_vals.append(float(pc_hat))
                        hit_vals.append(float(hits))
                        eval_vals.append(float(eval_count))
                        elapsed_vals.append(float(elapsed))
                        ess_vals.append(float(ess) if not np.isnan(ess) else np.nan)

                    pc_arr = np.asarray(pc_vals, dtype=float)
                    ci_low, ci_high = ci_from_replicates(pc_arr)

                    row: dict[str, float | int] = {
                        "method": method,
                        "N": int(n),
                        "pc_hat_mean": float(np.mean(pc_arr)),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "ci_width": float(ci_high - ci_low),
                        "hits_mean": float(np.mean(np.asarray(hit_vals, dtype=float))),
                        "eval_count_mean": float(
                            np.mean(np.asarray(eval_vals, dtype=float))
                        ),
                        "elapsed_s_mean": float(
                            np.mean(np.asarray(elapsed_vals, dtype=float))
                        ),
                        "n_reps": int(args.n_reps),
                        "seed_base": int(seed_base),
                    }

                    if method == "IS":
                        ess_arr = np.asarray(ess_vals, dtype=float)
                        ess_mean = float(np.nanmean(ess_arr))
                        row["ess_mean"] = ess_mean
                        row["ess_over_n"] = ess_mean / n

                    scenario_rows.append(row)

                tail = [
                    r
                    for r in scenario_rows
                    if r["method"] == method and r["N"] == n_list[-1]
                ][0]
                print(
                    f"  {scenario['scenario_name']:<24s} {method} "
                    f"N={n_list[-1]:<6d} ciw={tail['ci_width']:.3e} pc={tail['pc_hat_mean']:.3e}"
                )

            rows.extend(
                _rows_with_metadata(
                    rows=scenario_rows,
                    scenario=scenario,
                    n_list=n_list,
                    is_shift_frac=args.is_shift_frac,
                )
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
