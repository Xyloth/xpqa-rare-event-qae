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

from xq.distributions import (  # noqa: E402
    proposal_mean_toward_boundary,
    sample_rv,
    tier_params,
)
from xq.encounter import closest_approach_batch  # noqa: E402
from xq.importance_sampling import (  # noqa: E402
    estimate_pc_importance_sampling,
    importance_sampling_core,
    prefix_is_estimates,
    select_shift_frac,
)


DEFAULT_CI_TARGETS = {
    1: 2e-3,
    2: 2e-5,
    3: 2e-6,
}


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one integer")
    return values


def parse_n_list(text: str) -> list[int]:
    return parse_int_list(text)


def parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [float(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one float")
    return values


def default_n_list(nmax: int) -> list[int]:
    base = [1000, 3000, 10000, 30000]
    n_list = [n for n in base if n < nmax]
    if nmax not in n_list:
        n_list.append(nmax)
    return n_list


def default_shift_fracs() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mc_prefix_results(
    n_list: list[int],
    mean6: np.ndarray,
    cov6: np.ndarray,
    seed: int,
    T: float,
    d0: float,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    r0_w, v0_w = sample_rv(warmup_n, mean6, cov6, warmup_rng)
    _ = closest_approach_batch(r0_w, v0_w, T)

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    r0, v0 = sample_rv(n_max, mean6, cov6, rng)
    _, d_min = closest_approach_batch(r0, v0, T)
    indicators = d_min < d0
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
                "n": n,
                "pc_hat": pc_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "hits": hits,
                "elapsed_s": per_sample_time * n,
                "seed": seed,
                "n_max": n_max,
            }
        )
    return rows


def is_prefix_results(
    n_list: list[int],
    mean6: np.ndarray,
    cov6: np.ndarray,
    mean_prop: np.ndarray,
    seed: int,
    T: float,
    d0: float,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    _ = importance_sampling_core(
        n=warmup_n,
        mean=mean6,
        cov=cov6,
        mean_proposal=mean_prop,
        rng=warmup_rng,
        T=T,
        d0=d0,
    )

    rng = np.random.default_rng(seed)
    indicators, logw, elapsed_core = importance_sampling_core(
        n=n_max,
        mean=mean6,
        cov=cov6,
        mean_proposal=mean_prop,
        rng=rng,
        T=T,
        d0=d0,
    )

    rows = prefix_is_estimates(indicators, logw, n_list, elapsed_core)
    for row in rows:
        row["seed"] = seed
        row["n_max"] = n_max
    return rows


def sweep_shift_fracs(
    shift_fracs: list[float],
    n_tune: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    seed: int,
    T: float,
    d0: float,
    ess_min: float,
) -> tuple[pd.DataFrame, float, str]:
    rows: list[dict[str, float | int]] = []
    for i, shift_frac in enumerate(shift_fracs):
        rng = np.random.default_rng(seed + i)
        mean_prop = proposal_mean_toward_boundary(mean6, d0=d0, shift_frac=shift_frac)
        stats = estimate_pc_importance_sampling(
            n=n_tune,
            mean=mean6,
            cov=cov6,
            mean_proposal=mean_prop,
            rng=rng,
            T=T,
            d0=d0,
        )
        ci_width = stats["ci_high"] - stats["ci_low"]
        ess_n = stats["ess"] / n_tune
        rows.append(
            {
                "shift_frac": shift_frac,
                "pc_hat": stats["pc_hat"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "ci_width": ci_width,
                "ess": stats["ess"],
                "ess_n": ess_n,
                "hits": stats["hits"],
                "elapsed_s": stats["elapsed_s"],
            }
        )

    df = pd.DataFrame(rows)
    shift_frac, reason = select_shift_frac(df, ess_floor=ess_min, require_hits=True)
    return df, shift_frac, reason


def min_n_for_target(df: pd.DataFrame, ci_target: float, n_max: int) -> str | int:
    widths = df["ci_high"] - df["ci_low"]
    meets = df[widths <= ci_target]
    if meets.empty:
        return f">{n_max}"
    return int(meets.iloc[0]["n"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MC/IS bench suite across tiers")
    parser.add_argument(
        "--tiers",
        type=parse_int_list,
        default=parse_int_list("1,2,3"),
        help="Comma-separated list of tiers",
    )
    parser.add_argument("--d0", type=float, default=1.0, help="Event threshold")
    parser.add_argument("--T", type=float, default=3600.0, help="Time window")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed")
    parser.add_argument(
        "--n-list",
        type=parse_n_list,
        default=None,
        help="Comma-separated list of sample sizes",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=100000,
        help="Maximum sample size for the prefix batch",
    )
    parser.add_argument(
        "--shift-fracs",
        type=parse_float_list,
        default=None,
        help="Comma-separated shift_frac grid for IS tuning",
    )
    parser.add_argument(
        "--ess-min",
        type=float,
        default=0.005,
        help="Minimum ESS/N for shift_frac selection (default 0.005)",
    )
    parser.add_argument(
        "--ci-target",
        type=float,
        default=None,
        help="CI width target (overrides per-tier defaults)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "bench_suite_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.T < 0:
        raise ValueError("T must be non-negative")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")

    n_list = args.n_list if args.n_list is not None else default_n_list(args.nmax)
    if any(n <= 0 for n in n_list):
        raise ValueError("n-list must contain positive integers")
    if args.n_list is not None and args.nmax < max(n_list):
        raise ValueError("nmax must be >= max(n-list)")
    if args.n_list is not None and args.nmax > max(n_list):
        n_list = n_list + [args.nmax]
    if any(n_list[i] >= n_list[i + 1] for i in range(len(n_list) - 1)):
        raise ValueError("n-list must be strictly increasing for prefix estimates")

    shift_fracs = args.shift_fracs if args.shift_fracs is not None else default_shift_fracs()
    for sf in shift_fracs:
        if not (0.0 < sf < 1.0):
            raise ValueError("shift_frac values must be in (0, 1)")

    summary_rows: list[dict[str, float | int | str]] = []

    print("Bench suite:")
    for tier in args.tiers:
        r_sigma = args.d0
        v_sigma = 0.0 if args.T == 0 else 0.1 * args.d0 / args.T
        mean6, cov6 = tier_params(args.d0, tier, r_sigma=r_sigma, v_sigma=v_sigma)

        seed_tier = args.seed + tier * 1000
        n_max = n_list[-1]

        _sweep_df, best_shift, reason = sweep_shift_fracs(
            shift_fracs=shift_fracs,
            n_tune=n_max,
            mean6=mean6,
            cov6=cov6,
            seed=seed_tier,
            T=args.T,
            d0=args.d0,
            ess_min=args.ess_min,
        )
        print(
            f"  tier={tier} recommended shift_frac={best_shift:.2f} "
            f"(tune_n={n_max}, {reason})"
        )

        mc_rows = mc_prefix_results(n_list, mean6, cov6, seed_tier, args.T, args.d0)
        mc_df = pd.DataFrame(mc_rows)
        mc_df["tier"] = tier
        mc_df["d0"] = args.d0
        mc_df["T"] = args.T
        mc_df["r_sigma"] = r_sigma
        mc_df["v_sigma"] = v_sigma

        mean_prop = proposal_mean_toward_boundary(mean6, d0=args.d0, shift_frac=best_shift)
        is_rows = is_prefix_results(
            n_list=n_list,
            mean6=mean6,
            cov6=cov6,
            mean_prop=mean_prop,
            seed=seed_tier,
            T=args.T,
            d0=args.d0,
        )
        is_df = pd.DataFrame(is_rows)
        is_df["tier"] = tier
        is_df["d0"] = args.d0
        is_df["T"] = args.T
        is_df["r_sigma"] = r_sigma
        is_df["v_sigma"] = v_sigma
        is_df["shift_frac"] = best_shift

        mc_path = REPO_ROOT / "results" / "tables" / f"mc_tier{tier}.csv"
        is_path = REPO_ROOT / "results" / "tables" / f"is_tier{tier}.csv"
        mc_path.parent.mkdir(parents=True, exist_ok=True)
        mc_df.to_csv(mc_path, index=False)
        is_df.to_csv(is_path, index=False)
        print(f"  saved {mc_path} and {is_path}")

        ci_target = args.ci_target if args.ci_target is not None else DEFAULT_CI_TARGETS[tier]
        mc_n_hit = min_n_for_target(mc_df, ci_target, n_max)
        is_n_hit = min_n_for_target(is_df, ci_target, n_max)

        summary_rows.append(
            {
                "tier": tier,
                "method": "MC",
                "ci_target": ci_target,
                "n_achieved": mc_n_hit,
                "pc_hat_nmax": float(mc_df.iloc[-1]["pc_hat"]),
                "ci_width_nmax": float(
                    mc_df.iloc[-1]["ci_high"] - mc_df.iloc[-1]["ci_low"]
                ),
                "seed": seed_tier,
                "n_max": n_max,
            }
        )
        summary_rows.append(
            {
                "tier": tier,
                "method": "IS",
                "ci_target": ci_target,
                "n_achieved": is_n_hit,
                "pc_hat_nmax": float(is_df.iloc[-1]["pc_hat"]),
                "ci_width_nmax": float(
                    is_df.iloc[-1]["ci_high"] - is_df.iloc[-1]["ci_low"]
                ),
                "ess_nmax": float(is_df.iloc[-1]["ess"] / n_max),
                "shift_frac": best_shift,
                "seed": seed_tier,
                "n_max": n_max,
            }
        )

        print(
            f"  tier={tier} MC: N*={mc_n_hit} (target={ci_target:.1e}) | "
            f"IS: N*={is_n_hit} shift_frac={best_shift:.2f}"
        )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
