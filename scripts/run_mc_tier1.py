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

from xq.distributions import sample_rv, tier_params  # noqa: E402
from xq.encounter import closest_approach_batch  # noqa: E402


def parse_n_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("n-list must contain at least one integer")
    return values


def default_n_list(nmax: int) -> list[int]:
    base = [1000, 3000, 10000, 30000]
    n_list = [n for n in base if n < nmax]
    if nmax not in n_list:
        n_list.append(nmax)
    return n_list


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Tier-1 conjunction model benchmark"
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
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
    parser.add_argument("--tier", type=int, default=2, help="Difficulty tier (1-3)")
    parser.add_argument("--d0", type=float, default=1.0, help="Event threshold")
    parser.add_argument("--T", type=float, default=3600.0, help="Time window")
    parser.add_argument(
        "--r-sigma", type=float, default=None, help="Position std dev"
    )
    parser.add_argument(
        "--v-sigma", type=float, default=None, help="Velocity std dev"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "mc_tier1.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.T < 0:
        raise ValueError("T must be non-negative")
    n_list = args.n_list if args.n_list is not None else default_n_list(args.nmax)
    if any(n <= 0 for n in n_list):
        raise ValueError("n-list must contain positive integers")
    if args.n_list is not None and args.nmax < max(n_list):
        raise ValueError("nmax must be >= max(n-list)")
    if args.n_list is not None and args.nmax > max(n_list):
        n_list = n_list + [args.nmax]
    if any(
        n_list[i] >= n_list[i + 1] for i in range(len(n_list) - 1)
    ):
        raise ValueError("n-list must be strictly increasing for prefix estimates")

    r_sigma = args.r_sigma if args.r_sigma is not None else args.d0
    if args.v_sigma is None:
        v_sigma = 0.0 if args.T == 0 else 0.1 * args.d0 / args.T
    else:
        v_sigma = args.v_sigma

    mean6, cov6 = tier_params(args.d0, args.tier, r_sigma=r_sigma, v_sigma=v_sigma)
    n_max = n_list[-1]

    rows: list[dict[str, float | int]] = []
    print(
        "Tier-1 MC run:\n"
        f"  tier={args.tier} d0={args.d0} T={args.T} "
        f"r_sigma={r_sigma} v_sigma={v_sigma} seed={args.seed} nmax={n_max}"
    )

    # Warmup to avoid one-time overhead contaminating timings.
    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(args.seed + 1_000_003)
    r0_w, v0_w = sample_rv(warmup_n, mean6, cov6, warmup_rng)
    _ = closest_approach_batch(r0_w, v0_w, args.T)

    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    r0, v0 = sample_rv(n_max, mean6, cov6, rng)
    _, d_min = closest_approach_batch(r0, v0, args.T)
    indicators = d_min < args.d0
    elapsed_core = time.perf_counter() - start

    cum_hits = np.cumsum(indicators)
    per_sample_time = elapsed_core / n_max

    for n in n_list:
        hits = int(cum_hits[n - 1])
        pc_hat = hits / n
        ci_low, ci_high = wilson_ci(hits, n)
        # Scale the core time linearly to estimate per-prefix runtime.
        elapsed = per_sample_time * n

        rows.append(
            {
                "n": n,
                "pc_hat": pc_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "hits": hits,
                "elapsed_s": elapsed,
                "tier": args.tier,
                "d0": args.d0,
                "T": args.T,
                "r_sigma": r_sigma,
                "v_sigma": v_sigma,
                "n_max": n_max,
                "seed": args.seed,
            }
        )
        print(
            f"  N={n:<7d} Pc={pc_hat:.3e} "
            f"CI=[{ci_low:.3e}, {ci_high:.3e}] "
            f"hits={hits:<5d} time={elapsed:.3f}s"
        )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
