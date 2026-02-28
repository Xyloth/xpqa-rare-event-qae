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

from xq.distributions import proposal_mean_toward_boundary, tier_params  # noqa: E402
from xq.importance_sampling import (  # noqa: E402
    importance_sampling_core,
    prefix_is_estimates,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importance sampling Tier-1 conjunction model benchmark"
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
        "--shift-frac",
        type=float,
        default=0.7,
        help="Fractional shift of mean distance toward d0",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "is_tier1.csv",
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
    mean_prop = proposal_mean_toward_boundary(
        mean6, d0=args.d0, shift_frac=args.shift_frac
    )

    n_max = n_list[-1]
    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(args.seed + 1_000_003)
    _ = importance_sampling_core(
        n=warmup_n,
        mean=mean6,
        cov=cov6,
        mean_proposal=mean_prop,
        rng=warmup_rng,
        T=args.T,
        d0=args.d0,
    )
    rng = np.random.default_rng(args.seed)
    indicators, logw, elapsed_core = importance_sampling_core(
        n=n_max,
        mean=mean6,
        cov=cov6,
        mean_proposal=mean_prop,
        rng=rng,
        T=args.T,
        d0=args.d0,
    )

    rows = prefix_is_estimates(indicators, logw, n_list, elapsed_core)
    print(
        "Tier-1 IS run:\n"
        f"  tier={args.tier} d0={args.d0} T={args.T} "
        f"r_sigma={r_sigma} v_sigma={v_sigma} "
        f"shift_frac={args.shift_frac} seed={args.seed} nmax={n_max}"
    )

    for row in rows:
        print(
            f"  N={row['n']:<7d} Pc={row['pc_hat']:.3e} "
            f"CI=[{row['ci_low']:.3e}, {row['ci_high']:.3e}] "
            f"ESS={row['ess']:.1f} hits={row['hits']:<5d} "
            f"time={row['elapsed_s']:.3f}s"
        )

    for row in rows:
        row.update(
            {
                "tier": args.tier,
                "d0": args.d0,
                "T": args.T,
                "r_sigma": r_sigma,
                "v_sigma": v_sigma,
                "shift_frac": args.shift_frac,
                "n_max": n_max,
                "seed": args.seed,
            }
        )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
