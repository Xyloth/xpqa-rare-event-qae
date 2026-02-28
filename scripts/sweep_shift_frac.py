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
    estimate_pc_importance_sampling,
    select_shift_frac,
)


def parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [float(p) for p in parts]
    if not values:
        raise ValueError("list must contain at least one float")
    return values


def default_shift_fracs() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep IS shift_frac for Tier-1 conjunction model"
    )
    parser.add_argument("--tier", type=int, default=2, help="Difficulty tier (1-3)")
    parser.add_argument("--d0", type=float, default=1.0, help="Event threshold")
    parser.add_argument("--T", type=float, default=3600.0, help="Time window")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument(
        "--nmax", type=int, default=100000, help="Sample size for each sweep point"
    )
    parser.add_argument(
        "--r-sigma", type=float, default=None, help="Position std dev"
    )
    parser.add_argument(
        "--v-sigma", type=float, default=None, help="Velocity std dev"
    )
    parser.add_argument(
        "--shift-fracs",
        type=parse_float_list,
        default=None,
        help="Comma-separated shift_frac grid",
    )
    parser.add_argument(
        "--ess-min",
        type=float,
        default=0.005,
        help="Minimum ESS/N for selection (default 0.005)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.T < 0:
        raise ValueError("T must be non-negative")
    if args.nmax <= 0:
        raise ValueError("nmax must be positive")
    if args.ess_min <= 0:
        raise ValueError("ess-min must be positive")

    shift_fracs = args.shift_fracs if args.shift_fracs is not None else default_shift_fracs()
    for sf in shift_fracs:
        if not (0.0 < sf < 1.0):
            raise ValueError("shift_frac values must be in (0, 1)")

    r_sigma = args.r_sigma if args.r_sigma is not None else args.d0
    if args.v_sigma is None:
        v_sigma = 0.0 if args.T == 0 else 0.1 * args.d0 / args.T
    else:
        v_sigma = args.v_sigma

    mean6, cov6 = tier_params(args.d0, args.tier, r_sigma=r_sigma, v_sigma=v_sigma)

    rows: list[dict[str, float | int]] = []
    print(
        "Shift-frac sweep:\n"
        f"  tier={args.tier} d0={args.d0} T={args.T} "
        f"r_sigma={r_sigma} v_sigma={v_sigma} seed={args.seed} nmax={args.nmax}"
    )

    for i, shift_frac in enumerate(shift_fracs):
        rng = np.random.default_rng(args.seed + i)
        mean_prop = proposal_mean_toward_boundary(
            mean6, d0=args.d0, shift_frac=shift_frac
        )
        stats = estimate_pc_importance_sampling(
            n=args.nmax,
            mean=mean6,
            cov=cov6,
            mean_proposal=mean_prop,
            rng=rng,
            T=args.T,
            d0=args.d0,
        )
        ci_width = stats["ci_high"] - stats["ci_low"]
        ess_n = stats["ess"] / args.nmax

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
                "tier": args.tier,
                "d0": args.d0,
                "T": args.T,
                "r_sigma": r_sigma,
                "v_sigma": v_sigma,
                "n_max": args.nmax,
                "seed": args.seed + i,
            }
        )
        print(
            f"  shift_frac={shift_frac:.2f} Pc={stats['pc_hat']:.3e} "
            f"CIw={ci_width:.3e} ESS/N={ess_n:.3f} hits={stats['hits']:<5d}"
        )

    df = pd.DataFrame(rows)
    out_path = (
        args.out
        if args.out is not None
        else REPO_ROOT / "results" / "tables" / f"is_shift_sweep_tier{args.tier}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    shift_frac, reason = select_shift_frac(
        df, ess_floor=args.ess_min, require_hits=True
    )
    print(
        "Recommended shift_frac for tier={} is {:.2f} ({})".format(
            args.tier, shift_frac, reason
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
