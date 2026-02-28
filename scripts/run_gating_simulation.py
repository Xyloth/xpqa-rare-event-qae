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
from xq.encounter_plane import (  # noqa: E402
    importance_sampling_core,
    prefix_is_estimates,
    proposal_mean_toward_boundary,
    sample_positions_single,
)


def parse_n_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if not vals:
        raise ValueError("n-list must contain at least one integer")
    return vals


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def scenario_params(
    mu_norm: float,
    sigma_scale: float,
    d0: float,
    anisotropy: float,
    corr: float,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([mu_norm, 0.0], dtype=float)
    sx = sigma_scale * d0
    sy = anisotropy * sigma_scale * d0
    cov_xy = corr * sx * sy
    Sigma = np.array([[sx**2, cov_xy], [cov_xy, sy**2]], dtype=float)
    return mu, Sigma


def screen_classification(
    coarse_p: float,
    coarse_ciw: float,
    t_low: float,
    t_high: float,
    margin_mult: float,
    margin_floor: float,
) -> str:
    margin = max(margin_mult * coarse_ciw, margin_floor)
    if coarse_p + margin < t_low:
        return "SAFE"
    if coarse_p - margin > t_high:
        return "DANGER"
    return "GRAY"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run telescoping/gating simulation")
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--K", type=int, default=2000)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--mixture-frac", type=float, default=0.2)
    parser.add_argument("--mix-weight", type=float, default=0.05)
    parser.add_argument("--mix-inflation", type=float, default=10.0)
    parser.add_argument("--mix-shift-frac", type=float, default=0.2)
    parser.add_argument("--anisotropy", type=float, default=0.7)
    parser.add_argument("--corr", type=float, default=0.2)
    parser.add_argument("--coarse-n", type=int, default=300)
    parser.add_argument("--t-low", type=float, default=1e-3)
    parser.add_argument("--t-high", type=float, default=1e-2)
    parser.add_argument("--margin-mult", type=float, default=0.05)
    parser.add_argument("--margin-floor", type=float, default=1e-6)
    parser.add_argument("--refine-method", type=str, default="IS", choices=["IS"])
    parser.add_argument("--refine-ci-target", type=float, default=2e-6)
    parser.add_argument(
        "--refine-n-list", type=parse_n_list, default=parse_n_list("1000,3000,10000")
    )
    parser.add_argument("--is-shift-frac", type=float, default=0.6)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "gating_simulation.csv",
    )
    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError("K must be positive")
    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if not (0.0 <= args.mixture_frac <= 1.0):
        raise ValueError("mixture-frac must be in [0,1]")
    if not (0.0 <= args.mix_weight < 1.0):
        raise ValueError("mix-weight must be in [0,1)")
    if args.mix_inflation <= 0:
        raise ValueError("mix-inflation must be positive")
    if args.coarse_n <= 0:
        raise ValueError("coarse-n must be positive")
    if not (0 < args.t_low < args.t_high):
        raise ValueError("require 0 < t-low < t-high")

    refine_n_list = normalized_n_list(args.refine_n_list, max(args.refine_n_list))
    refine_n_max = refine_n_list[-1]

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str | bool]] = []

    print("Gating simulation:")
    print(
        f"  K={args.K} coarse_n={args.coarse_n} refine_n_list={refine_n_list} "
        f"t_low={args.t_low:.1e} t_high={args.t_high:.1e} margin_mult={args.margin_mult}"
    )

    for sid in range(args.K):
        tier_like = 2 if rng.random() < 0.6 else 3
        if tier_like == 2:
            mu_norm = float(rng.uniform(1.1, 4.6))
        else:
            mu_norm = float(rng.uniform(1.4, 5.6))

        sigma_scale = float(rng.uniform(0.25, 0.9))
        mixture_flag = bool(rng.random() < args.mixture_frac)

        mu, Sigma = scenario_params(
            mu_norm=mu_norm,
            sigma_scale=sigma_scale,
            d0=args.d0,
            anisotropy=args.anisotropy,
            corr=args.corr,
        )

        # Conservative coarse surrogate: inflate covariance for mixture scenarios.
        if mixture_flag:
            mu_coarse = mu.copy()
            mu_coarse[0] = max(0.0, mu_coarse[0] - args.mix_shift_frac * args.d0)
            Sigma_coarse = args.mix_inflation * Sigma
        else:
            mu_coarse = mu
            Sigma_coarse = Sigma

        coarse_seed = args.seed + sid * 1000 + 11
        coarse_rng = np.random.default_rng(coarse_seed)
        coarse_samples = sample_positions_single(
            args.coarse_n, mu_coarse, Sigma_coarse, coarse_rng
        )
        coarse_d = np.linalg.norm(coarse_samples, axis=1)
        coarse_hits = int(np.sum(coarse_d < args.d0))
        coarse_p = coarse_hits / args.coarse_n
        c_low, c_high = wilson_ci(coarse_hits, args.coarse_n)
        coarse_ciw = c_high - c_low

        classification = screen_classification(
            coarse_p=coarse_p,
            coarse_ciw=coarse_ciw,
            t_low=args.t_low,
            t_high=args.t_high,
            margin_mult=args.margin_mult,
            margin_floor=args.margin_floor,
        )

        refine_method = "NONE"
        refine_eval = 0.0
        refine_ciw = np.nan
        refined_p = np.nan
        refine_hits = np.nan

        if classification == "GRAY":
            refine_method = args.refine_method
            mean_prop = proposal_mean_toward_boundary(
                mu, d0=args.d0, shift_frac=args.is_shift_frac
            )

            target_kind = "mixture" if mixture_flag else "single"
            mixture_params = None
            if mixture_flag:
                mixture_params = {
                    "w": args.mix_weight,
                    "inflation_k": args.mix_inflation,
                    "mean_shift": np.array(
                        [-args.mix_shift_frac * args.d0, 0.0], dtype=float
                    ),
                }

            refine_seed = args.seed + sid * 1000 + 29
            refine_rng = np.random.default_rng(refine_seed)
            indicators, logw, elapsed_core = importance_sampling_core(
                n=refine_n_max,
                mu=mu,
                Sigma=Sigma,
                d0=args.d0,
                mu_proposal=mean_prop,
                rng=refine_rng,
                target=target_kind,
                mixture_params=mixture_params,
            )
            pref = prefix_is_estimates(indicators, logw, refine_n_list, elapsed_core)

            chosen = pref[-1]
            for row in pref:
                if float(row["ci_width"]) <= args.refine_ci_target:
                    chosen = row
                    break

            refine_eval = float(chosen["n"])
            refine_ciw = float(chosen["ci_width"])
            refined_p = float(chosen["pc_hat"])
            refine_hits = float(chosen["hits"])

        rows.append(
            {
                "scenario_id": sid,
                "tier_like": tier_like,
                "mixture_flag": mixture_flag,
                "mu_norm": mu_norm,
                "sigma_scale": sigma_scale,
                "classification": classification,
                "coarse_Pc": coarse_p,
                "coarse_ci_width": coarse_ciw,
                "coarse_eval_count": args.coarse_n,
                "refine_method": refine_method,
                "refine_eval_count": refine_eval,
                "refine_ci_width": refine_ciw,
                "refined_Pc": refined_p,
                "refine_hits": refine_hits,
                "t_low": args.t_low,
                "t_high": args.t_high,
                "margin_mult": args.margin_mult,
                "margin_floor": args.margin_floor,
                "refine_ci_target": args.refine_ci_target,
                "refine_n_max": refine_n_max,
                "mix_weight": args.mix_weight,
                "mix_inflation": args.mix_inflation,
                "mix_shift_frac": args.mix_shift_frac,
                "is_shift_frac": args.is_shift_frac,
            }
        )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    gray_frac = float(np.mean(out_df["classification"] == "GRAY"))
    print(f"Saved: {args.out}")
    print(
        f"  SAFE={np.mean(out_df['classification']=='SAFE'):.2%} "
        f"DANGER={np.mean(out_df['classification']=='DANGER'):.2%} "
        f"GRAY={gray_frac:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
