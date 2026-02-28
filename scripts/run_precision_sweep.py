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

from xq.encounter_plane import (  # noqa: E402
    miss_distance,
    sample_positions_mixture,
    sample_positions_single,
)
from xq.quantization import quantized_event_indicator_xy  # noqa: E402
from xq.scenarios import build_scenario  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if not vals:
        raise ValueError("list must contain at least one integer")
    return vals


def ci_from_replicates(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, mean
    std = float(np.std(values, ddof=1))
    half = z * std / np.sqrt(values.size)
    return max(0.0, mean - half), min(1.0, mean + half)


def sample_scenario(
    n: int,
    scenario: dict[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(scenario["mu"], dtype=float)
    Sigma = np.asarray(scenario["Sigma"], dtype=float)
    if str(scenario["distribution"]) == "single_gaussian":
        return sample_positions_single(n, mu, Sigma, rng)

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


def default_scenarios() -> list[tuple[int, str]]:
    return [
        (3, "baseline_single"),
        (3, "baseline_mixture"),
        (3, "near_single"),
        (2, "baseline_single"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-point precision calibration sweep"
    )
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=30000)
    parser.add_argument("--n-reps", type=int, default=25)
    parser.add_argument(
        "--bits", type=parse_int_list, default=parse_int_list("12,16,20,24,32")
    )
    parser.add_argument(
        "--frac-bits",
        type=int,
        default=None,
        help="Fixed fractional bits; default uses bits//2",
    )
    parser.add_argument(
        "--clip-mult",
        type=float,
        default=6.0,
        help="Clip bound multiplier on max sigma around mean",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "precision_sweep.csv",
    )
    args = parser.parse_args()

    if args.d0 <= 0:
        raise ValueError("d0 must be positive")
    if args.N <= 0:
        raise ValueError("N must be positive")
    if args.n_reps <= 0:
        raise ValueError("n-reps must be positive")
    if args.clip_mult <= 0:
        raise ValueError("clip-mult must be positive")

    bits_list = sorted(set(args.bits))
    rows: list[dict[str, float | int | str]] = []

    print("Precision sweep:")
    print(f"  bits={bits_list} N={args.N} n_reps={args.n_reps}")

    for tier, variant in default_scenarios():
        scenario = build_scenario(
            tier=tier, variant=variant, d0=args.d0, seed_base=args.seed
        )

        scenario_name = str(scenario["scenario_name"])
        mu = np.asarray(scenario["mu"], dtype=float)
        Sigma = np.asarray(scenario["Sigma"], dtype=float)
        sigma_ref = float(np.sqrt(np.max(np.diag(Sigma))))
        clip = float(np.linalg.norm(mu) + args.clip_mult * sigma_ref)

        float_pcs: list[float] = []
        quantized_pcs: dict[int, list[float]] = {b: [] for b in bits_list}

        scenario_seed = int(scenario["seed"]) + 7_000_000
        for rep in range(args.n_reps):
            rng = np.random.default_rng(scenario_seed + rep * 1000)
            samples = sample_scenario(args.N, scenario, rng)

            d = miss_distance(samples)
            pc_float = float(np.mean(d < float(scenario["d0"])))
            float_pcs.append(pc_float)

            for b in bits_list:
                ind_q, _ = quantized_event_indicator_xy(
                    samples=samples,
                    d0=float(scenario["d0"]),
                    bits=b,
                    frac_bits=args.frac_bits,
                    clip=clip,
                )
                quantized_pcs[b].append(float(np.mean(ind_q)))

        float_arr = np.asarray(float_pcs, dtype=float)
        pc_float_mean = float(np.mean(float_arr))

        for b in bits_list:
            arr = np.asarray(quantized_pcs[b], dtype=float)
            pc_mean = float(np.mean(arr))
            ci_low, ci_high = ci_from_replicates(arr)
            ci_width = float(ci_high - ci_low)
            abs_error = abs(pc_mean - pc_float_mean)
            rel_error = abs_error / max(pc_float_mean, 1e-12)

            frac_bits_used = args.frac_bits if args.frac_bits is not None else (b // 2)

            rows.append(
                {
                    "scenario_name": scenario_name,
                    "tier": tier,
                    "bits": b,
                    "frac_bits": int(frac_bits_used),
                    "N": args.N,
                    "n_reps": args.n_reps,
                    "Pc_mean": pc_mean,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "ci_width": ci_width,
                    "Pc_float_mean": pc_float_mean,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "clip": clip,
                    "variant": variant,
                }
            )

        print(
            f"  {scenario_name:<26s} Pc_float={pc_float_mean:.3e} "
            f"abs_err(b={bits_list[-1]})={abs(np.mean(quantized_pcs[bits_list[-1]]) - pc_float_mean):.3e}"
        )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["tier", "scenario_name", "bits"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
