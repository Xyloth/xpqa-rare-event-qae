from __future__ import annotations

import numpy as np

_ENCOUNTERPLANE_TIER_MU_FACTORS = {
    # Calibrated for sigma_scale ~= 0.7 and d0 = 1.0 to produce rare-event bands.
    2: 3.5,
    3: 4.0,
}

_DEFAULT_VARIANTS = {
    "baseline_single",
    "baseline_mixture",
    "near_single",
    "near_mixture",
}
_VARIANT_SEED_OFFSET = {
    "baseline_single": 0,
    "baseline_mixture": 100,
    "near_single": 200,
    "near_mixture": 300,
}


def build_scenario(
    tier: int,
    variant: str,
    *,
    d0: float = 1.0,
    sigma_scale: float = 0.7,
    anisotropy: float = 0.7,
    corr: float = 0.2,
    mixture_weight: float = 0.05,
    inflation_k: float = 10.0,
    near_margin_frac: float = 0.05,
    mean_shift_frac: float = 0.2,
    seed_base: int = 12345,
) -> dict[str, object]:
    """Build encounter-plane benchmark scenario parameters.

    Variants:
    - baseline_single
    - baseline_mixture
    - near_single
    - near_mixture
    """
    if tier not in _ENCOUNTERPLANE_TIER_MU_FACTORS:
        raise ValueError(
            f"tier must be one of {sorted(_ENCOUNTERPLANE_TIER_MU_FACTORS)}"
        )
    if variant not in _DEFAULT_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_DEFAULT_VARIANTS)}")
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")
    if anisotropy <= 0:
        raise ValueError("anisotropy must be positive")
    if not (-0.95 < corr < 0.95):
        raise ValueError("corr must be in (-0.95, 0.95)")
    if not (0.0 <= mixture_weight < 1.0):
        raise ValueError("mixture_weight must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")
    if near_margin_frac < 0:
        raise ValueError("near_margin_frac must be non-negative")
    if mean_shift_frac < 0:
        raise ValueError("mean_shift_frac must be non-negative")

    near = variant.startswith("near_")
    use_mixture = variant.endswith("mixture")

    if near:
        mu_mag = d0 * (1.0 + near_margin_frac)
    else:
        mu_mag = _ENCOUNTERPLANE_TIER_MU_FACTORS[tier] * d0

    mu = np.array([mu_mag, 0.0], dtype=float)

    sig_x = sigma_scale * d0
    sig_y = anisotropy * sigma_scale * d0
    cov_xy = corr * sig_x * sig_y
    Sigma = np.array(
        [
            [sig_x**2, cov_xy],
            [cov_xy, sig_y**2],
        ],
        dtype=float,
    )

    # Wide mode shift nudges probability mass toward boundary-crossing region.
    mean_shift = np.array([-mean_shift_frac * d0, 0.0], dtype=float)

    scenario_name = f"ep_tier{tier}_{variant}"
    scenario_seed = seed_base + tier * 1000 + _VARIANT_SEED_OFFSET[variant]

    return {
        "scenario_name": scenario_name,
        "tier": tier,
        "variant": variant,
        "family": "near_threshold" if near else "baseline",
        "distribution": "mixture" if use_mixture else "single_gaussian",
        "mu": mu,
        "Sigma": Sigma,
        "d0": float(d0),
        "seed": int(scenario_seed),
        "mixture_weight": float(mixture_weight if use_mixture else 0.0),
        "inflation_k": float(inflation_k if use_mixture else 1.0),
        "mean_shift": mean_shift if use_mixture else np.zeros(2, dtype=float),
        "mean_shift_flag": bool(use_mixture and np.linalg.norm(mean_shift) > 0.0),
        "near_margin_frac": float(near_margin_frac),
        "sigma_scale": float(sigma_scale),
        "anisotropy": float(anisotropy),
        "corr": float(corr),
    }


def default_variants() -> list[str]:
    return [
        "baseline_single",
        "baseline_mixture",
        "near_single",
        "near_mixture",
    ]
