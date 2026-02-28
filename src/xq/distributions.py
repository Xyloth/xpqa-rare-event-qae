from __future__ import annotations

import numpy as np

# Factors chosen so that with r_sigma == d0 and tiny v_sigma, Pc is roughly
# 1e-2 (tier 1), 1e-4 (tier 2), 1e-6 (tier 3).
_TIER_MEAN_FACTORS = {
    1: 2.7,
    2: 4.3,
    3: 5.4,
}


def sample_rv(
    n: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (r0, v0) from a 6D Gaussian.

    Args:
        n: Number of samples.
        mean6: Mean vector, shape (6,).
        cov6: Covariance matrix, shape (6, 6).
        rng: Numpy RNG.

    Returns:
        r0: Array of shape (n, 3).
        v0: Array of shape (n, 3).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    mean6 = np.asarray(mean6, dtype=float)
    cov6 = np.asarray(cov6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")
    if cov6.shape != (6, 6):
        raise ValueError("cov6 must be shape (6, 6)")

    samples = rng.multivariate_normal(mean6, cov6, size=n)
    r0 = samples[:, :3]
    v0 = samples[:, 3:]
    return r0, v0


def sample_rv_mixture(
    n: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from a two-component Gaussian mixture.

    Component A uses (mean6, cov6). Component B uses:
    - mean6 + mean_shift6 (or mean6 if shift is None)
    - inflation_k * cov6

    Returns:
        samples: Array shape (n, 6)
        labels: Component labels shape (n,), where 0=A and 1=B
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 <= w < 1.0):
        raise ValueError("w must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")
    if rng is None:
        raise ValueError("rng must be provided")

    mean6 = np.asarray(mean6, dtype=float)
    cov6 = np.asarray(cov6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")
    if cov6.shape != (6, 6):
        raise ValueError("cov6 must be shape (6, 6)")

    if mean_shift6 is None:
        mean_shift6 = np.zeros(6, dtype=float)
    else:
        mean_shift6 = np.asarray(mean_shift6, dtype=float)
        if mean_shift6.shape != (6,):
            raise ValueError("mean_shift6 must be shape (6,)")

    labels = (rng.random(n) < w).astype(int)
    samples = np.zeros((n, 6), dtype=float)

    n_a = int(np.sum(labels == 0))
    n_b = n - n_a
    if n_a > 0:
        samples[labels == 0] = rng.multivariate_normal(mean6, cov6, size=n_a)
    if n_b > 0:
        mean_b = mean6 + mean_shift6
        cov_b = inflation_k * cov6
        samples[labels == 1] = rng.multivariate_normal(mean_b, cov_b, size=n_b)

    return samples, labels


def sample_rv_from_mixture(
    n: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning r0, v0 from mixture samples."""
    samples, _ = sample_rv_mixture(
        n=n,
        mean6=mean6,
        cov6=cov6,
        w=w,
        inflation_k=inflation_k,
        mean_shift6=mean_shift6,
        rng=rng,
    )
    return samples[:, :3], samples[:, 3:]


def tier_params(
    d0: float,
    tier: int = 2,
    r_sigma: float | None = None,
    v_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean/cov for a difficulty tier by shifting mean distance.

    The mean distance is set to a tier-specific multiple of d0. The suggested
    multipliers are calibrated for r_sigma == d0 with small v_sigma (for
    example, v_sigma << d0 / T) to yield target probabilities roughly
    1e-2, 1e-4, 1e-6.
    """
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if tier not in _TIER_MEAN_FACTORS:
        raise ValueError(f"tier must be one of {sorted(_TIER_MEAN_FACTORS)}")

    if r_sigma is None:
        r_sigma = d0
    if v_sigma is None:
        v_sigma = 0.0

    mean_r = np.array([_TIER_MEAN_FACTORS[tier] * d0, 0.0, 0.0], dtype=float)
    mean_v = np.zeros(3, dtype=float)
    mean6 = np.concatenate([mean_r, mean_v])

    cov6 = np.diag([r_sigma**2] * 3 + [v_sigma**2] * 3)
    return mean6, cov6


def near_threshold_mean(
    mean6: np.ndarray, d0: float, margin_frac: float = 0.05
) -> np.ndarray:
    """Move mean position so ||r_mean|| is near d0*(1+margin_frac)."""
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if margin_frac < 0:
        raise ValueError("margin_frac must be non-negative")

    mean6 = np.asarray(mean6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")

    r_mean = mean6[:3]
    r_norm = float(np.linalg.norm(r_mean))
    target_norm = d0 * (1.0 + margin_frac)
    if r_norm == 0.0:
        r_new = np.array([target_norm, 0.0, 0.0], dtype=float)
    else:
        r_new = r_mean * (target_norm / r_norm)
    return np.concatenate([r_new, mean6[3:]])


def default_mixture_mean_shift(
    mean6: np.ndarray, d0: float, frac: float = 0.2
) -> np.ndarray:
    """Return a small risk-increasing mean shift for the wide-error mode."""
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if frac < 0:
        raise ValueError("frac must be non-negative")

    mean6 = np.asarray(mean6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")

    shift = np.zeros(6, dtype=float)
    r_mean = mean6[:3]
    r_norm = float(np.linalg.norm(r_mean))
    if r_norm == 0.0:
        shift[:3] = np.array([-frac * d0, 0.0, 0.0], dtype=float)
    else:
        # Shift toward the origin to increase event likelihood without extreme drift.
        shift[:3] = -(frac * d0) * (r_mean / r_norm)
    return shift


def mixture_scenario_params(
    mean6: np.ndarray,
    d0: float,
    mixture_weight: float = 0.05,
    inflation_k: float = 10.0,
    mean_shift6: np.ndarray | None = None,
    mean_shift_frac: float = 0.2,
) -> dict[str, np.ndarray | float]:
    """Return validated mixture configuration dictionary."""
    if not (0.0 <= mixture_weight < 1.0):
        raise ValueError("mixture_weight must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")
    if mean_shift6 is None:
        mean_shift6 = default_mixture_mean_shift(mean6, d0=d0, frac=mean_shift_frac)
    else:
        mean_shift6 = np.asarray(mean_shift6, dtype=float)
        if mean_shift6.shape != (6,):
            raise ValueError("mean_shift6 must be shape (6,)")

    return {
        "w": float(mixture_weight),
        "inflation_k": float(inflation_k),
        "mean_shift6": mean_shift6,
    }


def proposal_mean_toward_boundary(
    mean6: np.ndarray, d0: float, shift_frac: float = 0.7
) -> np.ndarray:
    """Shift the mean position toward the event boundary along a risk direction.

    The risk direction is toward the origin (smaller closest-approach distance).
    shift_frac in [0, 1] moves a fraction of the way from the current mean
    distance to the boundary at d0.
    """
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if not (0.0 <= shift_frac <= 1.0):
        raise ValueError("shift_frac must be in [0, 1]")
    mean6 = np.asarray(mean6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")

    r_mean = mean6[:3]
    r_norm = float(np.linalg.norm(r_mean))
    if r_norm == 0.0 or r_norm <= d0:
        return mean6.copy()

    target_norm = r_norm - shift_frac * (r_norm - d0)
    r_new = r_mean * (target_norm / r_norm)
    return np.concatenate([r_new, mean6[3:]])
