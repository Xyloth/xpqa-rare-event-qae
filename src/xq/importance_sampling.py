from __future__ import annotations

import time
from math import log

import numpy as np
import pandas as pd

from xq.encounter import closest_approach_batch


def log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute log N(x | mean, cov) for one or many samples.

    Args:
        x: Sample(s), shape (d,) or (n, d).
        mean: Mean vector, shape (d,).
        cov: Covariance matrix, shape (d, d).

    Returns:
        Log density value(s), shape (n,) or scalar for a single sample.
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mean.ndim != 1:
        raise ValueError("mean must be a 1D array")
    if cov.shape != (mean.size, mean.size):
        raise ValueError("cov must be shape (d, d) matching mean")

    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.shape[1] != mean.size:
        raise ValueError("x must have shape (n, d) matching mean")

    chol = np.linalg.cholesky(cov)
    diff = (x - mean).T
    sol = np.linalg.solve(chol, diff)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    log_norm = -0.5 * (mean.size * np.log(2.0 * np.pi) + log_det + quad)

    if single:
        return float(log_norm[0])
    return log_norm


def importance_sampling_core(
    n: int,
    mean: np.ndarray,
    cov: np.ndarray,
    mean_proposal: np.ndarray,
    rng: np.random.Generator,
    T: float,
    d0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample from proposal and return event indicators + log weights.

    Returns:
        indicators: Boolean array of shape (n,).
        log_weights: Array of shape (n,) for log p(x)/q(x).
        elapsed_s: Core runtime in seconds.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")

    mean = np.asarray(mean, dtype=float)
    mean_proposal = np.asarray(mean_proposal, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.shape != (6,) or mean_proposal.shape != (6,):
        raise ValueError("mean and mean_proposal must be shape (6,)")
    if cov.shape != (6, 6):
        raise ValueError("cov must be shape (6, 6)")

    start = time.perf_counter()
    samples = rng.multivariate_normal(mean_proposal, cov, size=n)
    r0 = samples[:, :3]
    v0 = samples[:, 3:]
    _, d_min = closest_approach_batch(r0, v0, T)
    indicators = d_min < d0

    log_p = log_gaussian_pdf(samples, mean, cov)
    log_q = log_gaussian_pdf(samples, mean_proposal, cov)
    log_weights = log_p - log_q
    elapsed = time.perf_counter() - start
    return indicators, log_weights, elapsed


def log_gaussian_mixture_pdf(
    x: np.ndarray,
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None = None,
) -> np.ndarray:
    """Compute log density for a two-component Gaussian mixture."""
    if not (0.0 <= w < 1.0):
        raise ValueError("w must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")
    mean6 = np.asarray(mean6, dtype=float)
    cov6 = np.asarray(cov6, dtype=float)
    if mean6.shape != (6,) or cov6.shape != (6, 6):
        raise ValueError("mean6/cov6 must be shape (6,) and (6,6)")
    if mean_shift6 is None:
        mean_shift6 = np.zeros(6, dtype=float)
    else:
        mean_shift6 = np.asarray(mean_shift6, dtype=float)
        if mean_shift6.shape != (6,):
            raise ValueError("mean_shift6 must be shape (6,)")

    log_a = log_gaussian_pdf(x, mean6, cov6)
    log_b = log_gaussian_pdf(x, mean6 + mean_shift6, inflation_k * cov6)

    if w == 0.0:
        return log_a
    if w >= 1.0:
        return log_b

    # Stable log-sum-exp for two terms: log((1-w)*exp(log_a) + w*exp(log_b))
    term_a = np.log1p(-w) + log_a
    term_b = log(w) + log_b
    m = np.maximum(term_a, term_b)
    return m + np.log(np.exp(term_a - m) + np.exp(term_b - m))


def importance_sampling_core_mixture_target(
    n: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None,
    mean_proposal: np.ndarray,
    cov_proposal: np.ndarray,
    rng: np.random.Generator,
    T: float,
    d0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """IS core where target is a Gaussian mixture and proposal is Gaussian."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")

    mean6 = np.asarray(mean6, dtype=float)
    cov6 = np.asarray(cov6, dtype=float)
    mean_proposal = np.asarray(mean_proposal, dtype=float)
    cov_proposal = np.asarray(cov_proposal, dtype=float)
    if mean6.shape != (6,) or mean_proposal.shape != (6,):
        raise ValueError("mean6 and mean_proposal must be shape (6,)")
    if cov6.shape != (6, 6) or cov_proposal.shape != (6, 6):
        raise ValueError("cov6 and cov_proposal must be shape (6, 6)")

    start = time.perf_counter()
    samples = rng.multivariate_normal(mean_proposal, cov_proposal, size=n)
    r0 = samples[:, :3]
    v0 = samples[:, 3:]
    _, d_min = closest_approach_batch(r0, v0, T)
    indicators = d_min < d0

    log_p = log_gaussian_mixture_pdf(
        x=samples,
        mean6=mean6,
        cov6=cov6,
        w=w,
        inflation_k=inflation_k,
        mean_shift6=mean_shift6,
    )
    log_q = log_gaussian_pdf(samples, mean_proposal, cov_proposal)
    log_weights = log_p - log_q
    elapsed = time.perf_counter() - start
    return indicators, log_weights, elapsed


def _stats_from_logw(
    indicators: np.ndarray, log_weights: np.ndarray, z: float = 1.96
) -> dict[str, float]:
    log_weights = np.asarray(log_weights, dtype=float)
    indicators = np.asarray(indicators, dtype=float)

    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)

    sum_w = float(np.sum(weights))
    sum_wi = float(np.sum(weights * indicators))

    pc_hat = sum_wi / sum_w
    w_norm = weights / sum_w
    ess = 1.0 / float(np.sum(w_norm**2))

    # Normal-approx CI using normalized-weight variance (approximate).
    var_hat = float(np.sum((w_norm**2) * (indicators - pc_hat) ** 2))
    half = z * np.sqrt(var_hat)
    ci_low = max(0.0, pc_hat - half)
    ci_high = min(1.0, pc_hat + half)

    return {
        "pc_hat": pc_hat,
        "ess": ess,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "hits": int(np.sum(indicators)),
    }


def estimate_pc_importance_sampling(
    n: int,
    mean: np.ndarray,
    cov: np.ndarray,
    mean_proposal: np.ndarray,
    rng: np.random.Generator,
    T: float,
    d0: float,
) -> dict[str, float]:
    """Estimate Pc with importance sampling using a Gaussian proposal."""
    indicators, logw, elapsed = importance_sampling_core(
        n=n,
        mean=mean,
        cov=cov,
        mean_proposal=mean_proposal,
        rng=rng,
        T=T,
        d0=d0,
    )
    stats = _stats_from_logw(indicators, logw)
    stats["elapsed_s"] = elapsed
    return stats


def prefix_is_estimates(
    indicators: np.ndarray,
    log_weights: np.ndarray,
    n_list: list[int],
    elapsed_core: float,
    z: float = 1.96,
) -> list[dict[str, float]]:
    """Compute prefix IS estimates using precomputed indicators and log-weights."""
    log_weights = np.asarray(log_weights, dtype=float)
    indicators = np.asarray(indicators, dtype=float)
    n_max = log_weights.size

    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)

    cum_w = np.cumsum(weights)
    cum_w2 = np.cumsum(weights**2)
    cum_wi = np.cumsum(weights * indicators)
    cum_hits = np.cumsum(indicators)

    per_sample_time = elapsed_core / n_max
    rows: list[dict[str, float]] = []
    for n in n_list:
        sum_w = float(cum_w[n - 1])
        sum_w2 = float(cum_w2[n - 1])
        sum_wi = float(cum_wi[n - 1])
        pc_hat = sum_wi / sum_w
        ess = (sum_w**2) / sum_w2

        # Normal-approx CI using normalized-weight variance (approximate).
        w_norm = weights[:n] / sum_w
        var_hat = float(np.sum((w_norm**2) * (indicators[:n] - pc_hat) ** 2))
        half = z * np.sqrt(var_hat)
        ci_low = max(0.0, pc_hat - half)
        ci_high = min(1.0, pc_hat + half)

        rows.append(
            {
                "n": n,
                "pc_hat": pc_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ess": ess,
                "hits": int(cum_hits[n - 1]),
                "elapsed_s": per_sample_time * n,
            }
        )

    return rows


def select_shift_frac(
    df: pd.DataFrame, ess_floor: float = 0.005, require_hits: bool = True
) -> tuple[float, str]:
    """Select shift_frac from a sweep dataframe using consistent criteria."""
    if ess_floor <= 0:
        raise ValueError("ess_floor must be positive")

    work = df.copy()
    if "ci_width" not in work.columns:
        if "ci_low" not in work.columns or "ci_high" not in work.columns:
            raise ValueError("df must include ci_low and ci_high to compute ci_width")
        work["ci_width"] = work["ci_high"] - work["ci_low"]

    if "ess_n" in work.columns:
        work["ess_ratio"] = work["ess_n"]
    elif "ess_over_n" in work.columns:
        work["ess_ratio"] = work["ess_over_n"]
    elif "ess" in work.columns and "n_max" in work.columns:
        work["ess_ratio"] = work["ess"] / work["n_max"]
    else:
        raise ValueError("df must include ess_n or ess_over_n (or ess with n_max)")

    criteria = []
    if require_hits:
        if "hits" not in work.columns:
            raise ValueError("df must include hits when require_hits is True")
        work = work[work["hits"] > 0]
        criteria.append("hits>0")
    work = work[work["ess_ratio"] >= ess_floor]
    criteria.append(f"ess/n>={ess_floor:g}")

    if work.empty:
        raise ValueError("no rows satisfy selection criteria")

    best = work.sort_values(["ci_width", "ess_ratio"], ascending=[True, False]).iloc[0]
    reason = (
        f"selected from {len(work)}/{len(df)} rows with "
        + " and ".join(criteria)
        + " by min ci_width"
    )
    return float(best["shift_frac"]), reason
