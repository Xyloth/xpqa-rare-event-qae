from __future__ import annotations

import time

import numpy as np

from xq.distributions import sample_rv, sample_rv_from_mixture
from xq.encounter import closest_approach_batch
from xq.importance_sampling import (
    importance_sampling_core,
    importance_sampling_core_mixture_target,
    prefix_is_estimates,
)


def default_n_list(nmax: int) -> list[int]:
    if nmax <= 0:
        raise ValueError("nmax must be positive")
    base = [1000, 3000, 10000, 30000]
    n_list = [n for n in base if n < nmax]
    if nmax not in n_list:
        n_list.append(nmax)
    return n_list


def normalized_n_list(n_list: list[int] | None, nmax: int) -> list[int]:
    if nmax <= 0:
        raise ValueError("nmax must be positive")
    if n_list is None:
        return default_n_list(nmax)

    out = list(n_list)
    if any(n <= 0 for n in out):
        raise ValueError("n-list must contain positive integers")
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError("n-list must be strictly increasing")
    if nmax < out[-1]:
        raise ValueError("nmax must be >= max(n-list)")
    if nmax > out[-1]:
        out.append(nmax)
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _mc_rows_from_indicators(
    indicators: np.ndarray,
    n_list: list[int],
    elapsed_core: float,
    seed: int,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]
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
                "ci_width": ci_high - ci_low,
                "hits": hits,
                "elapsed_s": per_sample_time * n,
                "seed": seed,
                "n_max": n_max,
            }
        )
    return rows


def run_mc_prefix_gaussian(
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

    return _mc_rows_from_indicators(indicators, n_list, elapsed_core, seed)


def run_mc_prefix_mixture(
    n_list: list[int],
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None,
    seed: int,
    T: float,
    d0: float,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    r0_w, v0_w = sample_rv_from_mixture(
        warmup_n,
        mean6,
        cov6,
        w,
        inflation_k,
        mean_shift6,
        warmup_rng,
    )
    _ = closest_approach_batch(r0_w, v0_w, T)

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    r0, v0 = sample_rv_from_mixture(
        n_max,
        mean6,
        cov6,
        w,
        inflation_k,
        mean_shift6,
        rng,
    )
    _, d_min = closest_approach_batch(r0, v0, T)
    indicators = d_min < d0
    elapsed_core = time.perf_counter() - start

    return _mc_rows_from_indicators(indicators, n_list, elapsed_core, seed)


def _decorate_is_rows(rows: list[dict[str, float]], seed: int, n_max: int) -> None:
    for row in rows:
        row["seed"] = seed
        row["n_max"] = n_max
        row["ci_width"] = float(row["ci_high"] - row["ci_low"])
        row["ess_over_n"] = float(row["ess"] / row["n"])


def run_is_prefix_gaussian(
    n_list: list[int],
    mean6: np.ndarray,
    cov6: np.ndarray,
    mean_proposal: np.ndarray,
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
        mean_proposal=mean_proposal,
        rng=warmup_rng,
        T=T,
        d0=d0,
    )

    rng = np.random.default_rng(seed)
    indicators, logw, elapsed_core = importance_sampling_core(
        n=n_max,
        mean=mean6,
        cov=cov6,
        mean_proposal=mean_proposal,
        rng=rng,
        T=T,
        d0=d0,
    )
    rows = prefix_is_estimates(indicators, logw, n_list, elapsed_core)
    _decorate_is_rows(rows, seed=seed, n_max=n_max)
    return rows


def run_is_prefix_mixture_target(
    n_list: list[int],
    mean6: np.ndarray,
    cov6: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift6: np.ndarray | None,
    mean_proposal: np.ndarray,
    cov_proposal: np.ndarray,
    seed: int,
    T: float,
    d0: float,
) -> list[dict[str, float | int]]:
    n_max = n_list[-1]

    warmup_n = min(1000, n_max)
    warmup_rng = np.random.default_rng(seed + 1_000_003)
    _ = importance_sampling_core_mixture_target(
        n=warmup_n,
        mean6=mean6,
        cov6=cov6,
        w=w,
        inflation_k=inflation_k,
        mean_shift6=mean_shift6,
        mean_proposal=mean_proposal,
        cov_proposal=cov_proposal,
        rng=warmup_rng,
        T=T,
        d0=d0,
    )

    rng = np.random.default_rng(seed)
    indicators, logw, elapsed_core = importance_sampling_core_mixture_target(
        n=n_max,
        mean6=mean6,
        cov6=cov6,
        w=w,
        inflation_k=inflation_k,
        mean_shift6=mean_shift6,
        mean_proposal=mean_proposal,
        cov_proposal=cov_proposal,
        rng=rng,
        T=T,
        d0=d0,
    )
    rows = prefix_is_estimates(indicators, logw, n_list, elapsed_core)
    _decorate_is_rows(rows, seed=seed, n_max=n_max)
    return rows
