from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class _GaussianModel:
    mean: np.ndarray
    cov: np.ndarray
    chol: np.ndarray
    log_norm_const: float

    @classmethod
    def from_mean_cov(cls, mean: np.ndarray, cov: np.ndarray) -> "_GaussianModel":
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if mean.shape != (6,):
            raise ValueError("mean must be shape (6,)")
        if cov.shape != (6, 6):
            raise ValueError("cov must be shape (6, 6)")

        # Small diagonal jitter keeps log-density numerically stable for very small
        # velocity variances while preserving the intended distribution.
        cov_reg = cov + 1e-12 * np.eye(6)
        chol = np.linalg.cholesky(cov_reg)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        log_norm_const = -0.5 * (6 * np.log(2.0 * np.pi) + log_det)
        return cls(mean=mean, cov=cov_reg, chol=chol, log_norm_const=log_norm_const)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def logpdf_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        diff = (x - self.mean).T
        sol = np.linalg.solve(self.chol, diff)
        quad = np.sum(sol**2, axis=0)
        return self.log_norm_const - 0.5 * quad

    def logpdf_single(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        diff = x - self.mean
        sol = np.linalg.solve(self.chol, diff)
        quad = float(np.dot(sol, sol))
        return float(self.log_norm_const - 0.5 * quad)


@dataclass(frozen=True)
class _MixtureModel:
    base: _GaussianModel
    wide: _GaussianModel
    w: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.random(n) < self.w
        out = np.empty((n, 6), dtype=float)
        n_wide = int(np.sum(labels))
        n_base = n - n_wide
        if n_base > 0:
            out[~labels] = self.base.sample(n_base, rng)
        if n_wide > 0:
            out[labels] = self.wide.sample(n_wide, rng)
        return out

    def logpdf_batch(self, x: np.ndarray) -> np.ndarray:
        log_a = self.base.logpdf_batch(x)
        log_b = self.wide.logpdf_batch(x)
        term_a = np.log1p(-self.w) + log_a
        term_b = np.log(self.w) + log_b
        m = np.maximum(term_a, term_b)
        return m + np.log(np.exp(term_a - m) + np.exp(term_b - m))

    def logpdf_single(self, x: np.ndarray) -> float:
        log_a = self.base.logpdf_single(x)
        log_b = self.wide.logpdf_single(x)
        term_a = np.log1p(-self.w) + log_a
        term_b = np.log(self.w) + log_b
        m = max(term_a, term_b)
        return float(m + np.log(np.exp(term_a - m) + np.exp(term_b - m)))


@dataclass
class _LevelSamples:
    samples: np.ndarray
    scores: np.ndarray
    logp: np.ndarray
    eval_count: int
    acceptance_rate: float


def _d_min_batch(samples: np.ndarray, T: float) -> np.ndarray:
    r0 = samples[:, :3]
    v0 = samples[:, 3:]
    v2 = np.sum(v0 * v0, axis=1)
    dot = np.sum(r0 * v0, axis=1)

    t_star = np.zeros_like(v2)
    moving = v2 > 0.0
    t_star[moving] = np.clip(-dot[moving] / v2[moving], 0.0, T)

    r_star = r0 + v0 * t_star[:, None]
    return np.linalg.norm(r_star, axis=1)


def _d_min_single(x: np.ndarray, T: float) -> float:
    r0 = x[:3]
    v0 = x[3:]
    v2 = float(np.dot(v0, v0))
    if v2 <= 0.0:
        t_star = 0.0
    else:
        t_star = float(np.clip(-np.dot(r0, v0) / v2, 0.0, T))
    r_star = r0 + v0 * t_star
    return float(np.linalg.norm(r_star))


def _build_target(
    target: str,
    mean6: np.ndarray,
    cov6: np.ndarray,
    mixture_params: dict[str, float | np.ndarray] | None,
) -> _GaussianModel | _MixtureModel:
    mean6 = np.asarray(mean6, dtype=float)
    cov6 = np.asarray(cov6, dtype=float)
    if mean6.shape != (6,):
        raise ValueError("mean6 must be shape (6,)")
    if cov6.shape != (6, 6):
        raise ValueError("cov6 must be shape (6, 6)")

    base = _GaussianModel.from_mean_cov(mean6, cov6)
    if target == "single":
        return base

    if target != "mixture":
        raise ValueError("target must be 'single' or 'mixture'")

    if mixture_params is None:
        raise ValueError("mixture_params must be provided when target='mixture'")

    w = float(mixture_params.get("w", 0.0))
    inflation_k = float(mixture_params.get("inflation_k", 1.0))
    mean_shift6 = mixture_params.get("mean_shift6", np.zeros(6, dtype=float))

    mean_shift6 = np.asarray(mean_shift6, dtype=float)
    if mean_shift6.shape != (6,):
        raise ValueError("mixture mean_shift6 must be shape (6,)")
    if not (0.0 < w < 1.0):
        raise ValueError("mixture weight w must be in (0, 1)")
    if inflation_k <= 0:
        raise ValueError("mixture inflation_k must be positive")

    wide = _GaussianModel.from_mean_cov(mean6 + mean_shift6, inflation_k * cov6)
    return _MixtureModel(base=base, wide=wide, w=w)


def _sample_conditional_level(
    seeds: np.ndarray,
    seed_scores: np.ndarray,
    seed_logp: np.ndarray,
    n: int,
    threshold: float,
    n_mcmc_steps: int,
    proposal_std: np.ndarray,
    logpdf_single,
    T: float,
    rng: np.random.Generator,
) -> _LevelSamples:
    dim = seeds.shape[1]
    n_seeds = seeds.shape[0]
    chain_len = int(ceil(n / n_seeds))

    out_samples = np.empty((n, dim), dtype=float)
    out_scores = np.empty(n, dtype=float)
    out_logp = np.empty(n, dtype=float)

    eval_count = 0
    accepts = 0
    proposals = 0

    idx = 0
    for i in range(n_seeds):
        x = seeds[i].copy()
        score_x = float(seed_scores[i])
        logp_x = float(seed_logp[i])

        for _ in range(chain_len):
            for _ in range(n_mcmc_steps):
                proposals += 1
                prop = x + proposal_std * rng.normal(size=dim)
                score_prop = _d_min_single(prop, T)
                eval_count += 1

                if score_prop <= threshold:
                    logp_prop = float(logpdf_single(prop))
                    log_alpha = min(0.0, logp_prop - logp_x)
                    if np.log(rng.random()) < log_alpha:
                        x = prop
                        score_x = score_prop
                        logp_x = logp_prop
                        accepts += 1

            if idx < n:
                out_samples[idx] = x
                out_scores[idx] = score_x
                out_logp[idx] = logp_x
                idx += 1
            else:
                break

        if idx >= n:
            break

    accept_rate = (accepts / proposals) if proposals > 0 else 0.0
    return _LevelSamples(
        samples=out_samples,
        scores=out_scores,
        logp=out_logp,
        eval_count=eval_count,
        acceptance_rate=accept_rate,
    )


def estimate_pc_splitting(
    n: int,
    mean6: np.ndarray,
    cov6: np.ndarray,
    T: float,
    d0: float,
    p0: float = 0.1,
    n_levels_max: int = 20,
    proposal_scale: float = 0.8,
    n_mcmc_steps: int = 2,
    seed: int = 12345,
    target: str = "single",
    mixture_params: dict[str, float | np.ndarray] | None = None,
) -> dict[str, float | int | list[float] | dict[str, object] | bool]:
    """Estimate rare-event probability with subset simulation / splitting.

    Event is d_min < d0 under the chosen target distribution over (r0, v0).

    The estimator uses adaptive intermediate levels based on the p0-elite score
    quantile and MCMC sampling restricted to each level set.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0, 1)")
    if n_levels_max <= 0:
        raise ValueError("n_levels_max must be positive")
    if proposal_scale <= 0:
        raise ValueError("proposal_scale must be positive")
    if n_mcmc_steps <= 0:
        raise ValueError("n_mcmc_steps must be positive")

    target_model = _build_target(target, mean6, cov6, mixture_params)
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(round(p0 * n)))

    diag = np.diag(np.asarray(cov6, dtype=float))
    proposal_std = proposal_scale * np.sqrt(np.maximum(diag, 1e-12))

    start = time.perf_counter()

    samples = target_model.sample(n, rng)
    scores = _d_min_batch(samples, T)
    logp = target_model.logpdf_batch(samples)

    eval_count = int(n)
    levels: list[float] = []
    hits_per_level: list[int] = []
    cond_probs: list[float] = []
    acceptance_rates: list[float] = []

    prob_product = 1.0
    terminated_by_max_levels = False

    for _level in range(n_levels_max):
        hits_final = int(np.sum(scores < d0))
        hits_per_level.append(hits_final)

        elite_idx = np.argpartition(scores, n_elite - 1)[:n_elite]
        level_threshold = float(np.max(scores[elite_idx]))

        if level_threshold <= d0:
            levels.append(level_threshold)
            q_final = hits_final / n
            pc_hat = prob_product * q_final
            break

        levels.append(level_threshold)
        cond_prob = n_elite / n
        cond_probs.append(cond_prob)
        prob_product *= cond_prob

        level_out = _sample_conditional_level(
            seeds=samples[elite_idx],
            seed_scores=scores[elite_idx],
            seed_logp=logp[elite_idx],
            n=n,
            threshold=level_threshold,
            n_mcmc_steps=n_mcmc_steps,
            proposal_std=proposal_std,
            logpdf_single=target_model.logpdf_single,
            T=T,
            rng=rng,
        )
        acceptance_rates.append(level_out.acceptance_rate)

        samples = level_out.samples
        scores = level_out.scores
        logp = level_out.logp
        eval_count += int(level_out.eval_count)

    else:
        terminated_by_max_levels = True
        hits_final = int(np.sum(scores < d0))
        q_final = hits_final / n
        pc_hat = prob_product * q_final

    elapsed_s = time.perf_counter() - start

    return {
        "pc_hat": float(pc_hat),
        "levels": levels,
        "n_levels": int(len(levels)),
        "eval_count": int(eval_count),
        "acceptance_rate": (
            float(np.mean(acceptance_rates)) if acceptance_rates else 0.0
        ),
        "elapsed_s": float(elapsed_s),
        "diagnostics": {
            "hits_per_level": hits_per_level,
            "conditional_probs": cond_probs,
            "n_elite": n_elite,
            "p0": p0,
            "terminated_by_max_levels": terminated_by_max_levels,
            "n_mcmc_steps": n_mcmc_steps,
            "proposal_scale": proposal_scale,
        },
    }
