from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil, log

import numpy as np


def _validate_mu_sigma(
    mu: np.ndarray, Sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if mu.shape != (2,):
        raise ValueError("mu must be shape (2,)")
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma must be shape (2, 2)")
    return mu, Sigma


def sample_positions_single(
    n: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample encounter-plane positions from a single Gaussian target."""
    if n <= 0:
        raise ValueError("n must be positive")
    mu, Sigma = _validate_mu_sigma(mu, Sigma)
    return rng.multivariate_normal(mu, Sigma, size=n)


def sample_positions_mixture(
    n: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a two-component Gaussian mixture and return samples + labels.

    Labels are 0 for nominal component and 1 for wide component.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if rng is None:
        raise ValueError("rng must be provided")
    if not (0.0 <= w < 1.0):
        raise ValueError("w must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")

    mu, Sigma = _validate_mu_sigma(mu, Sigma)

    if mean_shift is None:
        mean_shift = np.zeros(2, dtype=float)
    else:
        mean_shift = np.asarray(mean_shift, dtype=float)
        if mean_shift.shape != (2,):
            raise ValueError("mean_shift must be shape (2,)")

    labels = (rng.random(n) < w).astype(int)
    out = np.empty((n, 2), dtype=float)
    n_nominal = int(np.sum(labels == 0))
    n_wide = n - n_nominal

    if n_nominal > 0:
        out[labels == 0] = rng.multivariate_normal(mu, Sigma, size=n_nominal)
    if n_wide > 0:
        out[labels == 1] = rng.multivariate_normal(
            mu + mean_shift, inflation_k * Sigma, size=n_wide
        )

    return out, labels


def miss_distance(samples: np.ndarray) -> np.ndarray:
    """Compute Euclidean miss distance in encounter plane."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be shape (n, 2)")
    return np.linalg.norm(samples, axis=1)


def event_indicator(d: np.ndarray | float, d0: float) -> np.ndarray | int:
    """Return event indicator(s): 1 when miss distance is below threshold."""
    if d0 <= 0:
        raise ValueError("d0 must be positive")

    arr = np.asarray(d, dtype=float)
    out = (arr < d0).astype(int)
    if np.isscalar(d):
        return int(out)
    return out


def proposal_mean_toward_boundary(
    mu: np.ndarray,
    d0: float,
    shift_frac: float = 0.6,
) -> np.ndarray:
    """Shift proposal mean toward boundary along risk direction toward origin."""
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if not (0.0 <= shift_frac <= 1.0):
        raise ValueError("shift_frac must be in [0, 1]")

    mu = np.asarray(mu, dtype=float)
    if mu.shape != (2,):
        raise ValueError("mu must be shape (2,)")

    r = float(np.linalg.norm(mu))
    if r <= d0 or r == 0.0:
        return mu.copy()

    target_r = r - shift_frac * (r - d0)
    return mu * (target_r / r)


def log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute log N(x|mean,cov) for one or many samples."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mean.ndim != 1:
        raise ValueError("mean must be 1D")
    if cov.shape != (mean.size, mean.size):
        raise ValueError("cov shape mismatch")

    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != mean.size:
        raise ValueError("x shape mismatch")

    cov_reg = cov + 1e-12 * np.eye(mean.size)
    chol = np.linalg.cholesky(cov_reg)
    diff = (x - mean).T
    sol = np.linalg.solve(chol, diff)
    quad = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    log_norm = -0.5 * (mean.size * np.log(2.0 * np.pi) + log_det + quad)

    if single:
        return float(log_norm[0])
    return log_norm


def log_mixture_pdf(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    w: float,
    inflation_k: float,
    mean_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Compute log density of a two-component Gaussian mixture."""
    if not (0.0 <= w < 1.0):
        raise ValueError("w must be in [0, 1)")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")

    mu, Sigma = _validate_mu_sigma(mu, Sigma)
    if mean_shift is None:
        mean_shift = np.zeros(2, dtype=float)
    else:
        mean_shift = np.asarray(mean_shift, dtype=float)
        if mean_shift.shape != (2,):
            raise ValueError("mean_shift must be shape (2,)")

    log_a = log_gaussian_pdf(x, mu, Sigma)
    if w == 0.0:
        return log_a

    log_b = log_gaussian_pdf(x, mu + mean_shift, inflation_k * Sigma)
    term_a = np.log1p(-w) + log_a
    term_b = log(w) + log_b
    m = np.maximum(term_a, term_b)
    return m + np.log(np.exp(term_a - m) + np.exp(term_b - m))


def importance_sampling_core(
    n: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    d0: float,
    mu_proposal: np.ndarray,
    rng: np.random.Generator,
    target: str = "single",
    mixture_params: dict[str, float | np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample from proposal and return indicators, log-weights, elapsed seconds."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d0 <= 0:
        raise ValueError("d0 must be positive")

    mu, Sigma = _validate_mu_sigma(mu, Sigma)
    mu_proposal = np.asarray(mu_proposal, dtype=float)
    if mu_proposal.shape != (2,):
        raise ValueError("mu_proposal must be shape (2,)")

    start = time.perf_counter()
    samples = rng.multivariate_normal(mu_proposal, Sigma, size=n)
    d = miss_distance(samples)
    indicators = d < d0

    if target == "single":
        log_p = log_gaussian_pdf(samples, mu, Sigma)
    elif target == "mixture":
        if mixture_params is None:
            raise ValueError("mixture_params required for target='mixture'")
        log_p = log_mixture_pdf(
            samples,
            mu=mu,
            Sigma=Sigma,
            w=float(mixture_params["w"]),
            inflation_k=float(mixture_params["inflation_k"]),
            mean_shift=np.asarray(mixture_params.get("mean_shift", np.zeros(2))),
        )
    else:
        raise ValueError("target must be 'single' or 'mixture'")

    log_q = log_gaussian_pdf(samples, mu_proposal, Sigma)
    logw = log_p - log_q
    elapsed = time.perf_counter() - start
    return indicators, logw, elapsed


def prefix_is_estimates(
    indicators: np.ndarray,
    log_weights: np.ndarray,
    n_list: list[int],
    elapsed_core: float,
    z: float = 1.96,
) -> list[dict[str, float]]:
    """Prefix weighted estimates with approximate normal CI and ESS."""
    indicators = np.asarray(indicators, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    n_max = int(indicators.size)

    if log_weights.size != n_max:
        raise ValueError("indicators and log_weights must have same length")

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

        w_norm = weights[:n] / sum_w
        var_hat = float(np.sum((w_norm**2) * (indicators[:n] - pc_hat) ** 2))
        half = z * np.sqrt(var_hat)
        ci_low = max(0.0, pc_hat - half)
        ci_high = min(1.0, pc_hat + half)

        rows.append(
            {
                "n": float(n),
                "pc_hat": float(pc_hat),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_width": float(ci_high - ci_low),
                "ess": float(ess),
                "hits": float(cum_hits[n - 1]),
                "elapsed_s": float(per_sample_time * n),
            }
        )

    return rows


@dataclass(frozen=True)
class _PlaneGaussianModel:
    mean: np.ndarray
    cov: np.ndarray
    chol: np.ndarray
    log_norm_const: float

    @classmethod
    def from_mean_cov(cls, mean: np.ndarray, cov: np.ndarray) -> "_PlaneGaussianModel":
        mean, cov = _validate_mu_sigma(mean, cov)
        cov_reg = cov + 1e-12 * np.eye(2)
        chol = np.linalg.cholesky(cov_reg)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        log_norm_const = -0.5 * (2 * np.log(2.0 * np.pi) + log_det)
        return cls(mean=mean, cov=cov_reg, chol=chol, log_norm_const=log_norm_const)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def logpdf(self, x: np.ndarray) -> float:
        diff = np.asarray(x, dtype=float) - self.mean
        sol = np.linalg.solve(self.chol, diff)
        quad = float(np.dot(sol, sol))
        return float(self.log_norm_const - 0.5 * quad)


@dataclass(frozen=True)
class _PlaneMixtureModel:
    base: _PlaneGaussianModel
    wide: _PlaneGaussianModel
    w: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.random(n) < self.w
        out = np.empty((n, 2), dtype=float)
        n_wide = int(np.sum(labels))
        n_base = n - n_wide
        if n_base > 0:
            out[~labels] = self.base.sample(n_base, rng)
        if n_wide > 0:
            out[labels] = self.wide.sample(n_wide, rng)
        return out

    def logpdf(self, x: np.ndarray) -> float:
        log_a = self.base.logpdf(x)
        log_b = self.wide.logpdf(x)
        term_a = np.log1p(-self.w) + log_a
        term_b = log(self.w) + log_b
        m = max(term_a, term_b)
        return float(m + np.log(np.exp(term_a - m) + np.exp(term_b - m)))


def _build_target_model(
    target: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    mixture_params: dict[str, float | np.ndarray] | None,
) -> _PlaneGaussianModel | _PlaneMixtureModel:
    base = _PlaneGaussianModel.from_mean_cov(mu, Sigma)
    if target == "single":
        return base
    if target != "mixture":
        raise ValueError("target must be 'single' or 'mixture'")
    if mixture_params is None:
        raise ValueError("mixture_params required for target='mixture'")

    w = float(mixture_params["w"])
    inflation_k = float(mixture_params["inflation_k"])
    mean_shift = np.asarray(mixture_params.get("mean_shift", np.zeros(2)), dtype=float)
    if mean_shift.shape != (2,):
        raise ValueError("mean_shift must be shape (2,)")
    if not (0.0 < w < 1.0):
        raise ValueError("w must be in (0, 1) for mixture target")
    if inflation_k <= 0:
        raise ValueError("inflation_k must be positive")

    wide = _PlaneGaussianModel.from_mean_cov(mu + mean_shift, inflation_k * Sigma)
    return _PlaneMixtureModel(base=base, wide=wide, w=w)


def estimate_pc_splitting(
    n: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    d0: float,
    p0: float = 0.1,
    n_levels_max: int = 20,
    proposal_scale: float = 0.8,
    n_mcmc_steps: int = 2,
    seed: int = 12345,
    target: str = "single",
    mixture_params: dict[str, float | np.ndarray] | None = None,
) -> dict[str, float | int | list[float] | dict[str, object] | bool]:
    """Estimate Pc by subset simulation / splitting in encounter-plane coordinates."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0, 1)")
    if n_levels_max <= 0:
        raise ValueError("n_levels_max must be positive")
    if proposal_scale <= 0:
        raise ValueError("proposal_scale must be positive")
    if n_mcmc_steps <= 0:
        raise ValueError("n_mcmc_steps must be positive")

    model = _build_target_model(target, mu, Sigma, mixture_params)
    rng = np.random.default_rng(seed)

    n_elite = max(1, int(round(p0 * n)))
    proposal_std = proposal_scale * np.sqrt(
        np.maximum(np.diag(np.asarray(Sigma)), 1e-12)
    )

    start = time.perf_counter()

    samples = model.sample(n, rng)
    scores = miss_distance(samples)
    logp = np.array([model.logpdf(x) for x in samples], dtype=float)

    eval_count = int(n)
    levels: list[float] = []
    hits_per_level: list[int] = []
    cond_probs: list[float] = []
    acceptance_rates: list[float] = []
    prob_product = 1.0
    terminated_by_max_levels = False

    for _ in range(n_levels_max):
        hits_final = int(np.sum(scores < d0))
        hits_per_level.append(hits_final)

        elite_idx = np.argpartition(scores, n_elite - 1)[:n_elite]
        level_threshold = float(np.max(scores[elite_idx]))
        levels.append(level_threshold)

        if level_threshold <= d0:
            q_final = hits_final / n
            pc_hat = prob_product * q_final
            break

        cond_prob = n_elite / n
        cond_probs.append(cond_prob)
        prob_product *= cond_prob

        seeds = samples[elite_idx]
        seed_scores = scores[elite_idx]
        seed_logp = logp[elite_idx]

        chain_len = int(ceil(n / n_elite))
        next_samples = np.empty_like(samples)
        next_scores = np.empty_like(scores)
        next_logp = np.empty_like(logp)

        idx = 0
        accepts = 0
        proposals = 0

        for s_idx in range(n_elite):
            x = seeds[s_idx].copy()
            score_x = float(seed_scores[s_idx])
            logp_x = float(seed_logp[s_idx])

            for _ in range(chain_len):
                for _ in range(n_mcmc_steps):
                    proposals += 1
                    prop = x + proposal_std * rng.normal(size=2)
                    score_prop = float(np.linalg.norm(prop))
                    eval_count += 1

                    if score_prop <= level_threshold:
                        logp_prop = float(model.logpdf(prop))
                        log_alpha = min(0.0, logp_prop - logp_x)
                        if np.log(rng.random()) < log_alpha:
                            x = prop
                            score_x = score_prop
                            logp_x = logp_prop
                            accepts += 1

                if idx < n:
                    next_samples[idx] = x
                    next_scores[idx] = score_x
                    next_logp[idx] = logp_x
                    idx += 1
                else:
                    break

            if idx >= n:
                break

        acceptance_rates.append((accepts / proposals) if proposals > 0 else 0.0)
        samples = next_samples
        scores = next_scores
        logp = next_logp

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
