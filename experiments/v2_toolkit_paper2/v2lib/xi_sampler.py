"""Parameterised execution realization factor ξ sampling.

Supports the dimensions required for the P0-1 robustness sweep:
  - Distribution family: beta, truncnorm, logitnorm
  - Mean and std parameterisation (translated to distribution-specific params)
  - Persistence structure: none (per-event), half (50% user / 50% event), full (per-user)

Design rationale:
  ξ has mean ~0.67 (not 1.0), so it is a systematic execution derating factor,
  not centred "noise". The name "execution realization factor" reflects this.

  Per-event independence is the most conservative (pessimistic) assumption.
  Real resources have user-inherent reliability that creates within-user
  correlation across events. The persistence parameter lets us test whether
  conclusions are fragile to this assumption.
"""
from __future__ import annotations

import numpy as np


def _beta_params(mean: float, std: float) -> tuple[float, float]:
    """Solve Beta(a, b) from target mean and std."""
    var = std * std
    common = mean * (1 - mean) / var - 1
    a = mean * common
    b = (1 - mean) * common
    if a <= 0 or b <= 0:
        # Fallback for extreme parameter combinations
        a = max(a, 0.5)
        b = max(b, 0.5)
    return a, b


def _sample_beta(rng: np.random.Generator, n: int, mean: float, std: float) -> np.ndarray:
    a, b = _beta_params(mean, std)
    return rng.beta(a, b, size=n)


def _sample_truncnorm(rng: np.random.Generator, n: int, mean: float, std: float) -> np.ndarray:
    """Truncated normal on [0.01, 1.0]."""
    from scipy.stats import truncnorm
    lower = (0.01 - mean) / std
    upper = (1.0 - mean) / std
    return truncnorm.rvs(lower, upper, loc=mean, scale=std, size=n, random_state=rng)


def _sample_logitnorm(rng: np.random.Generator, n: int, mean: float, std: float) -> np.ndarray:
    """Logit-normal: logit(ξ) ~ N(μ, σ²).

    We solve μ and σ so that E[ξ] ≈ target_mean and SD[ξ] ≈ target_std
    via moment matching on the logit scale.
    """
    from scipy.special import expit, logit
    from scipy.optimize import brentq

    target_mean = mean
    target_var = std ** 2

    # For logit-normal: E[ξ] = E[expit(μ + σZ)]
    # No closed form; use numerical optimisation.
    def mean_error(sigma):
        # Given sigma, find mu such that E[expit(mu + sigma*Z)] = target_mean
        # Use Gauss-Hermite quadrature for E[expit(mu + sigma*Z)]
        nodes, weights = np.polynomial.hermite.hermgauss(20)
        nodes = nodes * np.sqrt(2)

        def pred_mean(mu):
            vals = expit(mu + sigma * nodes)
            return np.dot(vals, weights) / np.sqrt(np.pi)

        # Solve for mu
        try:
            mu = brentq(lambda m: pred_mean(m) - target_mean, -10, 10)
        except ValueError:
            mu = logit(target_mean)

        # Now compute variance
        vals_sq = expit(mu + sigma * nodes) ** 2
        pred_var = np.dot(vals_sq, weights) / np.sqrt(np.pi) - target_mean ** 2
        return pred_var - target_var, mu

    # Search for sigma that matches variance
    try:
        from scipy.optimize import brentq as _brentq
        sigma = _brentq(lambda s: mean_error(s)[0], 0.01, 5.0)
        _, mu = mean_error(sigma)
    except (ValueError, RuntimeError):
        # Fallback: simple parameterisation
        mu = logit(target_mean)
        sigma = std

    return expit(rng.normal(mu, sigma, size=n))


_DIST_FUNCS = {
    "beta": _sample_beta,
    "truncnorm": _sample_truncnorm,
    "logitnorm": _sample_logitnorm,
}


def sample_xi(
    rng: np.random.Generator,
    n: int,
    user_ids: np.ndarray | None = None,
    mean: float = 0.67,
    std: float = 0.15,
    dist: str = "beta",
    persistence: str = "none",
) -> np.ndarray:
    """Draw execution realization factor ξ.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator (seeded externally for reproducibility).
    n : int
        Number of rows (user-task-event instances).
    user_ids : np.ndarray, optional
        User ID per row. Required when persistence != "none".
    mean : float
        Target E[ξ]. Default 0.67 (calibrated to PJM Summer 2025 delivery ratio).
    std : float
        Target SD[ξ]. Default 0.15.
    dist : str
        Distribution family: "beta", "truncnorm", or "logitnorm".
    persistence : str
        "none": fully per-event (current default, most pessimistic).
        "half": 50% user-inherent + 50% per-event.
        "full": fully user-inherent (same ξ for all events of same user).

    Returns
    -------
    np.ndarray of shape (n,) with values in (0, 1].
    """
    if dist not in _DIST_FUNCS:
        raise ValueError(f"Unknown dist '{dist}'. Choose from {list(_DIST_FUNCS)}")
    sampler = _DIST_FUNCS[dist]

    if persistence == "none":
        return sampler(rng, n, mean, std)

    if user_ids is None:
        raise ValueError("user_ids required for persistence != 'none'")

    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    n_users = len(unique_users)

    if persistence == "full":
        # One ξ per user, broadcast to all rows
        xi_user = sampler(rng, n_users, mean, std)
        return xi_user[inverse]

    if persistence == "half":
        # Decompose variance: var_total = var_user + var_event
        # With 50/50 split: each component gets std / sqrt(2)
        std_component = std / np.sqrt(2)
        # User effect on logit scale (additive), then expit
        from scipy.special import expit
        mean_logit = float(np.log(mean / (1 - mean)))
        u_user = rng.normal(0, std_component, size=n_users)
        u_event = rng.normal(0, std_component, size=n)
        # Combine: logit(ξ) = logit(mean) + u_user[i] + u_event[t]
        # But this gives a different mean than target due to expit nonlinearity.
        # For robustness sweep, this approximation is acceptable — the point
        # is to test persistence structure, not exact moment matching.
        xi = expit(mean_logit + u_user[inverse] + u_event)
        return xi

    raise ValueError(f"Unknown persistence '{persistence}'. Use 'none', 'half', or 'full'.")
