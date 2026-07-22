"""Rust backend: plausible."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
)
from mirt.constants import PROB_EPSILON


def generate_plausible_values_posterior(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_plausible: int = 5,
    jitter_sd: float = 0.3,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate plausible values using posterior sampling."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.generate_plausible_values_posterior(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            n_plausible,
            jitter_sd,
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = responses.shape[0]
    n_quad = len(quad_points)

    pvs = np.zeros((n_persons, n_plausible))
    log_weights = np.log(quad_weights + 1e-300)

    for i in range(n_persons):
        log_likes = np.zeros(n_quad)
        for q in range(n_quad):
            ll = 0.0
            theta = quad_points[q]
            for j in range(responses.shape[1]):
                if responses[i, j] >= 0:
                    z = discrimination[j] * (theta - difficulty[j])
                    p = sigmoid(z)
                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                    if responses[i, j] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
            log_likes[q] = ll

        log_posterior = log_likes + log_weights
        log_posterior = log_posterior - np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()

        for p in range(n_plausible):
            idx = rng.choice(n_quad, p=posterior)
            pvs[i, p] = quad_points[idx] + rng.normal(0, jitter_sd)

    return pvs


def generate_plausible_values_mcmc(
    responses: NDArray[np.int_],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    n_plausible: int = 5,
    n_iter: int = 500,
    proposal_sd: float = 0.5,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate plausible values using MCMC."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.generate_plausible_values_mcmc(
            responses.astype(np.int32),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            n_plausible,
            n_iter,
            proposal_sd,
            int(seed),
        )

    from scipy import stats

    rng = np.random.default_rng(seed)
    n_persons = responses.shape[0]
    pvs = np.zeros((n_persons, n_plausible))

    def log_likelihood(resp: NDArray[np.int_], theta: float) -> float:
        ll = 0.0
        for j in range(len(resp)):
            if resp[j] >= 0:
                z = discrimination[j] * (theta - difficulty[j])
                p = sigmoid(z)
                p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                if resp[j] == 1:
                    ll += np.log(p)
                else:
                    ll += np.log(1 - p)
        return ll

    for i in range(n_persons):
        resp = responses[i]
        theta = 0.0

        for p in range(n_plausible):
            for _ in range(n_iter):
                proposal = theta + rng.normal(0, proposal_sd)

                ll_current = log_likelihood(resp, theta)
                ll_proposal = log_likelihood(resp, proposal)

                prior_current = stats.norm.logpdf(theta)
                prior_proposal = stats.norm.logpdf(proposal)

                log_alpha = (ll_proposal + prior_proposal) - (
                    ll_current + prior_current
                )

                if np.log(rng.random()) < log_alpha:
                    theta = proposal

            pvs[i, p] = theta

    return pvs


def compute_observed_margins(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute observed univariate and bivariate margins."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_observed_margins(responses.astype(np.int32))

    n_persons, n_items = responses.shape

    obs_uni = np.zeros(n_items)
    for j in range(n_items):
        valid = responses[:, j] >= 0
        if valid.any():
            obs_uni[j] = responses[valid, j].mean()

    obs_bi = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            if valid.any():
                obs_bi[i, j] = (responses[valid, i] * responses[valid, j]).mean()
                obs_bi[j, i] = obs_bi[i, j]

    return obs_uni, obs_bi


def compute_expected_margins(
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected margins under the model."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_margins(
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_items = len(discrimination)
    n_quad = len(quad_points)

    probs = np.zeros((n_items, n_quad))
    for j in range(n_items):
        z = discrimination[j] * (quad_points - difficulty[j])
        probs[j] = sigmoid(z)

    exp_uni = np.sum(probs * quad_weights, axis=1)

    exp_bi = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            exp_bi[i, j] = np.sum(probs[i] * probs[j] * quad_weights)
            exp_bi[j, i] = exp_bi[i, j]

    return exp_uni, exp_bi


def generate_bootstrap_indices(
    n_persons: int,
    n_bootstrap: int,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Generate bootstrap sample indices."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.generate_bootstrap_indices(n_persons, n_bootstrap, int(seed))

    rng = np.random.default_rng(seed)
    return rng.integers(0, n_persons, size=(n_bootstrap, n_persons))


def resample_responses(
    responses: NDArray[np.int_],
    indices: NDArray[np.int64],
) -> NDArray[np.int_]:
    """Resample responses matrix."""
    if RUST_AVAILABLE:
        return mirt_rs.resample_responses(
            responses.astype(np.int32),
            indices.astype(np.int64),
        )

    return responses[indices]


def impute_from_probabilities(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    missing_code: int = -1,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Impute missing responses using model probabilities."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.impute_from_probabilities(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            missing_code,
            int(seed),
        )

    rng = np.random.default_rng(seed)
    imputed = responses.copy()
    n_persons, n_items = responses.shape

    for i in range(n_persons):
        for j in range(n_items):
            if responses[i, j] == missing_code:
                z = discrimination[j] * (theta[i] - difficulty[j])
                p = sigmoid(z)
                imputed[i, j] = 1 if rng.random() < p else 0

    return imputed


def multiple_imputation(
    responses: NDArray[np.int_],
    theta_mean: NDArray[np.float64],
    theta_se: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    missing_code: int = -1,
    n_imputations: int = 5,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Multiple imputation in parallel."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.multiple_imputation(
            responses.astype(np.int32),
            theta_mean.astype(np.float64).ravel(),
            theta_se.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            missing_code,
            n_imputations,
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons, n_items = responses.shape
    imputations = np.zeros((n_imputations, n_persons, n_items), dtype=np.int_)

    for m in range(n_imputations):
        theta_draw = theta_mean + rng.standard_normal(n_persons) * theta_se
        imputations[m] = impute_from_probabilities(
            responses, theta_draw, discrimination, difficulty, missing_code, seed + m
        )

    return imputations
