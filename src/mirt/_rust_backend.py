"""Python interface to the Rust backend for MIRT.

This module provides a clean interface to the high-performance Rust
implementations. It automatically falls back to pure Python implementations
if the Rust extension is not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

try:
    from mirt import mirt_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    mirt_rs = None

if TYPE_CHECKING:
    pass


def is_rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE


def compute_log_likelihoods_2pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for 2PL model at all quadrature points.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Log-likelihoods (n_persons, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_2pl(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        z = discrimination * (theta - difficulty)
        probs = 1.0 / (1.0 + np.exp(-z))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                if responses[i, j] >= 0:
                    if responses[i, j] == 1:
                        ll += np.log(probs[j])
                    else:
                        ll += np.log(1 - probs[j])
            log_likes[i, q] = ll

    return log_likes


def compute_log_likelihoods_3pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for 3PL model at all quadrature points."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_3pl(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            guessing.astype(np.float64),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        z = discrimination * (theta - difficulty)
        p_star = 1.0 / (1.0 + np.exp(-z))
        probs = guessing + (1 - guessing) * p_star
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                if responses[i, j] >= 0:
                    if responses[i, j] == 1:
                        ll += np.log(probs[j])
                    else:
                        ll += np.log(1 - probs[j])
            log_likes[i, q] = ll

    return log_likes


def compute_log_likelihoods_mirt(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for multidimensional IRT model."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_mirt(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons = responses.shape[0]
    n_quad = quad_points.shape[0]

    disc_sums = discrimination.sum(axis=1)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta_q = quad_points[q]
        z = np.dot(discrimination, theta_q) - disc_sums * difficulty

        for i in range(n_persons):
            ll = 0.0
            for j in range(responses.shape[1]):
                if responses[i, j] >= 0:
                    p = 1.0 / (1.0 + np.exp(-z[j]))
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    if responses[i, j] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
            log_likes[i, q] = ll

    return log_likes


def e_step_complete(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Complete E-step computation with posterior weights.

    Returns
    -------
    tuple
        (posterior_weights, marginal_likelihood)
    """
    if RUST_AVAILABLE:
        return mirt_rs.e_step_complete(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            float(prior_mean),
            float(prior_var),
        )

    from mirt.utils.numeric import logsumexp

    log_likes = compute_log_likelihoods_2pl(
        responses, quad_points, discrimination, difficulty
    )

    log_prior = (
        -0.5 * np.log(2 * np.pi * prior_var)
        - 0.5 * ((quad_points - prior_mean) ** 2) / prior_var
    )

    log_joint = log_likes + log_prior[None, :] + np.log(quad_weights + 1e-300)[None, :]
    log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
    log_posterior = log_joint - log_marginal

    posterior_weights = np.exp(log_posterior)
    marginal_ll = np.exp(log_marginal.ravel())

    return posterior_weights, marginal_ll


def compute_expected_counts(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected counts for dichotomous items."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_counts(
            responses.astype(np.int32).ravel(),
            posterior_weights.astype(np.float64),
        )

    n_persons = len(responses)
    n_quad = posterior_weights.shape[1]
    valid_mask = responses >= 0

    r_k = np.zeros(n_quad)
    n_k = np.zeros(n_quad)

    for i in range(n_persons):
        if valid_mask[i]:
            n_k += posterior_weights[i]
            if responses[i] == 1:
                r_k += posterior_weights[i]

    return r_k, n_k


def compute_expected_counts_polytomous(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    n_categories: int,
) -> NDArray[np.float64]:
    """Compute expected counts per category for polytomous items."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_counts_polytomous(
            responses.astype(np.int32).ravel(),
            posterior_weights.astype(np.float64),
            n_categories,
        )

    n_quad = posterior_weights.shape[1]
    r_kc = np.zeros((n_quad, n_categories))

    for i, resp in enumerate(responses):
        if 0 <= resp < n_categories:
            r_kc[:, resp] += posterior_weights[i]

    return r_kc


def sibtest_compute_beta(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute SIBTEST beta statistic."""
    if RUST_AVAILABLE:
        return mirt_rs.sibtest_compute_beta(
            ref_data.astype(np.int32),
            focal_data.astype(np.int32),
            ref_scores.astype(np.int32),
            focal_scores.astype(np.int32),
            suspect_items.astype(np.int32),
        )

    all_scores = np.concatenate([ref_scores, focal_scores])
    unique_scores = np.unique(all_scores)

    beta_k = []
    n_k = []

    for k in unique_scores:
        ref_at_k = ref_data[ref_scores == k]
        focal_at_k = focal_data[focal_scores == k]

        n_ref_k = len(ref_at_k)
        n_focal_k = len(focal_at_k)

        if n_ref_k > 0 and n_focal_k > 0:
            mean_ref_k = ref_at_k[:, suspect_items].sum(axis=1).mean()
            mean_focal_k = focal_at_k[:, suspect_items].sum(axis=1).mean()
            beta_k.append(mean_ref_k - mean_focal_k)
            n_k.append(2 * n_ref_k * n_focal_k / (n_ref_k + n_focal_k))

    if not beta_k:
        return np.nan, np.nan, np.array([]), np.array([])

    beta_k = np.array(beta_k)
    n_k = np.array(n_k)
    beta = np.sum(n_k * beta_k) / np.sum(n_k)

    weighted_var = np.sum(n_k * (beta_k - beta) ** 2) / np.sum(n_k)
    n_total = len(ref_scores) + len(focal_scores)
    se = np.sqrt(weighted_var / n_total)

    return beta, se, beta_k, n_k


def sibtest_all_items(
    data: NDArray[np.int_],
    groups: NDArray[np.int_],
    anchor_items: NDArray[np.int_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run SIBTEST for all items in parallel."""
    if RUST_AVAILABLE:
        return mirt_rs.sibtest_all_items(
            data.astype(np.int32),
            groups.astype(np.int32),
            anchor_items.astype(np.int32) if anchor_items is not None else None,
        )

    from scipy import stats

    n_items = data.shape[1]
    unique_groups = np.unique(groups)
    ref_group, focal_group = unique_groups[0], unique_groups[1]

    ref_mask = groups == ref_group
    focal_mask = groups == focal_group

    betas = np.zeros(n_items)
    zs = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for i in range(n_items):
        if anchor_items is None:
            matching = [j for j in range(n_items) if j != i]
        else:
            matching = [j for j in anchor_items if j != i]

        if not matching:
            betas[i] = np.nan
            zs[i] = np.nan
            p_values[i] = np.nan
            continue

        ref_scores = data[ref_mask][:, matching].sum(axis=1)
        focal_scores = data[focal_mask][:, matching].sum(axis=1)

        beta, se, _, _ = sibtest_compute_beta(
            data[ref_mask],
            data[focal_mask],
            ref_scores,
            focal_scores,
            np.array([i]),
        )

        betas[i] = beta
        if se > 1e-10:
            zs[i] = beta / se
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(zs[i])))
        else:
            zs[i] = np.nan
            p_values[i] = np.nan

    return betas, zs, p_values


def simulate_grm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Graded Response Model."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_grm(
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = theta.shape[0]
    n_items = len(discrimination)
    n_categories = thresholds.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        cum_probs = np.ones((n_persons, n_categories))
        for k in range(n_categories - 1):
            z = discrimination[i] * (theta[:, 0] - thresholds[i, k])
            cum_probs[:, k + 1] = 1.0 / (1.0 + np.exp(-z))

        cat_probs = np.diff(
            np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1
        )
        cat_probs = np.maximum(cat_probs, 0)
        cat_probs = cat_probs / cat_probs.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses


def simulate_gpcm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Generalized Partial Credit Model."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_gpcm(
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = theta.shape[0]
    n_items = len(discrimination)
    n_categories = thresholds.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        numerators = np.zeros((n_persons, n_categories))
        for k in range(n_categories):
            cumsum = 0.0
            for v in range(k):
                cumsum += discrimination[i] * (theta[:, 0] - thresholds[i, v])
            numerators[:, k] = np.exp(cumsum)

        cat_probs = numerators / numerators.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses


def simulate_dichotomous(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64] | None = None,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate dichotomous responses (2PL/3PL)."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_dichotomous(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            guessing.astype(np.float64) if guessing is not None else None,
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = len(theta)
    n_items = len(discrimination)

    if guessing is None:
        guessing = np.zeros(n_items)

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    p_star = 1.0 / (1.0 + np.exp(-z))
    probs = guessing[None, :] + (1 - guessing[None, :]) * p_star

    u = rng.random((n_persons, n_items))
    return (u < probs).astype(np.int_)


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
                    p = 1.0 / (1.0 + np.exp(-z))
                    p = np.clip(p, 1e-10, 1 - 1e-10)
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
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-10, 1 - 1e-10)
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
        probs[j] = 1.0 / (1.0 + np.exp(-z))

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
                p = 1.0 / (1.0 + np.exp(-z))
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


def compute_eap_scores(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute EAP scores with standard errors."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_eap_scores(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons = responses.shape[0]
    n_quad = len(quad_points)

    log_weights = np.log(quad_weights + 1e-300)
    theta = np.zeros(n_persons)
    se = np.zeros(n_persons)

    for i in range(n_persons):
        log_likes = np.zeros(n_quad)
        for q in range(n_quad):
            ll = 0.0
            t = quad_points[q]
            for j in range(responses.shape[1]):
                if responses[i, j] >= 0:
                    z = discrimination[j] * (t - difficulty[j])
                    p = 1.0 / (1.0 + np.exp(-z))
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    if responses[i, j] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
            log_likes[q] = ll

        log_posterior = log_likes + log_weights
        log_posterior = log_posterior - np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()

        theta[i] = np.sum(posterior * quad_points)
        se[i] = np.sqrt(np.sum(posterior * (quad_points - theta[i]) ** 2))

    return theta, se


def em_fit_2pl(
    responses: NDArray[np.int_],
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int, bool]:
    """Fit 2PL model using EM algorithm in Rust.

    Returns
    -------
    tuple
        (discrimination, difficulty, log_likelihood, n_iterations, converged)
    """
    if RUST_AVAILABLE:
        return mirt_rs.em_fit_2pl(
            responses.astype(np.int32),
            n_quadpts,
            max_iter,
            tol,
        )

    raise RuntimeError("Rust backend required for em_fit_2pl")


def gibbs_sample_2pl(
    responses: NDArray[np.int_],
    n_iter: int = 5000,
    burnin: int = 1000,
    thin: int = 1,
    seed: int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Run Gibbs sampler for 2PL model in Rust.

    Returns
    -------
    tuple
        (disc_chain, diff_chain, theta_chain, ll_chain)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.gibbs_sample_2pl(
            responses.astype(np.int32),
            n_iter,
            burnin,
            thin,
            int(seed),
        )

    raise RuntimeError("Rust backend required for gibbs_sample_2pl")


def mhrm_fit_2pl(
    responses: NDArray[np.int_],
    n_cycles: int = 2000,
    burnin: int = 500,
    proposal_sd: float = 0.5,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit 2PL model using MHRM algorithm in Rust.

    Returns
    -------
    tuple
        (discrimination, difficulty, log_likelihood)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.mhrm_fit_2pl(
            responses.astype(np.int32),
            n_cycles,
            burnin,
            proposal_sd,
            int(seed),
        )

    raise RuntimeError("Rust backend required for mhrm_fit_2pl")


def bootstrap_fit_2pl(
    responses: NDArray[np.int_],
    n_bootstrap: int = 100,
    n_quadpts: int = 21,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run parallel bootstrap for 2PL model in Rust.

    Returns
    -------
    tuple
        (disc_samples, diff_samples) - arrays of shape (n_bootstrap, n_items)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.bootstrap_fit_2pl(
            responses.astype(np.int32),
            n_bootstrap,
            n_quadpts,
            max_iter,
            tol,
            int(seed),
        )

    raise RuntimeError("Rust backend required for bootstrap_fit_2pl")


def lord_wingersky_recursion(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute sum score distribution using Lord-Wingersky recursion in Rust.

    For dichotomous items, computes log P(sum_score = s | theta) for all s and theta.

    Parameters
    ----------
    theta : ndarray
        Quadrature points, shape (n_quad,)
    discrimination : ndarray
        Item discriminations, shape (n_items,)
    difficulty : ndarray
        Item difficulties, shape (n_items,)

    Returns
    -------
    ndarray or None
        Log probability distribution, shape (max_score + 1, n_quad).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.lord_wingersky_recursion(
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    return None


def lord_wingersky_polytomous(
    item_probs: NDArray[np.float64],
    max_score: int,
) -> NDArray[np.float64] | None:
    """Compute sum score distribution for polytomous items using Rust.

    Parameters
    ----------
    item_probs : ndarray
        Item probabilities, shape (n_items, n_quad, max_categories)
    max_score : int
        Maximum possible sum score

    Returns
    -------
    ndarray or None
        Log probability distribution, shape (max_score + 1, n_quad).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.lord_wingersky_polytomous(
            item_probs.astype(np.float64),
            max_score,
        )

    return None


def cat_compute_item_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute Fisher information for all items at a given theta.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters, shape (n_items,).
    difficulty : ndarray
        Item difficulty parameters, shape (n_items,).

    Returns
    -------
    ndarray or None
        Fisher information for each item, shape (n_items,).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.cat_compute_item_info(
            float(theta),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    z = discrimination * (theta - difficulty)
    p = 1.0 / (1.0 + np.exp(-z))
    q = 1.0 - p
    return (discrimination**2) * p * q


def cat_select_max_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    available_mask: NDArray[np.bool_],
) -> int:
    """Select item with maximum Fisher information.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    available_mask : ndarray
        Boolean mask of available items.

    Returns
    -------
    int
        Index of selected item, or -1 if no items available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.cat_select_max_info(
            float(theta),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            available_mask.astype(np.bool_),
        )

    info = cat_compute_item_info(theta, discrimination, difficulty)
    info = np.where(available_mask, info, -np.inf)
    return int(np.argmax(info))


def cat_eap_update(
    administered_items: NDArray[np.int32],
    responses: NDArray[np.int32],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Incremental EAP update after responses.

    Parameters
    ----------
    administered_items : ndarray
        Indices of administered items.
    responses : ndarray
        Responses to administered items.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points.
    quad_weights : ndarray
        Quadrature weights.

    Returns
    -------
    tuple[float, float]
        (theta_estimate, standard_error).
    """
    if RUST_AVAILABLE:
        theta, se = mirt_rs.cat_eap_update(
            administered_items.astype(np.int32),
            responses.astype(np.int32),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
        )
        return float(theta[0]), float(se[0])

    n_quad = len(quad_points)

    log_likes = np.zeros(n_quad)
    for q in range(n_quad):
        theta_q = quad_points[q]
        ll = 0.0
        for i, item_idx in enumerate(administered_items):
            j = int(item_idx)
            r = responses[i]
            if r >= 0:
                z = discrimination[j] * (theta_q - difficulty[j])
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                if r == 1:
                    ll += np.log(p)
                else:
                    ll += np.log(1 - p)
        log_likes[q] = ll

    log_prior = -0.5 * quad_points**2 - 0.5 * np.log(2 * np.pi)
    log_posterior = log_likes + log_prior + np.log(quad_weights + 1e-300)

    log_norm = np.max(log_posterior) + np.log(
        np.sum(np.exp(log_posterior - np.max(log_posterior)))
    )
    posterior = np.exp(log_posterior - log_norm)

    theta_eap = np.sum(posterior * quad_points)
    variance = np.sum(posterior * (quad_points - theta_eap) ** 2)
    se = np.sqrt(variance)

    return float(theta_eap), float(se)


def cat_simulate_batch(
    true_thetas: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    se_threshold: float,
    max_items: int,
    min_items: int,
    n_replications: int,
    seed: int | None = None,
) -> (
    tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]
    ]
    | None
):
    """Run batch CAT simulations in parallel using Rust.

    Parameters
    ----------
    true_thetas : ndarray
        True ability values to simulate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points for EAP scoring.
    quad_weights : ndarray
        Quadrature weights.
    se_threshold : float
        SE stopping threshold.
    max_items : int
        Maximum items per test.
    min_items : int
        Minimum items before stopping.
    n_replications : int
        Number of replications per theta.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple or None
        (theta_estimates, se_estimates, n_items, true_thetas_expanded).
        Returns None if Rust backend not available.
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.cat_simulate_batch(
            true_thetas.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            float(se_threshold),
            int(max_items),
            int(min_items),
            int(n_replications),
            int(seed),
        )

    return None


def cat_conditional_mse(
    eval_thetas: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    se_threshold: float,
    max_items: int,
    min_items: int,
    n_replications: int,
    seed: int | None = None,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None
):
    """Compute conditional MSE at specified theta values using Rust.

    Parameters
    ----------
    eval_thetas : ndarray
        Theta values to evaluate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points for EAP scoring.
    quad_weights : ndarray
        Quadrature weights.
    se_threshold : float
        SE stopping threshold.
    max_items : int
        Maximum items per test.
    min_items : int
        Minimum items before stopping.
    n_replications : int
        Number of replications per theta.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple or None
        (thetas, biases, mses, avg_items).
        Returns None if Rust backend not available.
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.cat_conditional_mse(
            eval_thetas.astype(np.float64),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            float(se_threshold),
            int(max_items),
            int(min_items),
            int(n_replications),
            int(seed),
        )

    return None


def compute_standardized_residuals(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute standardized residuals for each person-item combination.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Standardized residuals (n_persons, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_standardized_residuals(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons, n_items = responses.shape
    residuals = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        z = discrimination[j] * (theta - difficulty[j])
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        variance = p * (1 - p)

        valid = responses[:, j] >= 0
        residuals[valid, j] = (responses[valid, j] - p[valid]) / np.sqrt(
            variance[valid] + 1e-10
        )

    return residuals


def compute_q3_matrix(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Yen's Q3 (residual correlation) matrix.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Q3 correlation matrix (n_items, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_q3_matrix(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    residuals = compute_standardized_residuals(
        responses, theta, discrimination, difficulty
    )

    n_items = responses.shape[1]
    q3_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            valid &= ~np.isnan(residuals[:, i]) & ~np.isnan(residuals[:, j])

            if valid.sum() > 2:
                r_i = residuals[valid, i]
                r_j = residuals[valid, j]
                q3 = np.corrcoef(r_i, r_j)[0, 1]
                q3_matrix[i, j] = q3
                q3_matrix[j, i] = q3

    return q3_matrix


def compute_ld_chi2_matrix(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute LD chi-square statistics for all item pairs.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        LD chi-square matrix (n_items, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_ld_chi2_matrix(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons, n_items = responses.shape
    chi2_matrix = np.full((n_items, n_items), np.nan)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            n_valid = valid.sum()

            if n_valid < 10:
                continue

            resp_i = responses[valid, i]
            resp_j = responses[valid, j]
            theta_valid = theta[valid]

            z_i = discrimination[i] * (theta_valid - difficulty[i])
            z_j = discrimination[j] * (theta_valid - difficulty[j])
            prob_i = 1.0 / (1.0 + np.exp(-z_i))
            prob_j = 1.0 / (1.0 + np.exp(-z_j))

            resp_i_bin = (resp_i > 0).astype(int)
            resp_j_bin = (resp_j > 0).astype(int)

            obs_00 = np.sum((resp_i_bin == 0) & (resp_j_bin == 0))
            obs_01 = np.sum((resp_i_bin == 0) & (resp_j_bin == 1))
            obs_10 = np.sum((resp_i_bin == 1) & (resp_j_bin == 0))
            obs_11 = np.sum((resp_i_bin == 1) & (resp_j_bin == 1))

            exp_00 = np.sum((1 - prob_i) * (1 - prob_j))
            exp_01 = np.sum((1 - prob_i) * prob_j)
            exp_10 = np.sum(prob_i * (1 - prob_j))
            exp_11 = np.sum(prob_i * prob_j)

            observed = np.array([obs_00, obs_01, obs_10, obs_11])
            expected = np.array([exp_00, exp_01, exp_10, exp_11])
            expected = np.maximum(expected, 0.5)

            chi2 = np.sum((observed - expected) ** 2 / expected)
            chi2_matrix[i, j] = chi2
            chi2_matrix[j, i] = chi2

    return chi2_matrix


def m_step_dichotomous_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    max_iter: int = 10,
    tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    damping: float = 0.5,
    regularization: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parallel M-step optimization for dichotomous items.

    Uses Newton-Raphson optimization for each item in parallel using Rayon.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    posterior_weights : NDArray
        Posterior weights from E-step (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Initial discrimination parameters (n_items,)
    difficulty : NDArray
        Initial difficulty parameters (n_items,)
    max_iter : int
        Maximum Newton-Raphson iterations per item
    tol : float
        Convergence tolerance
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max)
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max)
    damping : float
        Damping factor for Newton-Raphson updates
    regularization : float
        Regularization for Hessian diagonal

    Returns
    -------
    tuple
        (new_discrimination, new_difficulty)
    """
    if RUST_AVAILABLE:
        return mirt_rs.m_step_dichotomous_parallel(
            responses.astype(np.int32),
            posterior_weights.astype(np.float64),
            quad_points.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
            max_iter,
            tol,
            disc_bounds,
            diff_bounds,
            damping,
            regularization,
        )

    from scipy.optimize import minimize

    n_items = responses.shape[1]
    new_disc = np.zeros(n_items)
    new_diff = np.zeros(n_items)

    for j in range(n_items):
        item_responses = responses[:, j]
        valid_mask = item_responses >= 0

        r_k = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )
        n_k = np.sum(posterior_weights[valid_mask], axis=0)

        def neg_ll(params):
            a, b = params
            z = a * (quad_points - b)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll = np.sum(r_k * np.log(p) + (n_k - r_k) * np.log(1 - p))
            return -ll

        result = minimize(
            neg_ll,
            x0=[discrimination[j], difficulty[j]],
            method="L-BFGS-B",
            bounds=[disc_bounds, diff_bounds],
        )
        new_disc[j], new_diff[j] = result.x

    return new_disc, new_diff


def compute_item_se_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    h: float = 1e-5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-item standard errors in parallel.

    Exploits block diagonal structure of the Hessian.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    posterior_weights : NDArray
        Posterior weights from E-step (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Discrimination parameters (n_items,)
    difficulty : NDArray
        Difficulty parameters (n_items,)
    h : float
        Step size for finite difference

    Returns
    -------
    tuple
        (se_discrimination, se_difficulty)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_item_se_parallel(
            responses.astype(np.int32),
            posterior_weights.astype(np.float64),
            quad_points.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
            h,
        )

    n_items = responses.shape[1]
    se_disc = np.zeros(n_items)
    se_diff = np.zeros(n_items)

    for j in range(n_items):
        item_responses = responses[:, j]
        valid_mask = item_responses >= 0

        r_k = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )
        n_k = np.sum(posterior_weights[valid_mask], axis=0)

        def item_ll(a, b):
            z = a * (quad_points - b)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return np.sum(r_k * np.log(p) + (n_k - r_k) * np.log(1 - p))

        a, b = discrimination[j], difficulty[j]
        ll_center = item_ll(a, b)

        ll_a_plus = item_ll(a + h, b)
        ll_a_minus = item_ll(a - h, b)
        hess_aa = (ll_a_plus - 2 * ll_center + ll_a_minus) / (h**2)
        se_disc[j] = np.sqrt(-1.0 / hess_aa) if hess_aa < -1e-10 else np.nan

        ll_b_plus = item_ll(a, b + h)
        ll_b_minus = item_ll(a, b - h)
        hess_bb = (ll_b_plus - 2 * ll_center + ll_b_minus) / (h**2)
        se_diff[j] = np.sqrt(-1.0 / hess_bb) if hess_bb < -1e-10 else np.nan

    return se_disc, se_diff


def compute_hessian_block_diagonal(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute full Hessian matrix exploiting block diagonal structure.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    posterior_weights : NDArray
        Posterior weights (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Discrimination parameters (n_items,)
    difficulty : NDArray
        Difficulty parameters (n_items,)
    h : float
        Step size for finite difference

    Returns
    -------
    NDArray
        Hessian matrix (n_params, n_params) where n_params = n_items * 2
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_hessian_block_diagonal(
            responses.astype(np.int32),
            posterior_weights.astype(np.float64),
            quad_points.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
            h,
        )

    n_items = len(discrimination)
    n_params = n_items * 2
    hessian = np.zeros((n_params, n_params))

    se_disc, se_diff = compute_item_se_parallel(
        responses, posterior_weights, quad_points, discrimination, difficulty, h
    )

    for j in range(n_items):
        idx_a = j * 2
        idx_b = j * 2 + 1

        if not np.isnan(se_disc[j]):
            hessian[idx_a, idx_a] = -1.0 / (se_disc[j] ** 2)
        if not np.isnan(se_diff[j]):
            hessian[idx_b, idx_b] = -1.0 / (se_diff[j] ** 2)

    return hessian


def compute_expected_counts_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected counts for all items in parallel.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    posterior_weights : NDArray
        Posterior weights (n_persons, n_quad)

    Returns
    -------
    tuple
        (r_k_all, n_k_all) both shape (n_items, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_counts_parallel(
            responses.astype(np.int32),
            posterior_weights.astype(np.float64),
        )

    n_items = responses.shape[1]
    n_quad = posterior_weights.shape[1]

    r_k_all = np.zeros((n_items, n_quad))
    n_k_all = np.zeros((n_items, n_quad))

    for j in range(n_items):
        item_responses = responses[:, j]
        valid_mask = item_responses >= 0

        r_k_all[j] = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )
        n_k_all[j] = np.sum(posterior_weights[valid_mask], axis=0)

    return r_k_all, n_k_all


def compute_fit_statistics(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute item and person fit statistics (infit/outfit).

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    tuple
        (item_outfit, item_infit, person_outfit, person_infit)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_fit_statistics(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons, n_items = responses.shape

    z_sq = np.full((n_persons, n_items), np.nan)
    variance = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        z = discrimination[j] * (theta - difficulty[j])
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        var = p * (1 - p)

        valid = responses[:, j] >= 0
        raw_resid = responses[valid, j] - p[valid]
        z_sq[valid, j] = (raw_resid**2) / (var[valid] + 1e-10)
        variance[valid, j] = var[valid]

    item_outfit = np.nanmean(z_sq, axis=0)
    item_infit = np.nansum(z_sq * variance, axis=0) / np.nansum(variance, axis=0)

    person_outfit = np.nanmean(z_sq, axis=1)
    person_infit = np.nansum(z_sq * variance, axis=1) / np.nansum(variance, axis=1)

    return item_outfit, item_infit, person_outfit, person_infit


def compute_probabilities_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute probabilities for all items in batch (2PL model).

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Probabilities (n_persons, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_probabilities_batch(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
        )

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    return 1.0 / (1.0 + np.exp(-z))


def compute_probabilities_batch_3pl(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute probabilities for all items in batch (3PL model).

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)
    guessing : NDArray
        Item guessing parameters (n_items,)

    Returns
    -------
    NDArray
        Probabilities (n_persons, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_probabilities_batch_3pl(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
            guessing.astype(np.float64).ravel(),
        )

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    p_star = 1.0 / (1.0 + np.exp(-z))
    return guessing[None, :] + (1 - guessing[None, :]) * p_star


def compute_expected_variance_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected values and variances for all items in batch.

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    tuple
        (expected, variance) both shape (n_persons, n_items)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_variance_batch(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
        )

    probs = compute_probabilities_batch(theta, discrimination, difficulty)
    expected = probs
    variance = probs * (1 - probs)
    return expected, variance


def eapsum_from_distribution(
    log_p_score_theta: NDArray[np.float64],
    log_prior: NDArray[np.float64],
    sum_scores: NDArray[np.int_],
    theta_points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Compute EAPsum scores from pre-computed distribution using Rust.

    Parameters
    ----------
    log_p_score_theta : ndarray
        Log P(score | theta), shape (max_score + 1, n_quad)
    log_prior : ndarray
        Log prior weights, shape (n_quad,)
    sum_scores : ndarray
        Observed sum scores, shape (n_persons,)
    theta_points : ndarray
        Quadrature points, shape (n_quad,)

    Returns
    -------
    tuple or None
        (theta_estimates, standard_errors) or None if Rust not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.eapsum_from_distribution(
            log_p_score_theta.astype(np.float64),
            log_prior.astype(np.float64),
            sum_scores.astype(np.int32),
            theta_points.astype(np.float64),
        )

    return None


def fixed_calib_em(
    responses: NDArray[np.int_],
    anchor_items: list[int],
    new_items: list[int],
    anchor_disc: NDArray[np.float64],
    anchor_diff: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    max_iter: int = 500,
    tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.2, 5.0),
    diff_bounds: tuple[float, float] = (-5.0, 5.0),
    prob_clamp: tuple[float, float] = (0.01, 0.99),
    init_disc: float = 1.0,
    init_diff: float = 0.0,
    min_count: float = 1.0,
    min_valid_points: int = 3,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int, bool
]:
    """Fixed-item calibration EM algorithm in Rust.

    Calibrates new items to an existing scale defined by anchor items
    with fixed parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    anchor_items : list[int]
        Indices of anchor items in responses
    new_items : list[int]
        Indices of new items to calibrate
    anchor_disc : NDArray
        Fixed discrimination parameters for anchor items
    anchor_diff : NDArray
        Fixed difficulty parameters for anchor items
    theta_grid : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max)
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max)
    prob_clamp : tuple[float, float]
        Bounds for probability clipping (min, max)
    init_disc : float
        Initial discrimination value
    init_diff : float
        Initial difficulty value
    min_count : float
        Minimum count threshold for valid quadrature points
    min_valid_points : int
        Minimum number of valid points for regression

    Returns
    -------
    tuple
        (new_disc, new_diff, theta, log_likelihood, n_iterations, converged)
    """
    if RUST_AVAILABLE:
        return mirt_rs.fixed_calib_em(
            responses.astype(np.int32),
            anchor_items,
            new_items,
            anchor_disc.astype(np.float64),
            anchor_diff.astype(np.float64),
            theta_grid.astype(np.float64),
            quad_weights.astype(np.float64),
            max_iter,
            tol,
            disc_bounds,
            diff_bounds,
            prob_clamp,
            init_disc,
            init_diff,
            min_count,
            min_valid_points,
        )

    raise RuntimeError("Rust backend required for fixed_calib_em")


def stocking_lord_criterion(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    a: float,
    b: float,
    theta_grid: NDArray[np.float64],
) -> float:
    """Compute Stocking-Lord equating criterion using Rust.

    Parameters
    ----------
    disc_old : NDArray
        Discrimination parameters for old form
    diff_old : NDArray
        Difficulty parameters for old form
    disc_new : NDArray
        Discrimination parameters for new form
    diff_new : NDArray
        Difficulty parameters for new form
    a : float
        Slope transformation constant
    b : float
        Intercept transformation constant
    theta_grid : NDArray
        Theta values for integration

    Returns
    -------
    float
        Criterion value (sum of squared probability differences)
    """
    if RUST_AVAILABLE:
        return mirt_rs.stocking_lord_criterion(
            disc_old.astype(np.float64),
            diff_old.astype(np.float64),
            disc_new.astype(np.float64),
            diff_new.astype(np.float64),
            float(a),
            float(b),
            theta_grid.astype(np.float64),
        )

    n_items = len(disc_old)

    total_diff = 0.0
    for j in range(n_items):
        for theta in theta_grid:
            p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta - diff_old[j])))
            theta_trans = a * theta + b
            p_new = 1.0 / (1.0 + np.exp(-disc_new[j] * (theta_trans - diff_new[j])))
            total_diff += (p_old - p_new) ** 2

    return total_diff


def compute_log_likelihoods_grm(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    n_categories: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for GRM at all quadrature points.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    thresholds : NDArray
        Threshold parameters (n_items, max_categories-1)
    n_categories : NDArray
        Number of categories per item (n_items,)

    Returns
    -------
    NDArray
        Log-likelihoods (n_persons, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_grm(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            n_categories.astype(np.int32),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))
    eps = 1e-10

    for q in range(n_quad):
        theta = quad_points[q]
        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                resp = responses[i, j]
                if resp < 0:
                    continue
                n_cat = n_categories[j]
                if resp == 0:
                    z = discrimination[j] * (theta - thresholds[j, 0])
                    p_above = 1.0 / (1.0 + np.exp(-z))
                    prob = max(1.0 - p_above, eps)
                elif resp == n_cat - 1:
                    z = discrimination[j] * (theta - thresholds[j, resp - 1])
                    prob = max(1.0 / (1.0 + np.exp(-z)), eps)
                else:
                    z_upper = discrimination[j] * (theta - thresholds[j, resp - 1])
                    z_lower = discrimination[j] * (theta - thresholds[j, resp])
                    p_upper = 1.0 / (1.0 + np.exp(-z_upper))
                    p_lower = 1.0 / (1.0 + np.exp(-z_lower))
                    prob = max(p_upper - p_lower, eps)
                ll += np.log(prob)
            log_likes[i, q] = ll

    return log_likes


def compute_log_likelihoods_gpcm(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    steps: NDArray[np.float64],
    n_categories: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for GPCM at all quadrature points.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    steps : NDArray
        Step parameters (n_items, max_categories)
    n_categories : NDArray
        Number of categories per item (n_items,)

    Returns
    -------
    NDArray
        Log-likelihoods (n_persons, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_gpcm(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            steps.astype(np.float64),
            n_categories.astype(np.int32),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                resp = responses[i, j]
                if resp < 0:
                    continue
                n_cat = n_categories[j]
                a = discrimination[j]
                numerators = np.zeros(n_cat)
                for k in range(1, n_cat):
                    numerators[k] = numerators[k - 1] + a * (theta - steps[j, k])
                max_num = np.max(numerators)
                sum_exp = np.sum(np.exp(numerators - max_num))
                log_denom = max_num + np.log(sum_exp)
                prob = np.exp(numerators[resp] - log_denom)
                ll += np.log(max(prob, 1e-10))
            log_likes[i, q] = ll

    return log_likes


def compute_alpha_if_deleted(
    responses: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Cronbach's alpha if each item is deleted.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), NaN for missing

    Returns
    -------
    NDArray
        Alpha-if-deleted for each item (n_items,)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_alpha_if_deleted(
            responses.astype(np.float64),
        )

    n_persons, n_items = responses.shape

    total_scores = np.nansum(responses, axis=1)

    item_variances = np.zeros(n_items)
    for j in range(n_items):
        col = responses[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            mean = np.nanmean(col)
            item_variances[j] = np.nansum((col[valid] - mean) ** 2) / (valid.sum() - 1)

    alpha_if_deleted = np.zeros(n_items)
    for j in range(n_items):
        remaining_scores = total_scores - np.where(
            np.isnan(responses[:, j]), 0, responses[:, j]
        )
        remaining_var_sum = np.sum(item_variances[np.arange(n_items) != j])
        remaining_mean = np.mean(remaining_scores)
        remaining_total_var = np.sum((remaining_scores - remaining_mean) ** 2) / max(
            n_persons - 1, 1
        )

        k = n_items - 1
        if remaining_total_var > 0 and k > 1:
            alpha_if_deleted[j] = (k / (k - 1)) * (
                1 - remaining_var_sum / remaining_total_var
            )
        else:
            alpha_if_deleted[j] = 0.0

    return alpha_if_deleted


def multigroup_e_step_2pl(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    diff_list: list[NDArray[np.float64]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for 2PL models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    diff_list : list of NDArray
        Difficulty arrays, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
        - posterior_weights: list of (n_persons_g, n_quad) arrays
        - group_log_likelihoods: (n_groups,) array
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        diff_float = [d.astype(np.float64) for d in diff_list]

        return mirt_rs.multigroup_e_step_2pl(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            diff_float,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_3pl(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    diff_list: list[NDArray[np.float64]],
    guess_list: list[NDArray[np.float64]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for 3PL models using Rust backend.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    diff_list : list of NDArray
        Difficulty arrays, one per group (n_items,)
    guess_list : list of NDArray
        Guessing arrays, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        diff_float = [d.astype(np.float64) for d in diff_list]
        guess_float = [g.astype(np.float64) for g in guess_list]

        return mirt_rs.multigroup_e_step_3pl(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            diff_float,
            guess_float,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_grm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    thresh_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for GRM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    thresh_list : list of NDArray
        Threshold matrices, one per group (n_items, max_categories-1)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        thresh_float = [t.astype(np.float64) for t in thresh_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_grm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            thresh_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_gpcm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    disc_list: list[NDArray[np.float64]],
    steps_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for GPCM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    disc_list : list of NDArray
        Discrimination arrays, one per group (n_items,)
    steps_list : list of NDArray
        Step matrices, one per group (n_items, max_categories)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        disc_float = [d.astype(np.float64) for d in disc_list]
        steps_float = [s.astype(np.float64) for s in steps_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_gpcm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            disc_float,
            steps_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_e_step_nrm(
    responses_list: list[NDArray[np.int_]],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    slopes_list: list[NDArray[np.float64]],
    intercepts_list: list[NDArray[np.float64]],
    n_categories_list: list[NDArray[np.int_]],
    prior_means: NDArray[np.float64],
    prior_vars: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]] | None:
    """Compute multigroup E-step for NRM models using Rust backend.

    Processes all groups in parallel using Rayon.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    slopes_list : list of NDArray
        Slope matrices, one per group (n_items, max_categories)
    intercepts_list : list of NDArray
        Intercept matrices, one per group (n_items, max_categories)
    n_categories_list : list of NDArray
        Number of categories per item, one per group (n_items,)
    prior_means : NDArray
        Prior means per group (n_groups,)
    prior_vars : NDArray
        Prior variances per group (n_groups,)

    Returns
    -------
    tuple or None
        (posterior_weights, group_log_likelihoods) or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        slopes_float = [s.astype(np.float64) for s in slopes_list]
        intercepts_float = [i.astype(np.float64) for i in intercepts_list]
        n_cats_int = [n.astype(np.int32) for n in n_categories_list]

        return mirt_rs.multigroup_e_step_nrm(
            responses_int,
            quad_points.astype(np.float64),
            quad_weights.astype(np.float64),
            slopes_float,
            intercepts_float,
            n_cats_int,
            prior_means.astype(np.float64),
            prior_vars.astype(np.float64),
        )

    return None


def multigroup_expected_counts(
    responses_list: list[NDArray[np.int_]],
    posterior_weights_list: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]] | None:
    """Compute expected counts for all groups in parallel.

    Parameters
    ----------
    responses_list : list of NDArray
        Response matrices, one per group (n_persons_g, n_items)
    posterior_weights_list : list of NDArray
        Posterior weights, one per group (n_persons_g, n_quad)

    Returns
    -------
    tuple or None
        (r_k_list, n_k_list) or None if Rust unavailable
        - r_k_list: list of (n_items, n_quad) expected correct counts
        - n_k_list: list of (n_items, n_quad) expected total counts
    """
    if RUST_AVAILABLE:
        responses_int = [r.astype(np.int32) for r in responses_list]
        weights_float = [w.astype(np.float64) for w in posterior_weights_list]

        return mirt_rs.multigroup_expected_counts(responses_int, weights_float)

    return None


def gvem_e_step(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_cov_inv: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    n_inner_iter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """GVEM E-step: update variational parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)
    prior_cov_inv : NDArray
        Inverse of prior covariance (n_factors, n_factors)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    n_inner_iter : int
        Number of inner iterations

    Returns
    -------
    tuple or None
        (mu, sigma, xi) updated arrays, or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        return mirt_rs.gvem_e_step(
            responses.astype(np.int32),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
            prior_cov_inv.astype(np.float64),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            n_inner_iter,
        )

    return None


def gvem_m_step(
    responses: NDArray[np.int_],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """GVEM M-step: update item parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)

    Returns
    -------
    tuple or None
        (loadings, intercepts) updated arrays, or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        return mirt_rs.gvem_m_step(
            responses.astype(np.int32),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
        )

    return None


def gvem_compute_elbo(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
) -> float | None:
    """Compute ELBO for GVEM.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    prior_mean : NDArray
        Prior mean (n_factors,)
    prior_cov : NDArray
        Prior covariance (n_factors, n_factors)

    Returns
    -------
    float or None
        ELBO value, or None if Rust unavailable
    """
    if RUST_AVAILABLE:
        return mirt_rs.gvem_compute_elbo(
            responses.astype(np.int32),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            prior_mean.astype(np.float64),
            prior_cov.astype(np.float64),
        )

    return None
