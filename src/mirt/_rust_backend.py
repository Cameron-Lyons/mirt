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
