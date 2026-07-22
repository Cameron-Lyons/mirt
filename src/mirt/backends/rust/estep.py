"""Rust backend: estep."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
    _ensure_f64,
    _ensure_i32,
)

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
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
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
        resp = _ensure_i32(responses)
        return mirt_rs.compute_expected_counts(
            resp.ravel() if resp is not None else responses.astype(np.int32).ravel(),
            _ensure_f64(posterior_weights),
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
        resp = _ensure_i32(responses)
        return mirt_rs.compute_expected_counts_polytomous(
            resp.ravel() if resp is not None else responses.astype(np.int32).ravel(),
            _ensure_f64(posterior_weights),
            n_categories,
        )

    n_quad = posterior_weights.shape[1]
    r_kc = np.zeros((n_quad, n_categories))

    for i, resp in enumerate(responses):
        if 0 <= resp < n_categories:
            r_kc[:, resp] += posterior_weights[i]

    return r_kc

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
