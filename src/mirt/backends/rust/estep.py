"""Rust backend: estep.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    _entry_chunk_size,
    _prepare_binary_response_components,
    mirt_rs,
    rust_enabled,
)
from mirt.backends.rust.likelihood import compute_log_likelihoods_2pl

FALLBACK_MODE = "numpy"


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
    if rust_enabled():
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

    log_prior_mass = gaussian_log_quadrature_mass(
        quad_points,
        quad_weights,
        np.array([prior_mean]),
        np.array([[prior_var]]),
    )
    log_joint = log_likes + log_prior_mass[None, :]
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
    if rust_enabled():
        resp = _ensure_i32(responses)
        return mirt_rs.compute_expected_counts(
            resp.ravel() if resp is not None else responses.astype(np.int32).ravel(),
            _ensure_f64(posterior_weights),
        )

    responses_array = np.asarray(responses)
    weights = np.asarray(posterior_weights, dtype=np.float64)
    n_persons = len(responses_array)
    n_quad = weights.shape[1]
    r_k = np.zeros(n_quad, dtype=np.float64)
    n_k = np.zeros(n_quad, dtype=np.float64)
    chunk_size = _entry_chunk_size(n_persons, 2)

    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        correct, valid = _prepare_binary_response_components(
            responses_array[start:stop]
        )
        weight_chunk = weights[start:stop]
        r_k += correct @ weight_chunk
        n_k += valid @ weight_chunk
    return r_k, n_k


def compute_expected_counts_polytomous(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    n_categories: int,
) -> NDArray[np.float64]:
    """Compute expected counts per category for polytomous items."""
    if rust_enabled():
        resp = _ensure_i32(responses)
        return mirt_rs.compute_expected_counts_polytomous(
            resp.ravel() if resp is not None else responses.astype(np.int32).ravel(),
            _ensure_f64(posterior_weights),
            n_categories,
        )

    responses_array = np.asarray(responses)
    weights = np.asarray(posterior_weights, dtype=np.float64)
    n_quad = weights.shape[1]
    r_kc = np.zeros((n_quad, n_categories), dtype=np.float64)
    valid_rows = np.flatnonzero(
        (responses_array >= 0) & (responses_array < n_categories)
    )
    chunk_size = _entry_chunk_size(
        len(valid_rows),
        n_quad + n_categories,
    )

    for start in range(0, len(valid_rows), chunk_size):
        rows = valid_rows[start : start + chunk_size]
        categories = responses_array[rows].astype(np.intp, copy=False)
        indicators = np.zeros((len(rows), n_categories), dtype=np.float64)
        indicators[np.arange(len(rows)), categories] = 1.0
        r_kc += weights[rows].T @ indicators

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
    if rust_enabled():
        return mirt_rs.compute_expected_counts_parallel(
            responses.astype(np.int32),
            posterior_weights.astype(np.float64),
        )

    responses_array = np.asarray(responses)
    weights = np.asarray(posterior_weights, dtype=np.float64)
    n_persons, n_items = responses_array.shape
    n_quad = weights.shape[1]

    r_k_all = np.zeros((n_items, n_quad), dtype=np.float64)
    n_k_all = np.zeros((n_items, n_quad), dtype=np.float64)
    chunk_size = _entry_chunk_size(n_persons, 2 * n_items)

    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        correct, valid = _prepare_binary_response_components(
            responses_array[start:stop]
        )
        weight_chunk = weights[start:stop]
        r_k_all += correct.T @ weight_chunk
        n_k_all += valid.T @ weight_chunk

    return r_k_all, n_k_all
