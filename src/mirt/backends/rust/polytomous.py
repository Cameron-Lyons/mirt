"""Rust backend: polytomous.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    _entry_chunk_size,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"


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
    if rust_enabled():
        return mirt_rs.compute_log_likelihoods_grm(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(discrimination),
            _ensure_f64(thresholds),
            _ensure_i32(n_categories),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    n_categories = np.asarray(n_categories)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]
    max_categories = int(np.max(n_categories, initial=1))
    chunk_size = _entry_chunk_size(
        n_quad,
        n_persons + max_categories,
    )
    log_likes = np.zeros((n_persons, n_quad), dtype=np.float64)

    for start in range(0, n_quad, chunk_size):
        stop = min(start + chunk_size, n_quad)
        theta_chunk = quad_points[start:stop]

        for item_idx in range(n_items):
            item_responses = responses[:, item_idx]
            observed = item_responses >= 0
            if not np.any(observed):
                continue

            n_cat = int(n_categories[item_idx])
            cumulative = sigmoid(
                discrimination[item_idx]
                * (theta_chunk[:, None] - thresholds[item_idx, : n_cat - 1][None, :])
            )
            probabilities = np.empty((stop - start, n_cat), dtype=np.float64)
            probabilities[:, 0] = 1.0 - cumulative[:, 0]
            probabilities[:, -1] = cumulative[:, -1]
            if n_cat > 2:
                probabilities[:, 1:-1] = cumulative[:, :-1] - cumulative[:, 1:]
            np.maximum(probabilities, PROB_EPSILON, out=probabilities)
            log_probabilities = np.log(probabilities)

            log_likes[observed, start:stop] += log_probabilities[
                :, item_responses[observed]
            ].T

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
    if rust_enabled():
        return mirt_rs.compute_log_likelihoods_gpcm(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(discrimination),
            _ensure_f64(steps),
            _ensure_i32(n_categories),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    n_categories = np.asarray(n_categories)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]
    max_categories = int(np.max(n_categories, initial=1))
    chunk_size = _entry_chunk_size(
        n_quad,
        n_persons + max_categories,
    )
    log_likes = np.zeros((n_persons, n_quad), dtype=np.float64)
    log_epsilon = np.log(PROB_EPSILON)

    for start in range(0, n_quad, chunk_size):
        stop = min(start + chunk_size, n_quad)
        theta_chunk = quad_points[start:stop]

        for item_idx in range(n_items):
            item_responses = responses[:, item_idx]
            observed = item_responses >= 0
            if not np.any(observed):
                continue

            n_cat = int(n_categories[item_idx])
            numerators = np.zeros((stop - start, n_cat), dtype=np.float64)
            numerators[:, 1:] = np.cumsum(
                discrimination[item_idx]
                * (theta_chunk[:, None] - steps[item_idx, 1:n_cat][None, :]),
                axis=1,
            )
            log_probabilities = numerators - np.logaddexp.reduce(
                numerators,
                axis=1,
                keepdims=True,
            )
            np.maximum(log_probabilities, log_epsilon, out=log_probabilities)

            log_likes[observed, start:stop] += log_probabilities[
                :, item_responses[observed]
            ].T

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
    if rust_enabled():
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
