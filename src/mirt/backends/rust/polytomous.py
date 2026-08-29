"""Rust backend: polytomous.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from mirt._classical import _alpha_if_deleted_numpy
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


def _observed_item_groups(
    responses: NDArray[np.int_],
    n_categories: NDArray[np.int_],
) -> Iterator[tuple[int, NDArray[np.intp], NDArray[np.int_]]]:
    """Yield observed items grouped by their shared category count."""
    if responses.shape[0] == 0:
        observed_items = np.zeros(responses.shape[1], dtype=bool)
    else:
        observed_items = np.max(responses, axis=0) >= 0
    for n_cat_value in np.unique(n_categories):
        n_cat = int(n_cat_value)
        item_indices = np.flatnonzero(n_categories == n_cat)
        item_indices = item_indices[observed_items[item_indices]]
        item_chunk_size = _entry_chunk_size(
            item_indices.size,
            responses.shape[0] + n_cat,
        )
        for start in range(0, item_indices.size, item_chunk_size):
            grouped_indices = item_indices[start : start + item_chunk_size]
            grouped_responses = responses[:, grouped_indices]
            if np.any(grouped_responses >= n_cat):
                raise IndexError("response category is outside the item category range")
            yield n_cat, grouped_indices, grouped_responses


def _accumulate_category_log_probabilities(
    log_likes: NDArray[np.float64],
    responses: NDArray[np.int_],
    log_probabilities: NDArray[np.float64],
    start: int,
    stop: int,
) -> None:
    """Accumulate grouped item contributions with category indicator products."""
    for category in range(log_probabilities.shape[2]):
        indicators = responses == category
        if np.any(indicators):
            log_likes[:, start:stop] += (
                indicators.astype(np.float64) @ log_probabilities[:, :, category].T
            )


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

    n_persons = responses.shape[0]
    n_quad = quad_points.shape[0]
    log_likes = np.zeros((n_persons, n_quad), dtype=np.float64)

    for n_cat, item_indices, grouped_responses in _observed_item_groups(
        responses,
        n_categories,
    ):
        chunk_size = _entry_chunk_size(
            n_quad,
            n_persons + item_indices.size * n_cat,
        )

        for start in range(0, n_quad, chunk_size):
            stop = min(start + chunk_size, n_quad)
            theta_chunk = quad_points[start:stop]
            cumulative = sigmoid(
                discrimination[None, item_indices, None]
                * (
                    theta_chunk[:, None, None]
                    - thresholds[None, item_indices, : n_cat - 1]
                )
            )
            probabilities = np.empty(
                (stop - start, item_indices.size, n_cat),
                dtype=np.float64,
            )
            probabilities[:, :, 0] = 1.0 - cumulative[:, :, 0]
            probabilities[:, :, -1] = cumulative[:, :, -1]
            if n_cat > 2:
                probabilities[:, :, 1:-1] = cumulative[:, :, :-1] - cumulative[:, :, 1:]
            np.maximum(probabilities, PROB_EPSILON, out=probabilities)
            np.log(probabilities, out=probabilities)
            _accumulate_category_log_probabilities(
                log_likes,
                grouped_responses,
                probabilities,
                start,
                stop,
            )

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

    n_persons = responses.shape[0]
    n_quad = quad_points.shape[0]
    log_likes = np.zeros((n_persons, n_quad), dtype=np.float64)
    log_epsilon = np.log(PROB_EPSILON)

    for n_cat, item_indices, grouped_responses in _observed_item_groups(
        responses,
        n_categories,
    ):
        chunk_size = _entry_chunk_size(
            n_quad,
            n_persons + item_indices.size * n_cat,
        )

        for start in range(0, n_quad, chunk_size):
            stop = min(start + chunk_size, n_quad)
            theta_chunk = quad_points[start:stop]
            numerators = np.zeros(
                (stop - start, item_indices.size, n_cat),
                dtype=np.float64,
            )
            numerators[:, :, 1:] = np.cumsum(
                discrimination[None, item_indices, None]
                * (theta_chunk[:, None, None] - steps[None, item_indices, 1:n_cat]),
                axis=2,
            )
            numerators -= np.logaddexp.reduce(
                numerators,
                axis=2,
                keepdims=True,
            )
            np.maximum(numerators, log_epsilon, out=numerators)
            _accumulate_category_log_probabilities(
                log_likes,
                grouped_responses,
                numerators,
                start,
                stop,
            )

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

    return _alpha_if_deleted_numpy(np.asarray(responses, dtype=np.float64))
