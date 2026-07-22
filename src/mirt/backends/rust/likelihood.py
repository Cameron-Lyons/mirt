"""Rust backend: likelihood.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    _prepare_binary_response_components,
    _quad_chunk_size,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"


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
    if rust_enabled():
        return mirt_rs.compute_log_likelihoods_2pl(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]
    chunk_size = _quad_chunk_size(n_quad, n_items)

    correct, valid = _prepare_binary_response_components(responses)
    log_likes = np.empty((n_persons, n_quad), dtype=np.float64)

    for start in range(0, n_quad, chunk_size):
        stop = min(start + chunk_size, n_quad)
        theta_chunk = quad_points[start:stop]

        z = discrimination[None, :] * (theta_chunk[:, None] - difficulty[None, :])
        probs = sigmoid(z)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        log_p1 = np.log(probs)
        log_p0 = np.log1p(-probs)

        log_likes[:, start:stop] = correct @ log_p1.T + (valid - correct) @ log_p0.T

    return log_likes


def compute_log_likelihoods_3pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for 3PL model at all quadrature points."""
    if rust_enabled():
        return mirt_rs.compute_log_likelihoods_3pl(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(guessing),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    guessing = np.asarray(guessing, dtype=np.float64)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]
    chunk_size = _quad_chunk_size(n_quad, n_items)

    correct, valid = _prepare_binary_response_components(responses)
    log_likes = np.empty((n_persons, n_quad), dtype=np.float64)

    for start in range(0, n_quad, chunk_size):
        stop = min(start + chunk_size, n_quad)
        theta_chunk = quad_points[start:stop]

        z = discrimination[None, :] * (theta_chunk[:, None] - difficulty[None, :])
        p_star = sigmoid(z)
        probs = guessing[None, :] + (1 - guessing[None, :]) * p_star
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        log_p1 = np.log(probs)
        log_p0 = np.log1p(-probs)

        log_likes[:, start:stop] = correct @ log_p1.T + (valid - correct) @ log_p0.T

    return log_likes


def compute_log_likelihoods_mirt(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for multidimensional IRT model."""
    if rust_enabled():
        return mirt_rs.compute_log_likelihoods_mirt(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)

    if quad_points.ndim == 1:
        quad_points = quad_points.reshape(-1, 1)
    if discrimination.ndim == 1:
        discrimination = discrimination.reshape(-1, 1)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]
    chunk_size = _quad_chunk_size(n_quad, n_items)

    correct, valid = _prepare_binary_response_components(responses)
    log_likes = np.empty((n_persons, n_quad), dtype=np.float64)

    item_offsets = discrimination.sum(axis=1) * difficulty

    for start in range(0, n_quad, chunk_size):
        stop = min(start + chunk_size, n_quad)
        theta_chunk = quad_points[start:stop]

        z = theta_chunk @ discrimination.T - item_offsets[None, :]
        probs = sigmoid(z)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        log_p1 = np.log(probs)
        log_p0 = np.log1p(-probs)

        log_likes[:, start:stop] = correct @ log_p1.T + (valid - correct) @ log_p0.T

    return log_likes
