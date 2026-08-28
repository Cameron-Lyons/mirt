"""Rust backend: scoring.

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
    _prepare_binary_response_components,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"


def compute_eap_scores(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute EAP scores with standard errors."""
    if rust_enabled():
        return mirt_rs.compute_eap_scores(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    responses = np.asarray(responses)
    quad_points = np.asarray(quad_points, dtype=np.float64)
    quad_weights = np.asarray(quad_weights, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)

    n_persons, n_items = responses.shape
    n_quad = quad_points.shape[0]

    z = discrimination[None, :] * (quad_points[:, None] - difficulty[None, :])
    probabilities = sigmoid(z)
    probabilities = np.clip(
        probabilities,
        PROB_EPSILON,
        1 - PROB_EPSILON,
    )
    log_correct = np.log(probabilities)
    log_incorrect = np.log1p(-probabilities)
    log_weights = np.log(quad_weights + 1e-300)

    correct, valid = _prepare_binary_response_components(responses)
    chunk_size = _entry_chunk_size(n_persons, n_items + 2 * n_quad)
    theta = np.empty(n_persons, dtype=np.float64)
    se = np.empty(n_persons, dtype=np.float64)

    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        correct_chunk = correct[start:stop]
        valid_chunk = valid[start:stop]
        log_posterior = (
            correct_chunk @ log_correct.T
            + (valid_chunk - correct_chunk) @ log_incorrect.T
            + log_weights[None, :]
        )
        log_posterior -= np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)

        theta_chunk = posterior @ quad_points
        theta[start:stop] = theta_chunk
        se[start:stop] = np.sqrt(
            np.sum(
                posterior * (quad_points[None, :] - theta_chunk[:, None]) ** 2,
                axis=1,
            )
        )

    return theta, se
