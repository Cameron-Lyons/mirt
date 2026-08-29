"""Rust backend: scoring.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import os

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


def _validate_wle_inputs(
    responses: NDArray[np.int_],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    theta_min: float,
    theta_max: float,
    tol: float,
    n_jobs: int,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
    int,
]:
    """Validate and normalize the batched 2PL WLE contract."""
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional array")
    if raw_responses.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")
    if raw_responses.dtype.kind == "f":
        if not np.all(np.isfinite(raw_responses)):
            raise ValueError("responses must contain only finite values")
        observed = raw_responses >= 0.0
        if np.any(raw_responses[observed] != np.floor(raw_responses[observed])):
            raise ValueError("observed responses must be integer-valued")
    else:
        observed = raw_responses >= 0
    if np.any(observed & (raw_responses > 1)):
        raise ValueError("observed responses must contain only 0 or 1")
    response_values = np.where(observed, raw_responses, -1).astype(np.int32, copy=False)

    discrimination_values = np.asarray(discrimination, dtype=np.float64)
    difficulty_values = np.asarray(difficulty, dtype=np.float64)
    expected_shape = (response_values.shape[1],)
    if discrimination_values.shape != expected_shape:
        raise ValueError(f"discrimination must have shape {expected_shape}")
    if difficulty_values.shape != expected_shape:
        raise ValueError(f"difficulty must have shape {expected_shape}")
    if not np.all(np.isfinite(discrimination_values)) or not np.all(
        np.isfinite(difficulty_values)
    ):
        raise ValueError("item parameters must contain only finite values")

    lower = float(theta_min)
    upper = float(theta_max)
    tolerance = float(tol)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta bounds must be finite and strictly increasing")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and positive")
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(
        n_jobs, (int, np.integer)
    ):
        raise ValueError("n_jobs must be -1 or a positive integer")
    if n_jobs == -1:
        worker_count = os.cpu_count() or 1
    elif n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    else:
        worker_count = int(n_jobs)

    return (
        np.ascontiguousarray(response_values),
        np.ascontiguousarray(discrimination_values),
        np.ascontiguousarray(difficulty_values),
        lower,
        upper,
        tolerance,
        worker_count,
    )


def _wle_numpy_chunk(
    responses: NDArray[np.int32],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    theta_min: float,
    theta_max: float,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Score one response chunk with vectorized golden-section searches."""
    n_persons = responses.shape[0]
    valid = responses >= 0
    correct = responses == 1
    has_responses = np.any(valid, axis=1)
    lower = np.full(n_persons, theta_min, dtype=np.float64)
    upper = np.full(n_persons, theta_max, dtype=np.float64)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    def criterion(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
        log_correct = -np.logaddexp(0.0, -logits)
        log_incorrect = -np.logaddexp(0.0, logits)
        log_likelihood = np.sum(
            np.where(correct, log_correct, np.where(valid, log_incorrect, 0.0)),
            axis=1,
        )
        probabilities = sigmoid(logits)
        information = np.sum(
            valid
            * discrimination[None, :] ** 2
            * probabilities
            * (1.0 - probabilities),
            axis=1,
        )
        correction = np.zeros(n_persons, dtype=np.float64)
        positive = information > PROB_EPSILON
        correction[positive] = 0.5 * np.log(information[positive])
        return log_likelihood + correction

    left = upper - (upper - lower) / phi
    right = lower + (upper - lower) / phi
    left_value = criterion(left)
    right_value = criterion(right)
    for _ in range(256):
        if float(np.max(upper - lower, initial=0.0)) <= tol:
            break
        choose_left = left_value > right_value
        next_lower = np.where(choose_left, lower, left)
        next_upper = np.where(choose_left, right, upper)
        if np.array_equal(next_lower, lower) and np.array_equal(next_upper, upper):
            break
        span = next_upper - next_lower
        next_left = np.where(choose_left, next_upper - span / phi, right)
        next_right = np.where(choose_left, left, next_lower + span / phi)
        proposal_value = criterion(np.where(choose_left, next_left, next_right))
        next_left_value = np.where(choose_left, proposal_value, right_value)
        next_right_value = np.where(choose_left, left_value, proposal_value)
        lower, upper = next_lower, next_upper
        left, right = next_left, next_right
        left_value, right_value = next_left_value, next_right_value

    theta = (lower + upper) / 2.0
    logits = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    probabilities = sigmoid(logits)
    information = np.sum(
        valid * discrimination[None, :] ** 2 * probabilities * (1.0 - probabilities),
        axis=1,
    )
    standard_error = np.full(n_persons, np.inf, dtype=np.float64)
    positive = information > PROB_EPSILON
    standard_error[positive] = 1.0 / np.sqrt(information[positive])
    theta[~has_responses] = 0.0
    standard_error[~has_responses] = np.inf
    return theta, standard_error


def compute_wle_scores(
    responses: NDArray[np.int_],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    theta_min: float = -6.0,
    theta_max: float = 6.0,
    tol: float = 1e-6,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute batched 2PL WLE scores with observed-item information."""
    (
        response_values,
        discrimination_values,
        difficulty_values,
        lower,
        upper,
        tolerance,
        worker_count,
    ) = _validate_wle_inputs(
        responses,
        discrimination,
        difficulty,
        theta_min,
        theta_max,
        tol,
        n_jobs,
    )
    if rust_enabled():
        return mirt_rs.compute_wle_scores(
            response_values,
            discrimination_values,
            difficulty_values,
            lower,
            upper,
            tolerance,
            worker_count,
        )

    n_persons, n_items = response_values.shape
    theta = np.empty(n_persons, dtype=np.float64)
    standard_error = np.empty(n_persons, dtype=np.float64)
    chunk_size = _entry_chunk_size(n_persons, max(1, 8 * n_items))
    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        theta[start:stop], standard_error[start:stop] = _wle_numpy_chunk(
            response_values[start:stop],
            discrimination_values,
            difficulty_values,
            lower,
            upper,
            tolerance,
        )
    return theta, standard_error
