"""Rust backend: simulation.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._categorical import sample_categorical_rows
from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "numpy"
_MAX_SEED = int(np.iinfo(np.int64).max)
_MAX_STABLE_LOGIT = np.finfo(np.float64).max


def _float_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """Convert a numeric input and reject values unsafe for either backend."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _theta_column(theta: ArrayLike) -> NDArray[np.float64]:
    """Normalize unidimensional abilities to one contiguous column."""
    values = _float_array("theta", theta)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("theta must be one-dimensional or have shape (n_persons, 1)")
    if values.shape[0] == 0:
        raise ValueError("theta must contain at least one person")
    return np.ascontiguousarray(values)


def _item_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """Validate a non-empty one-dimensional item parameter."""
    array = _float_array(name, values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return np.ascontiguousarray(array)


def _matching_item_vector(
    name: str,
    values: ArrayLike,
    n_items: int,
) -> NDArray[np.float64]:
    """Validate an item vector against the established item count."""
    array = _float_array(name, values)
    if array.shape != (n_items,):
        raise ValueError(f"{name} must have shape ({n_items},)")
    return np.ascontiguousarray(array)


def _threshold_matrix(
    thresholds: ArrayLike,
    n_items: int,
) -> NDArray[np.float64]:
    """Validate a common rectangular threshold matrix."""
    values = _float_array("thresholds", thresholds)
    if values.ndim != 2 or values.shape[0] != n_items or values.shape[1] == 0:
        raise ValueError(
            "thresholds must have shape (n_items, n_categories - 1) with "
            "at least two categories"
        )
    return np.ascontiguousarray(values)


def _seed_value(seed: int | None) -> int:
    """Return a seed accepted consistently by NumPy and the native boundary."""
    if seed is None:
        return int(np.random.default_rng().integers(0, _MAX_SEED))
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer or None")
    value = int(seed)
    if value < 0 or value > _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    return value


def _gpcm_native_safe(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> bool:
    """Return whether cumulative native GPCM logits stay safely finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        largest_increment = np.max(np.abs(discrimination)) * (
            np.max(np.abs(theta)) + np.max(np.abs(thresholds))
        )
        largest_logit = largest_increment * (thresholds.shape[1] + 1)
    return bool(np.isfinite(largest_logit) and largest_logit < 700.0)


def _linear_predictors_native_safe(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    location: NDArray[np.float64],
) -> bool:
    """Return whether a native discrimination-times-distance stays finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        largest_predictor = np.max(np.abs(discrimination)) * (
            np.max(np.abs(theta)) + np.max(np.abs(location))
        )
    return bool(np.isfinite(largest_predictor))


def _stable_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize finite or saturated category logits without warnings."""
    with np.errstate(over="ignore", invalid="ignore"):
        finite_logits = np.nan_to_num(
            logits,
            nan=0.0,
            posinf=_MAX_STABLE_LOGIT,
            neginf=-_MAX_STABLE_LOGIT,
        )
        shifted = finite_logits - np.max(finite_logits, axis=1, keepdims=True)
        weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / np.sum(weights, axis=1, keepdims=True)


def simulate_grm(
    theta: ArrayLike,
    discrimination: ArrayLike,
    thresholds: ArrayLike,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Graded Response Model."""
    theta_array = _theta_column(theta)
    discrimination_array = _item_vector("discrimination", discrimination)
    thresholds_array = _threshold_matrix(thresholds, len(discrimination_array))
    seed_value = _seed_value(seed)

    if rust_enabled() and _linear_predictors_native_safe(
        theta_array,
        discrimination_array,
        thresholds_array,
    ):
        return mirt_rs.simulate_grm(
            theta_array,
            discrimination_array,
            thresholds_array,
            seed_value,
        )

    rng = np.random.default_rng(seed_value)
    n_persons = theta_array.shape[0]
    n_items = len(discrimination_array)
    n_categories = thresholds_array.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        cum_probs = np.ones((n_persons, n_categories))
        for k in range(n_categories - 1):
            with np.errstate(over="ignore", invalid="ignore"):
                z = discrimination_array[i] * (
                    theta_array[:, 0] - thresholds_array[i, k]
                )
            z = np.nan_to_num(z, nan=0.0, posinf=np.inf, neginf=-np.inf)
            cum_probs[:, k + 1] = sigmoid(z)

        cat_probs = -np.diff(
            np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1
        )
        cat_probs = np.maximum(cat_probs, 0)
        totals = cat_probs.sum(axis=1, keepdims=True)
        cat_probs = np.divide(
            cat_probs,
            totals,
            out=np.zeros_like(cat_probs),
            where=totals > 0,
        )

        responses[:, i] = sample_categorical_rows(cat_probs, rng)

    return responses


def simulate_gpcm(
    theta: ArrayLike,
    discrimination: ArrayLike,
    thresholds: ArrayLike,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Generalized Partial Credit Model."""
    theta_array = _theta_column(theta)
    discrimination_array = _item_vector("discrimination", discrimination)
    thresholds_array = _threshold_matrix(thresholds, len(discrimination_array))
    seed_value = _seed_value(seed)

    if rust_enabled() and _gpcm_native_safe(
        theta_array,
        discrimination_array,
        thresholds_array,
    ):
        return mirt_rs.simulate_gpcm(
            theta_array,
            discrimination_array,
            thresholds_array,
            seed_value,
        )

    rng = np.random.default_rng(seed_value)
    n_persons = theta_array.shape[0]
    n_items = len(discrimination_array)
    n_categories = thresholds_array.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        with np.errstate(over="ignore", invalid="ignore"):
            increments = discrimination_array[i] * (
                theta_array[:, :1] - thresholds_array[i][None, :]
            )
        increment_limit = _MAX_STABLE_LOGIT / n_categories
        increments = np.nan_to_num(
            increments,
            nan=0.0,
            posinf=increment_limit,
            neginf=-increment_limit,
        )
        increments = np.clip(increments, -increment_limit, increment_limit)
        logits = np.zeros((n_persons, n_categories), dtype=np.float64)
        logits[:, 1:] = np.cumsum(increments, axis=1)
        cat_probs = _stable_softmax(logits)

        responses[:, i] = sample_categorical_rows(cat_probs, rng)

    return responses


def simulate_dichotomous(
    theta: ArrayLike,
    discrimination: ArrayLike,
    difficulty: ArrayLike,
    guessing: ArrayLike | None = None,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate dichotomous responses (2PL/3PL)."""
    theta_array = _theta_column(theta)[:, 0]
    discrimination_array = _item_vector("discrimination", discrimination)
    n_items = len(discrimination_array)
    difficulty_array = _matching_item_vector("difficulty", difficulty, n_items)
    seed_value = _seed_value(seed)
    if guessing is None:
        guessing_array = np.zeros(n_items, dtype=np.float64)
        native_guessing = None
    else:
        guessing_array = _matching_item_vector("guessing", guessing, n_items)
        if np.any((guessing_array < 0.0) | (guessing_array > 1.0)):
            raise ValueError("guessing values must be between 0 and 1")
        native_guessing = guessing_array

    if rust_enabled() and _linear_predictors_native_safe(
        theta_array,
        discrimination_array,
        difficulty_array,
    ):
        return mirt_rs.simulate_dichotomous(
            theta_array,
            discrimination_array,
            difficulty_array,
            native_guessing,
            seed_value,
        )

    rng = np.random.default_rng(seed_value)
    n_persons = len(theta_array)

    with np.errstate(over="ignore", invalid="ignore"):
        z = discrimination_array[None, :] * (
            theta_array[:, None] - difficulty_array[None, :]
        )
    z = np.nan_to_num(z, nan=0.0, posinf=np.inf, neginf=-np.inf)
    p_star = sigmoid(z)
    probs = guessing_array[None, :] + (1 - guessing_array[None, :]) * p_star

    u = rng.random((n_persons, n_items))
    return (u < probs).astype(np.int_)
