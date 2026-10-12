"""Rust backend: cat.

Fallback mode: optional. Batch simulation and conditional MSE helpers return
``None`` when Rust is unavailable; callers own Python paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "optional"

CATBatchDetails = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
]


def _numeric_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return a finite, non-empty contiguous float vector."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector)


def _finite_scalar(value: float, name: str, *, non_negative: bool = False) -> float:
    """Return a finite real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    if non_negative and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _positive_integer(value: int, name: str) -> int:
    """Return a positive integer control value."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _item_parameters(
    discrimination: NDArray[np.float64], difficulty: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize a unidimensional item bank."""
    disc = _numeric_vector(discrimination, "discrimination")
    diff = _numeric_vector(difficulty, "difficulty")
    if disc.shape != diff.shape:
        raise ValueError("discrimination and difficulty must have the same length")
    return disc, diff


def _quadrature(
    quad_points: NDArray[np.float64], quad_weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize quadrature points and prior masses."""
    points = _numeric_vector(quad_points, "quad_points")
    weights = _numeric_vector(quad_weights, "quad_weights")
    if points.shape != weights.shape:
        raise ValueError("quad_weights must have one value per quad_points entry")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("quad_weights must be non-negative with a positive sum")
    return points, weights


def _seed_value(seed: int | None) -> int:
    """Resolve a seed accepted by the native unsigned 64-bit interface."""
    if seed is None:
        return int(np.random.default_rng().integers(0, 2**31))
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    return int(seed)


def _simulation_inputs(
    evaluation_thetas: NDArray[np.float64],
    theta_name: str,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    se_threshold: float,
    max_items: int,
    min_items: int,
    n_replications: int,
    seed: int | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    int,
    int,
    int,
    int,
]:
    """Validate common inputs for native adaptive-test simulations."""
    thetas = _numeric_vector(evaluation_thetas, theta_name)
    disc, diff = _item_parameters(discrimination, difficulty)
    points, weights = _quadrature(quad_points, quad_weights)
    threshold = _finite_scalar(se_threshold, "se_threshold", non_negative=True)
    maximum = _positive_integer(max_items, "max_items")
    minimum = _positive_integer(min_items, "min_items")
    replications = _positive_integer(n_replications, "n_replications")
    if minimum > maximum:
        raise ValueError("min_items must not exceed max_items")
    resolved_seed = _seed_value(seed)
    return (
        thetas,
        disc,
        diff,
        points,
        weights,
        threshold,
        maximum,
        minimum,
        replications,
        resolved_seed,
    )


def cat_simulate_batch_full(
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
) -> CATBatchDetails | None:
    """Run native batch simulations and retain item and response paths.

    The final two arrays have one row per simulation and up to ``max_items``
    columns, capped by the item-bank size. They use ``-1`` only for padding
    beyond each simulation's reported item count.
    Returns ``None`` when the native backend is unavailable or disabled.

    Replication ``r`` of theta ``t`` draws from its own random stream,
    mixed from ``seed``, ``t`` and ``r``, so the sessions of a call and of
    neighbouring seeds are independent.
    """
    (
        true_thetas,
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
        seed,
    ) = _simulation_inputs(
        true_thetas,
        "true_thetas",
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
        seed,
    )

    if not rust_enabled():
        return None
    result = mirt_rs.cat_simulate_batch(
        true_thetas,
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
        seed,
    )
    if not isinstance(result, tuple) or len(result) != 6:
        raise RuntimeError("native CAT batch simulation returned an invalid payload")
    return result


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

    Replication ``r`` of theta ``t`` draws from its own random stream,
    mixed from ``seed``, ``t`` and ``r``, as in :func:`cat_simulate_batch_full`.

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
    (
        eval_thetas,
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
        seed,
    ) = _simulation_inputs(
        eval_thetas,
        "eval_thetas",
        discrimination,
        difficulty,
        quad_points,
        quad_weights,
        se_threshold,
        max_items,
        min_items,
        n_replications,
        seed,
    )

    if rust_enabled():
        return mirt_rs.cat_conditional_mse(
            eval_thetas,
            discrimination,
            difficulty,
            quad_points,
            quad_weights,
            se_threshold,
            max_items,
            min_items,
            n_replications,
            seed,
        )

    return None
