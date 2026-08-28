"""Rust backend: cat.

Fallback mode: mixed. cat_compute_item_info / cat_select_max_info / cat_eap_update are numpy; cat_simulate_batch / cat_conditional_mse are optional.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt._prior_mass import normalize_log_mass
from mirt.backends.rust._helpers import (
    _quad_chunk_size,
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "mixed"


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


def cat_compute_item_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Fisher information for all items at a given theta.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters, shape (n_items,).
    difficulty : ndarray
        Item difficulty parameters, shape (n_items,).

    Returns
    -------
    ndarray
        Fisher information for each item, shape (n_items,).
    """
    theta = _finite_scalar(theta, "theta")
    discrimination, difficulty = _item_parameters(discrimination, difficulty)

    if rust_enabled():
        return mirt_rs.cat_compute_item_info(
            theta,
            discrimination,
            difficulty,
        )

    z = discrimination * (theta - difficulty)
    p = sigmoid(z)
    q = 1.0 - p
    return (discrimination**2) * p * q


def cat_select_max_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    available_mask: NDArray[np.bool_],
) -> int:
    """Select item with maximum Fisher information.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    available_mask : ndarray
        Boolean mask of available items.

    Returns
    -------
    int
        Index of selected item, or -1 if no items available.
    """
    theta = _finite_scalar(theta, "theta")
    discrimination, difficulty = _item_parameters(discrimination, difficulty)
    mask = np.asarray(available_mask)
    if mask.dtype != np.bool_ or mask.shape != discrimination.shape:
        raise ValueError(
            "available_mask must be a boolean vector with one value per item"
        )
    mask = np.ascontiguousarray(mask)

    if rust_enabled():
        return mirt_rs.cat_select_max_info(
            theta,
            discrimination,
            difficulty,
            mask,
        )

    if not np.any(mask):
        return -1
    z = discrimination * (theta - difficulty)
    probability = sigmoid(z)
    info = np.where(
        mask, discrimination**2 * probability * (1.0 - probability), -np.inf
    )
    return int(np.argmax(info))


def cat_eap_update(
    administered_items: NDArray[np.int32],
    responses: NDArray[np.int32],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Incremental EAP update after responses.

    Parameters
    ----------
    administered_items : ndarray
        Indices of administered items.
    responses : ndarray
        Responses to administered items.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points.
    quad_weights : ndarray
        Quadrature weights.

    Returns
    -------
    tuple[float, float]
        (theta_estimate, standard_error).
    """
    discrimination, difficulty = _item_parameters(discrimination, difficulty)
    points, weights = _quadrature(quad_points, quad_weights)

    item_values = np.asarray(administered_items)
    if item_values.ndim != 1 or not np.issubdtype(item_values.dtype, np.integer):
        raise ValueError("administered_items must be a one-dimensional integer vector")
    if np.any(item_values < 0) or np.any(item_values >= discrimination.size):
        raise ValueError("administered_items contains an out-of-bounds item index")
    items = np.ascontiguousarray(item_values, dtype=np.int32)

    try:
        response_values = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must be a numeric vector") from exc
    if response_values.ndim != 1 or response_values.shape != items.shape:
        raise ValueError("responses must have one value per administered item")
    if not np.all(np.isfinite(response_values)):
        raise ValueError("responses must contain only finite values")
    observed = response_values >= 0.0
    if np.any(observed & (response_values != 0.0) & (response_values != 1.0)):
        raise ValueError("observed responses must be coded as 0 or 1")
    normalized_responses = np.ascontiguousarray(
        np.where(observed, response_values, -1.0), dtype=np.int32
    )

    if rust_enabled():
        theta, se = mirt_rs.cat_eap_update(
            items,
            normalized_responses,
            discrimination,
            difficulty,
            points,
            weights,
        )
        return float(theta[0]), float(se[0])

    observed_items = items[observed]
    observed_responses = normalized_responses[observed]
    selected_disc = discrimination[observed_items]
    selected_diff = difficulty[observed_items]

    log_likes = np.zeros(points.size, dtype=np.float64)
    chunk_size = _quad_chunk_size(points.size, observed_items.size)
    signed_responses = 2.0 * observed_responses - 1.0
    for start in range(0, points.size, chunk_size):
        point_chunk = points[start : start + chunk_size]
        logits = selected_disc[:, None] * (
            point_chunk[None, :] - selected_diff[:, None]
        )
        log_likes[start : start + chunk_size] = np.sum(
            -np.logaddexp(0.0, -signed_responses[:, None] * logits),
            axis=0,
        )

    log_weights = np.full_like(weights, -np.inf)
    positive_weight = weights > 0.0
    log_weights[positive_weight] = np.log(weights[positive_weight])
    posterior = np.exp(normalize_log_mass(log_likes + log_weights))

    theta_eap = float(posterior @ points)
    variance = max(0.0, float(posterior @ np.square(points - theta_eap)))
    se = np.sqrt(variance)

    return float(theta_eap), float(se)


def cat_simulate_batch(
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
) -> (
    tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]
    ]
    | None
):
    """Run batch CAT simulations in parallel using Rust.

    Parameters
    ----------
    true_thetas : ndarray
        True ability values to simulate.
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
        (theta_estimates, se_estimates, n_items, true_thetas_expanded).
        Returns None if Rust backend not available.
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

    if rust_enabled():
        return mirt_rs.cat_simulate_batch(
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

    return None


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
