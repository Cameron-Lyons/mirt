"""Rust backend: calibration.

Fallback mode: mixed. fixed_calib_em is required; stocking_lord_criterion has a NumPy fallback.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _quad_chunk_size,
    mirt_rs,
    rust_enabled,
    rust_required,
)

FALLBACK_MODE = "mixed"


def _numeric_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return a finite contiguous float vector."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector)


def _finite_scalar(value: float, name: str) -> float:
    """Return a finite real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    """Return an integer no smaller than ``minimum``."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _bounds(
    values: tuple[float, float], name: str, *, positive: bool = False
) -> tuple[float, float]:
    """Validate a pair of finite ordered bounds."""
    try:
        limits = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numeric bounds") from exc
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError(f"{name} must contain two finite numeric bounds")
    lower, upper = float(limits[0]), float(limits[1])
    if lower >= upper:
        raise ValueError(f"{name} must satisfy lower < upper")
    if positive and lower <= 0.0:
        raise ValueError(f"{name} must have a positive lower bound")
    return lower, upper


def _item_indices(values: list[int], name: str, n_items: int) -> list[int]:
    """Validate item indices before native array indexing."""
    try:
        indices = list(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a non-empty sequence of item indices"
        ) from exc
    if not indices:
        raise ValueError(f"{name} must contain at least one item index")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise ValueError(f"{name} must contain only integer item indices")
    normalized = [int(index) for index in indices]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate item indices")
    if any(index < 0 or index >= n_items for index in normalized):
        raise ValueError(f"{name} contains an out-of-bounds item index")
    return normalized


def _prepare_fixed_calib_inputs(
    responses: NDArray[np.int_],
    anchor_items: list[int],
    new_items: list[int],
    anchor_disc: NDArray[np.float64],
    anchor_diff: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    max_iter: int,
    tol: float,
    disc_bounds: tuple[float, float],
    diff_bounds: tuple[float, float],
    prob_clamp: tuple[float, float],
    init_disc: float,
    init_diff: float,
    min_count: float,
    min_valid_points: int,
) -> tuple[
    NDArray[np.int32],
    list[int],
    list[int],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    float,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
    float,
    float,
    int,
]:
    """Normalize fixed-calibration inputs before dispatching to Rust."""
    try:
        response_values = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("responses must be a numeric matrix") from exc
    if response_values.ndim != 2 or 0 in response_values.shape:
        raise ValueError("responses must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(response_values)):
        raise ValueError("responses must contain only finite values")
    observed = response_values >= 0.0
    if np.any(observed & (response_values != 0.0) & (response_values != 1.0)):
        raise ValueError("observed responses must be coded as 0 or 1")
    normalized_responses = np.ascontiguousarray(
        np.where(observed, response_values, -1.0), dtype=np.int32
    )

    n_items = response_values.shape[1]
    anchors = _item_indices(anchor_items, "anchor_items", n_items)
    new = _item_indices(new_items, "new_items", n_items)
    if set(anchors) & set(new):
        raise ValueError("anchor_items and new_items must be disjoint")

    anchor_discrimination = _numeric_vector(anchor_disc, "anchor_disc")
    anchor_difficulty = _numeric_vector(anchor_diff, "anchor_diff")
    expected_anchor_shape = (len(anchors),)
    if anchor_discrimination.shape != expected_anchor_shape:
        raise ValueError(f"anchor_disc must have shape {expected_anchor_shape}")
    if anchor_difficulty.shape != expected_anchor_shape:
        raise ValueError(f"anchor_diff must have shape {expected_anchor_shape}")
    if np.any(anchor_discrimination <= 0.0):
        raise ValueError("anchor_disc must contain only positive values")

    points = _numeric_vector(theta_grid, "theta_grid")
    weights = _numeric_vector(quad_weights, "quad_weights")
    if weights.shape != points.shape:
        raise ValueError("quad_weights must have one value per theta_grid point")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("quad_weights must be non-negative with a positive sum")

    iterations = _positive_integer(max_iter, "max_iter")
    tolerance = _finite_scalar(tol, "tol")
    if tolerance <= 0.0:
        raise ValueError("tol must be positive")
    disc_limits = _bounds(disc_bounds, "disc_bounds", positive=True)
    diff_limits = _bounds(diff_bounds, "diff_bounds")
    probability_limits = _bounds(prob_clamp, "prob_clamp")
    if not 0.0 < probability_limits[0] < probability_limits[1] < 1.0:
        raise ValueError("prob_clamp values must lie strictly between 0 and 1")

    initial_disc = _finite_scalar(init_disc, "init_disc")
    initial_diff = _finite_scalar(init_diff, "init_diff")
    if not disc_limits[0] <= initial_disc <= disc_limits[1]:
        raise ValueError("init_disc must lie within disc_bounds")
    if not diff_limits[0] <= initial_diff <= diff_limits[1]:
        raise ValueError("init_diff must lie within diff_bounds")

    count_threshold = _finite_scalar(min_count, "min_count")
    if count_threshold < 0.0:
        raise ValueError("min_count must be non-negative")
    valid_points = _positive_integer(min_valid_points, "min_valid_points", minimum=2)
    if valid_points > points.size:
        raise ValueError("min_valid_points must not exceed the theta_grid size")

    return (
        normalized_responses,
        anchors,
        new,
        anchor_discrimination,
        anchor_difficulty,
        points,
        weights,
        iterations,
        tolerance,
        disc_limits,
        diff_limits,
        probability_limits,
        initial_disc,
        initial_diff,
        count_threshold,
        valid_points,
    )


def fixed_calib_em(
    responses: NDArray[np.int_],
    anchor_items: list[int],
    new_items: list[int],
    anchor_disc: NDArray[np.float64],
    anchor_diff: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    max_iter: int = 500,
    tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.2, 5.0),
    diff_bounds: tuple[float, float] = (-5.0, 5.0),
    prob_clamp: tuple[float, float] = (0.01, 0.99),
    init_disc: float = 1.0,
    init_diff: float = 0.0,
    min_count: float = 1.0,
    min_valid_points: int = 3,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int, bool
]:
    """Fixed-item calibration EM algorithm in Rust.

    Calibrates new items to an existing scale defined by anchor items
    with fixed parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    anchor_items : list[int]
        Indices of anchor items in responses
    new_items : list[int]
        Indices of new items to calibrate
    anchor_disc : NDArray
        Fixed discrimination parameters for anchor items
    anchor_diff : NDArray
        Fixed difficulty parameters for anchor items
    theta_grid : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max)
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max)
    prob_clamp : tuple[float, float]
        Bounds for probability clipping (min, max)
    init_disc : float
        Initial discrimination value
    init_diff : float
        Initial difficulty value
    min_count : float
        Minimum count threshold for valid quadrature points
    min_valid_points : int
        Minimum number of valid points for regression

    Returns
    -------
    tuple
        (new_disc, new_diff, theta, log_likelihood, n_iterations, converged)
    """
    (
        responses,
        anchor_items,
        new_items,
        anchor_disc,
        anchor_diff,
        theta_grid,
        quad_weights,
        max_iter,
        tol,
        disc_bounds,
        diff_bounds,
        prob_clamp,
        init_disc,
        init_diff,
        min_count,
        min_valid_points,
    ) = _prepare_fixed_calib_inputs(
        responses,
        anchor_items,
        new_items,
        anchor_disc,
        anchor_diff,
        theta_grid,
        quad_weights,
        max_iter,
        tol,
        disc_bounds,
        diff_bounds,
        prob_clamp,
        init_disc,
        init_diff,
        min_count,
        min_valid_points,
    )

    if rust_enabled():
        return mirt_rs.fixed_calib_em(
            responses,
            anchor_items,
            new_items,
            anchor_disc,
            anchor_diff,
            theta_grid,
            quad_weights,
            max_iter,
            tol,
            disc_bounds,
            diff_bounds,
            prob_clamp,
            init_disc,
            init_diff,
            min_count,
            min_valid_points,
        )

    rust_required("fixed_calib_em")


def stocking_lord_criterion(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    a: float,
    b: float,
    theta_grid: NDArray[np.float64],
) -> float:
    """Compute Stocking-Lord equating criterion using Rust.

    Parameters
    ----------
    disc_old : NDArray
        Discrimination parameters for old form
    diff_old : NDArray
        Difficulty parameters for old form
    disc_new : NDArray
        Discrimination parameters for new form
    diff_new : NDArray
        Difficulty parameters for new form
    a : float
        Slope transformation constant
    b : float
        Intercept transformation constant
    theta_grid : NDArray
        Theta values for integration

    Returns
    -------
    float
        Criterion value (sum of squared probability differences)
    """
    disc_old = _numeric_vector(disc_old, "disc_old")
    diff_old = _numeric_vector(diff_old, "diff_old")
    disc_new = _numeric_vector(disc_new, "disc_new")
    diff_new = _numeric_vector(diff_new, "diff_new")
    if not (disc_old.shape == diff_old.shape == disc_new.shape == diff_new.shape):
        raise ValueError("item parameter vectors must have the same length")
    a = _finite_scalar(a, "a")
    b = _finite_scalar(b, "b")
    theta_grid = _numeric_vector(theta_grid, "theta_grid")

    if rust_enabled():
        return mirt_rs.stocking_lord_criterion(
            disc_old,
            diff_old,
            disc_new,
            diff_new,
            a,
            b,
            theta_grid,
        )

    n_items = len(disc_old)
    total_diff = 0.0
    chunk_size = _quad_chunk_size(theta_grid.size, n_items)
    for start in range(0, theta_grid.size, chunk_size):
        theta = theta_grid[start : start + chunk_size]
        transformed_theta = a * theta + b
        p_old = sigmoid(disc_old[:, None] * (theta - diff_old[:, None]))
        p_new = sigmoid(disc_new[:, None] * (transformed_theta - diff_new[:, None]))
        total_diff += float(np.sum(np.square(p_old - p_new), dtype=np.float64))

    return float(total_diff)
