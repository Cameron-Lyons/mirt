"""Rust backend: calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
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
    if RUST_AVAILABLE:
        return mirt_rs.fixed_calib_em(
            responses.astype(np.int32),
            anchor_items,
            new_items,
            anchor_disc.astype(np.float64),
            anchor_diff.astype(np.float64),
            theta_grid.astype(np.float64),
            quad_weights.astype(np.float64),
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

    raise RuntimeError("Rust backend required for fixed_calib_em")


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
    if RUST_AVAILABLE:
        return mirt_rs.stocking_lord_criterion(
            disc_old.astype(np.float64),
            diff_old.astype(np.float64),
            disc_new.astype(np.float64),
            diff_new.astype(np.float64),
            float(a),
            float(b),
            theta_grid.astype(np.float64),
        )

    n_items = len(disc_old)

    total_diff = 0.0
    for j in range(n_items):
        for theta in theta_grid:
            p_old = sigmoid(disc_old[j] * (theta - diff_old[j]))
            theta_trans = a * theta + b
            p_new = sigmoid(disc_new[j] * (theta_trans - diff_new[j]))
            total_diff += (p_old - p_new) ** 2

    return total_diff
