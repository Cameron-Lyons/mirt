"""Validated native and NumPy M-step kernels for dichotomous items.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import mirt_rs, rust_enabled
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"

_MAX_LINE_SEARCH_STEPS = 24

_PreparedMstepInputs: TypeAlias = tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    float,
    tuple[float, float],
    tuple[float, float],
    float,
    float,
]


def _validate_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to 1")
    return int(value)


def _validate_scalar(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < minimum:
        qualifier = "positive" if minimum > 0.0 else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}")
    return normalized


def _validate_bounds(
    bounds: tuple[float, float],
    name: str,
    *,
    positive_lower: bool = False,
) -> tuple[float, float]:
    try:
        values = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two numeric bounds") from exc
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain two finite numeric bounds")
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError(f"{name} must satisfy lower < upper")
    if positive_lower and lower <= 0.0:
        raise ValueError(f"{name} must have a positive lower bound")
    return lower, upper


def _prepare_m_step_inputs(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    max_iter: int,
    tol: float,
    disc_bounds: tuple[float, float],
    diff_bounds: tuple[float, float],
    damping: float,
    regularization: float,
) -> _PreparedMstepInputs:
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

    try:
        weights = np.asarray(posterior_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("posterior_weights must be a numeric matrix") from exc
    if weights.ndim != 2 or weights.shape[0] != response_values.shape[0]:
        raise ValueError("posterior_weights must have one row per response pattern")
    if weights.shape[1] == 0:
        raise ValueError("posterior_weights must contain at least one quadrature point")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("posterior_weights must be finite and non-negative")
    if np.any(np.sum(weights, axis=1) <= 0.0):
        raise ValueError("each posterior_weights row must have a positive sum")
    weights = np.ascontiguousarray(weights)

    try:
        points = np.asarray(quad_points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("quad_points must be a numeric vector") from exc
    if points.ndim != 1 or points.shape != (weights.shape[1],):
        raise ValueError(
            "quad_points must be one-dimensional with one value per posterior column"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("quad_points must contain only finite values")
    points = np.ascontiguousarray(points)

    n_items = response_values.shape[1]
    parameters: list[NDArray[np.float64]] = []
    for name, raw_values in (
        ("discrimination", discrimination),
        ("difficulty", difficulty),
    ):
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector") from exc
        if values.shape != (n_items,):
            raise ValueError(f"{name} must have shape ({n_items},)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        parameters.append(np.ascontiguousarray(values))
    disc, diff = parameters

    iterations = _validate_integer(max_iter, "max_iter")
    tolerance = _validate_scalar(tol, "tol", minimum=np.nextafter(0.0, 1.0))
    disc_limits = _validate_bounds(disc_bounds, "disc_bounds", positive_lower=True)
    diff_limits = _validate_bounds(diff_bounds, "diff_bounds")
    damping_value = _validate_scalar(
        damping,
        "damping",
        minimum=np.nextafter(0.0, 1.0),
        maximum=1.0,
    )
    regularization_value = _validate_scalar(
        regularization, "regularization", minimum=0.0
    )

    if np.any((disc < disc_limits[0]) | (disc > disc_limits[1])):
        raise ValueError("discrimination values must lie within disc_bounds")
    if np.any((diff < diff_limits[0]) | (diff > diff_limits[1])):
        raise ValueError("difficulty values must lie within diff_bounds")

    return (
        normalized_responses,
        weights,
        points,
        disc,
        diff,
        iterations,
        tolerance,
        disc_limits,
        diff_limits,
        damping_value,
        regularization_value,
    )


def _expected_log_likelihood(
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    correct_counts: NDArray[np.float64],
    total_counts: NDArray[np.float64],
    quad_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    logits = discrimination[:, None] * (quad_points[None, :] - difficulty[:, None])
    log_probability = -np.logaddexp(0.0, -logits)
    log_complement = -np.logaddexp(0.0, logits)
    return np.sum(
        correct_counts * log_probability
        + (total_counts - correct_counts) * log_complement,
        axis=1,
    )


def _m_step_numpy(
    responses: NDArray[np.int32],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    max_iter: int,
    tol: float,
    disc_bounds: tuple[float, float],
    diff_bounds: tuple[float, float],
    damping: float,
    regularization: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run a vectorized, monotone Newton M-step in slope-intercept space."""
    observed = responses >= 0
    correct = responses == 1
    total_counts = observed.T @ posterior_weights
    correct_counts = correct.T @ posterior_weights

    disc = discrimination.copy()
    diff = difficulty.copy()
    active = np.sum(total_counts, axis=1) > PROB_EPSILON

    for _ in range(max_iter):
        item_indices = np.flatnonzero(active)
        if item_indices.size == 0:
            break

        item_disc = disc[item_indices]
        item_diff = diff[item_indices]
        item_total = total_counts[item_indices]
        item_correct = correct_counts[item_indices]
        logits = item_disc[:, None] * (quad_points[None, :] - item_diff[:, None])
        probability = sigmoid(logits)
        residual = item_correct - item_total * probability
        information = item_total * probability * (1.0 - probability)

        gradient_slope = residual @ quad_points
        gradient_intercept = np.sum(residual, axis=1)
        hessian_slope = -(information @ (quad_points**2)) - regularization
        hessian_intercept = -np.sum(information, axis=1) - regularization
        hessian_cross = -(information @ quad_points)
        determinant = hessian_slope * hessian_intercept - hessian_cross * hessian_cross
        solvable = np.abs(determinant) >= PROB_EPSILON
        if not np.all(solvable):
            active[item_indices[~solvable]] = False
            item_indices = item_indices[solvable]
            if item_indices.size == 0:
                continue
            item_disc = item_disc[solvable]
            item_diff = item_diff[solvable]
            item_total = item_total[solvable]
            item_correct = item_correct[solvable]
            gradient_slope = gradient_slope[solvable]
            gradient_intercept = gradient_intercept[solvable]
            hessian_slope = hessian_slope[solvable]
            hessian_intercept = hessian_intercept[solvable]
            hessian_cross = hessian_cross[solvable]
            determinant = determinant[solvable]

        delta_slope = (
            hessian_intercept * gradient_slope - hessian_cross * gradient_intercept
        ) / determinant
        delta_intercept = (
            -hessian_cross * gradient_slope + hessian_slope * gradient_intercept
        ) / determinant
        intercept = -item_disc * item_diff
        current_ll = _expected_log_likelihood(
            item_disc,
            item_diff,
            item_correct,
            item_total,
            quad_points,
        )

        step = np.full(item_indices.size, damping, dtype=np.float64)
        accepted = np.zeros(item_indices.size, dtype=np.bool_)
        candidate_disc = item_disc.copy()
        candidate_diff = item_diff.copy()
        likelihood_tolerance = 1e-12 * np.maximum(1.0, np.abs(current_ll))

        for _ in range(_MAX_LINE_SEARCH_STEPS):
            pending = ~accepted
            if not np.any(pending):
                break
            proposed_disc = np.clip(
                item_disc[pending] - step[pending] * delta_slope[pending],
                *disc_bounds,
            )
            proposed_intercept = (
                intercept[pending] - step[pending] * delta_intercept[pending]
            )
            proposed_diff = np.clip(
                -proposed_intercept / proposed_disc,
                *diff_bounds,
            )
            proposed_ll = _expected_log_likelihood(
                proposed_disc,
                proposed_diff,
                item_correct[pending],
                item_total[pending],
                quad_points,
            )
            improved = (
                proposed_ll >= current_ll[pending] - likelihood_tolerance[pending]
            )
            pending_indices = np.flatnonzero(pending)
            accepted_indices = pending_indices[improved]
            candidate_disc[accepted_indices] = proposed_disc[improved]
            candidate_diff[accepted_indices] = proposed_diff[improved]
            accepted[accepted_indices] = True
            step[pending_indices[~improved]] *= 0.5

        if not np.all(accepted):
            active[item_indices[~accepted]] = False

        parameter_change = np.maximum(
            np.abs(candidate_disc - item_disc),
            np.abs(candidate_diff - item_diff),
        )
        disc[item_indices[accepted]] = candidate_disc[accepted]
        diff[item_indices[accepted]] = candidate_diff[accepted]
        converged = accepted & (parameter_change < tol)
        active[item_indices[converged]] = False

    return disc, diff


def m_step_dichotomous_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    max_iter: int = 10,
    tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    damping: float = 0.5,
    regularization: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Optimize independent 2PL items with a monotone Newton M-step.

    The optimizer uses the exact expected-likelihood curvature in the
    canonical slope-intercept parameterization, then converts each update
    back to discrimination and difficulty. Backtracking guarantees that an
    accepted update does not reduce the expected log-likelihood. Items with
    no observed responses retain their initial values.

    Parameters
    ----------
    responses : ndarray
        Response matrix with shape (n_persons, n_items). Zero and one are
        observed responses; any finite negative value denotes missing data.
    posterior_weights : ndarray
        Non-negative posterior weights with shape (n_persons, n_quad).
        Every row must have positive total weight.
    quad_points : ndarray
        Finite one-dimensional quadrature grid with n_quad values.
    discrimination, difficulty : ndarray
        Finite initial item parameters with shape (n_items,) and values
        inside their respective bounds.
    max_iter : int, default=10
        Maximum Newton iterations per item.
    tol : float, default=1e-4
        Absolute convergence tolerance for both item parameters.
    disc_bounds, diff_bounds : tuple of float
        Finite lower and upper bounds. The discrimination lower bound must be
        positive.
    damping : float, default=0.5
        Initial Newton step fraction in the interval (0, 1].
    regularization : float, default=0.01
        Non-negative diagonal Hessian stabilization.

    Returns
    -------
    tuple of ndarray
        Updated discrimination and difficulty vectors.
    """
    prepared = _prepare_m_step_inputs(
        responses,
        posterior_weights,
        quad_points,
        discrimination,
        difficulty,
        max_iter,
        tol,
        disc_bounds,
        diff_bounds,
        damping,
        regularization,
    )
    (
        response_values,
        weights,
        points,
        disc,
        diff,
        iterations,
        tolerance,
        disc_limits,
        diff_limits,
        damping_value,
        regularization_value,
    ) = prepared

    if rust_enabled():
        return mirt_rs.m_step_dichotomous_parallel(
            response_values,
            weights,
            points,
            disc,
            diff,
            iterations,
            tolerance,
            disc_limits,
            diff_limits,
            damping_value,
            regularization_value,
        )

    return _m_step_numpy(*prepared)
