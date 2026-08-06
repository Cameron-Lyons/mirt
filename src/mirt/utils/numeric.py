"""Shared numeric utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10


def logsumexp(
    a: NDArray[np.float64],
    axis: int | None = None,
    keepdims: bool = False,
) -> NDArray[np.float64]:
    """Compute log(sum(exp(a))) in a numerically stable way."""
    values = np.asarray(a, dtype=np.float64)
    if values.size == 0:
        raise ValueError("a must contain at least one value")

    a_max = np.max(values, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        exp_sum = np.sum(
            np.exp(values - a_max), axis=axis, keepdims=True, dtype=np.float64
        )
        result = a_max + np.log(exp_sum)

    result = np.where(np.isposinf(a_max), np.inf, result)
    result = np.where(np.isneginf(a_max), -np.inf, result)

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return np.asarray(result, dtype=np.float64)


def logsumexp_axis1(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute logsumexp along axis 1, returning a 1D array."""
    values = np.asarray(a, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("a must be a two-dimensional array")
    return logsumexp(values, axis=1).ravel()


def compute_hessian_se(
    func: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute standard errors from a finite-difference Hessian.

    Parameters
    ----------
    func : callable
        Function to compute Hessian of (should be negative log-likelihood or similar).
    x : array
        Point at which to compute Hessian.
    h : float
        Step size for finite differences.

    Returns
    -------
    se : array
        Standard errors (sqrt of diagonal of inverse Hessian).
    """
    point = np.asarray(x, dtype=np.float64)
    if point.ndim != 1 or point.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(point)):
        raise ValueError("x must contain only finite values")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be finite and positive")

    def evaluate(candidate: NDArray[np.float64]) -> float:
        value = float(func(candidate))
        if not np.isfinite(value):
            raise ValueError("func must return finite scalar values near x")
        return value

    n_parameters = len(point)
    steps = h * np.maximum(1.0, np.abs(point))
    hessian = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    f_center = evaluate(point)

    for row in range(n_parameters):
        row_plus = point.copy()
        row_minus = point.copy()
        row_plus[row] += steps[row]
        row_minus[row] -= steps[row]
        hessian[row, row] = (
            evaluate(row_plus) - 2.0 * f_center + evaluate(row_minus)
        ) / (steps[row] ** 2)

        for column in range(row + 1, n_parameters):
            plus_plus = point.copy()
            plus_minus = point.copy()
            minus_plus = point.copy()
            minus_minus = point.copy()
            plus_plus[[row, column]] += steps[[row, column]]
            plus_minus[row] += steps[row]
            plus_minus[column] -= steps[column]
            minus_plus[row] -= steps[row]
            minus_plus[column] += steps[column]
            minus_minus[[row, column]] -= steps[[row, column]]

            cross_derivative = (
                evaluate(plus_plus)
                - evaluate(plus_minus)
                - evaluate(minus_plus)
                + evaluate(minus_minus)
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = cross_derivative
            hessian[column, row] = cross_derivative

    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    # Finite-difference Hessians are only accurate to roughly sqrt(eps).
    # Treat smaller or non-positive eigenvalues as numerically singular so
    # platform FD noise on rank-deficient objectives cannot yield huge SEs.
    tolerance = np.sqrt(np.finfo(np.float64).eps) * scale
    if np.any(eigenvalues <= tolerance):
        return np.full(n_parameters, np.nan, dtype=np.float64)

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.full(n_parameters, np.nan, dtype=np.float64)

    variances = np.diag(covariance)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        return np.full(n_parameters, np.nan, dtype=np.float64)
    return np.sqrt(variances)


def compute_expected_variance(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_items: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected values and variances for all items.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    theta : array of shape (n_persons, n_factors)
        Person ability estimates.
    n_items : int
        Number of items.

    Returns
    -------
    expected : array of shape (n_persons, n_items)
        Expected scores for each person-item combination.
    variance : array of shape (n_persons, n_items)
        Variance of scores for each person-item combination.
    """
    if isinstance(n_items, (bool, np.bool_)) or not isinstance(
        n_items, (int, np.integer)
    ):
        raise ValueError("n_items must be an integer")
    if int(n_items) != model.n_items:
        raise ValueError(
            f"n_items ({n_items}) must match model.n_items ({model.n_items})"
        )

    theta_array = np.asarray(theta, dtype=np.float64)
    if theta_array.ndim != 2 or theta_array.shape[0] == 0:
        raise ValueError("theta must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values")

    probabilities = np.asarray(model.probability(theta_array), dtype=np.float64)
    n_persons = theta_array.shape[0]

    if model.is_polytomous:
        if probabilities.ndim != 3 or probabilities.shape[:2] != (
            n_persons,
            model.n_items,
        ):
            raise ValueError("model returned invalid polytomous probabilities")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < -_PROBABILITY_TOLERANCE)
            | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
        ):
            raise ValueError("model returned probabilities outside [0, 1]")

        np.clip(probabilities, 0.0, 1.0, out=probabilities)
        probability_mass = np.sum(probabilities, axis=2, keepdims=True)
        if np.any(np.abs(probability_mass - 1.0) > _PROBABILITY_TOLERANCE):
            raise ValueError("model category probabilities must sum to one")
        probabilities /= probability_mass

        categories = np.arange(probabilities.shape[2], dtype=np.float64)
        expected = probabilities @ categories
        expected_squared = probabilities @ (categories**2)
        variance = np.maximum(expected_squared - expected**2, 0.0)
        return expected, variance

    if probabilities.shape != (n_persons, model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -_PROBABILITY_TOLERANCE)
        | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")

    expected = np.clip(probabilities, 0.0, 1.0, out=probabilities)
    return expected, expected * (1.0 - expected)


def compute_fit_stats(
    responses: NDArray[np.int_],
    expected: NDArray[np.float64],
    variance: NDArray[np.float64],
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute infit and outfit statistics.

    Parameters
    ----------
    responses : array
        Observed responses.
    expected : array
        Expected responses.
    variance : array
        Variance of responses.
    axis : int
        Axis along which to compute statistics (0 for items, 1 for persons).

    Returns
    -------
    infit : array
        Infit mean square statistics.
    outfit : array
        Outfit mean square statistics.
    """
    if isinstance(axis, (bool, np.bool_)) or axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    response_array = np.asarray(responses)
    expected_array = np.asarray(expected, dtype=np.float64)
    variance_array = np.asarray(variance, dtype=np.float64)
    if response_array.ndim != 2:
        raise ValueError("responses must be a two-dimensional array")
    if response_array.dtype.kind not in "biuf" or not np.all(
        np.isfinite(response_array)
    ):
        raise ValueError("responses must contain only finite numeric values")
    if expected_array.shape != response_array.shape:
        raise ValueError("expected must have the same shape as responses")
    if variance_array.shape != response_array.shape:
        raise ValueError("variance must have the same shape as responses")
    if not np.all(np.isfinite(expected_array)):
        raise ValueError("expected must contain only finite values")
    if not np.all(np.isfinite(variance_array)) or np.any(
        variance_array < -_PROBABILITY_TOLERANCE
    ):
        raise ValueError("variance must contain finite non-negative values")

    variance_array = np.maximum(variance_array, 0.0)
    valid_mask = response_array >= 0
    residuals_squared = (response_array - expected_array) ** 2

    outfit_mask = valid_mask & (variance_array > PROB_EPSILON)
    standardized_squared = np.zeros_like(variance_array)
    np.divide(
        residuals_squared,
        variance_array,
        out=standardized_squared,
        where=outfit_mask,
    )
    outfit_count = np.sum(outfit_mask, axis=axis)
    outfit_sum = np.sum(standardized_squared, axis=axis)
    outfit = np.full_like(outfit_sum, np.nan, dtype=np.float64)
    np.divide(outfit_sum, outfit_count, out=outfit, where=outfit_count > 0)

    infit_numerator = np.sum(np.where(valid_mask, residuals_squared, 0.0), axis=axis)
    infit_denominator = np.sum(np.where(valid_mask, variance_array, 0.0), axis=axis)
    infit = np.full_like(infit_numerator, np.nan, dtype=np.float64)
    np.divide(
        infit_numerator,
        infit_denominator,
        out=infit,
        where=infit_denominator > PROB_EPSILON,
    )

    return np.asarray(infit), np.asarray(outfit)
