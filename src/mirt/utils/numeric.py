"""Shared numeric utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10
_FIT_TARGET_CHUNK_ELEMENTS = 262_144


@lru_cache(maxsize=16)
def standard_normal_quadrature(
    n_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return immutable Gauss--Hermite nodes and standard-normal weights."""
    if (
        isinstance(n_points, (bool, np.bool_))
        or not isinstance(n_points, (int, np.integer))
        or n_points < 1
    ):
        raise ValueError("n_points must be a positive integer")

    nodes, weights = hermgauss(int(n_points))
    normal_nodes = np.asarray(nodes * np.sqrt(2.0), dtype=np.float64)
    normal_weights = np.asarray(weights / np.sqrt(np.pi), dtype=np.float64)
    normal_nodes.setflags(write=False)
    normal_weights.setflags(write=False)
    return normal_nodes, normal_weights


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


def compute_probability_moments(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_items: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate probabilities and compute score moments for all items.

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
    probabilities : array
        Validated probabilities. Dichotomous models return shape
        ``(n_persons, n_items)`` and polytomous models return shape
        ``(n_persons, n_items, n_categories)``.
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
        return probabilities, expected, variance

    if probabilities.shape != (n_persons, model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -_PROBABILITY_TOLERANCE)
        | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")

    expected = np.clip(probabilities, 0.0, 1.0, out=probabilities)
    return probabilities, expected, expected * (1.0 - expected)


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
    _, expected, variance = compute_probability_moments(model, theta, n_items)
    return expected, variance


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

    Notes
    -----
    Temporary arrays are limited to a block of rows (at least one row).
    Missing responses and zero-variance entries retain separate infit and
    outfit eligibility rules; final ratios use sums across all blocks.
    """
    if isinstance(axis, (bool, np.bool_)) or axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    response_array = np.asarray(responses)
    expected_array = np.asarray(expected, dtype=np.float64)
    variance_array = np.asarray(variance, dtype=np.float64)
    if response_array.ndim != 2:
        raise ValueError("responses must be a two-dimensional array")
    if response_array.dtype.kind not in "biuf":
        raise ValueError("responses must contain only finite numeric values")
    if expected_array.shape != response_array.shape:
        raise ValueError("expected must have the same shape as responses")
    if variance_array.shape != response_array.shape:
        raise ValueError("variance must have the same shape as responses")

    n_persons, n_items = response_array.shape
    accumulator = _FitStatsAccumulator(response_array.shape[1 - axis])
    rows_per_chunk = max(1, _FIT_TARGET_CHUNK_ELEMENTS // max(n_items, 1))

    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        accumulator.add(
            response_array[start:stop],
            expected_array[start:stop],
            variance_array[start:stop],
            axis=axis,
            target=slice(start, stop) if axis == 1 else slice(None),
        )

    return accumulator.finish()


class _FitStatsAccumulator:
    """Accumulate bounded response blocks without averaging partial ratios."""

    def __init__(self, output_size: int) -> None:
        self.infit_numerator = np.zeros(output_size)
        self.infit_denominator = np.zeros(output_size)
        self.outfit_sum = np.zeros(output_size)
        self.outfit_count = np.zeros(output_size, dtype=np.intp)

    def add(
        self,
        responses: NDArray,
        expected: NDArray[np.float64],
        variance: NDArray[np.float64],
        *,
        axis: int = 0,
        target: slice = slice(None),
    ) -> None:
        """Add an aligned block to item totals or a slice of person totals."""
        if responses.dtype.kind not in "biuf" or not np.all(np.isfinite(responses)):
            raise ValueError("responses must contain only finite numeric values")
        if not np.all(np.isfinite(expected)):
            raise ValueError("expected must contain only finite values")
        if not np.all(np.isfinite(variance)) or np.any(
            variance < -_PROBABILITY_TOLERANCE
        ):
            raise ValueError("variance must contain finite non-negative values")

        block_variance = np.maximum(variance, 0.0)
        valid = responses >= 0
        squared = np.where(valid, responses, expected)
        squared -= expected
        np.square(squared, out=squared)
        self.infit_numerator[target] += np.sum(squared, axis=axis)
        block_variance = np.where(valid, block_variance, 0.0)
        self.infit_denominator[target] += np.sum(block_variance, axis=axis)

        eligible = block_variance > PROB_EPSILON
        self.outfit_count[target] += np.sum(eligible, axis=axis)
        squared = np.where(eligible, squared, 0.0)
        np.maximum(block_variance, PROB_EPSILON, out=block_variance)
        np.divide(squared, block_variance, out=squared)
        self.outfit_sum[target] += np.sum(squared, axis=axis)

    def finish(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply eligibility thresholds after all blocks have accumulated."""
        outfit = np.full_like(self.outfit_sum, np.nan)
        np.divide(
            self.outfit_sum,
            self.outfit_count,
            out=outfit,
            where=self.outfit_count > 0,
        )

        infit = np.full_like(self.infit_numerator, np.nan)
        np.divide(
            self.infit_numerator,
            self.infit_denominator,
            out=infit,
            where=self.infit_denominator > PROB_EPSILON,
        )

        return infit, outfit
