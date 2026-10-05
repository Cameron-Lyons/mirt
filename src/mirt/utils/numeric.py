"""Shared numeric utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


def _fourth_central_moment(
    probabilities: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the fourth central moment of each modeled item score.

    Parameters
    ----------
    probabilities : array
        Category probabilities. Binary items use the positive-response
        probability with the same shape as ``expected``; polytomous items add a
        trailing, zero-padded category axis.
    expected : array
        Expected item scores.

    Returns
    -------
    array
        ``E[(X - E[X])^4]`` with the shape of ``expected``.
    """
    if probabilities.shape == expected.shape:
        variance = probabilities * (1.0 - probabilities)
        return variance * (1.0 - 3.0 * variance)
    deviations = np.arange(probabilities.shape[-1], dtype=np.float64)
    deviations = np.square(deviations - expected[..., None])
    return np.einsum("...k,...k->...", probabilities, np.square(deviations))


def _wilson_hilferty_z(
    mean_square: NDArray[np.float64],
    variance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Standardize mean-square fit statistics with the cube-root transform.

    ``t = (MS^(1/3) - 1)(3/q) + q/3`` where ``q^2`` is the modeled variance of
    the mean square (Wright & Masters, 1982). Entries with an undefined mean
    square or a nonpositive variance are ``NaN``.
    """
    result = np.full(np.shape(mean_square), np.nan)
    defined = np.isfinite(mean_square) & np.isfinite(variance) & (variance > 0.0)
    q = np.sqrt(variance[defined])
    result[defined] = (np.cbrt(mean_square[defined]) - 1.0) * (3.0 / q) + q / 3.0
    return result


@dataclass(frozen=True)
class _FitCellTerms:
    """Per-cell mean-square terms shared by item and person reductions."""

    observed: NDArray[np.bool_]
    squared: NDArray[np.float64]
    variance: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    outfit: NDArray[np.float64]
    infit_kurtosis: NDArray[np.float64] | None = None
    outfit_kurtosis: NDArray[np.float64] | None = None


def _fit_cell_terms(
    responses: NDArray,
    expected: NDArray[np.float64],
    variance: NDArray[np.float64],
    fourth_moment: NDArray[np.float64] | None = None,
) -> _FitCellTerms:
    """Validate a block and return its infit, outfit and kurtosis terms.

    Missing (negative) responses contribute nothing. Outfit and its variance
    use only observed entries whose variance exceeds ``PROB_EPSILON``.
    """
    if responses.dtype.kind not in "biuf" or not np.all(np.isfinite(responses)):
        raise ValueError("responses must contain only finite numeric values")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected must contain only finite values")
    if not np.all(np.isfinite(variance)) or np.any(variance < -_PROBABILITY_TOLERANCE):
        raise ValueError("variance must contain finite non-negative values")
    if fourth_moment is not None and not np.all(np.isfinite(fourth_moment)):
        raise ValueError("fourth_moment must contain only finite values")

    observed = responses >= 0
    squared = np.where(observed, responses, expected)
    squared -= expected
    np.square(squared, out=squared)
    cell_variance = np.where(observed, np.maximum(variance, 0.0), 0.0)
    eligible = cell_variance > PROB_EPSILON
    outfit = np.divide(
        squared, cell_variance, out=np.zeros_like(squared), where=eligible
    )
    if fourth_moment is None:
        return _FitCellTerms(observed, squared, cell_variance, eligible, outfit)
    fourth = np.where(observed, np.maximum(fourth_moment, 0.0), 0.0)
    squared_variance = np.square(cell_variance)
    return _FitCellTerms(
        observed,
        squared,
        cell_variance,
        eligible,
        outfit,
        infit_kurtosis=fourth - squared_variance,
        outfit_kurtosis=np.divide(
            fourth, squared_variance, out=np.zeros_like(fourth), where=eligible
        ),
    )


class _FitStatsAccumulator:
    """Accumulate bounded response blocks without averaging partial ratios.

    With ``standardized=True`` the accumulator also sums the fourth-moment
    terms that give the modeled variance of each mean square, so
    :meth:`standardized` can return Wilson-Hilferty z statistics.
    """

    def __init__(self, output_size: int, *, standardized: bool = False) -> None:
        self.infit_numerator = np.zeros(output_size)
        self.infit_denominator = np.zeros(output_size)
        self.outfit_sum = np.zeros(output_size)
        self.outfit_count = np.zeros(output_size, dtype=np.intp)
        self.infit_kurtosis = np.zeros(output_size) if standardized else None
        self.outfit_kurtosis = np.zeros(output_size) if standardized else None

    def add(
        self,
        responses: NDArray,
        expected: NDArray[np.float64],
        variance: NDArray[np.float64],
        *,
        axis: int = 0,
        target: slice = slice(None),
        fourth_moment: NDArray[np.float64] | None = None,
    ) -> None:
        """Add an aligned block to item totals or a slice of person totals."""
        self.add_terms(
            _fit_cell_terms(responses, expected, variance, fourth_moment),
            axis=axis,
            target=target,
        )

    def add_terms(
        self,
        terms: _FitCellTerms,
        *,
        axis: int = 0,
        target: slice = slice(None),
    ) -> None:
        """Reduce precomputed cell terms along ``axis`` into ``target``."""
        self.infit_numerator[target] += np.sum(terms.squared, axis=axis)
        self.infit_denominator[target] += np.sum(terms.variance, axis=axis)
        self.outfit_count[target] += np.count_nonzero(terms.eligible, axis=axis)
        self.outfit_sum[target] += np.sum(terms.outfit, axis=axis)
        if self.infit_kurtosis is None or self.outfit_kurtosis is None:
            return
        if terms.infit_kurtosis is None or terms.outfit_kurtosis is None:
            raise ValueError("fourth_moment is required for standardized statistics")
        self.infit_kurtosis[target] += np.sum(terms.infit_kurtosis, axis=axis)
        self.outfit_kurtosis[target] += np.sum(terms.outfit_kurtosis, axis=axis)

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

    def statistics(self) -> dict[str, NDArray[np.float64]]:
        """Return outfit and infit, each followed by its z statistic if tracked.

        Standardized accumulators add Wilson-Hilferty ``z_outfit`` and
        ``z_infit``. The outfit variance is ``sum(C / W^2) / N^2 - 1 / N`` over
        the entries that enter the outfit mean, and the infit variance is
        ``sum(C - W^2) / (sum W)^2`` over observed entries, where ``W`` and
        ``C`` are the second and fourth central moments of each modeled score.
        """
        infit, outfit = self.finish()
        if self.infit_kurtosis is None or self.outfit_kurtosis is None:
            return {"outfit": outfit, "infit": infit}
        count = self.outfit_count.astype(np.float64)
        outfit_variance = np.full_like(self.outfit_sum, np.nan)
        np.divide(
            self.outfit_kurtosis / np.maximum(count, 1.0) - 1.0,
            count,
            out=outfit_variance,
            where=count > 0,
        )
        infit_variance = np.full_like(self.infit_numerator, np.nan)
        np.divide(
            self.infit_kurtosis,
            np.square(self.infit_denominator),
            out=infit_variance,
            where=self.infit_denominator > PROB_EPSILON,
        )
        return {
            "outfit": outfit,
            "z_outfit": _wilson_hilferty_z(outfit, outfit_variance),
            "infit": infit,
            "z_infit": _wilson_hilferty_z(infit, infit_variance),
        }
