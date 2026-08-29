"""Nonparametric and semiparametric item response models.

The implementations in this module provide monotone spline and Bernstein
response curves plus kernel-smoothed empirical item response functions.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.special import comb, expit

from mirt.constants import PROB_EPSILON
from mirt.models.base import DichotomousItemModel

_POWER_BASIS_MAX_DEGREE = 12


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _item_index(item_idx: int, n_items: int) -> int:
    if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
        item_idx, (int, np.integer)
    ):
        raise TypeError("item_idx must be an integer")
    result = int(item_idx)
    if result < 0 or result >= n_items:
        raise IndexError(f"item_idx {result} out of range [0, {n_items})")
    return result


def _theta_vector(
    model: DichotomousItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = model._ensure_theta_2d(theta).ravel()
    if not np.all(np.isfinite(result)):
        raise ValueError("theta must contain only finite values")
    return result


def _parameter_update(
    parameters: dict[str, NDArray[np.float64]],
    updates: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Validate and copy a parameter update without partial mutation."""
    result = {name: values.copy() for name, values in parameters.items()}
    for name, values in updates.items():
        if name not in parameters:
            valid = ", ".join(parameters)
            raise ValueError(f"Unknown parameter: {name}. Valid parameters: {valid}")
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != parameters[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {parameters[name].shape}, "
                f"got {values_array.shape}"
            )
        if not np.all(np.isfinite(values_array)):
            raise ValueError(f"{name} must contain only finite values")
        result[name] = values_array.copy()
    return result


def _item_parameter_update(
    parameters: dict[str, NDArray[np.float64]],
    n_items: int,
    item_idx: int,
    param_name: str,
    value: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a validated-shape single-item parameter update."""
    item_idx = _item_index(item_idx, n_items)
    if param_name not in parameters:
        valid = ", ".join(parameters)
        raise ValueError(f"Unknown parameter: {param_name}. Valid parameters: {valid}")
    values = parameters[param_name].copy()
    value_array = np.asarray(value, dtype=np.float64)
    if values.ndim == 1 and values.shape == (n_items,):
        if value_array.ndim != 0:
            raise ValueError(f"{param_name} requires a scalar item value")
        values[item_idx] = float(value_array)
    elif values.ndim == 2 and values.shape[0] == n_items:
        if value_array.shape != values.shape[1:]:
            raise ValueError(
                f"{param_name} item value shape {value_array.shape} != "
                f"{values.shape[1:]}"
            )
        values[item_idx] = value_array
    else:
        raise ValueError(f"{param_name} does not contain per-item values")
    return values


def _validate_curve_bounds(parameters: dict[str, NDArray[np.float64]]) -> None:
    lower = parameters["lower"]
    upper = parameters["upper"]
    if np.any(lower < 0) or np.any(lower >= 1):
        raise ValueError("lower must be in [0, 1)")
    if np.any(upper <= 0) or np.any(upper > 1):
        raise ValueError("upper must be in (0, 1]")
    if np.any(lower >= upper):
        raise ValueError("lower must be strictly less than upper for every item")


def _relative_positive(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate log values after removing an unidentified row scale."""
    centered = log_values - np.max(log_values, axis=1, keepdims=True)
    centered = np.maximum(centered, np.log(np.nextafter(0.0, 1.0)))
    return np.exp(centered)


def _stable_gaussian_weights(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    bandwidth: float,
    sample_weight: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return stable Gaussian and person weights relative to each grid maximum."""
    sample_matrix = samples[:, None]
    grid_matrix = grid[None, :]
    log_sample_weight = np.zeros(samples.size, dtype=np.float64)
    if sample_weight is not None:
        log_sample_weight.fill(-np.inf)
        np.log(
            sample_weight,
            out=log_sample_weight,
            where=sample_weight > 0.0,
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_distance = (sample_matrix - grid_matrix) / bandwidth
        log_weights = -0.5 * scaled_distance**2 + log_sample_weight[:, None]
    column_maximum = np.max(log_weights, axis=0)
    if np.all(np.isfinite(column_maximum)):
        return np.exp(log_weights - column_maximum)

    scale = np.maximum(np.abs(sample_matrix), np.abs(grid_matrix))
    nonzero_scale = scale > 0
    scaled_samples = np.divide(
        sample_matrix, scale, out=np.zeros_like(scale), where=nonzero_scale
    )
    scaled_grid = np.divide(
        grid_matrix, scale, out=np.zeros_like(scale), where=nonzero_scale
    )
    normalized_distance = np.abs(scaled_samples - scaled_grid)
    with np.errstate(divide="ignore"):
        log_distance = np.log(scale) + np.log(normalized_distance)

    nearest_log_distance = np.min(log_distance, axis=0)
    non_nearest = log_distance > nearest_log_distance[None, :]
    log_squared_gap = np.full_like(log_distance, -np.inf)
    relative_log_square = np.zeros_like(log_distance)
    np.subtract(
        nearest_log_distance[None, :],
        log_distance,
        out=relative_log_square,
        where=non_nearest,
    )
    relative_log_square *= 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_squared_gap[non_nearest] = 2.0 * log_distance[non_nearest] + np.log1p(
            -np.exp(relative_log_square[non_nearest])
        )

    log_penalty = log_squared_gap - np.log(2.0) - 2.0 * np.log(bandwidth)
    max_log_penalty = np.log(-np.log(np.nextafter(0.0, 1.0)))
    penalty = np.exp(np.minimum(log_penalty, max_log_penalty))
    log_weights = -penalty + log_sample_weight[:, None]
    column_maximum = np.max(log_weights, axis=0)
    return np.exp(log_weights - column_maximum)


def _fisher_information(
    probability: NDArray[np.float64],
    derivative: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return dichotomous Fisher information from an exact curve derivative."""
    return np.square(derivative) / (probability * (1.0 - probability) + PROB_EPSILON)


class MonotonicSplineModel(DichotomousItemModel):
    """Monotone item response curves represented by I-spline bases.

    The basis functions are exact normalized integrals of B-splines over
    ``[-4, 4]``. Positive item weights therefore produce nondecreasing curves.
    Values outside that interval use the corresponding saturated endpoint.
    """

    model_name = "MonotonicSpline"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_knots: int = 4,
        degree: int = 3,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Spline model only supports unidimensional analysis")
        self.n_knots = _positive_integer(n_knots, "n_knots")
        self.degree = _positive_integer(degree, "degree")
        self._n_basis = self.n_knots + self.degree + 1
        self._knots = np.empty(0, dtype=np.float64)
        self._mspline_basis: BSpline
        self._ispline_antiderivative: BSpline
        self._ispline_origin = np.empty(0, dtype=np.float64)
        self._ispline_scale = np.empty(0, dtype=np.float64)
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["log_weights"] = np.zeros((self.n_items, self._n_basis))
        self._parameters["lower"] = np.zeros(self.n_items)
        self._parameters["upper"] = np.ones(self.n_items)

        self._knots = np.linspace(-3.0, 3.0, self.n_knots + 2)[1:-1]
        knot_vector = np.concatenate(
            [
                np.full(self.degree + 1, -4.0),
                self._knots,
                np.full(self.degree + 1, 4.0),
            ]
        )
        self._mspline_basis = BSpline(
            knot_vector,
            np.eye(self._n_basis),
            self.degree,
            extrapolate=False,
        )
        self._ispline_antiderivative = self._mspline_basis.antiderivative()
        self._ispline_origin = np.asarray(
            self._ispline_antiderivative(-4.0), dtype=np.float64
        )
        self._ispline_scale = np.asarray(
            self._ispline_antiderivative(4.0) - self._ispline_origin,
            dtype=np.float64,
        )

    @property
    def knots(self) -> NDArray[np.float64]:
        """Interior knots used by the spline basis."""
        return self._knots.copy()

    @property
    def weights(self) -> NDArray[np.float64]:
        """Finite positive relative spline weights."""
        return _relative_positive(self._parameters["log_weights"])

    @property
    def lower(self) -> NDArray[np.float64]:
        return self._parameters["lower"].copy()

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"].copy()

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        updated = _parameter_update(self._parameters, params)
        _validate_curve_bounds(updated)
        self._parameters = updated
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        values = _item_parameter_update(
            self._parameters, self.n_items, item_idx, param_name, value
        )
        self.set_parameters(**{param_name: values})

    def _basis_matrix(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped_theta = np.clip(np.asarray(theta, dtype=np.float64).ravel(), -4.0, 4.0)
        integrated = self._ispline_antiderivative(clipped_theta) - self._ispline_origin
        basis = integrated / self._ispline_scale
        return np.clip(basis, 0.0, 1.0)

    def _basis_derivative_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate exact derivatives of the normalized I-spline bases."""
        theta_vector = np.asarray(theta, dtype=np.float64).ravel()
        clipped_theta = np.clip(theta_vector, -4.0, 4.0)
        derivative = (
            np.asarray(self._mspline_basis(clipped_theta)) / self._ispline_scale
        )
        outside = (theta_vector < -4.0) | (theta_vector > 4.0)
        derivative[outside] = 0.0
        return derivative

    def _ispline_basis(
        self,
        theta: NDArray[np.float64],
        knot_idx: int,
    ) -> NDArray[np.float64]:
        """Compute one normalized I-spline basis function."""
        knot_idx = _item_index(knot_idx, self._n_basis)
        return self._basis_matrix(theta)[:, knot_idx]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta_vector = _theta_vector(self, theta)
        basis = self._basis_matrix(theta_vector)
        weights = self.weights
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            curve = basis @ normalized_weights[item_idx]
            return lower[item_idx] + (upper[item_idx] - lower[item_idx]) * curve

        curves = basis @ normalized_weights.T
        return lower + (upper - lower) * curves

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta_vector = _theta_vector(self, theta)
        basis = self._basis_matrix(theta_vector)
        basis_derivative = self._basis_derivative_matrix(theta_vector)
        weights = self.weights
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            scale = upper[item_idx] - lower[item_idx]
            probability = lower[item_idx] + scale * (
                basis @ normalized_weights[item_idx]
            )
            derivative = scale * (basis_derivative @ normalized_weights[item_idx])
            return _fisher_information(probability, derivative)

        scales = upper - lower
        probability = lower + scales * (basis @ normalized_weights.T)
        derivative = scales * (basis_derivative @ normalized_weights.T)
        return _fisher_information(probability, derivative)

    def copy(self) -> Self:
        new_model = MonotonicSplineModel(
            n_items=self.n_items,
            n_knots=self.n_knots,
            degree=self.degree,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class MonotonicPolynomialModel(DichotomousItemModel):
    """Monotone item response curves represented in a Bernstein basis.

    Positive increments are accumulated into ordered Bernstein coefficients.
    Ordered coefficients and a positive scale guarantee nondecreasing curves.
    """

    model_name = "MonotonicPolynomial"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        degree: int = 5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Polynomial model only supports unidimensional analysis")
        self.degree = _positive_integer(degree, "degree")
        self._bernstein_to_power = np.empty((0, 0), dtype=np.float64)
        super().__init__(n_items, n_factors=1, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["log_coefficients"] = np.zeros((self.n_items, self.degree + 1))
        self._parameters["location"] = np.zeros(self.n_items)
        self._parameters["scale"] = np.ones(self.n_items)
        self._parameters["lower"] = np.zeros(self.n_items)
        self._parameters["upper"] = np.ones(self.n_items)
        if self.degree <= _POWER_BASIS_MAX_DEGREE:
            self._bernstein_to_power = np.zeros(
                (self.degree + 1, self.degree + 1), dtype=np.float64
            )
            for power in range(self.degree + 1):
                for basis_idx in range(power + 1):
                    self._bernstein_to_power[power, basis_idx] = (
                        comb(self.degree, power)
                        * comb(power, basis_idx)
                        * (-1.0) ** (power - basis_idx)
                    )

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Ordered Bernstein coefficients in ``(0, 1]``."""
        increments = _relative_positive(self._parameters["log_coefficients"])
        return np.cumsum(increments, axis=1) / np.sum(increments, axis=1, keepdims=True)

    @property
    def location(self) -> NDArray[np.float64]:
        return self._parameters["location"].copy()

    @property
    def scale(self) -> NDArray[np.float64]:
        return self._parameters["scale"].copy()

    @property
    def lower(self) -> NDArray[np.float64]:
        return self._parameters["lower"].copy()

    @property
    def upper(self) -> NDArray[np.float64]:
        return self._parameters["upper"].copy()

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        updated = _parameter_update(self._parameters, params)
        _validate_curve_bounds(updated)
        if np.any(updated["scale"] <= 0):
            raise ValueError("scale must be positive for every item")
        self._parameters = updated
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        values = _item_parameter_update(
            self._parameters, self.n_items, item_idx, param_name, value
        )
        self.set_parameters(**{param_name: values})

    def _bernstein_basis(
        self,
        t: NDArray[np.float64],
        k: int,
        n: int,
    ) -> NDArray[np.float64]:
        """Compute one Bernstein basis polynomial."""
        if n < 1:
            raise ValueError("n must be positive")
        k = _item_index(k, n + 1)
        clipped = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        return comb(n, k) * (clipped**k) * ((1.0 - clipped) ** (n - k))

    def _basis_matrix(
        self, transformed_theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._basis_matrix_for_degree(transformed_theta, self.degree)

    @staticmethod
    def _basis_matrix_for_degree(
        transformed_theta: NDArray[np.float64],
        degree: int,
    ) -> NDArray[np.float64]:
        """Evaluate a complete Bernstein basis of the requested degree."""
        t = np.clip(np.asarray(transformed_theta, dtype=np.float64), 0.0, 1.0)
        orders = np.arange(degree + 1)
        binomial = comb(degree, orders)
        return (
            binomial
            * (t[..., None] ** orders)
            * ((1.0 - t[..., None]) ** (degree - orders))
        )

    def _evaluate_bernstein(
        self,
        transformed_theta: NDArray[np.float64],
        coefficients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate Bernstein curves with a degree-appropriate stable form."""
        t = np.asarray(transformed_theta, dtype=np.float64)
        if self.degree > _POWER_BASIS_MAX_DEGREE:
            if coefficients.ndim == 1:
                result = self._basis_matrix(t) @ coefficients
                return np.clip(result, coefficients[0], coefficients[-1])
            result = np.empty_like(t)
            for item_idx in range(self.n_items):
                result[:, item_idx] = (
                    self._basis_matrix(t[:, item_idx]) @ coefficients[item_idx]
                )
            return np.clip(result, coefficients[:, 0], coefficients[:, -1])

        if coefficients.ndim == 1:
            power_coefficients = self._bernstein_to_power @ coefficients
            result = np.full_like(t, power_coefficients[-1])
            for power in range(self.degree - 1, -1, -1):
                result = result * t + power_coefficients[power]
            return np.clip(result, coefficients[0], coefficients[-1])

        power_coefficients = coefficients @ self._bernstein_to_power.T
        result = np.broadcast_to(power_coefficients[:, -1], t.shape).copy()
        for power in range(self.degree - 1, -1, -1):
            result *= t
            result += power_coefficients[:, power]
        return np.clip(result, coefficients[:, 0], coefficients[:, -1])

    def _evaluate_bernstein_derivative(
        self,
        transformed_theta: NDArray[np.float64],
        coefficients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the exact derivative of Bernstein curves with respect to t."""
        t = np.asarray(transformed_theta, dtype=np.float64)
        if self.degree > _POWER_BASIS_MAX_DEGREE:
            derivative_coefficients = self.degree * np.diff(coefficients, axis=-1)
            if coefficients.ndim == 1:
                return (
                    self._basis_matrix_for_degree(t, self.degree - 1)
                    @ derivative_coefficients
                )
            result = np.empty_like(t)
            for item_idx in range(self.n_items):
                result[:, item_idx] = (
                    self._basis_matrix_for_degree(t[:, item_idx], self.degree - 1)
                    @ derivative_coefficients[item_idx]
                )
            return result

        orders = np.arange(1, self.degree + 1, dtype=np.float64)
        if coefficients.ndim == 1:
            power_coefficients = self._bernstein_to_power @ coefficients
            derivative_coefficients = orders * power_coefficients[1:]
            result = np.full_like(t, derivative_coefficients[-1])
            for power in range(self.degree - 2, -1, -1):
                result = result * t + derivative_coefficients[power]
            return np.maximum(result, 0.0)

        power_coefficients = coefficients @ self._bernstein_to_power.T
        derivative_coefficients = power_coefficients[:, 1:] * orders
        result = np.broadcast_to(derivative_coefficients[:, -1], t.shape).copy()
        for power in range(self.degree - 2, -1, -1):
            result *= t
            result += derivative_coefficients[:, power]
        return np.maximum(result, 0.0)

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta_vector = _theta_vector(self, theta)
        location = self._parameters["location"]
        scale = self._parameters["scale"]
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]
        coefficients = self.coefficients

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            transformed = expit(scale[item_idx] * (theta_vector - location[item_idx]))
            curve = self._evaluate_bernstein(transformed, coefficients[item_idx])
            return lower[item_idx] + (upper[item_idx] - lower[item_idx]) * curve

        transformed = expit(
            scale[None, :] * (theta_vector[:, None] - location[None, :])
        )
        curves = self._evaluate_bernstein(transformed, coefficients)
        return lower + (upper - lower) * curves

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta_vector = _theta_vector(self, theta)
        location = self._parameters["location"]
        scale = self._parameters["scale"]
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]
        coefficients = self.coefficients

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            transformed = expit(scale[item_idx] * (theta_vector - location[item_idx]))
            curve = self._evaluate_bernstein(transformed, coefficients[item_idx])
            curve_derivative = self._evaluate_bernstein_derivative(
                transformed, coefficients[item_idx]
            )
            response_scale = upper[item_idx] - lower[item_idx]
            probability = lower[item_idx] + response_scale * curve
            derivative = (
                response_scale
                * curve_derivative
                * scale[item_idx]
                * transformed
                * (1.0 - transformed)
            )
            return _fisher_information(probability, derivative)

        transformed = expit(
            scale[None, :] * (theta_vector[:, None] - location[None, :])
        )
        curves = self._evaluate_bernstein(transformed, coefficients)
        curve_derivatives = self._evaluate_bernstein_derivative(
            transformed, coefficients
        )
        response_scales = upper - lower
        probability = lower + response_scales * curves
        derivative = (
            response_scales
            * curve_derivatives
            * scale
            * transformed
            * (1.0 - transformed)
        )
        return _fisher_information(probability, derivative)

    def copy(self) -> Self:
        new_model = MonotonicPolynomialModel(
            n_items=self.n_items,
            degree=self.degree,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class KernelSmoothingModel(DichotomousItemModel):
    """Gaussian-kernel empirical item response model.

    Calibration uses a numerically stable Nadaraya-Watson estimator on a
    configurable ability grid. Missing responses may be represented by any
    negative integer code, and person weights support frequency, survey, or
    inverse-probability weighted calibration.
    """

    model_name = "KernelSmoothing"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        bandwidth: float = 0.5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
        theta_grid: NDArray[np.float64] | None = None,
    ) -> None:
        if n_factors != 1:
            raise ValueError("Kernel smoothing only supports unidimensional analysis")
        self.bandwidth = _positive_finite(bandwidth, "bandwidth")
        self._configured_theta_grid = self._validate_theta_grid(theta_grid)
        self._theta_grid = np.empty(0, dtype=np.float64)
        self._irf_values = np.empty((0, 0), dtype=np.float64)
        self._calibration_counts = np.empty(0, dtype=np.intp)
        self._calibration_weight_sums = np.empty(0, dtype=np.float64)
        super().__init__(n_items, n_factors=1, item_names=item_names)

    @staticmethod
    def _validate_theta_grid(
        theta_grid: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if theta_grid is None:
            return np.linspace(-4.0, 4.0, 81)
        result = np.asarray(theta_grid, dtype=np.float64)
        if result.ndim != 1 or result.size < 2:
            raise ValueError("theta_grid must be 1D with at least two points")
        if not np.all(np.isfinite(result)):
            raise ValueError("theta_grid must contain only finite values")
        if np.any(result[1:] <= result[:-1]):
            raise ValueError("theta_grid must be strictly increasing")
        return result.copy()

    def _initialize_parameters(self) -> None:
        self._theta_grid = self._configured_theta_grid.copy()
        self._irf_values = np.full((self.n_items, self._theta_grid.size), 0.5)
        self._calibration_counts = np.zeros(self.n_items, dtype=np.intp)
        self._calibration_weight_sums = np.zeros(self.n_items, dtype=np.float64)

    @property
    def theta_grid(self) -> NDArray[np.float64]:
        return self._theta_grid.copy()

    @property
    def irf_values(self) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise ValueError("Model must be calibrated before accessing IRF values")
        return self._irf_values.copy()

    @property
    def calibration_counts(self) -> NDArray[np.intp]:
        """Number of observed responses used for each item."""
        return self._calibration_counts.copy()

    @property
    def calibration_weight_sums(self) -> NDArray[np.float64]:
        """Sum of person weights contributing to each item calibration."""
        return self._calibration_weight_sums.copy()

    def calibrate(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> Self:
        """Calibrate item response functions with stable Gaussian kernels.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Dichotomous responses. Any negative value denotes a missing response.
        theta : ndarray of shape (n_persons,) or (n_persons, 1)
            Finite person ability values.
        sample_weight : ndarray of shape (n_persons,), optional
            Nonnegative finite person weights. Multiplying every weight by the
            same positive constant leaves the fitted curves unchanged. Every
            item must retain positive total weight after excluding its missing
            responses.

        Returns
        -------
        KernelSmoothingModel
            This fitted model.
        """
        responses_array = np.asarray(responses)
        if responses_array.ndim != 2:
            raise ValueError("responses must be a 2D array")
        if responses_array.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {responses_array.shape[1]} items; "
                f"expected {self.n_items}"
            )
        if responses_array.shape[0] == 0:
            raise ValueError("responses must contain at least one person")
        if responses_array.dtype.kind not in "biuf":
            raise ValueError("responses must contain numeric response codes")
        if responses_array.dtype.kind == "f" and (
            not np.all(np.isfinite(responses_array))
            or not np.all(responses_array == np.trunc(responses_array))
        ):
            raise ValueError("responses must contain finite integer response codes")
        valid = responses_array >= 0
        if np.any(valid & (responses_array != 0) & (responses_array != 1)):
            raise ValueError(
                "responses must contain only 0, 1, or negative missing codes"
            )

        theta_array = np.asarray(theta, dtype=np.float64)
        if theta_array.ndim == 2 and theta_array.shape[1] == 1:
            theta_array = theta_array[:, 0]
        elif theta_array.ndim != 1:
            raise ValueError("theta must be 1D or a single-column 2D array")
        if theta_array.size != responses_array.shape[0]:
            raise ValueError(
                f"theta length ({theta_array.size}) must match response rows "
                f"({responses_array.shape[0]})"
            )
        if not np.all(np.isfinite(theta_array)):
            raise ValueError("theta must contain only finite values")

        if sample_weight is None:
            sample_weight_array = np.ones(theta_array.size, dtype=np.float64)
        else:
            sample_weight_array = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight_array.shape != (theta_array.size,):
                raise ValueError(f"sample_weight must have shape ({theta_array.size},)")
            if not np.all(np.isfinite(sample_weight_array)):
                raise ValueError("sample_weight must contain only finite values")
            if np.any(sample_weight_array < 0.0):
                raise ValueError("sample_weight must be nonnegative")

        counts = np.sum(valid, axis=0).astype(np.intp, copy=False)
        if np.any(counts == 0):
            missing_items = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"items without observed responses: {missing_items}")
        all_observed = bool(np.all(valid))
        valid_values: NDArray[np.float64] | None = None
        with np.errstate(over="ignore", invalid="ignore"):
            if all_observed:
                calibration_weight_sums = np.full(
                    self.n_items,
                    np.sum(sample_weight_array),
                )
            else:
                valid_values = valid.astype(np.float64)
                calibration_weight_sums = valid_values.T @ sample_weight_array
        if not np.all(np.isfinite(calibration_weight_sums)):
            raise ValueError("sample_weight totals must be finite; rescale the weights")
        if np.any(calibration_weight_sums <= 0.0):
            missing_items = np.flatnonzero(calibration_weight_sums <= 0.0).tolist()
            raise ValueError(
                f"items without positive calibration weight: {missing_items}"
            )

        kernel_weights = _stable_gaussian_weights(
            theta_array,
            self._theta_grid,
            self.bandwidth,
            sample_weight_array,
        )
        response_values = np.where(valid, responses_array, 0.0)
        if all_observed:
            new_irf_values = (
                response_values.T @ kernel_weights / np.sum(kernel_weights, axis=0)
            )
        else:
            assert valid_values is not None
            denominator = valid_values.T @ kernel_weights
            new_irf_values = np.divide(
                response_values.T @ kernel_weights,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0.0,
            )
            fallback_items = np.flatnonzero(
                np.any(
                    (denominator <= np.finfo(np.float64).tiny)
                    | ~np.isfinite(denominator),
                    axis=1,
                )
            )
            for item_idx in fallback_items:
                item_valid = valid[:, item_idx] & (sample_weight_array > 0.0)
                item_kernel_weights = _stable_gaussian_weights(
                    theta_array[item_valid],
                    self._theta_grid,
                    self.bandwidth,
                    sample_weight_array[item_valid],
                )
                new_irf_values[item_idx] = (
                    response_values[item_valid, item_idx]
                    @ item_kernel_weights
                    / np.sum(item_kernel_weights, axis=0)
                )

        self._irf_values = np.clip(new_irf_values, 0.0, 1.0)
        self._calibration_counts = counts
        self._calibration_weight_sums = calibration_weight_sums
        self._is_fitted = True
        return self

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise ValueError("Model must be calibrated before computing probabilities")
        theta_vector = _theta_vector(self, theta)
        left, right, fraction, _ = self._interpolation_components(theta_vector)

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            lower_values = self._irf_values[item_idx, left]
            upper_values = self._irf_values[item_idx, right]
            return lower_values + fraction * (upper_values - lower_values)

        lower_values = self._irf_values[:, left].T
        upper_values = self._irf_values[:, right].T
        return lower_values + fraction[:, None] * (upper_values - lower_values)

    def _interpolation_components(
        self,
        theta_vector: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """Resolve piecewise-linear interpolation positions and active slopes."""
        clipped_theta = np.clip(theta_vector, self._theta_grid[0], self._theta_grid[-1])
        left = np.searchsorted(self._theta_grid, clipped_theta, side="right") - 1
        left = np.clip(left, 0, self._theta_grid.size - 2)
        right = left + 1
        fraction = (clipped_theta - self._theta_grid[left]) / (
            self._theta_grid[right] - self._theta_grid[left]
        )
        active = (theta_vector >= self._theta_grid[0]) & (
            theta_vector <= self._theta_grid[-1]
        )
        return left, right, fraction, active

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        if not self._is_fitted:
            raise ValueError("Model must be calibrated before computing information")
        theta_vector = _theta_vector(self, theta)
        left, right, fraction, active = self._interpolation_components(theta_vector)
        interval_width = self._theta_grid[right] - self._theta_grid[left]

        if item_idx is not None:
            item_idx = _item_index(item_idx, self.n_items)
            lower_values = self._irf_values[item_idx, left]
            value_difference = self._irf_values[item_idx, right] - lower_values
            probability = lower_values + fraction * value_difference
            derivative = np.divide(value_difference, interval_width)
            derivative[~active] = 0.0
            return _fisher_information(probability, derivative)

        lower_values = self._irf_values[:, left].T
        value_difference = self._irf_values[:, right].T - lower_values
        probability = lower_values + fraction[:, None] * value_difference
        derivative = value_difference / interval_width[:, None]
        derivative[~active] = 0.0
        return _fisher_information(probability, derivative)

    def copy(self) -> Self:
        new_model = KernelSmoothingModel(
            n_items=self.n_items,
            bandwidth=self.bandwidth,
            item_names=self.item_names.copy(),
            theta_grid=self._theta_grid,
        )
        new_model._irf_values = self._irf_values.copy()
        new_model._calibration_counts = self._calibration_counts.copy()
        new_model._calibration_weight_sums = self._calibration_weight_sums.copy()
        new_model._is_fitted = self._is_fitted
        return new_model
