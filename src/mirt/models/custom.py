"""User-defined item and group models.

Callbacks make it possible to add response functions and latent group
distributions without implementing a new model class.  The wrappers in this
module validate callback output at the boundary so malformed probabilities or
covariance matrices fail close to their source.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import BaseItemModel

FloatArray = NDArray[np.float64]


def _validate_name(value: str, parameter: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise MirtValidationError(
            f"{parameter} must be a non-empty string",
            parameter=parameter,
            value=value,
        )
    return value.strip()


def _validate_parameter_metadata(
    par_names: list[str],
    par_bounds: Mapping[str, tuple[float, float]],
    par_defaults: Mapping[str, float],
) -> tuple[list[str], dict[str, tuple[float, float]], dict[str, float]]:
    names = list(par_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise MirtValidationError(
            "par_names must contain non-empty strings", parameter="par_names"
        )
    if len(names) != len(set(names)):
        raise MirtValidationError(
            "par_names must be unique", parameter="par_names", value=names
        )

    unknown_bounds = set(par_bounds) - set(names)
    unknown_defaults = set(par_defaults) - set(names)
    if unknown_bounds:
        raise MirtValidationError(
            f"Bounds supplied for unknown parameters: {sorted(unknown_bounds)}",
            parameter="par_bounds",
        )
    if unknown_defaults:
        raise MirtValidationError(
            f"Defaults supplied for unknown parameters: {sorted(unknown_defaults)}",
            parameter="par_defaults",
        )

    bounds: dict[str, tuple[float, float]] = {}
    defaults: dict[str, float] = {}
    for name in names:
        raw_bounds = par_bounds.get(name, (-np.inf, np.inf))
        if not isinstance(raw_bounds, (tuple, list)) or len(raw_bounds) != 2:
            raise MirtValidationError(
                f"Bounds for {name} must be a (lower, upper) pair",
                parameter="par_bounds",
                value=raw_bounds,
            )
        lower, upper = (float(raw_bounds[0]), float(raw_bounds[1]))
        if np.isnan(lower) or np.isnan(upper) or lower > upper:
            raise MirtValidationError(
                f"Invalid bounds for {name}: ({lower}, {upper})",
                parameter="par_bounds",
                expected="lower <= upper",
            )
        bounds[name] = (lower, upper)

        default = float(par_defaults.get(name, 0.0))
        if not np.isfinite(default) or not lower <= default <= upper:
            raise MirtValidationError(
                f"Default for {name} must be finite and within its bounds",
                parameter="par_defaults",
                value=default,
                expected=f"[{lower}, {upper}]",
            )
        defaults[name] = default

    return names, bounds, defaults


def _inferred_item_parameters(function: Callable[..., Any]) -> list[str]:
    parameters = [
        parameter
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name != "self"
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not parameters:
        raise MirtValidationError(
            "icc_function must accept theta as its first argument",
            parameter="icc_function",
        )
    return [parameter.name for parameter in parameters[1:]]


def _inferred_group_parameters(
    *functions: Callable[..., Any] | None,
) -> list[str]:
    names: list[str] = []
    for function in functions:
        if function is None:
            continue
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.name not in names:
                names.append(parameter.name)
    return names


@dataclass
class ItemTypeSpec:
    """Specification for a callback-based item type.

    The probability callback receives theta followed by the named item
    parameters.  It returns ``(n_theta,)`` for dichotomous items or
    ``(n_theta, n_categories)`` for polytomous items.
    """

    name: str
    icc_function: Callable[..., FloatArray]
    info_function: Callable[..., FloatArray] | None = None
    par_names: list[str] = field(default_factory=list)
    par_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    par_defaults: dict[str, float] = field(default_factory=dict)
    n_categories: int = 2
    gradient_function: Callable[..., dict[str, FloatArray]] | None = None

    def __post_init__(self) -> None:
        self.name = _validate_name(self.name)
        if not callable(self.icc_function):
            raise MirtValidationError(
                "icc_function must be callable", parameter="icc_function"
            )
        if self.info_function is not None and not callable(self.info_function):
            raise MirtValidationError(
                "info_function must be callable", parameter="info_function"
            )
        if self.gradient_function is not None and not callable(self.gradient_function):
            raise MirtValidationError(
                "gradient_function must be callable", parameter="gradient_function"
            )
        if (
            isinstance(self.n_categories, bool)
            or not isinstance(self.n_categories, int)
            or self.n_categories < 2
        ):
            raise MirtValidationError(
                "n_categories must be an integer of at least 2",
                parameter="n_categories",
                value=self.n_categories,
            )
        self.par_names, self.par_bounds, self.par_defaults = (
            _validate_parameter_metadata(
                self.par_names, self.par_bounds, self.par_defaults
            )
        )


def create_item_type(
    name: str,
    icc_function: Callable[..., FloatArray],
    info_function: Callable[..., FloatArray] | None = None,
    par_names: list[str] | None = None,
    par_bounds: dict[str, tuple[float, float]] | None = None,
    par_defaults: dict[str, float] | None = None,
    n_categories: int = 2,
    gradient_function: Callable[..., dict[str, FloatArray]] | None = None,
) -> ItemTypeSpec:
    """Create a custom item specification.

    Parameter names are inferred from every callback argument after theta when
    ``par_names`` is omitted.
    """

    if par_names is None:
        par_names = _inferred_item_parameters(icc_function)
    return ItemTypeSpec(
        name=name,
        icc_function=icc_function,
        info_function=info_function,
        par_names=par_names,
        par_bounds={} if par_bounds is None else par_bounds,
        par_defaults={} if par_defaults is None else par_defaults,
        n_categories=n_categories,
        gradient_function=gradient_function,
    )


class CustomItemModel(BaseItemModel):
    """IRT item model backed by user-defined probability callbacks."""

    model_name = "Custom"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        item_type: ItemTypeSpec | Callable[..., FloatArray],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if callable(item_type) and not isinstance(item_type, ItemTypeSpec):
            item_type = create_item_type("Custom", item_type)
        if not isinstance(item_type, ItemTypeSpec):
            raise MirtValidationError(
                "item_type must be an ItemTypeSpec or callable",
                parameter="item_type",
            )
        self.item_type = item_type
        self.model_name = item_type.name
        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        self._parameters = {
            name: np.full(self.n_items, self.item_type.par_defaults[name])
            for name in self.item_type.par_names
        }

    @property
    def is_polytomous(self) -> bool:
        return self.item_type.n_categories > 2

    @property
    def n_categories(self) -> int:
        return self.item_type.n_categories

    def _validated_theta(self, theta: FloatArray) -> FloatArray:
        theta_values = np.asarray(theta, dtype=np.float64)
        if theta_values.ndim == 0:
            theta_values = theta_values.reshape(1)
        theta_2d = self._ensure_theta_2d(theta_values)
        if theta_2d.shape[0] == 0 or not np.all(np.isfinite(theta_2d)):
            raise MirtValidationError(
                "theta must contain at least one finite value", parameter="theta"
            )
        return theta_2d

    def _callback_theta(self, theta: FloatArray) -> FloatArray:
        return theta[:, 0] if self.n_factors == 1 else theta

    def _validated_item(self, item_idx: int) -> int:
        if isinstance(item_idx, bool) or not isinstance(item_idx, (int, np.integer)):
            raise MirtValidationError(
                "item_idx must be an integer", parameter="item_idx", value=item_idx
            )
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return int(item_idx)

    def _validate_probability_output(
        self, values: Any, n_theta: int, callback_name: str = "icc_function"
    ) -> FloatArray:
        probabilities = np.asarray(values, dtype=np.float64)
        if self.is_polytomous:
            expected = (n_theta, self.n_categories)
            if probabilities.shape != expected:
                raise MirtValidationError(
                    f"{callback_name} returned shape {probabilities.shape}, expected {expected}",
                    parameter=callback_name,
                )
        else:
            if probabilities.shape == (n_theta, 1):
                probabilities = probabilities[:, 0]
            if probabilities.ndim == 0 and n_theta == 1:
                probabilities = probabilities.reshape(1)
            if probabilities.shape != (n_theta,):
                raise MirtValidationError(
                    f"{callback_name} returned shape {probabilities.shape}, expected {(n_theta,)}",
                    parameter=callback_name,
                )
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise MirtValidationError(
                f"{callback_name} must return finite probabilities in [0, 1]",
                parameter=callback_name,
            )
        if self.is_polytomous and not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9
        ):
            raise MirtValidationError(
                f"Rows returned by {callback_name} must sum to 1",
                parameter=callback_name,
            )
        return probabilities

    def _item_probability(self, theta: FloatArray, item_idx: int) -> FloatArray:
        params = self.get_item_parameters(item_idx)
        result = self.item_type.icc_function(self._callback_theta(theta), **params)
        return self._validate_probability_output(result, theta.shape[0])

    def probability(self, theta: FloatArray, item_idx: int | None = None) -> FloatArray:
        """Compute success probabilities or full category traces."""

        theta_2d = self._validated_theta(theta)
        if item_idx is not None:
            return self._item_probability(theta_2d, self._validated_item(item_idx))
        item_probabilities = [
            self._item_probability(theta_2d, item) for item in range(self.n_items)
        ]
        return np.stack(item_probabilities, axis=1)

    def icc(self, theta: FloatArray, item_idx: int) -> FloatArray:
        """Alias for a single item's probability trace."""

        return self.probability(theta, item_idx)

    def category_probability(
        self, theta: FloatArray, item_idx: int, category: int
    ) -> FloatArray:
        """Compute a single category probability for one item."""

        if (
            isinstance(category, bool)
            or not isinstance(category, (int, np.integer))
            or category < 0
            or category >= self.n_categories
        ):
            raise MirtValidationError(
                f"category must be in [0, {self.n_categories})",
                parameter="category",
                value=category,
            )
        probabilities = self.probability(theta, item_idx)
        if self.is_polytomous:
            return probabilities[:, int(category)]
        return probabilities if category == 1 else 1.0 - probabilities

    def _validate_information_output(self, values: Any, n_theta: int) -> FloatArray:
        information = np.asarray(values, dtype=np.float64)
        if information.shape == (n_theta, 1):
            information = information[:, 0]
        if information.ndim == 0 and n_theta == 1:
            information = information.reshape(1)
        if information.shape != (n_theta,):
            raise MirtValidationError(
                f"info_function returned shape {information.shape}, expected {(n_theta,)}",
                parameter="info_function",
            )
        if not np.all(np.isfinite(information)) or np.any(information < 0.0):
            raise MirtValidationError(
                "info_function must return finite, non-negative values",
                parameter="info_function",
            )
        return information

    def information(self, theta: FloatArray, item_idx: int | None = None) -> FloatArray:
        """Compute scalar item information at each theta point.

        For multidimensional callbacks, numerical information is the trace of
        the Fisher information matrix.
        """

        theta_2d = self._validated_theta(theta)
        if item_idx is None:
            return np.column_stack(
                [self.information(theta_2d, item) for item in range(self.n_items)]
            )
        item_idx = self._validated_item(item_idx)
        if self.item_type.info_function is None:
            return self._numerical_information(theta_2d, item_idx)
        result = self.item_type.info_function(
            self._callback_theta(theta_2d), **self.get_item_parameters(item_idx)
        )
        return self._validate_information_output(result, theta_2d.shape[0])

    def _numerical_information(self, theta: FloatArray, item_idx: int) -> FloatArray:
        h = 1e-5
        probability = self._item_probability(theta, item_idx)
        information = np.zeros(theta.shape[0], dtype=np.float64)
        for factor in range(self.n_factors):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[:, factor] += h
            theta_minus[:, factor] -= h
            derivative = (
                self._item_probability(theta_plus, item_idx)
                - self._item_probability(theta_minus, item_idx)
            ) / (2.0 * h)
            if self.is_polytomous:
                denominator = np.clip(probability, PROB_EPSILON, None)
                information += np.sum(derivative**2 / denominator, axis=1)
            else:
                clipped = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)
                information += derivative**2 / (clipped * (1.0 - clipped))
        return information

    def expected_score(
        self, theta: FloatArray, item_idx: int | None = None
    ) -> FloatArray:
        """Compute expected item or total score."""

        probabilities = self.probability(theta, item_idx)
        if not self.is_polytomous:
            return probabilities.sum(axis=1) if item_idx is None else probabilities
        scores = np.arange(self.n_categories, dtype=np.float64)
        expected = probabilities @ scores
        return expected.sum(axis=1) if item_idx is None else expected

    def _validated_responses(
        self, responses: NDArray[np.int_], n_persons: int | None = None
    ) -> NDArray[np.int64]:
        values = np.asarray(responses)
        if values.ndim != 2 or values.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses must have shape (n_persons, {self.n_items})",
                n_items=values.shape[1] if values.ndim == 2 else None,
            )
        if n_persons is not None and values.shape[0] != n_persons:
            raise MirtDataError(
                f"responses has {values.shape[0]} persons, expected {n_persons}"
            )
        if not np.issubdtype(values.dtype, np.number) or not np.all(
            np.isfinite(values)
        ):
            raise MirtDataError("responses must contain finite numeric values")
        integer_values = values.astype(np.int64)
        if not np.array_equal(values, integer_values):
            raise MirtDataError("responses must contain integer category values")
        if np.any(integer_values < -1) or np.any(integer_values >= self.n_categories):
            raise MirtDataError(
                f"responses must use -1 for missing or categories 0 through {self.n_categories - 1}"
            )
        return integer_values

    def log_likelihood(
        self, responses: NDArray[np.int_], theta: FloatArray
    ) -> FloatArray:
        """Compute each person's log likelihood at their theta value."""

        theta_2d = self._validated_theta(theta)
        response_values = self._validated_responses(responses, theta_2d.shape[0])
        valid = response_values >= 0
        probabilities = self.probability(theta_2d)
        if not self.is_polytomous:
            clipped = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
            values = np.where(
                valid,
                response_values * np.log(clipped)
                + (1 - response_values) * np.log1p(-clipped),
                0.0,
            )
            return values.sum(axis=1)

        safe_responses = np.where(valid, response_values, 0)
        selected = np.take_along_axis(
            probabilities, safe_responses[:, :, None], axis=2
        )[:, :, 0]
        return np.where(valid, np.log(np.clip(selected, PROB_EPSILON, None)), 0.0).sum(
            axis=1
        )

    def log_likelihood_batch(
        self, responses: NDArray[np.int_], theta: FloatArray
    ) -> FloatArray:
        """Compute every response pattern's likelihood at every theta point."""

        response_values = self._validated_responses(responses)
        theta_2d = self._validated_theta(theta)
        valid = response_values >= 0
        probabilities = self.probability(theta_2d)
        if not self.is_polytomous:
            clipped = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
            observed = np.where(valid, response_values, 0)
            missed = np.where(valid, 1 - response_values, 0)
            return observed @ np.log(clipped).T + missed @ np.log1p(-clipped).T

        likelihood = np.zeros((response_values.shape[0], theta_2d.shape[0]))
        safe_responses = np.where(valid, response_values, 0)
        for item in range(self.n_items):
            log_probabilities = np.log(
                np.clip(probabilities[:, item, :], PROB_EPSILON, None)
            )
            contribution = log_probabilities[:, safe_responses[:, item]].T
            likelihood += np.where(valid[:, item, None], contribution, 0.0)
        return likelihood

    def parameter_gradient(
        self, theta: FloatArray, item_idx: int
    ) -> dict[str, FloatArray]:
        """Compute probability gradients with respect to item parameters."""

        theta_2d = self._validated_theta(theta)
        item_idx = self._validated_item(item_idx)
        params = self.get_item_parameters(item_idx)
        if self.item_type.gradient_function is not None:
            raw = self.item_type.gradient_function(
                self._callback_theta(theta_2d), **params
            )
            if not isinstance(raw, Mapping) or set(raw) != set(
                self.item_type.par_names
            ):
                raise MirtValidationError(
                    "gradient_function must return every named parameter",
                    parameter="gradient_function",
                )
            gradients = {
                name: np.asarray(raw[name], dtype=np.float64)
                for name in self.item_type.par_names
            }
        else:
            gradients = {}
            for name in self.item_type.par_names:
                value = float(params[name])
                h = 1e-5 * max(1.0, abs(value))
                lower, upper = self.item_type.par_bounds[name]
                plus_value = min(value + h, upper)
                minus_value = max(value - h, lower)
                if plus_value == minus_value:
                    shape = (
                        (theta_2d.shape[0], self.n_categories)
                        if self.is_polytomous
                        else (theta_2d.shape[0],)
                    )
                    gradients[name] = np.zeros(shape, dtype=np.float64)
                    continue
                plus = dict(params)
                minus = dict(params)
                plus[name] = plus_value
                minus[name] = minus_value
                plus_probability = self._validate_probability_output(
                    self.item_type.icc_function(self._callback_theta(theta_2d), **plus),
                    theta_2d.shape[0],
                )
                minus_probability = self._validate_probability_output(
                    self.item_type.icc_function(
                        self._callback_theta(theta_2d), **minus
                    ),
                    theta_2d.shape[0],
                )
                gradients[name] = (plus_probability - minus_probability) / (
                    plus_value - minus_value
                )

        expected_shape = (
            (theta_2d.shape[0], self.n_categories)
            if self.is_polytomous
            else (theta_2d.shape[0],)
        )
        for name, gradient in gradients.items():
            if gradient.shape != expected_shape or not np.all(np.isfinite(gradient)):
                raise MirtValidationError(
                    f"Gradient for {name} must have shape {expected_shape} and be finite",
                    parameter="gradient_function",
                )
        return gradients

    def _coerce_parameter_values(self, name: str, values: Any) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.n_items, float(array))
        if array.shape != (self.n_items,):
            raise MirtValidationError(
                f"Parameter {name} must be scalar or have shape {(self.n_items,)}",
                parameter=name,
                value=array.shape,
            )
        lower, upper = self.item_type.par_bounds[name]
        if not np.all(np.isfinite(array)) or np.any((array < lower) | (array > upper)):
            raise MirtValidationError(
                f"Parameter {name} must be finite and within [{lower}, {upper}]",
                parameter=name,
            )
        return array.copy()

    def set_parameters(self, **params: Any) -> Self:
        """Set one or more item parameters atomically."""

        unknown = set(params) - set(self._parameters)
        if unknown:
            raise MirtValidationError(
                f"Unknown parameters: {sorted(unknown)}",
                parameter="parameters",
                expected=", ".join(self._parameters),
            )
        validated = {
            name: self._coerce_parameter_values(name, values)
            for name, values in params.items()
        }
        self._parameters.update(validated)
        return self

    def set_item_parameter(self, item_idx: int, param_name: str, value: Any) -> None:
        """Set one bounded scalar parameter for a single item."""

        item_idx = self._validated_item(item_idx)
        if param_name not in self._parameters:
            raise MirtValidationError(
                f"Unknown parameter: {param_name}", parameter=param_name
            )
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 0:
            raise MirtValidationError(
                "Item parameter value must be scalar", parameter=param_name
            )
        scalar = float(array)
        lower, upper = self.item_type.par_bounds[param_name]
        if not np.isfinite(scalar) or not lower <= scalar <= upper:
            raise MirtValidationError(
                f"Parameter {param_name} must be finite and within [{lower}, {upper}]",
                parameter=param_name,
                value=scalar,
            )
        self._parameters[param_name][item_idx] = scalar

    def copy(self) -> Self:
        """Return an independent model copy."""

        new_model = self.__class__(
            self.n_items,
            self.item_type,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: value.copy() for name, value in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


STANDARD_2PL = create_item_type(
    name="Standard2PL",
    icc_function=lambda theta, a, b: 1 / (1 + np.exp(-a * (theta - b))),
    info_function=lambda theta, a, b: (
        a**2
        * (1 / (1 + np.exp(-a * (theta - b))))
        * (1 - 1 / (1 + np.exp(-a * (theta - b))))
    ),
    par_names=["a", "b"],
    par_bounds={"a": (0.01, 5.0), "b": (-5.0, 5.0)},
    par_defaults={"a": 1.0, "b": 0.0},
)

STANDARD_3PL = create_item_type(
    name="Standard3PL",
    icc_function=lambda theta, a, b, c: c + (1 - c) / (1 + np.exp(-a * (theta - b))),
    par_names=["a", "b", "c"],
    par_bounds={"a": (0.01, 5.0), "b": (-5.0, 5.0), "c": (0.0, 0.5)},
    par_defaults={"a": 1.0, "b": 0.0, "c": 0.0},
)

LOGISTIC_DEVIATION = create_item_type(
    name="LogisticDeviation",
    icc_function=lambda theta, alpha, delta: 1 / (1 + np.exp(-(alpha + delta * theta))),
    par_names=["alpha", "delta"],
    par_bounds={"alpha": (-5.0, 5.0), "delta": (0.01, 5.0)},
    par_defaults={"alpha": 0.0, "delta": 1.0},
)


def list_standard_item_types() -> list[str]:
    """List bundled callback-based item specifications."""

    return ["STANDARD_2PL", "STANDARD_3PL", "LOGISTIC_DEVIATION"]


def get_standard_item_type(name: str) -> ItemTypeSpec:
    """Return a bundled item specification by name."""

    item_types = {
        "STANDARD_2PL": STANDARD_2PL,
        "STANDARD_3PL": STANDARD_3PL,
        "LOGISTIC_DEVIATION": LOGISTIC_DEVIATION,
    }
    if name not in item_types:
        raise MirtValidationError(
            f"Unknown item type: {name}",
            parameter="name",
            expected=", ".join(item_types),
        )
    return item_types[name]


@dataclass
class GroupSpec:
    """Specification for a callback-based latent group distribution."""

    name: str
    mean_function: Callable[..., float | FloatArray] | None = None
    cov_function: Callable[..., FloatArray] | None = None
    par_names: list[str] = field(default_factory=list)
    par_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    par_defaults: dict[str, float] = field(default_factory=dict)
    n_factors: int = 1

    def __post_init__(self) -> None:
        self.name = _validate_name(self.name)
        if self.mean_function is not None and not callable(self.mean_function):
            raise MirtValidationError(
                "mean_function must be callable", parameter="mean_function"
            )
        if self.cov_function is not None and not callable(self.cov_function):
            raise MirtValidationError(
                "cov_function must be callable", parameter="cov_function"
            )
        if (
            isinstance(self.n_factors, bool)
            or not isinstance(self.n_factors, int)
            or self.n_factors < 1
        ):
            raise MirtValidationError(
                "n_factors must be a positive integer",
                parameter="n_factors",
                value=self.n_factors,
            )
        self.par_names, self.par_bounds, self.par_defaults = (
            _validate_parameter_metadata(
                self.par_names, self.par_bounds, self.par_defaults
            )
        )


def create_group(
    name: str,
    mean_function: Callable[..., float | FloatArray] | None = None,
    cov_function: Callable[..., FloatArray] | None = None,
    par_names: list[str] | None = None,
    par_bounds: dict[str, tuple[float, float]] | None = None,
    par_defaults: dict[str, float] | None = None,
    n_factors: int = 1,
) -> GroupSpec:
    """Create a custom latent group distribution specification."""

    if par_names is None:
        par_names = _inferred_group_parameters(mean_function, cov_function)
    return GroupSpec(
        name=name,
        mean_function=mean_function,
        cov_function=cov_function,
        par_names=par_names,
        par_bounds={} if par_bounds is None else par_bounds,
        par_defaults={} if par_defaults is None else par_defaults,
        n_factors=n_factors,
    )


def createGroup(
    name: str,
    mean_function: Callable[..., float | FloatArray] | None = None,
    cov_function: Callable[..., FloatArray] | None = None,
    par_names: list[str] | None = None,
    par_bounds: dict[str, tuple[float, float]] | None = None,
    par_defaults: dict[str, float] | None = None,
    n_factors: int = 1,
) -> GroupSpec:
    """Compatibility alias for :func:`create_group`."""

    return create_group(
        name=name,
        mean_function=mean_function,
        cov_function=cov_function,
        par_names=par_names,
        par_bounds=par_bounds,
        par_defaults=par_defaults,
        n_factors=n_factors,
    )


def _call_group_function(
    function: Callable[..., Any], parameters: Mapping[str, float]
) -> Any:
    signature = inspect.signature(function)
    positional: list[float] = []
    keyword: dict[str, float] = {}
    consumed: set[str] = set()
    has_var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if parameter.name in parameters:
                positional.append(parameters[parameter.name])
                consumed.add(parameter.name)
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            if parameter.name in parameters:
                keyword[parameter.name] = parameters[parameter.name]
                consumed.add(parameter.name)
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
    if has_var_keyword:
        keyword.update(
            {name: value for name, value in parameters.items() if name not in consumed}
        )
    return function(*positional, **keyword)


class CustomGroupModel:
    """Latent group model backed by mean and covariance callbacks."""

    def __init__(self, group_spec: GroupSpec) -> None:
        if not isinstance(group_spec, GroupSpec):
            raise MirtValidationError(
                "group_spec must be a GroupSpec", parameter="group_spec"
            )
        self.spec = group_spec
        self.name = group_spec.name
        self.n_factors = group_spec.n_factors
        self._parameters = dict(group_spec.par_defaults)

    @property
    def parameters(self) -> dict[str, float]:
        """Return an independent parameter mapping."""

        return self._parameters.copy()

    def set_parameters(self, **params: float) -> Self:
        """Set one or more group parameters atomically."""

        unknown = set(params) - set(self._parameters)
        if unknown:
            raise MirtValidationError(
                f"Unknown parameters: {sorted(unknown)}", parameter="parameters"
            )
        validated: dict[str, float] = {}
        for name, value in params.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 0:
                raise MirtValidationError(
                    f"Parameter {name} must be scalar", parameter=name
                )
            scalar = float(array)
            lower, upper = self.spec.par_bounds[name]
            if not np.isfinite(scalar) or not lower <= scalar <= upper:
                raise MirtValidationError(
                    f"Parameter {name} must be finite and within [{lower}, {upper}]",
                    parameter=name,
                    value=scalar,
                )
            validated[name] = scalar
        self._parameters.update(validated)
        return self

    def get_mean(self) -> FloatArray:
        """Return the validated latent mean vector."""

        if self.spec.mean_function is None:
            return np.zeros(self.n_factors)
        mean = np.asarray(
            _call_group_function(self.spec.mean_function, self._parameters),
            dtype=np.float64,
        )
        if mean.ndim == 0 and self.n_factors == 1:
            mean = mean.reshape(1)
        if mean.shape != (self.n_factors,) or not np.all(np.isfinite(mean)):
            raise MirtValidationError(
                f"mean_function must return shape {(self.n_factors,)} with finite values",
                parameter="mean_function",
            )
        return mean

    def get_cov(self) -> FloatArray:
        """Return a finite, symmetric, positive-semidefinite covariance."""

        if self.spec.cov_function is None:
            return np.eye(self.n_factors)
        covariance = np.asarray(
            _call_group_function(self.spec.cov_function, self._parameters),
            dtype=np.float64,
        )
        expected = (self.n_factors, self.n_factors)
        if covariance.ndim == 0 and self.n_factors == 1:
            covariance = covariance.reshape(1, 1)
        if covariance.shape != expected or not np.all(np.isfinite(covariance)):
            raise MirtValidationError(
                f"cov_function must return shape {expected} with finite values",
                parameter="cov_function",
            )
        if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
            raise MirtValidationError(
                "cov_function must return a symmetric matrix",
                parameter="cov_function",
            )
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
            raise MirtValidationError(
                "cov_function must return a positive-semidefinite matrix",
                parameter="cov_function",
            )
        return covariance

    def sample(self, n: int, rng: np.random.Generator | None = None) -> FloatArray:
        """Draw ``n`` samples from the custom latent distribution."""

        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise MirtValidationError(
                "n must be a non-negative integer", parameter="n", value=n
            )
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise MirtValidationError(
                "rng must be a numpy.random.Generator", parameter="rng"
            )
        generator = np.random.default_rng() if rng is None else rng
        samples = generator.multivariate_normal(
            self.get_mean(), self.get_cov(), size=int(n)
        )
        return np.asarray(samples, dtype=np.float64).reshape(int(n), self.n_factors)

    def copy(self) -> Self:
        """Return an independent group model copy."""

        copied = self.__class__(self.spec)
        copied._parameters = self._parameters.copy()
        return copied
