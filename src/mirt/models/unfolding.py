"""Unfolding and ideal-point item response models.

These models describe preference or attitude responses whose probability is
highest near an item location and decreases as the respondent moves away from
that ideal point.
"""

from __future__ import annotations

from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import DichotomousItemModel, PolytomousItemModel


def _validate_item_index(n_items: int, item_idx: int) -> int:
    if (
        isinstance(item_idx, bool)
        or not isinstance(item_idx, Integral)
        or item_idx < 0
        or item_idx >= n_items
    ):
        raise IndexError(f"Item index {item_idx} out of range [0, {n_items})")
    return int(item_idx)


def _theta_values(
    model: DichotomousItemModel | PolytomousItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    theta_array = model._ensure_theta_2d(theta)
    if not np.all(np.isfinite(theta_array)):
        raise MirtValidationError(
            "theta must contain only finite values", parameter="theta"
        )
    return theta_array[:, 0]


def _validate_response_matrix(
    responses: NDArray[np.int_],
    *,
    n_items: int,
    category_limits: NDArray[np.intp],
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    try:
        response_array = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtDataError("responses must contain numeric values") from exc
    if response_array.ndim != 2:
        raise MirtDataError(
            "responses must be two-dimensional", ndim=response_array.ndim
        )
    if response_array.shape[1] != n_items:
        raise MirtDataError(
            "response item count does not match the model",
            n_items=response_array.shape[1],
            expected=n_items,
        )

    missing = np.isnan(response_array) | (
        np.isfinite(response_array) & (response_array < 0)
    )
    observed = ~missing
    observed_values = response_array[observed]
    if not np.all(np.isfinite(observed_values)):
        raise MirtDataError("observed responses must be finite")
    if np.any(observed_values != np.floor(observed_values)):
        raise MirtDataError("observed responses must be integer category codes")
    if np.any(observed & (response_array >= category_limits[None, :])):
        raise MirtDataError("observed response is outside the item's category range")

    codes = np.zeros(response_array.shape, dtype=np.intp)
    codes[observed] = observed_values.astype(np.intp)
    return codes, observed


class GeneralizedGradedUnfolding(PolytomousItemModel):
    """Generalized Graded Unfolding Model (GGUM).

    For an item with observed categories ``0, ..., C``, define
    ``M = 2C + 1`` and

    ``f(w) = exp(alpha * (w * (theta - delta) - sum(tau[0:w+1])))``.

    The observed category probability is proportional to
    ``f(z) + f(M-z)``. ``tau_0`` is fixed to zero, ``tau_(C+1)`` is
    fixed to zero, and the remaining subjective thresholds are symmetric:
    ``tau_z = -tau_(M-z+1)``.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_categories : int or list[int]
        Number of observed categories, common to all items or per item.
    n_factors : int
        Must be one; GGUM is unidimensional.
    item_names : list[str], optional
        Item names.

    References
    ----------
    Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000).
    A general item response theory model for unfolding unidimensional
    polytomous responses. Applied Psychological Measurement, 24(1), 3-32.
    """

    model_name = "GGUM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int] = 5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise MirtValidationError(
                "GGUM only supports unidimensional models",
                parameter="n_factors",
                value=n_factors,
                expected="1",
            )

        if isinstance(n_categories, bool):
            raise MirtValidationError(
                "n_categories must contain integers of at least 2",
                parameter="n_categories",
                value=n_categories,
            )
        if isinstance(n_categories, Integral):
            normalized_categories: int | list[int] = int(n_categories)
        else:
            try:
                raw_categories = list(n_categories)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    "n_categories must be an integer or sequence of integers",
                    parameter="n_categories",
                ) from exc
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in raw_categories
            ):
                raise MirtValidationError(
                    "n_categories must contain only integers",
                    parameter="n_categories",
                )
            normalized_categories = [int(value) for value in raw_categories]

        super().__init__(
            n_items=n_items,
            n_categories=normalized_categories,
            n_factors=1,
            item_names=item_names,
        )

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)

        max_thresholds = 2 * (max(self._n_categories) - 1) + 1
        thresholds = np.zeros((self.n_items, max_thresholds), dtype=np.float64)
        for item_idx, n_categories in enumerate(self._n_categories):
            c = n_categories - 1
            m = 2 * c + 1
            independent = np.linspace(-2.0, -0.5, c, dtype=np.float64)
            thresholds[item_idx, :c] = independent
            thresholds[item_idx, c] = 0.0
            thresholds[item_idx, c + 1 : m] = -independent[::-1]
        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Discrimination parameters (alpha)."""
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        """Item ideal points (delta)."""
        return self._parameters["location"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Padded subjective thresholds ``tau_1, ..., tau_M``."""
        return self._parameters["thresholds"]

    def thresholds_for_item(
        self, item_idx: int, *, include_tau_zero: bool = False
    ) -> NDArray[np.float64]:
        """Return the active subjective thresholds for one item."""
        item = _validate_item_index(self.n_items, item_idx)
        c = self._n_categories[item] - 1
        m = 2 * c + 1
        values = self._parameters["thresholds"][item, :m].copy()
        if include_tau_zero:
            values = np.concatenate((np.zeros(1, dtype=np.float64), values))
        return values

    def independent_thresholds(self, item_idx: int) -> NDArray[np.float64]:
        """Return the freely specified first-half thresholds for one item."""
        item = _validate_item_index(self.n_items, item_idx)
        c = self._n_categories[item] - 1
        return self._parameters["thresholds"][item, :c].copy()

    def set_independent_thresholds(
        self, item_idx: int, values: NDArray[np.float64]
    ) -> None:
        """Set first-half thresholds and construct their symmetric partners."""
        item = _validate_item_index(self.n_items, item_idx)
        c = self._n_categories[item] - 1
        try:
            independent = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "thresholds must contain numeric values", parameter="thresholds"
            ) from exc
        if independent.shape != (c,):
            raise MirtValidationError(
                "independent thresholds have the wrong shape",
                parameter="thresholds",
                value=independent.shape,
                expected=str((c,)),
            )
        if not np.all(np.isfinite(independent)):
            raise MirtValidationError(
                "thresholds must contain only finite values", parameter="thresholds"
            )

        updated = self._parameters["thresholds"].copy()
        m = 2 * c + 1
        updated[item, :c] = independent
        updated[item, c] = 0.0
        updated[item, c + 1 : m] = -independent[::-1]
        self.set_parameters(thresholds=updated)

    def _item_components(
        self,
        theta_values: NDArray[np.float64],
        item_idx: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return category probabilities and log-weight derivatives."""
        n_categories = self._n_categories[item_idx]
        c = n_categories - 1
        m = 2 * c + 1
        categories = np.arange(n_categories, dtype=np.float64)
        complements = m - categories

        thresholds = self._parameters["thresholds"][item_idx, :m]
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(thresholds))
        )
        alpha = self._parameters["discrimination"][item_idx]
        distance = theta_values - self._parameters["location"][item_idx]

        first = alpha * (
            distance[:, None] * categories[None, :]
            - cumulative[categories.astype(np.intp)][None, :]
        )
        second = alpha * (
            distance[:, None] * complements[None, :]
            - cumulative[complements.astype(np.intp)][None, :]
        )
        log_weights = np.logaddexp(first, second)
        centered = log_weights - np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(centered)
        probabilities = weights / np.sum(weights, axis=1, keepdims=True)

        first_fraction = np.exp(first - log_weights)
        derivative = alpha * (
            first_fraction * categories[None, :]
            + (1.0 - first_fraction) * complements[None, :]
        )
        return probabilities, derivative

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute stable GGUM category probabilities."""
        values = _theta_values(self, theta)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            return self._item_components(values, item)[0]

        probabilities = np.zeros(
            (len(values), self.n_items, max(self._n_categories)), dtype=np.float64
        )
        for item in range(self.n_items):
            item_probabilities, _ = self._item_components(values, item)
            probabilities[:, item, : self._n_categories[item]] = item_probabilities
        return probabilities

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute the probability of one observed response category."""
        item = _validate_item_index(self.n_items, item_idx)
        if (
            isinstance(category, bool)
            or not isinstance(category, Integral)
            or category < 0
            or category >= self._n_categories[item]
        ):
            raise MirtValidationError(
                f"category must be in [0, {self._n_categories[item]})",
                parameter="category",
                value=category,
            )
        values = _theta_values(self, theta)
        return self._item_components(values, item)[0][:, int(category)]

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected Fisher information for theta."""
        values = _theta_values(self, theta)
        if item_idx is not None:
            return self._item_information_from_values(
                values, _validate_item_index(self.n_items, item_idx)
            )

        total = np.zeros(len(values), dtype=np.float64)
        for item in range(self.n_items):
            total += self._item_information_from_values(values, item)
        return total

    def _item_information_from_values(
        self, theta_values: NDArray[np.float64], item_idx: int
    ) -> NDArray[np.float64]:
        probabilities, derivatives = self._item_components(theta_values, item_idx)
        expected_derivative = np.sum(probabilities * derivatives, axis=1)
        return np.sum(
            probabilities * (derivatives - expected_derivative[:, None]) ** 2,
            axis=1,
        )

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        values = _theta_values(self, theta)
        return self._item_information_from_values(
            values, _validate_item_index(self.n_items, item_idx)
        )

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected observed-category scores."""
        probabilities = self.probability(theta, item_idx)
        if item_idx is not None:
            return probabilities @ np.arange(probabilities.shape[1], dtype=np.float64)
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        return np.sum(probabilities * category_scores[None, None, :], axis=(1, 2))

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return all response-category curves for one item."""
        return self.probability(theta, item_idx)

    def _validated_responses(
        self, responses: NDArray[np.int_]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        return _validate_response_matrix(
            responses,
            n_items=self.n_items,
            category_limits=np.asarray(self._n_categories, dtype=np.intp),
        )

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute personwise log likelihoods."""
        response_codes, observed = self._validated_responses(responses)
        values = _theta_values(self, theta)
        if len(values) != response_codes.shape[0]:
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(values),
                response_persons=response_codes.shape[0],
            )

        probabilities = self.probability(values)
        selected = np.take_along_axis(probabilities, response_codes[..., None], axis=2)[
            ..., 0
        ]
        contributions = np.where(
            observed,
            np.log(np.clip(selected, PROB_EPSILON, 1.0)),
            0.0,
        )
        return np.sum(contributions, axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log likelihood for every person and theta point."""
        response_codes, observed = self._validated_responses(responses)
        probabilities = np.clip(self.probability(theta), PROB_EPSILON, 1.0)
        log_probabilities = np.log(probabilities)
        likelihood = np.zeros(
            (response_codes.shape[0], log_probabilities.shape[0]), dtype=np.float64
        )
        for item in range(self.n_items):
            valid = observed[:, item]
            if np.any(valid):
                likelihood[valid] += log_probabilities[
                    :, item, response_codes[valid, item]
                ].T
        return likelihood

    def _validate_threshold_matrix(self, values: NDArray[np.float64]) -> None:
        for item, n_categories in enumerate(self._n_categories):
            c = n_categories - 1
            m = 2 * c + 1
            active = values[item, :m]
            if not np.isclose(active[c], 0.0, atol=1e-12, rtol=0.0):
                raise MirtValidationError(
                    f"item {item} center threshold must be zero",
                    parameter="thresholds",
                )
            if not np.allclose(
                active[c + 1 :], -active[:c][::-1], atol=1e-12, rtol=0.0
            ):
                raise MirtValidationError(
                    f"item {item} thresholds must be symmetric",
                    parameter="thresholds",
                )
            if np.any(values[item, m:] != 0.0):
                raise MirtValidationError(
                    f"item {item} padded thresholds must be zero",
                    parameter="thresholds",
                )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Atomically set validated GGUM parameters."""
        unknown = set(params) - set(self._parameters)
        if unknown:
            name = sorted(unknown)[0]
            raise MirtValidationError(f"Unknown parameter: {name}", parameter=name)
        updated = {name: value.copy() for name, value in self._parameters.items()}
        for name, value in params.items():
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must contain numeric values", parameter=name
                ) from exc
            if array.shape != updated[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}",
                    parameter=name,
                    value=array.shape,
                    expected=str(updated[name].shape),
                )
            if not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"{name} must contain only finite values", parameter=name
                )
            if name == "discrimination" and np.any(array <= 0.0):
                raise MirtValidationError(
                    "discrimination values must be positive",
                    parameter=name,
                    expected="> 0",
                )
            if name == "thresholds":
                self._validate_threshold_matrix(array)
            updated[name] = array.copy()
        self._parameters = updated
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set a validated parameter for one item."""
        item = _validate_item_index(self.n_items, item_idx)
        if param_name not in self._parameters:
            raise MirtValidationError(
                f"Unknown parameter: {param_name}", parameter=param_name
            )
        updated = self._parameters[param_name].copy()
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{param_name} must contain numeric values", parameter=param_name
            ) from exc
        if updated.ndim == 1:
            if array.ndim != 0:
                raise MirtValidationError(
                    f"{param_name} requires a scalar value",
                    parameter=param_name,
                )
            updated[item] = float(array)
        else:
            if array.shape != updated[item].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {param_name}",
                    parameter=param_name,
                    value=array.shape,
                    expected=str(updated[item].shape),
                )
            updated[item] = array
        if param_name == "discrimination":
            self.set_parameters(discrimination=updated)
        elif param_name == "location":
            self.set_parameters(location=updated)
        else:
            self.set_parameters(thresholds=updated)

    def copy(self) -> Self:
        """Create an independent model copy."""
        new_model = self.__class__(
            n_items=self.n_items,
            n_categories=list(self._n_categories),
            n_factors=1,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class _UnfoldingDichotomousModel(DichotomousItemModel):
    """Shared validation and likelihood operations for binary ideal-point models."""

    def _validate_parameter_values(
        self, name: str, values: NDArray[np.float64]
    ) -> None:
        if name == "discrimination" and np.any(values <= 0.0):
            raise MirtValidationError(
                "discrimination values must be positive",
                parameter=name,
                expected="> 0",
            )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Atomically set validated item parameters."""
        unknown = set(params) - set(self._parameters)
        if unknown:
            name = sorted(unknown)[0]
            raise MirtValidationError(f"Unknown parameter: {name}", parameter=name)
        updated = {name: value.copy() for name, value in self._parameters.items()}
        for name, value in params.items():
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must contain numeric values", parameter=name
                ) from exc
            if array.shape != updated[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}",
                    parameter=name,
                    value=array.shape,
                    expected=str(updated[name].shape),
                )
            if not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"{name} must contain only finite values", parameter=name
                )
            self._validate_parameter_values(name, array)
            updated[name] = array.copy()
        self._parameters = updated
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one validated scalar item parameter."""
        item = _validate_item_index(self.n_items, item_idx)
        if param_name not in self._parameters:
            raise MirtValidationError(
                f"Unknown parameter: {param_name}", parameter=param_name
            )
        try:
            scalar = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{param_name} must be numeric", parameter=param_name
            ) from exc
        if scalar.ndim != 0:
            raise MirtValidationError(
                f"{param_name} requires a scalar value", parameter=param_name
            )
        updated = self._parameters[param_name].copy()
        updated[item] = float(scalar)
        if param_name == "discrimination":
            self.set_parameters(discrimination=updated)
        elif param_name == "location":
            self.set_parameters(location=updated)
        elif param_name == "peak_height":
            self.set_parameters(peak_height=updated)
        else:
            self.set_parameters(asymmetry=updated)

    def _validated_responses(
        self, responses: NDArray[np.int_]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        return _validate_response_matrix(
            responses,
            n_items=self.n_items,
            category_limits=np.full(self.n_items, 2, dtype=np.intp),
        )

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute personwise binary log likelihoods."""
        response_codes, observed = self._validated_responses(responses)
        values = _theta_values(self, theta)
        if len(values) != response_codes.shape[0]:
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(values),
                response_persons=response_codes.shape[0],
            )
        probabilities = np.clip(
            self.probability(values), PROB_EPSILON, 1.0 - PROB_EPSILON
        )
        selected = np.where(response_codes == 1, probabilities, 1.0 - probabilities)
        return np.sum(np.where(observed, np.log(selected), 0.0), axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute binary log likelihood for every person and theta point."""
        response_codes, observed = self._validated_responses(responses)
        probabilities = np.clip(
            self.probability(theta), PROB_EPSILON, 1.0 - PROB_EPSILON
        )
        successes = observed & (response_codes == 1)
        failures = observed & (response_codes == 0)
        return (
            successes @ np.log(probabilities).T + failures @ np.log1p(-probabilities).T
        )


class IdealPointModel(_UnfoldingDichotomousModel):
    """Gaussian ideal-point model for dichotomous responses.

    ``P(X=1 | theta) = h * exp(-a * (theta-delta)^2)``, where ``a`` is
    positive and the peak height ``h`` is in ``(0, 1]``.
    """

    model_name = "IdealPoint"
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["peak_height"] = np.ones(self.n_items, dtype=np.float64)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        return self._parameters["location"]

    @property
    def peak_height(self) -> NDArray[np.float64]:
        return self._parameters["peak_height"]

    @property
    def peak_location(self) -> NDArray[np.float64]:
        return self._parameters["location"].copy()

    def _validate_parameter_values(
        self, name: str, values: NDArray[np.float64]
    ) -> None:
        super()._validate_parameter_values(name, values)
        if name == "peak_height" and np.any((values <= 0.0) | (values > 1.0)):
            raise MirtValidationError(
                "peak_height values must be in (0, 1]",
                parameter=name,
                expected="0 < peak_height <= 1",
            )

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute stable Gaussian ideal-point probabilities."""
        values = _theta_values(self, theta)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            distance = values - self._parameters["location"][item]
            with np.errstate(over="ignore"):
                exponent = -self._parameters["discrimination"][item] * distance**2
            return self._parameters["peak_height"][item] * np.exp(exponent)

        distance = values[:, None] - self._parameters["location"][None, :]
        with np.errstate(over="ignore"):
            exponent = -self._parameters["discrimination"][None, :] * distance**2
        return self._parameters["peak_height"][None, :] * np.exp(exponent)

    @staticmethod
    def _information_from_scaled_distance(
        scaled_distance: NDArray[np.float64],
        discrimination: NDArray[np.float64],
        peak_height: NDArray[np.float64],
        probability: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        information = np.zeros_like(probability)
        unit_peak = peak_height == 1.0
        zero_distance = scaled_distance == 0.0
        information[unit_peak & zero_distance] = (
            4.0 * discrimination[unit_peak & zero_distance]
        )

        stable_unit = unit_peak & (scaled_distance > 0.0) & (scaled_distance < 700.0)
        information[stable_unit] = (
            4.0
            * discrimination[stable_unit]
            * scaled_distance[stable_unit]
            / np.expm1(scaled_distance[stable_unit])
        )

        stable_nonunit = ~unit_peak & np.isfinite(scaled_distance) & (probability > 0.0)
        numerator = (
            4.0
            * discrimination[stable_nonunit]
            * scaled_distance[stable_nonunit]
            * probability[stable_nonunit]
        )
        information[stable_nonunit] = numerator / (1.0 - probability[stable_nonunit])
        return information

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information with the peak limit handled exactly."""
        values = _theta_values(self, theta)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            discrimination = np.full_like(
                values, self._parameters["discrimination"][item]
            )
            peak_height = np.full_like(values, self._parameters["peak_height"][item])
            distance = values - self._parameters["location"][item]
            with np.errstate(over="ignore"):
                scaled_distance = discrimination * distance**2
            probability = peak_height * np.exp(-scaled_distance)
            return self._information_from_scaled_distance(
                scaled_distance, discrimination, peak_height, probability
            )

        discrimination = np.broadcast_to(
            self._parameters["discrimination"][None, :],
            (len(values), self.n_items),
        )
        peak_height = np.broadcast_to(
            self._parameters["peak_height"][None, :],
            (len(values), self.n_items),
        )
        distance = values[:, None] - self._parameters["location"][None, :]
        with np.errstate(over="ignore"):
            scaled_distance = discrimination * distance**2
        probability = peak_height * np.exp(-scaled_distance)
        return self._information_from_scaled_distance(
            scaled_distance, discrimination, peak_height, probability
        )

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class HyperbolicCosineModel(_UnfoldingDichotomousModel):
    """Hyperbolic-cosine unfolding model.

    ``P(X=1 | theta) = 1 / (1 + cosh(z))`` with
    ``z = a * (theta-delta) - gamma``. ``gamma`` shifts the response peak,
    whose location is ``delta + gamma/a``.
    """

    model_name = "HCM"
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["asymmetry"] = np.zeros(self.n_items, dtype=np.float64)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        return self._parameters["location"]

    @property
    def asymmetry(self) -> NDArray[np.float64]:
        return self._parameters["asymmetry"]

    @property
    def peak_location(self) -> NDArray[np.float64]:
        return self._parameters["location"] + (
            self._parameters["asymmetry"] / self._parameters["discrimination"]
        )

    @staticmethod
    def _stable_probability(
        linear_predictor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        tail = np.exp(-np.abs(linear_predictor))
        return 2.0 * tail / (1.0 + tail) ** 2

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute hyperbolic-cosine probabilities without overflow."""
        values = _theta_values(self, theta)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            predictor = (
                self._parameters["discrimination"][item]
                * (values - self._parameters["location"][item])
                - self._parameters["asymmetry"][item]
            )
            return self._stable_probability(predictor)

        predictor = (
            self._parameters["discrimination"][None, :]
            * (values[:, None] - self._parameters["location"][None, :])
            - self._parameters["asymmetry"][None, :]
        )
        return self._stable_probability(predictor)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute stable Fisher information."""
        values = _theta_values(self, theta)
        probability = self.probability(values, item_idx)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            predictor = (
                self._parameters["discrimination"][item]
                * (values - self._parameters["location"][item])
                - self._parameters["asymmetry"][item]
            )
            log_slope = -self._parameters["discrimination"][item] * np.tanh(
                predictor / 2.0
            )
        else:
            predictor = (
                self._parameters["discrimination"][None, :]
                * (values[:, None] - self._parameters["location"][None, :])
                - self._parameters["asymmetry"][None, :]
            )
            log_slope = -self._parameters["discrimination"][None, :] * np.tanh(
                predictor / 2.0
            )
        return log_slope**2 * probability / (1.0 - probability)

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model
