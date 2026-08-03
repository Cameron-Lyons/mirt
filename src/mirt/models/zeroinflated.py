"""Zero-inflated response models.

These models separate an observed zero into a structural process and the
ordinary item-response process.  The shared implementation keeps probability,
information, and structural-zero diagnostics on the same likelihood.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.exceptions import MirtValidationError
from mirt.models.base import DichotomousItemModel

_NORMAL_NODES, _NORMAL_WEIGHTS = np.polynomial.hermite.hermgauss(41)
_NORMAL_NODES = np.sqrt(2.0) * _NORMAL_NODES
_NORMAL_WEIGHTS = _NORMAL_WEIGHTS / np.sqrt(np.pi)


def _safe_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the shared sigmoid without surfacing harmless tail overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        return sigmoid(values)


class _ZeroResponseModel(DichotomousItemModel):
    """Shared likelihood and diagnostics for excess-zero response models."""

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discriminations."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulties."""
        return self._parameters["difficulty"]

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        for name, values in parameters.items():
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                    expected="finite values",
                )

        discrimination = parameters["discrimination"]
        if np.any(discrimination <= 0.0):
            raise MirtValidationError(
                "discrimination must be strictly positive",
                parameter="discrimination",
                value=discrimination,
                expected="> 0",
            )

        for name in ("zero_inflation", "guessing"):
            if name not in parameters:
                continue
            values = parameters[name]
            if np.any((values < 0.0) | (values >= 1.0)):
                raise MirtValidationError(
                    f"{name} must be in [0, 1)",
                    parameter=name,
                    value=values,
                    expected="[0, 1)",
                )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set model parameters atomically after validating their domains."""
        candidate = {name: values.copy() for name, values in self._parameters.items()}

        for name, value in params.items():
            if name not in candidate:
                valid_params = ", ".join(candidate)
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}",
                    parameter=name,
                    expected=valid_params,
                )

            value_arr = np.asarray(value, dtype=np.float64)
            expected_shape = candidate[name].shape
            if value_arr.shape != expected_shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {expected_shape}, "
                    f"got {value_arr.shape}",
                    parameter=name,
                    value=value_arr.shape,
                    expected=str(expected_shape),
                )
            candidate[name] = value_arr.copy()

        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item parameter while preserving model validity."""
        item_idx = self._validate_item_idx(item_idx)
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        value_arr = np.asarray(value, dtype=np.float64)
        if value_arr.ndim != 0:
            raise MirtValidationError(
                f"{param_name} must be a scalar for one item",
                parameter=param_name,
                value=value_arr.shape,
                expected="scalar",
            )

        updated = self._parameters[param_name].copy()
        updated[item_idx] = float(value_arr)
        self.set_parameters(**{param_name: updated})

    def _validate_item_idx(self, item_idx: int) -> int:
        if isinstance(item_idx, bool) or not isinstance(item_idx, (int, np.integer)):
            raise MirtValidationError(
                "item_idx must be an integer",
                parameter="item_idx",
                value=item_idx,
                expected="integer",
            )
        item_idx = int(item_idx)
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return item_idx

    def _prepare_evaluation(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], int | None]:
        theta_2d = self._ensure_theta_2d(theta)
        if theta_2d.shape[0] == 0:
            raise MirtValidationError(
                "theta must contain at least one value",
                parameter="theta",
                value=theta_2d.shape,
                expected="at least one row",
            )
        if not np.all(np.isfinite(theta_2d)):
            raise MirtValidationError(
                "theta must contain only finite values",
                parameter="theta",
                value=theta_2d,
                expected="finite values",
            )
        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
        self._validate_parameter_state(self._parameters)
        return theta_2d[:, 0], item_idx

    def _parameter_for(
        self,
        name: str,
        item_idx: int | None,
    ) -> NDArray[np.float64] | np.float64:
        values = self._parameters[name]
        if item_idx is not None:
            return values[item_idx]
        return values[None, :]

    def probability_2pl(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the logistic response-process probability."""
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        a = self._parameter_for("discrimination", item_idx)
        b = self._parameter_for("difficulty", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        return _safe_sigmoid(a * (theta_eval - b))

    def _evaluate(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        derivative: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        raise NotImplementedError

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the marginal probability of a correct response."""
        probability, _ = self._evaluate(theta, item_idx, derivative=False)
        return probability

    def probability_zero(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the marginal probability of an observed zero."""
        return 1.0 - self.probability(theta, item_idx)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute exact Fisher information for the marginal Bernoulli model."""
        probability, derivative = self._evaluate(theta, item_idx, derivative=True)
        assert derivative is not None
        denominator = probability * (1.0 - probability)
        return np.divide(
            derivative**2,
            denominator,
            out=np.zeros_like(probability),
            where=denominator > 0.0,
        )

    def structural_zero_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return the prior probability of the structural-zero process."""
        raise NotImplementedError

    def structural_zero_posterior(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return P(structural process | observed zero, theta)."""
        structural = self.structural_zero_probability(theta, item_idx)
        probability_zero = self.probability_zero(theta, item_idx)
        return np.divide(
            structural,
            probability_zero,
            out=np.zeros_like(probability_zero),
            where=probability_zero > 0.0,
        )

    def expected_structural_zero_fraction(
        self,
        theta: NDArray[np.float64] | None = None,
        item_idx: int | None = None,
    ) -> NDArray[np.float64] | np.float64:
        """Estimate the fraction of observed zeros that are structural.

        When ``theta`` is omitted, expectations are integrated over a standard
        normal distribution with Gauss-Hermite quadrature.  Supplied ability
        values instead define an equally weighted empirical distribution.
        """
        if theta is None:
            theta_values = _NORMAL_NODES.reshape(-1, 1)
            weights = _NORMAL_WEIGHTS
        else:
            theta_values = self._ensure_theta_2d(theta)
            if theta_values.shape[0] == 0:
                raise MirtValidationError(
                    "theta must contain at least one value",
                    parameter="theta",
                    value=theta_values.shape,
                    expected="at least one row",
                )
            weights = np.full(theta_values.shape[0], 1.0 / theta_values.shape[0])

        structural = self.structural_zero_probability(theta_values, item_idx)
        probability_zero = self.probability_zero(theta_values, item_idx)
        if item_idx is None:
            weights = weights[:, None]
        numerator = np.sum(weights * structural, axis=0)
        denominator = np.sum(weights * probability_zero, axis=0)
        result = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        if item_idx is not None:
            return np.float64(result)
        return result

    def expected_proportion_zeros_from_inflation(
        self,
        theta: NDArray[np.float64] | None = None,
        item_idx: int | None = None,
    ) -> NDArray[np.float64] | np.float64:
        """Alias for :meth:`expected_structural_zero_fraction`."""
        return self.expected_structural_zero_fraction(theta, item_idx)


class _ConstantZeroInflationModel(_ZeroResponseModel):
    """Shared structural-zero process with an item-specific constant mass."""

    @property
    def zero_inflation(self) -> NDArray[np.float64]:
        """Item structural-zero probabilities."""
        return self._parameters["zero_inflation"]

    def structural_zero_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        pi = self._parameter_for("zero_inflation", item_idx)
        shape = (
            theta_1d.shape if item_idx is not None else (theta_1d.size, self.n_items)
        )
        return np.broadcast_to(pi, shape).copy()


class ZeroInflated2PL(_ConstantZeroInflationModel):
    """Two-parameter logistic model with item-specific structural zeros.

    ``P(X=1 | theta) = (1 - pi) * logistic(a * (theta - b))``
    """

    model_name = "ZI-2PL"
    n_params_per_item = 3
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["zero_inflation"] = np.full(
            self.n_items, 0.1, dtype=np.float64
        )

    def _evaluate(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        derivative: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        a = self._parameter_for("discrimination", item_idx)
        b = self._parameter_for("difficulty", item_idx)
        pi = self._parameter_for("zero_inflation", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        response_probability = _safe_sigmoid(a * (theta_eval - b))
        probability = (1.0 - pi) * response_probability
        if not derivative:
            return probability, None
        probability_derivative = (
            (1.0 - pi) * a * response_probability * (1.0 - response_probability)
        )
        return probability, probability_derivative


class ZeroInflated3PL(_ConstantZeroInflationModel):
    """Three-parameter logistic model with item-specific structural zeros.

    ``P(X=1 | theta) = (1 - pi) * (c + (1 - c) * logistic(a*(theta-b)))``
    """

    model_name = "ZI-3PL"
    n_params_per_item = 4
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["guessing"] = np.full(self.n_items, 0.2, dtype=np.float64)
        self._parameters["zero_inflation"] = np.full(
            self.n_items, 0.1, dtype=np.float64
        )

    @property
    def guessing(self) -> NDArray[np.float64]:
        """Item lower asymptotes."""
        return self._parameters["guessing"]

    def probability_3pl(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the response-process probability without zero inflation."""
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        a = self._parameter_for("discrimination", item_idx)
        b = self._parameter_for("difficulty", item_idx)
        c = self._parameter_for("guessing", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        response_probability = _safe_sigmoid(a * (theta_eval - b))
        return c + (1.0 - c) * response_probability

    def _evaluate(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        derivative: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        a = self._parameter_for("discrimination", item_idx)
        b = self._parameter_for("difficulty", item_idx)
        c = self._parameter_for("guessing", item_idx)
        pi = self._parameter_for("zero_inflation", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        logistic_probability = _safe_sigmoid(a * (theta_eval - b))
        response_probability = c + (1.0 - c) * logistic_probability
        probability = (1.0 - pi) * response_probability
        if not derivative:
            return probability, None
        probability_derivative = (
            (1.0 - pi)
            * (1.0 - c)
            * a
            * logistic_probability
            * (1.0 - logistic_probability)
        )
        return probability, probability_derivative


class HurdleIRT(_ZeroResponseModel):
    """IRT model with an ability-dependent engagement hurdle.

    ``P(engage) = logistic(alpha_0 + alpha_1 * theta)`` and
    ``P(X=1 | theta) = P(engage) * logistic(a * (theta - b))``.
    """

    model_name = "Hurdle"
    n_params_per_item = 4
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["engagement_intercept"] = np.full(
            self.n_items, 2.0, dtype=np.float64
        )
        self._parameters["engagement_slope"] = np.full(
            self.n_items, 0.5, dtype=np.float64
        )

    @property
    def engagement_intercept(self) -> NDArray[np.float64]:
        """Item engagement intercepts."""
        return self._parameters["engagement_intercept"]

    @property
    def engagement_slope(self) -> NDArray[np.float64]:
        """Item engagement slopes."""
        return self._parameters["engagement_slope"]

    def engagement_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the probability that the response process is engaged."""
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        intercept = self._parameter_for("engagement_intercept", item_idx)
        slope = self._parameter_for("engagement_slope", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        return _safe_sigmoid(intercept + slope * theta_eval)

    def _evaluate(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None,
        *,
        derivative: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        theta_1d, item_idx = self._prepare_evaluation(theta, item_idx)
        a = self._parameter_for("discrimination", item_idx)
        b = self._parameter_for("difficulty", item_idx)
        intercept = self._parameter_for("engagement_intercept", item_idx)
        slope = self._parameter_for("engagement_slope", item_idx)
        theta_eval = theta_1d if item_idx is not None else theta_1d[:, None]
        engagement = _safe_sigmoid(intercept + slope * theta_eval)
        response_probability = _safe_sigmoid(a * (theta_eval - b))
        probability = engagement * response_probability
        if not derivative:
            return probability, None
        probability_derivative = probability * (
            slope * (1.0 - engagement) + a * (1.0 - response_probability)
        )
        return probability, probability_derivative

    def structural_zero_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the probability of a zero caused by non-engagement."""
        return 1.0 - self.engagement_probability(theta, item_idx)
