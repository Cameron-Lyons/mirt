"""Nested-logit models for keyed multiple-choice items.

The models separate correctness from distractor choice.  A logistic item
curve determines the probability of the keyed response, while a nominal
softmax describes which distractor is selected conditional on an incorrect
response.

References
----------
Suh, Y., & Bolt, D. M. (2010). Nested logit models for multiple-choice item
response data. *Psychometrika, 75*(3), 454-473.

Thissen, D., & Steinberg, L. (1984). A response model for multiple-choice
items. *Psychometrika, 49*(4), 501-519.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import PolytomousItemModel


class TwoPLNestedLogit(PolytomousItemModel):
    """Two-parameter nested-logit model for multiple-choice items.

    The probability of a keyed response follows a 2PL curve.  Conditional
    on an incorrect response, distractors follow a nominal softmax with one
    slope and intercept per option.
    """

    model_name = "2PLNRM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        correct_response: int | list[int] = 0,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors != 1:
            raise MirtValidationError(
                "2PLNRM only supports unidimensional analysis",
                parameter="n_factors",
                value=n_factors,
                expected="1",
            )
        if isinstance(correct_response, (bool, np.bool_)):
            raw_correct: list[object] = [correct_response] * n_items
        elif isinstance(correct_response, (int, np.integer)):
            raw_correct = [correct_response] * n_items
        else:
            try:
                raw_correct = list(correct_response)
            except TypeError as exc:
                raise MirtValidationError(
                    "correct_response must be an integer or a sequence of integers",
                    parameter="correct_response",
                    value=correct_response,
                ) from exc

        self._correct = raw_correct
        super().__init__(n_items, n_categories, n_factors=1, item_names=item_names)
        self._correct = self._validate_correct_responses(raw_correct)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)

        max_categories = max(self._n_categories)
        self._parameters["distractor_slopes"] = np.zeros(
            (self.n_items, max_categories), dtype=np.float64
        )
        self._parameters["distractor_intercepts"] = np.zeros(
            (self.n_items, max_categories), dtype=np.float64
        )
        for item_idx, n_categories in enumerate(self._n_categories):
            self._parameters["distractor_intercepts"][item_idx, :n_categories] = (
                np.linspace(-0.5, 0.5, n_categories)
            )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Correct-versus-incorrect discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Correct-versus-incorrect difficulty parameters."""
        return self._parameters["difficulty"]

    @property
    def distractor_slopes(self) -> NDArray[np.float64]:
        """Conditional nominal slopes, padded to the maximum category count."""
        return self._parameters["distractor_slopes"]

    @property
    def distractor_intercepts(self) -> NDArray[np.float64]:
        """Conditional nominal intercepts, padded to the maximum category count."""
        return self._parameters["distractor_intercepts"]

    @property
    def correct_response(self) -> list[int]:
        """Return the keyed category for every item."""
        return self._correct.copy()

    def _validate_correct_responses(self, values: list[object]) -> list[int]:
        if len(values) != self.n_items:
            raise MirtValidationError(
                f"Length of correct_response ({len(values)}) must match "
                f"n_items ({self.n_items})",
                parameter="correct_response",
                value=len(values),
                expected=str(self.n_items),
            )

        result: list[int] = []
        for item_idx, (value, n_categories) in enumerate(
            zip(values, self._n_categories, strict=True)
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise MirtValidationError(
                    f"correct_response[{item_idx}] must be an integer",
                    parameter="correct_response",
                    value=value,
                    expected=f"integer in [0, {n_categories})",
                )
            key = int(value)
            if key < 0 or key >= n_categories:
                raise MirtValidationError(
                    f"correct_response[{item_idx}]={key} is out of range",
                    parameter="correct_response",
                    value=key,
                    expected=f"[0, {n_categories})",
                )
            result.append(key)
        return result

    def _validate_item_index(self, item_idx: int) -> int:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
            item_idx, (int, np.integer)
        ):
            raise IndexError("item_idx must be an integer")
        index = int(item_idx)
        if index < 0 or index >= self.n_items:
            raise IndexError(f"item_idx {index} out of range [0, {self.n_items})")
        return index

    def _validate_category(self, item_idx: int, category: int) -> int:
        if isinstance(category, (bool, np.bool_)) or not isinstance(
            category, (int, np.integer)
        ):
            raise MirtValidationError(
                "category must be an integer", parameter="category", value=category
            )
        value = int(category)
        n_categories = self._n_categories[item_idx]
        if value < 0 or value >= n_categories:
            raise MirtValidationError(
                f"Category {value} out of range [0, {n_categories})",
                parameter="category",
                value=value,
                expected=f"[0, {n_categories})",
            )
        return value

    def _validate_parameter_values(
        self, parameters: dict[str, NDArray[np.float64]]
    ) -> None:
        for name, values in parameters.items():
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                )
        if np.any(parameters["discrimination"] <= 0.0):
            raise MirtValidationError(
                "discrimination must be positive",
                parameter="discrimination",
                value=parameters["discrimination"],
                expected="> 0",
            )

        guessing = parameters.get("guessing")
        if guessing is not None and (np.any(guessing < 0.0) or np.any(guessing >= 1.0)):
            raise MirtValidationError(
                "guessing must lie in [0, 1)",
                parameter="guessing",
                value=guessing,
                expected="[0, 1)",
            )

        upper = parameters.get("upper")
        if upper is not None:
            if np.any(upper <= 0.0) or np.any(upper > 1.0):
                raise MirtValidationError(
                    "upper must lie in (0, 1]",
                    parameter="upper",
                    value=upper,
                    expected="(0, 1]",
                )
            if guessing is not None and np.any(guessing >= upper):
                raise MirtValidationError(
                    "guessing must be strictly less than upper",
                    parameter="guessing",
                    value=guessing,
                    expected="guessing < upper",
                )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set statistically valid model parameters atomically."""
        candidates = self._parameters.copy()
        converted: dict[str, NDArray[np.float64]] = {}
        for name, value in params.items():
            if name not in self._parameters:
                valid = ", ".join(self._parameters)
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid}",
                    parameter=name,
                    expected=valid,
                )
            array = np.asarray(value, dtype=np.float64)
            expected_shape = self._parameters[name].shape
            if array.shape != expected_shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {expected_shape}, "
                    f"got {array.shape}",
                    parameter=name,
                    value=array.shape,
                    expected=str(expected_shape),
                )
            candidates[name] = array
            converted[name] = array
        self._validate_parameter_values(candidates)
        return super().set_parameters(**converted)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item's parameter while preserving domain validation."""
        index = self._validate_item_index(item_idx)
        if param_name not in self._parameters:
            valid = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid}",
                parameter=param_name,
                expected=valid,
            )
        candidate = self._parameters[param_name].copy()
        try:
            candidate[index] = value
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"Invalid per-item value for {param_name}",
                parameter=param_name,
                value=value,
            ) from exc
        self.set_parameters(**{param_name: candidate})

    def _lower_asymptote(self, item_idx: int) -> float:
        return 0.0

    def _upper_asymptote(self, item_idx: int) -> float:
        return 1.0

    def _validate_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        values = self._ensure_theta_2d(theta)
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "theta must contain only finite values",
                parameter="theta",
                value=theta,
            )
        return values[:, 0]

    def _item_curves_from_theta(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return category probabilities and their ability derivatives."""
        index = self._validate_item_index(item_idx)
        n_categories = self._n_categories[index]
        correct = self._correct[index]
        discrimination = float(self._parameters["discrimination"][index])
        difficulty = float(self._parameters["difficulty"][index])
        slopes = self._parameters["distractor_slopes"][index, :n_categories]
        intercepts = self._parameters["distractor_intercepts"][index, :n_categories]
        lower = self._lower_asymptote(index)
        upper = self._upper_asymptote(index)

        if (
            not np.all(np.isfinite([discrimination, difficulty, lower, upper]))
            or not np.all(np.isfinite(slopes))
            or not np.all(np.isfinite(intercepts))
            or discrimination <= 0.0
            or lower < 0.0
            or upper > 1.0
            or lower >= upper
        ):
            raise MirtValidationError(
                f"Stored parameters for item {index} are invalid",
                parameter="parameters",
            )

        with np.errstate(over="ignore", invalid="ignore"):
            logistic = sigmoid(discrimination * (theta - difficulty))
        probability_correct = lower + (upper - lower) * logistic
        derivative_correct = (
            (upper - lower) * discrimination * logistic * (1.0 - logistic)
        )

        logits = theta[:, None] * slopes[None, :] + intercepts[None, :]
        logits[:, correct] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        conditional = np.exp(logits)
        conditional /= conditional.sum(axis=1, keepdims=True)
        mean_slope = np.sum(conditional * slopes[None, :], axis=1, keepdims=True)
        conditional_derivative = conditional * (slopes[None, :] - mean_slope)

        probability = (1.0 - probability_correct[:, None]) * conditional
        derivative = (
            -derivative_correct[:, None] * conditional
            + (1.0 - probability_correct[:, None]) * conditional_derivative
        )
        probability[:, correct] = probability_correct
        derivative[:, correct] = derivative_correct
        return probability, derivative

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute the probability of one response category."""
        theta_values = self._validate_theta(theta)
        index = self._validate_item_index(item_idx)
        selected = self._validate_category(index, category)
        probability, _ = self._item_curves_from_theta(theta_values, index)
        return probability[:, selected]

    def category_derivative(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute category-probability derivatives with respect to ability."""
        theta_values = self._validate_theta(theta)
        index = self._validate_item_index(item_idx)
        _, derivative = self._item_curves_from_theta(theta_values, index)
        if category is None:
            return derivative
        return derivative[:, self._validate_category(index, category)]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities with one softmax per item."""
        theta_values = self._validate_theta(theta)
        return self._probability_from_theta(theta_values, item_idx)

    def _probability_from_theta(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        if item_idx is not None:
            probability, _ = self._item_curves_from_theta(theta, item_idx)
            return probability

        result = np.zeros(
            (theta.size, self.n_items, self.max_categories),
            dtype=np.float64,
        )
        for index, n_categories in enumerate(self._n_categories):
            probability, _ = self._item_curves_from_theta(theta, index)
            result[:, index, :n_categories] = probability
        return result

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return all category curves for one item."""
        return self.probability(theta, item_idx)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute exact nominal-response Fisher information."""
        theta_values = self._validate_theta(theta)
        probability, derivative = self._item_curves_from_theta(theta_values, item_idx)
        denominator = np.clip(probability, PROB_EPSILON, None)
        return np.sum(derivative**2 / denominator, axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute exact Fisher information without repeated theta validation."""
        theta_values = self._validate_theta(theta)
        if item_idx is not None:
            probability, derivative = self._item_curves_from_theta(
                theta_values, item_idx
            )
            return np.sum(
                derivative**2 / np.clip(probability, PROB_EPSILON, None), axis=1
            )

        result = np.zeros(theta_values.size, dtype=np.float64)
        for index in range(self.n_items):
            probability, derivative = self._item_curves_from_theta(theta_values, index)
            result += np.sum(
                derivative**2 / np.clip(probability, PROB_EPSILON, None), axis=1
            )
        return result

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected numeric category scores."""
        theta_values = self._validate_theta(theta)
        if item_idx is not None:
            index = self._validate_item_index(item_idx)
            probability, _ = self._item_curves_from_theta(theta_values, index)
            return probability @ np.arange(self._n_categories[index])

        result = np.zeros(theta_values.size, dtype=np.float64)
        for index, n_categories in enumerate(self._n_categories):
            probability, _ = self._item_curves_from_theta(theta_values, index)
            result += probability @ np.arange(n_categories)
        return result

    def _validate_responses(self, responses: NDArray[np.int_]) -> NDArray[np.int_]:
        raw = np.asarray(responses)
        if raw.ndim != 2:
            raise MirtDataError(f"responses must be 2D, got {raw.ndim}D")
        if raw.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {raw.shape[1]} items, expected {self.n_items}",
                n_items=raw.shape[1],
                expected_n_items=self.n_items,
            )
        try:
            numeric = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise MirtDataError(
                "responses must contain numeric category codes"
            ) from exc
        if not np.all(np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise MirtDataError("responses must contain integer category codes")
        values = numeric.astype(np.int_, copy=False)
        if np.any(values < -1):
            raise MirtDataError("missing responses must be coded as -1")
        for item_idx, n_categories in enumerate(self._n_categories):
            observed = values[:, item_idx] >= 0
            if np.any(values[observed, item_idx] >= n_categories):
                raise MirtDataError(
                    f"Item {item_idx} responses must lie in [0, {n_categories})",
                    item_idx=item_idx,
                    n_categories=n_categories,
                )
        return values

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute paired response-pattern log likelihoods."""
        values = self._validate_responses(responses)
        theta_values = self._validate_theta(theta)
        if values.shape[0] != theta_values.size:
            raise MirtDataError(
                "responses and theta must have equal row counts",
                n_persons=values.shape[0],
                n_theta=theta_values.size,
            )

        probabilities = self._probability_from_theta(theta_values)
        observed = values >= 0
        category_indices = np.where(observed, values, 0)
        selected = np.take_along_axis(
            probabilities, category_indices[:, :, None], axis=2
        )[:, :, 0]
        log_selected = np.log(np.clip(selected, PROB_EPSILON, 1.0))
        return np.where(observed, log_selected, 0.0).sum(axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate all response patterns at every supplied ability point."""
        values = self._validate_responses(responses)
        theta_values = self._validate_theta(theta)
        result = np.zeros((values.shape[0], theta_values.size), dtype=np.float64)
        for item_idx in range(self.n_items):
            probability, _ = self._item_curves_from_theta(theta_values, item_idx)
            log_probability = np.log(np.clip(probability, PROB_EPSILON, 1.0))
            responses_item = values[:, item_idx]
            observed = responses_item >= 0
            if np.any(observed):
                result[observed] += log_probability[:, responses_item[observed]].T
        return result

    def copy(self) -> Self:
        """Create a deep copy while preserving the item answer key."""
        new_model = self.__class__(
            n_items=self.n_items,
            n_categories=self._n_categories.copy(),
            correct_response=self._correct.copy(),
            n_factors=1,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class ThreePLNestedLogit(TwoPLNestedLogit):
    """Nested-logit model with a lower asymptote for keyed responses."""

    model_name = "3PLNRM"

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self._parameters["guessing"] = np.full(self.n_items, 0.2, dtype=np.float64)

    @property
    def guessing(self) -> NDArray[np.float64]:
        """Lower asymptotes for keyed-response curves."""
        return self._parameters["guessing"]

    def _lower_asymptote(self, item_idx: int) -> float:
        return float(self._parameters["guessing"][item_idx])


class FourPLNestedLogit(ThreePLNestedLogit):
    """Nested-logit model with lower and upper keyed-response asymptotes."""

    model_name = "4PLNRM"

    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self._parameters["upper"] = np.ones(self.n_items, dtype=np.float64)

    @property
    def upper(self) -> NDArray[np.float64]:
        """Upper asymptotes for keyed-response curves."""
        return self._parameters["upper"]

    def _upper_asymptote(self, item_idx: int) -> float:
        return float(self._parameters["upper"][item_idx])
