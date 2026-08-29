"""Sequential and adjacent-category ordinal response models.

Sequential models factor an ordinal response into conditional binary steps.
Adjacent-category models instead parameterize neighboring category log odds.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError
from mirt.models.base import PolytomousItemModel


def _safe_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the shared sigmoid without harmless tail warnings."""
    with np.errstate(over="ignore", invalid="ignore"):
        return sigmoid(values)


class _OrdinalLogitModel(PolytomousItemModel):
    """Shared validation, likelihood, and summaries for ordinal logits."""

    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)):
            raise MirtValidationError(
                "n_items must be an integer",
                parameter="n_items",
                value=n_items,
                expected="positive integer",
            )
        if n_factors != 1:
            raise MirtValidationError(
                f"{self.model_name} only supports one factor",
                parameter="n_factors",
                value=n_factors,
                expected="1",
            )

        if isinstance(n_categories, bool):
            category_counts: int | list[int] = int(n_categories)
        elif isinstance(n_categories, (int, np.integer)):
            category_counts = int(n_categories)
        else:
            try:
                category_counts = list(n_categories)
            except TypeError as exc:
                raise MirtValidationError(
                    "n_categories must be an integer or one value per item",
                    parameter="n_categories",
                    value=n_categories,
                    expected="integer or list of integers",
                ) from exc
            for value in category_counts:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise MirtValidationError(
                        "n_categories must contain integers",
                        parameter="n_categories",
                        value=value,
                        expected="integers >= 2",
                    )
            category_counts = [int(value) for value in category_counts]

        super().__init__(int(n_items), category_counts, 1, item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        thresholds = np.zeros((self.n_items, self.max_categories - 1), dtype=np.float64)
        for item_idx, n_categories in enumerate(self._n_categories):
            thresholds[item_idx, : n_categories - 1] = np.linspace(
                -2.0, 2.0, n_categories - 1
            )
        self._parameters["thresholds"] = thresholds

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discrimination parameters."""
        return self._parameters["discrimination"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Threshold matrix, padded to the largest category count."""
        return self._parameters["thresholds"]

    def thresholds_for_item(self, item_idx: int) -> NDArray[np.float64]:
        """Return only the active thresholds for one item."""
        item_idx = self._validate_item_idx(item_idx)
        self._validate_parameter_state(self._parameters)
        return self.thresholds[item_idx, : self._n_categories[item_idx] - 1].copy()

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        expected_shapes = {
            "discrimination": (self.n_items,),
            "thresholds": (self.n_items, self.max_categories - 1),
        }
        for name, expected_shape in expected_shapes.items():
            values = parameters[name]
            if values.shape != expected_shape:
                raise MirtValidationError(
                    f"{name} must have shape {expected_shape}",
                    parameter=name,
                    value=values.shape,
                    expected=str(expected_shape),
                )
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                    expected="finite values",
                )
        if np.any(parameters["discrimination"] <= 0.0):
            raise MirtValidationError(
                "discrimination must be strictly positive",
                parameter="discrimination",
                value=parameters["discrimination"],
                expected="> 0",
            )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters atomically after validating their full state."""
        candidate = {name: values.copy() for name, values in self._parameters.items()}
        for name, value in params.items():
            if name not in candidate:
                valid = ", ".join(candidate)
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid}",
                    parameter=name,
                    expected=valid,
                )
            value_array = np.asarray(value, dtype=np.float64)
            if value_array.shape != candidate[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {candidate[name].shape}, "
                    f"got {value_array.shape}",
                    parameter=name,
                    value=value_array.shape,
                    expected=str(candidate[name].shape),
                )
            candidate[name] = value_array.copy()
        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item parameter and retain a valid model state."""
        item_idx = self._validate_item_idx(item_idx)
        if param_name not in self._parameters:
            valid = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid}",
                parameter=param_name,
                expected=valid,
            )

        updated = self._parameters[param_name].copy()
        value_array = np.asarray(value, dtype=np.float64)
        if param_name == "discrimination":
            if value_array.ndim != 0:
                raise MirtValidationError(
                    "discrimination must be scalar for one item",
                    parameter=param_name,
                    value=value_array.shape,
                    expected="scalar",
                )
            updated[item_idx] = float(value_array)
        else:
            n_active = self._n_categories[item_idx] - 1
            if value_array.shape == (n_active,):
                updated[item_idx, :n_active] = value_array
            elif value_array.shape == (self.max_categories - 1,):
                updated[item_idx] = value_array
            else:
                raise MirtValidationError(
                    f"thresholds for item {item_idx} must have {n_active} active values",
                    parameter=param_name,
                    value=value_array.shape,
                    expected=f"({n_active},)",
                )
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

    def _validate_category(self, item_idx: int, category: int) -> int:
        if isinstance(category, bool) or not isinstance(category, (int, np.integer)):
            raise MirtValidationError(
                "category must be an integer",
                parameter="category",
                value=category,
                expected="integer",
            )
        category = int(category)
        n_categories = self._n_categories[item_idx]
        if category < 0 or category >= n_categories:
            raise IndexError(
                f"Category {category} out of range [0, {n_categories}) for item {item_idx}"
            )
        return category

    def _prepare_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
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
        self._validate_parameter_state(self._parameters)
        return theta_2d[:, 0]

    def _item_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def _all_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def _all_item_information(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute category probabilities for one item or all items."""
        theta_1d = self._prepare_theta(theta)
        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
            return self._item_probabilities(theta_1d, item_idx)
        return self._all_probabilities(theta_1d)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute one category response curve."""
        item_idx = self._validate_item_idx(item_idx)
        category = self._validate_category(item_idx, category)
        return self.probability(theta, item_idx)[:, category]

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all category curves for one item."""
        return self.probability(theta, item_idx)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        theta_1d = self._prepare_theta(theta)
        item_idx = self._validate_item_idx(item_idx)
        return self._all_item_information(theta_1d)[:, item_idx]

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute exact Fisher information for one item or the test."""
        theta_1d = self._prepare_theta(theta)
        information = self._all_item_information(theta_1d)
        if item_idx is not None:
            return information[:, self._validate_item_idx(item_idx)]
        return np.sum(information, axis=1)

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute expected item or total score."""
        probabilities = self.probability(theta, item_idx)
        if item_idx is not None:
            categories = np.arange(probabilities.shape[1])
            return np.sum(probabilities * categories[None, :], axis=1)
        categories = np.arange(self.max_categories)
        return np.sum(probabilities * categories[None, None, :], axis=(1, 2))

    def _validated_responses(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
        values = np.asarray(responses)
        if values.ndim != 2 or values.shape[1:] != (self.n_items,):
            raise MirtValidationError(
                f"responses must have shape (n_persons, {self.n_items})",
                parameter="responses",
                value=values.shape,
                expected=f"(n_persons, {self.n_items})",
            )
        if values.shape[0] == 0:
            raise MirtValidationError(
                "responses must contain at least one person",
                parameter="responses",
                value=values.shape,
                expected="at least one row",
            )
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "responses must be numeric",
                parameter="responses",
                value=values,
                expected="integer categories or negative/NaN missing values",
            ) from exc

        finite = np.isfinite(numeric)
        missing = np.isnan(numeric) | (finite & (numeric < 0.0))
        observed = finite & (numeric >= 0.0)
        invalid = (~missing & ~observed) | (observed & (numeric != np.floor(numeric)))
        safe = np.where(observed, numeric, 0.0).astype(np.int64)
        category_counts = np.asarray(self._n_categories)[None, :]
        invalid |= observed & (safe >= category_counts)
        if np.any(invalid):
            raise MirtValidationError(
                "observed responses must be valid integer categories",
                parameter="responses",
                value=numeric[invalid],
                expected="an item category or negative/NaN missing value",
            )
        return safe, observed

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute response log likelihood per person."""
        safe_responses, observed = self._validated_responses(responses)
        probabilities = self.probability(theta)
        if probabilities.shape[0] != safe_responses.shape[0]:
            raise MirtValidationError(
                "theta and responses must contain the same number of persons",
                parameter="theta",
                value=probabilities.shape[0],
                expected=str(safe_responses.shape[0]),
            )
        selected = np.take_along_axis(
            probabilities, safe_responses[:, :, None], axis=2
        )[:, :, 0]
        log_probability = np.log(np.clip(selected, PROB_EPSILON, 1.0))
        return np.sum(np.where(observed, log_probability, 0.0), axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute each response pattern's likelihood over an ability grid."""
        safe_responses, observed = self._validated_responses(responses)
        probabilities = self.probability(theta)
        log_probabilities = np.log(np.clip(probabilities, PROB_EPSILON, 1.0))
        result = np.zeros((safe_responses.shape[0], probabilities.shape[0]))
        category_mask = np.empty_like(observed)
        for category in range(self.max_categories):
            np.equal(safe_responses, category, out=category_mask)
            np.logical_and(category_mask, observed, out=category_mask)
            if np.any(category_mask):
                result += category_mask @ log_probabilities[:, :, category].T
        return result


class _SequentialProcessModel(_OrdinalLogitModel):
    """Shared forward continuation-ratio likelihood."""

    def _item_step_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_steps = self._n_categories[item_idx] - 1
        logits = self.discrimination[item_idx] * (
            theta[:, None] - self.thresholds[item_idx, :n_steps][None, :]
        )
        return _safe_sigmoid(logits)

    def _all_step_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        logits = self.discrimination[None, :, None] * (
            theta[:, None, None] - self.thresholds[None, :, :]
        )
        return _safe_sigmoid(logits)

    def step_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        step_idx: int,
    ) -> NDArray[np.float64]:
        """Return the conditional probability of passing one reached step."""
        theta_1d = self._prepare_theta(theta)
        item_idx = self._validate_item_idx(item_idx)
        step_idx = self._validate_category(item_idx, step_idx)
        if step_idx == self._n_categories[item_idx] - 1:
            raise IndexError(
                f"Step {step_idx} out of range [0, {self._n_categories[item_idx] - 1})"
            )
        return self._item_step_probabilities(theta_1d, item_idx)[:, step_idx]

    def _step_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        step_idx: int,
    ) -> NDArray[np.float64]:
        """Backward-compatible protected alias for :meth:`step_probability`."""
        return self.step_probability(theta, item_idx, step_idx)

    def reach_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute ``P(X >= category | theta)``."""
        theta_1d = self._prepare_theta(theta)
        item_idx = self._validate_item_idx(item_idx)
        category = self._validate_category(item_idx, category)
        if category == 0:
            return np.ones_like(theta_1d)
        steps = self._item_step_probabilities(theta_1d, item_idx)
        return np.prod(steps[:, :category], axis=1)

    def _item_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        steps = self._item_step_probabilities(theta, item_idx)
        reach = np.concatenate(
            [np.ones((theta.size, 1)), np.cumprod(steps, axis=1)], axis=1
        )
        failure = np.concatenate([1.0 - steps, np.ones((theta.size, 1))], axis=1)
        return reach * failure

    def _all_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        steps = self._all_step_probabilities(theta)
        reach = np.concatenate(
            [
                np.ones((theta.size, self.n_items, 1)),
                np.cumprod(steps, axis=2),
            ],
            axis=2,
        )
        failure = np.concatenate(
            [1.0 - steps, np.ones((theta.size, self.n_items, 1))], axis=2
        )
        categories = np.arange(self.max_categories)[None, None, :]
        counts = np.asarray(self._n_categories)[None, :, None]
        failure = np.where(categories < counts - 1, failure, 1.0)
        return reach * failure * (categories < counts)

    def _all_item_information(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        steps = self._all_step_probabilities(theta)
        reach = np.concatenate(
            [
                np.ones((theta.size, self.n_items, 1)),
                np.cumprod(steps[:, :, :-1], axis=2),
            ],
            axis=2,
        )
        step_indices = np.arange(self.max_categories - 1)[None, None, :]
        active = step_indices < np.asarray(self._n_categories)[None, :, None] - 1
        conditional_information = reach * steps * (1.0 - steps) * active
        return self.discrimination[None, :] ** 2 * np.sum(
            conditional_information, axis=2
        )


class SequentialResponseModel(_SequentialProcessModel):
    """Sequential model for ordered responses built from conditional steps."""

    model_name = "Sequential"


class ContinuationRatioModel(_SequentialProcessModel):
    """Forward continuation-ratio model for a stopping process."""

    model_name = "ContinuationRatio"


class AdjacentCategoryModel(_OrdinalLogitModel):
    """Ordinal model for neighboring-category log odds."""

    model_name = "AdjacentCategory"

    def _item_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        n_categories = self._n_categories[item_idx]
        increments = self.discrimination[item_idx] * (
            theta[:, None] - self.thresholds[item_idx, : n_categories - 1][None, :]
        )
        logits = np.concatenate(
            [np.zeros((theta.size, 1)), np.cumsum(increments, axis=1)], axis=1
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def _all_probabilities(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        increments = self.discrimination[None, :, None] * (
            theta[:, None, None] - self.thresholds[None, :, :]
        )
        logits = np.concatenate(
            [
                np.zeros((theta.size, self.n_items, 1)),
                np.cumsum(increments, axis=2),
            ],
            axis=2,
        )
        categories = np.arange(self.max_categories)[None, None, :]
        active = categories < np.asarray(self._n_categories)[None, :, None]
        logits = np.where(active, logits, -np.inf)
        logits -= np.max(logits, axis=2, keepdims=True)
        weights = np.where(active, np.exp(logits), 0.0)
        return weights / np.sum(weights, axis=2, keepdims=True)

    def _all_item_information(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        probabilities = self._all_probabilities(theta)
        categories = np.arange(self.max_categories)[None, None, :]
        expected = np.sum(probabilities * categories, axis=2)
        variance = np.sum(probabilities * categories**2, axis=2) - expected**2
        return self.discrimination[None, :] ** 2 * variance
