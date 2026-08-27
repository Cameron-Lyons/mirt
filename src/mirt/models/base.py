from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError


class BaseItemModel(ABC):
    model_name: str = "BaseModel"
    n_params_per_item: int = 0
    supports_multidimensional: bool = False

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_items <= 0:
            raise MirtValidationError(
                "n_items must be positive",
                parameter="n_items",
                value=n_items,
                expected="> 0",
            )
        if n_factors <= 0:
            raise MirtValidationError(
                "n_factors must be positive",
                parameter="n_factors",
                value=n_factors,
                expected="> 0",
            )
        if n_factors > 1 and not self.supports_multidimensional:
            raise MirtModelError(
                f"{self.model_name} does not support multidimensional models",
                model_type=self.model_name,
                n_factors=n_factors,
            )

        self.n_items = n_items
        self.n_factors = n_factors
        self.item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if len(self.item_names) != n_items:
            raise MirtValidationError(
                f"Length of item_names ({len(self.item_names)}) must match n_items ({n_items})",
                parameter="item_names",
                value=len(self.item_names),
                expected=str(n_items),
            )

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._is_fitted: bool = False
        self._initialize_parameters()

    @property
    def is_polytomous(self) -> bool:
        return hasattr(self, "_n_categories")

    @abstractmethod
    def _initialize_parameters(self) -> None: ...

    @abstractmethod
    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]: ...

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._parameters.items()}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        return sum(
            int(np.count_nonzero(mask)) for mask in self.free_parameter_masks.values()
        )

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Boolean masks identifying statistically free stored parameters."""
        return {
            name: np.ones(values.shape, dtype=np.bool_)
            for name, values in self._parameters.items()
        }

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return an identified full-storage representation for estimation."""
        return np.asarray(values, dtype=np.float64).copy()

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        for name, value in params.items():
            if name not in self._parameters:
                valid_params = ", ".join(self._parameters.keys())
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}",
                    parameter=name,
                    expected=valid_params,
                )
            value_arr = np.asarray(value, dtype=np.float64)
            if value_arr.shape != self._parameters[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {self._parameters[name].shape}, "
                    f"got {value_arr.shape}",
                    parameter=name,
                    value=value_arr.shape,
                    expected=str(self._parameters[name].shape),
                )
            self._parameters[name] = value_arr
        return self

    def get_item_parameters(
        self, item_idx: int
    ) -> dict[str, float | NDArray[np.float64]]:
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")

        result: dict[str, float | NDArray[np.float64]] = {}
        for name, values in self._parameters.items():
            if values.ndim == 1 and len(values) == self.n_items:
                result[name] = float(values[item_idx])
            elif values.ndim == 2 and values.shape[0] == self.n_items:
                result[name] = values[item_idx].copy()
            else:
                result[name] = values.copy()
        return result

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set a parameter value for a specific item.

        Args:
            item_idx: Index of the item (0-based).
            param_name: Name of the parameter to set.
            value: New value for the parameter.

        Raises:
            IndexError: If item_idx is out of range.
            MirtValidationError: If param_name is not a valid parameter.
        """
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters.keys())
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        values = self._parameters[param_name]
        if values.ndim == 1 and len(values) == self.n_items:
            values[item_idx] = float(value)
        elif values.ndim == 2 and values.shape[0] == self.n_items:
            values[item_idx] = np.asarray(value, dtype=np.float64)
        else:
            raise MirtValidationError(
                f"Parameter {param_name} does not have per-item values",
                parameter=param_name,
            )

    def _ensure_theta_2d(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2:
            raise MirtValidationError(
                f"theta must be 1D or 2D, got {theta.ndim}D",
                parameter="theta",
                value=theta.ndim,
                expected="1 or 2",
            )
        if theta.shape[1] != self.n_factors:
            raise MirtValidationError(
                f"theta has {theta.shape[1]} factors, expected {self.n_factors}",
                parameter="theta",
                value=theta.shape[1],
                expected=str(self.n_factors),
            )
        return theta

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._is_fitted = self._is_fitted
        return new_model

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"n_items={self.n_items}, "
            f"n_factors={self.n_factors}, "
            f"{status})"
        )


class DichotomousItemModel(BaseItemModel):
    def icc(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Item characteristic curve (alias for probability)."""
        return self.probability(theta, item_idx)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        if responses.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}",
                n_items=responses.shape[1],
            )

        p = self.probability(theta)
        p = np.clip(p, PROB_EPSILON, 1.0 - PROB_EPSILON)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at all theta points.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.
        theta : ndarray of shape (n_theta, n_factors)
            Ability values at which to compute likelihood.

        Returns
        -------
        ndarray of shape (n_persons, n_theta)
            Log-likelihood for each person at each theta point.
        """
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta)
        p = np.clip(p, PROB_EPSILON, 1.0 - PROB_EPSILON)
        log_p = np.log(p)
        log_1_minus_p = np.log1p(-p)

        valid = responses >= 0
        response_values = np.where(valid, responses, 0).astype(np.float64, copy=False)
        observed = valid.astype(np.float64)

        return (
            response_values @ log_p.T + (observed - response_values) @ log_1_minus_p.T
        )

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        probs = self.probability(theta, item_idx)
        if item_idx is None:
            return np.sum(probs, axis=1)
        return probs


class PolytomousItemModel(BaseItemModel):
    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if isinstance(n_categories, int):
            self._n_categories = [n_categories] * n_items
        else:
            if len(n_categories) != n_items:
                raise MirtValidationError(
                    f"Length of n_categories ({len(n_categories)}) must match n_items ({n_items})",
                    parameter="n_categories",
                    value=len(n_categories),
                    expected=str(n_items),
                )
            self._n_categories = list(n_categories)

        for i, n_cat in enumerate(self._n_categories):
            if n_cat < 2:
                raise MirtValidationError(
                    f"Item {i} has {n_cat} categories; minimum is 2",
                    parameter="n_categories",
                    value=n_cat,
                    expected=">= 2",
                )

        super().__init__(n_items, n_factors, item_names)

    @property
    def n_categories(self) -> list[int]:
        return self._n_categories.copy()

    @property
    def max_categories(self) -> int:
        return max(self._n_categories)

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_categories=self._n_categories.copy(),
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        new_model._is_fitted = self._is_fitted
        return new_model

    @abstractmethod
    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]: ...

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all category probabilities for one item."""
        n_cat = self._n_categories[item_idx]
        probabilities = np.empty((theta.shape[0], n_cat), dtype=np.float64)
        for category in range(n_cat):
            probabilities[:, category] = self.category_probability(
                theta, item_idx, category
            )
        return probabilities

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._category_probabilities(theta, item_idx)

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            probs[:, i, :n_cat] = self._category_probabilities(theta, i)

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._item_information(theta, item_idx)

        info = np.zeros(n_persons)
        for i in range(self.n_items):
            info += self._item_information(theta, i)

        return info

    @abstractmethod
    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]: ...

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probabilities = self._category_probabilities(theta, item_idx)
            return probabilities @ np.arange(n_cat)

        total_expected = np.zeros(n_persons)
        for i in range(self.n_items):
            total_expected += self.expected_score(theta, i)
        return total_expected

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        return self._category_probabilities(theta, item_idx)

    def _validate_polytomous_responses(
        self,
        responses: NDArray[np.int_],
    ) -> NDArray:
        """Validate a polytomous response matrix for likelihood evaluation."""
        responses = np.asarray(responses)
        if responses.ndim != 2:
            raise MirtDataError(f"responses must be 2D, got {responses.ndim}D")
        if responses.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}",
                n_items=responses.shape[1],
            )
        if responses.dtype.kind not in "biuf":
            raise MirtDataError("responses must contain numeric category codes")

        observed = responses >= 0
        if responses.dtype.kind == "f" and np.any(
            observed & (~np.isfinite(responses) | (responses != np.trunc(responses)))
        ):
            raise MirtDataError("responses must contain integer category codes")

        n_categories = np.asarray(self._n_categories)
        invalid = observed & (responses >= n_categories[None, :])
        if np.any(invalid):
            item_idx = int(np.flatnonzero(np.any(invalid, axis=0))[0])
            raise MirtDataError(
                f"responses for item {item_idx} must be below "
                f"{self._n_categories[item_idx]}"
            )
        return responses

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = self._validate_polytomous_responses(responses)
        theta = self._ensure_theta_2d(theta)
        n_response_rows = responses.shape[0]
        n_theta_rows = theta.shape[0]

        if (
            n_response_rows != n_theta_rows
            and n_response_rows != 1
            and n_theta_rows != 1
        ):
            raise MirtDataError(
                "responses and theta must have matching row counts or a single row"
            )

        n_rows = max(n_response_rows, n_theta_rows)
        ll = np.zeros(n_rows, dtype=np.float64)
        row_indices = np.arange(n_rows)

        for item_idx in range(self.n_items):
            item_responses = np.broadcast_to(responses[:, item_idx], (n_rows,))
            valid = item_responses >= 0
            if not np.any(valid):
                continue

            probabilities = self._category_probabilities(theta, item_idx)
            probabilities = np.broadcast_to(
                probabilities, (n_rows, probabilities.shape[1])
            )
            response_indices = np.where(valid, item_responses, 0).astype(
                np.intp, copy=False
            )
            selected = probabilities[row_indices[valid], response_indices[valid]]
            ll[valid] += np.log(np.clip(selected, PROB_EPSILON, 1.0))

        return ll

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at all theta points.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.
        theta : ndarray of shape (n_theta, n_factors)
            Ability values at which to compute likelihood.

        Returns
        -------
        ndarray of shape (n_persons, n_theta)
            Log-likelihood for each person at each theta point.
        """
        responses = self._validate_polytomous_responses(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = responses.shape[0]
        n_theta = theta.shape[0]

        ll = np.zeros((n_persons, n_theta))

        for item_idx in range(self.n_items):
            probs = self._category_probabilities(theta, item_idx)
            probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)
            log_probs = np.log(probs)

            item_resp = responses[:, item_idx]
            valid_mask = item_resp >= 0

            if np.any(valid_mask):
                response_indices = item_resp[valid_mask].astype(np.intp, copy=False)
                ll[valid_mask, :] += log_probs[:, response_indices].T

        return ll
