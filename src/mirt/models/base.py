from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


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
            raise ValueError("n_items must be positive")
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        if n_factors > 1 and not self.supports_multidimensional:
            raise ValueError(
                f"{self.model_name} does not support multidimensional models"
            )

        self.n_items = n_items
        self.n_factors = n_factors
        self.item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if len(self.item_names) != n_items:
            raise ValueError(
                f"Length of item_names ({len(self.item_names)}) must match n_items ({n_items})"
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
        return sum(p.size for p in self._parameters.values())

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        for name, value in params.items():
            if name not in self._parameters:
                valid_params = ", ".join(self._parameters.keys())
                raise ValueError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}"
                )
            value_arr = np.asarray(value, dtype=np.float64)
            if value_arr.shape != self._parameters[name].shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {self._parameters[name].shape}, "
                    f"got {value_arr.shape}"
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
            ValueError: If param_name is not a valid parameter.
        """
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters.keys())
            raise ValueError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}"
            )

        values = self._parameters[param_name]
        if values.ndim == 1 and len(values) == self.n_items:
            values[item_idx] = float(value)
        elif values.ndim == 2 and values.shape[0] == self.n_items:
            values[item_idx] = np.asarray(value, dtype=np.float64)
        else:
            raise ValueError(f"Parameter {param_name} does not have per-item values")

    def _ensure_theta_2d(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2:
            raise ValueError(f"theta must be 1D or 2D, got {theta.ndim}D")
        if theta.shape[1] != self.n_factors:
            raise ValueError(
                f"theta has {theta.shape[1]} factors, expected {self.n_factors}"
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
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}"
            )

        p = self.probability(theta)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

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
                raise ValueError(
                    f"Length of n_categories ({len(n_categories)}) must match n_items ({n_items})"
                )
            self._n_categories = list(n_categories)

        for i, n_cat in enumerate(self._n_categories):
            if n_cat < 2:
                raise ValueError(f"Item {i} has {n_cat} categories; minimum is 2")

        super().__init__(n_items, n_factors, item_names)

    @property
    def n_categories(self) -> list[int]:
        return self._n_categories.copy()

    @property
    def max_categories(self) -> int:
        return max(self._n_categories)

    @abstractmethod
    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]: ...

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            n_cat = self._n_categories[item_idx]
            probs = np.zeros((n_persons, n_cat))
            for k in range(n_cat):
                probs[:, k] = self.category_probability(theta, item_idx, k)
            return probs

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            for k in range(n_cat):
                probs[:, i, k] = self.category_probability(theta, i, k)

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
            expected = np.zeros(n_persons)
            for k in range(n_cat):
                expected += k * self.category_probability(theta, item_idx, k)
            return expected

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
        n_persons = theta.shape[0]
        n_cat = self._n_categories[item_idx]

        curves = np.zeros((n_persons, n_cat))
        for k in range(n_cat):
            curves[:, k] = self.category_probability(theta, item_idx, k)

        return curves

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        ll = np.zeros(n_persons)

        for i in range(self.n_items):
            for person in range(n_persons):
                resp = responses[person, i]
                if resp >= 0:
                    prob = self.category_probability(
                        theta[person : person + 1], i, resp
                    )
                    ll[person] += np.log(prob[0] + 1e-10)

        return ll
