from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.models.base import DichotomousItemModel


class MultidimensionalModel(DichotomousItemModel):
    model_name = "MIRT"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
        model_type: Literal["exploratory", "confirmatory"] = "exploratory",
        loading_pattern: NDArray[np.float64] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("MultidimensionalModel requires n_factors >= 2")

        self.model_type = model_type

        if model_type == "confirmatory":
            if loading_pattern is None:
                raise ValueError("loading_pattern required for confirmatory model")
            loading_pattern = np.asarray(loading_pattern)
            if loading_pattern.shape != (n_items, n_factors):
                raise ValueError(
                    f"loading_pattern shape {loading_pattern.shape} doesn't match "
                    f"(n_items={n_items}, n_factors={n_factors})"
                )
            self._loading_pattern = loading_pattern
        else:
            self._loading_pattern = np.ones((n_items, n_factors))

        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        slopes = np.ones((self.n_items, self.n_factors)) * 0.8
        slopes = slopes * self._loading_pattern

        self._parameters["slopes"] = slopes
        self._parameters["intercepts"] = np.zeros(self.n_items)

    @property
    def slopes(self) -> NDArray[np.float64]:
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    @property
    def loading_pattern(self) -> NDArray[np.float64]:
        return self._loading_pattern.copy()

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["slopes"]
        d = self._parameters["intercepts"]

        if item_idx is not None:
            z = np.dot(theta, a[item_idx]) + d[item_idx]
            return sigmoid(z)

        z = np.dot(theta, a.T) + d[None, :]
        return sigmoid(z)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["slopes"]

        if item_idx is not None:
            a_sq_sum = np.sum(a[item_idx] ** 2)
            return a_sq_sum * p * q

        a_sq_sum = np.sum(a**2, axis=1)
        return a_sq_sum[None, :] * p * q

    def to_irt_parameterization(self) -> dict[str, NDArray[np.float64]]:
        a = self._parameters["slopes"]
        d = self._parameters["intercepts"]

        a_sum = np.sum(a, axis=1)
        b = -d / (a_sum + PROB_EPSILON)

        return {
            "discrimination": a.copy(),
            "difficulty": b,
        }

    def get_factor_loadings(
        self,
        standardized: bool = True,
    ) -> NDArray[np.float64]:
        a = self._parameters["slopes"]

        if not standardized:
            return a.copy()

        a_sq_sum = np.sum(a**2, axis=1, keepdims=True)
        denominator = np.sqrt(1 + a_sq_sum)
        return a / denominator

    def communalities(self) -> NDArray[np.float64]:
        loadings = self.get_factor_loadings(standardized=True)
        return np.sum(loadings**2, axis=1)

    def set_parameters(self, **params: NDArray[np.float64]) -> "MultidimensionalModel":
        if "slopes" in params:
            slopes = np.asarray(params["slopes"])
            slopes = slopes * self._loading_pattern
            params["slopes"] = slopes

        return super().set_parameters(**params)
