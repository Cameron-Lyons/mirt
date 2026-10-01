from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._logistic import (
    _affine_logits,
    _affine_probability,
    _information,
    _item_information,
    _test_information,
)
from mirt._model_defaults import register_builtin_model as _register_builtin_model
from mirt.constants import PROB_EPSILON
from mirt.models.base import DichotomousItemModel


@_register_builtin_model
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
            self._loading_pattern = loading_pattern.copy()
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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        masks["slopes"] &= self._loading_pattern != 0.0
        return masks

    def _curve_parameters(
        self, item_idx: int | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        slopes, intercepts = self.slopes, self.intercepts
        if item_idx is not None:
            return slopes[item_idx], intercepts[item_idx]
        return slopes, intercepts

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts = self._curve_parameters(item_idx)
        return _affine_probability(theta, slopes, intercepts)

    def _logits(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        slopes, intercepts = self._curve_parameters(item_idx)
        return _affine_logits(theta, slopes, intercepts)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs in bounded vectorized batches."""
        theta, indices = self._prepare_probability_pairs(theta, item_indices)
        return _affine_probability(
            theta, self.slopes, self.intercepts, item_indices=indices
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        slopes, intercepts = self._curve_parameters(item_idx)
        return _information(
            theta, slopes, lambda points: _affine_logits(points, slopes, intercepts)
        )

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return item Fisher matrices across multidimensional theta points."""
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"item_idx {item_idx} out of range [0, {self.n_items})")

        theta = self._ensure_theta_2d(theta)
        return _item_information(
            self._logits(theta, item_idx), self._parameters["slopes"][item_idx]
        )

    def test_information_matrix(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return summed Fisher matrices across all items and theta points."""
        theta = self._ensure_theta_2d(theta)
        return _test_information(theta, self._parameters["slopes"], self._logits)

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

        scale = np.maximum(1.0, np.max(np.abs(a), axis=1, keepdims=True))
        with np.errstate(under="ignore"):
            scaled = a / scale
            denominator = np.sqrt(
                (1.0 / scale) ** 2 + np.sum(scaled**2, axis=1, keepdims=True)
            )
            return scaled / denominator

    def communalities(self) -> NDArray[np.float64]:
        a = self._parameters["slopes"]
        scale = np.maximum(1.0, np.max(np.abs(a), axis=1))
        with np.errstate(under="ignore"):
            norm = np.sum((a / scale[:, None]) ** 2, axis=1)
            return norm / ((1.0 / scale) ** 2 + norm)

    def set_parameters(self, **params: NDArray[np.float64]) -> "MultidimensionalModel":
        if "slopes" in params:
            slopes = np.asarray(params["slopes"])
            slopes = slopes * self._loading_pattern
            params["slopes"] = slopes

        return super().set_parameters(**params)

    def copy(self) -> "MultidimensionalModel":
        """Copy parameters and retain the confirmatory loading constraints."""
        model = self.__class__(
            n_items=self.n_items,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
            model_type=self.model_type,
            loading_pattern=self._loading_pattern.copy(),
        )
        model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        model._is_fitted = self._is_fitted
        return model
