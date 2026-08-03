import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import DichotomousItemModel


class BifactorModel(DichotomousItemModel):
    """Bifactor model with one general and one specific loading per item.

    Specific-factor labels may be any non-negative integers. Labels are kept
    for reporting while an internal contiguous index determines theta columns.
    """

    model_name = "Bifactor"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
        item_names: list[str] | None = None,
    ) -> None:
        factor_values = np.asarray(specific_factors)

        if factor_values.ndim != 1 or factor_values.shape[0] != n_items:
            raise ValueError(
                f"Length of specific_factors ({factor_values.size}) "
                f"must match n_items ({n_items})"
            )

        if not np.issubdtype(factor_values.dtype, np.integer):
            if not np.issubdtype(factor_values.dtype, np.floating) or not np.all(
                np.isfinite(factor_values) & (factor_values == np.floor(factor_values))
            ):
                raise ValueError("specific_factors must contain integer labels")

        factor_values = factor_values.astype(np.int_, copy=False)
        if np.any(factor_values < 0):
            raise ValueError("specific_factors must be non-negative integers")

        factor_labels, factor_indices = np.unique(factor_values, return_inverse=True)
        self._specific_factors = factor_values.copy()
        self._specific_factor_labels = factor_labels
        self._specific_factor_indices = factor_indices
        self._n_specific_factors = len(factor_labels)

        n_factors = 1 + self._n_specific_factors

        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["general_loadings"] = np.ones(self.n_items) * 0.7

        self._parameters["specific_loadings"] = np.ones(self.n_items) * 0.5

        self._parameters["intercepts"] = np.zeros(self.n_items)

    @property
    def general_loadings(self) -> NDArray[np.float64]:
        return self._parameters["general_loadings"]

    @property
    def specific_loadings(self) -> NDArray[np.float64]:
        return self._parameters["specific_loadings"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        return self._parameters["intercepts"]

    @property
    def specific_factors(self) -> NDArray[np.int_]:
        return self._specific_factors.copy()

    @property
    def n_specific_factors(self) -> int:
        return self._n_specific_factors

    @property
    def specific_factor_labels(self) -> NDArray[np.int_]:
        """Return the sorted external labels for the specific factors."""
        return self._specific_factor_labels.copy()

    def get_factor_structure(self) -> dict[int, list[int]]:
        """Map each external specific-factor label to its item indices."""
        return {
            int(label): np.flatnonzero(self._specific_factors == label).tolist()
            for label in self._specific_factor_labels
        }

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]
        d = self._parameters["intercepts"]

        theta_g = theta[:, 0]

        if item_idx is not None:
            factor_idx = self._specific_factor_indices[item_idx]
            theta_s = theta[:, 1 + factor_idx]

            z = a_g[item_idx] * theta_g + a_s[item_idx] * theta_s + d[item_idx]
            return sigmoid(z)

        theta_s = theta[:, 1 + self._specific_factor_indices]
        z = a_g[None, :] * theta_g[:, None] + a_s[None, :] * theta_s + d[None, :]
        return sigmoid(z)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        if item_idx is not None:
            a_sq_total = a_g[item_idx] ** 2 + a_s[item_idx] ** 2
            return a_sq_total * p * q

        a_sq_total = a_g**2 + a_s**2
        return a_sq_total[None, :] * p * q

    def omega_hierarchical(self) -> float:
        """Estimate general-factor reliability for the total score."""
        general_variance, specific_variance = self._score_variance_components()
        total_variance = general_variance + specific_variance + self.n_items
        return float(general_variance / total_variance)

    def omega_total(self) -> float:
        """Estimate reliability due to all general and specific factors."""
        general_variance, specific_variance = self._score_variance_components()
        common_variance = general_variance + specific_variance
        return float(common_variance / (common_variance + self.n_items))

    def _score_variance_components(
        self,
        items: NDArray[np.int_] | None = None,
    ) -> tuple[float, float]:
        """Return general and group-specific score variance components."""
        if items is None:
            items = np.arange(self.n_items)

        a_g = self._parameters["general_loadings"][items]
        general_variance = float(np.sum(a_g) ** 2)
        specific_variance = 0.0
        for label in self._specific_factor_labels:
            factor_items = items[self._specific_factors[items] == label]
            if factor_items.size:
                factor_loadings = self._parameters["specific_loadings"][factor_items]
                specific_variance += float(np.sum(factor_loadings) ** 2)

        return general_variance, specific_variance

    def omega_subscale(self, specific_factor: int) -> float:
        """Estimate total reliability for one externally labeled subscale."""
        items = np.where(self._specific_factors == specific_factor)[0]

        if len(items) == 0:
            return np.nan

        general_variance, specific_variance = self._score_variance_components(items)
        common_variance = general_variance + specific_variance
        return float(common_variance / (common_variance + len(items)))

    def explained_common_variance(self) -> dict[str, float]:
        """Return general and group-specific shares of common variance."""
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        sum_a_g_sq = np.sum(a_g**2)
        sum_a_s_sq = np.sum(a_s**2)
        total_common = sum_a_g_sq + sum_a_s_sq

        result = {"general": sum_a_g_sq / total_common if total_common > 0 else np.nan}

        for label in self._specific_factor_labels:
            items = np.where(self._specific_factors == label)[0]
            sf_variance = np.sum(a_s[items] ** 2)
            result[f"specific_{int(label)}"] = (
                sf_variance / total_common if total_common > 0 else np.nan
            )

        return result

    def get_loading_matrix(self) -> NDArray[np.float64]:
        """Return the sparse item-by-factor loading matrix."""
        loadings = np.zeros((self.n_items, 1 + self._n_specific_factors))

        loadings[:, 0] = self._parameters["general_loadings"]
        loadings[
            np.arange(self.n_items),
            1 + self._specific_factor_indices,
        ] = self._parameters["specific_loadings"]

        return loadings

    def get_item_parameters(
        self,
        item_idx: int,
    ) -> dict[str, float | NDArray[np.float64]]:
        """Return item parameters plus its full multidimensional slope vector."""
        parameters = super().get_item_parameters(item_idx)
        slopes = np.zeros(self.n_factors, dtype=np.float64)
        slopes[0] = self._parameters["general_loadings"][item_idx]
        slopes[1 + self._specific_factor_indices[item_idx]] = self._parameters[
            "specific_loadings"
        ][item_idx]
        parameters["slopes"] = slopes
        return parameters

    def copy(self) -> "BifactorModel":
        """Return an independent copy that preserves the bifactor structure."""
        new_model = BifactorModel(
            n_items=self.n_items,
            specific_factors=self._specific_factors.copy(),
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model
