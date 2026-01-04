import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class BifactorModel(DichotomousItemModel):
    model_name = "Bifactor"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
        item_names: list[str] | None = None,
    ) -> None:
        specific_factors = np.asarray(specific_factors, dtype=np.int_)

        if len(specific_factors) != n_items:
            raise ValueError(
                f"Length of specific_factors ({len(specific_factors)}) "
                f"must match n_items ({n_items})"
            )

        if np.min(specific_factors) < 0:
            raise ValueError("specific_factors must be non-negative integers")

        self._specific_factors = specific_factors
        self._n_specific_factors = len(np.unique(specific_factors))

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

    def get_factor_structure(self) -> dict[int, list[int]]:
        structure: dict[int, list[int]] = {}
        for i, sf in enumerate(self._specific_factors):
            if sf not in structure:
                structure[sf] = []
            structure[sf].append(i)
        return structure

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]
        d = self._parameters["intercepts"]

        theta_g = theta[:, 0]

        if item_idx is not None:
            sf = self._specific_factors[item_idx]
            theta_s = theta[:, 1 + sf]

            z = a_g[item_idx] * theta_g + a_s[item_idx] * theta_s + d[item_idx]
            return 1.0 / (1.0 + np.exp(-z))

        probs = np.zeros((n_persons, self.n_items))

        for i in range(self.n_items):
            sf = self._specific_factors[i]
            theta_s = theta[:, 1 + sf]

            z = a_g[i] * theta_g + a_s[i] * theta_s + d[i]
            probs[:, i] = 1.0 / (1.0 + np.exp(-z))

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = np.asarray(responses)
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(p) + (1 - responses) * np.log(1 - p),
            0.0,
        )

        return ll.sum(axis=1)

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
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        sum_a_g_sq = np.sum(a_g) ** 2
        sum_a_s_sq = np.sum(a_s**2)
        n = self.n_items

        omega_h = sum_a_g_sq / (sum_a_g_sq + sum_a_s_sq + n)
        return float(omega_h)

    def omega_subscale(self, specific_factor: int) -> float:
        items = np.where(self._specific_factors == specific_factor)[0]

        if len(items) == 0:
            return np.nan

        a_g = self._parameters["general_loadings"][items]
        a_s = self._parameters["specific_loadings"][items]

        sum_a_g = np.sum(a_g)
        sum_a_s = np.sum(a_s)
        sum_a_g_sq = np.sum(a_g**2)
        sum_a_s_sq = np.sum(a_s**2)
        n = len(items)

        omega = (sum_a_g + sum_a_s) ** 2 / (
            (sum_a_g + sum_a_s) ** 2 + n - sum_a_g_sq - sum_a_s_sq
        )

        return float(omega)

    def explained_common_variance(self) -> dict[str, float]:
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        sum_a_g_sq = np.sum(a_g**2)
        sum_a_s_sq = np.sum(a_s**2)
        total_common = sum_a_g_sq + sum_a_s_sq

        result = {"general": sum_a_g_sq / total_common}

        for sf in range(self._n_specific_factors):
            items = np.where(self._specific_factors == sf)[0]
            sf_variance = np.sum(a_s[items] ** 2)
            result[f"specific_{sf}"] = sf_variance / total_common

        return result

    def get_loading_matrix(self) -> NDArray[np.float64]:
        loadings = np.zeros((self.n_items, 1 + self._n_specific_factors))

        loadings[:, 0] = self._parameters["general_loadings"]

        for i in range(self.n_items):
            sf = self._specific_factors[i]
            loadings[i, 1 + sf] = self._parameters["specific_loadings"][i]

        return loadings
