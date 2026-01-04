"""Bifactor IRT models."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class BifactorModel(DichotomousItemModel):
    """Bifactor IRT Model.

    The bifactor model includes a general factor that all items load on,
    plus specific (group) factors that subsets of items load on. Each item
    loads on exactly one specific factor in addition to the general factor.

    P(X=1|θ_g, θ_s) = 1 / (1 + exp(-(a_g × θ_g + a_s × θ_s + d)))

    where:
    - θ_g is the general factor
    - θ_s is the specific factor for this item's group
    - a_g is the general factor loading
    - a_s is the specific factor loading
    - d is the intercept

    Parameters
    ----------
    n_items : int
        Number of items.
    specific_factors : array-like of shape (n_items,)
        Assignment of each item to a specific factor (0, 1, 2, ...).
        Items with the same value belong to the same specific factor.
    item_names : list of str, optional
        Names for each item.

    Attributes
    ----------
    general_loadings : ndarray of shape (n_items,)
        General factor loadings.
    specific_loadings : ndarray of shape (n_items,)
        Specific factor loadings.
    intercepts : ndarray of shape (n_items,)
        Item intercepts.
    n_specific_factors : int
        Number of specific factors.

    Examples
    --------
    >>> # 15 items, 3 specific factors (5 items each)
    >>> specific = [0]*5 + [1]*5 + [2]*5
    >>> model = BifactorModel(n_items=15, specific_factors=specific)

    >>> # Check factor structure
    >>> print(f"General factor: 1")
    >>> print(f"Specific factors: {model.n_specific_factors}")

    Notes
    -----
    The bifactor model is useful when:
    - Items measure a general trait plus specific subdimensions
    - You want to control for method effects
    - Testing unidimensionality while allowing for local dependencies

    The model uses dimension reduction: instead of integrating over
    (1 + n_specific) dimensions, it integrates over each specific factor
    separately, reducing computational complexity.
    """

    model_name = "Bifactor"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        specific_factors: NDArray[np.int_] | list[int],
        item_names: Optional[list[str]] = None,
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

        # Total factors = 1 general + n_specific
        n_factors = 1 + self._n_specific_factors

        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        # General factor loadings
        self._parameters["general_loadings"] = np.ones(self.n_items) * 0.7

        # Specific factor loadings
        self._parameters["specific_loadings"] = np.ones(self.n_items) * 0.5

        # Intercepts
        self._parameters["intercepts"] = np.zeros(self.n_items)

    @property
    def general_loadings(self) -> NDArray[np.float64]:
        """General factor loadings."""
        return self._parameters["general_loadings"]

    @property
    def specific_loadings(self) -> NDArray[np.float64]:
        """Specific factor loadings."""
        return self._parameters["specific_loadings"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        """Item intercepts."""
        return self._parameters["intercepts"]

    @property
    def specific_factors(self) -> NDArray[np.int_]:
        """Specific factor assignment for each item."""
        return self._specific_factors.copy()

    @property
    def n_specific_factors(self) -> int:
        """Number of specific factors."""
        return self._n_specific_factors

    def get_factor_structure(self) -> dict[int, list[int]]:
        """Get which items belong to each specific factor.

        Returns
        -------
        dict
            Mapping from specific factor index to list of item indices.
        """
        structure: dict[int, list[int]] = {}
        for i, sf in enumerate(self._specific_factors):
            if sf not in structure:
                structure[sf] = []
            structure[sf].append(i)
        return structure

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute P(X=1|θ) for bifactor model.

        Parameters
        ----------
        theta : ndarray of shape (n_persons, n_factors)
            Latent trait values. First column is general factor,
            remaining columns are specific factors.
        item_idx : int, optional
            If specified, compute for this item only.

        Returns
        -------
        ndarray
            Response probabilities.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a_g = self._parameters["general_loadings"]  # (n_items,)
        a_s = self._parameters["specific_loadings"]  # (n_items,)
        d = self._parameters["intercepts"]  # (n_items,)

        # Extract general and specific theta
        theta_g = theta[:, 0]  # (n_persons,)

        if item_idx is not None:
            # Which specific factor does this item belong to?
            sf = self._specific_factors[item_idx]
            theta_s = theta[:, 1 + sf]  # (n_persons,)

            z = a_g[item_idx] * theta_g + a_s[item_idx] * theta_s + d[item_idx]
            return 1.0 / (1.0 + np.exp(-z))

        # All items
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
        """Compute log-likelihood for given responses."""
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
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information.

        Returns the sum of information from general and specific factors.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        if item_idx is not None:
            # Total information = a_g² + a_s²
            a_sq_total = a_g[item_idx] ** 2 + a_s[item_idx] ** 2
            return a_sq_total * p * q

        # All items
        a_sq_total = a_g ** 2 + a_s ** 2  # (n_items,)
        return a_sq_total[None, :] * p * q

    def omega_hierarchical(self) -> float:
        """Compute omega hierarchical (ω_h).

        Omega hierarchical measures the proportion of total variance
        attributable to the general factor.

        Returns
        -------
        float
            Omega hierarchical coefficient.

        Notes
        -----
        ω_h = (Σ a_g)² / [(Σ a_g)² + Σ a_s² + n]

        where a_g and a_s are the general and specific loadings.
        """
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        sum_a_g_sq = np.sum(a_g) ** 2
        sum_a_s_sq = np.sum(a_s ** 2)
        n = self.n_items

        omega_h = sum_a_g_sq / (sum_a_g_sq + sum_a_s_sq + n)
        return float(omega_h)

    def omega_subscale(self, specific_factor: int) -> float:
        """Compute omega for a specific factor subscale.

        Parameters
        ----------
        specific_factor : int
            Index of the specific factor.

        Returns
        -------
        float
            Omega coefficient for the subscale.
        """
        items = np.where(self._specific_factors == specific_factor)[0]

        if len(items) == 0:
            return np.nan

        a_g = self._parameters["general_loadings"][items]
        a_s = self._parameters["specific_loadings"][items]

        sum_a_g = np.sum(a_g)
        sum_a_s = np.sum(a_s)
        sum_a_g_sq = np.sum(a_g ** 2)
        sum_a_s_sq = np.sum(a_s ** 2)
        n = len(items)

        # Total omega for subscale
        omega = (sum_a_g + sum_a_s) ** 2 / \
                ((sum_a_g + sum_a_s) ** 2 + n - sum_a_g_sq - sum_a_s_sq)

        return float(omega)

    def explained_common_variance(self) -> dict[str, float]:
        """Compute explained common variance (ECV) for each factor.

        Returns
        -------
        dict
            ECV for general factor and each specific factor.
        """
        a_g = self._parameters["general_loadings"]
        a_s = self._parameters["specific_loadings"]

        sum_a_g_sq = np.sum(a_g ** 2)
        sum_a_s_sq = np.sum(a_s ** 2)
        total_common = sum_a_g_sq + sum_a_s_sq

        result = {"general": sum_a_g_sq / total_common}

        for sf in range(self._n_specific_factors):
            items = np.where(self._specific_factors == sf)[0]
            sf_variance = np.sum(a_s[items] ** 2)
            result[f"specific_{sf}"] = sf_variance / total_common

        return result

    def get_loading_matrix(self) -> NDArray[np.float64]:
        """Get full factor loading matrix.

        Returns
        -------
        ndarray of shape (n_items, 1 + n_specific_factors)
            Loading matrix. First column is general factor,
            remaining columns are specific factors.
        """
        loadings = np.zeros((self.n_items, 1 + self._n_specific_factors))

        # General factor (first column)
        loadings[:, 0] = self._parameters["general_loadings"]

        # Specific factors
        for i in range(self.n_items):
            sf = self._specific_factors[i]
            loadings[i, 1 + sf] = self._parameters["specific_loadings"][i]

        return loadings
