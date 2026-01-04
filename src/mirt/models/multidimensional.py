"""Multidimensional IRT models."""

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class MultidimensionalModel(DichotomousItemModel):
    """Multidimensional IRT Model (MIRT).

    This model extends the 2PL to multiple latent dimensions. Each item
    can load on multiple factors with different discrimination parameters.

    The probability of correct response is:

    P(X=1|θ) = 1 / (1 + exp(-(a·θ + d)))

    where:
    - a is a vector of discrimination/slope parameters (one per factor)
    - θ is the vector of latent traits
    - d is the intercept

    This is the "slope-intercept" parameterization. The traditional
    IRT parameterization uses difficulty b where d = -Σ(a_j × b).

    Parameters
    ----------
    n_items : int
        Number of items.
    n_factors : int
        Number of latent factors/dimensions.
    item_names : list of str, optional
        Names for each item.
    model_type : {'exploratory', 'confirmatory'}, default='exploratory'
        - 'exploratory': All items load on all factors
        - 'confirmatory': Use loading_pattern to specify structure

    loading_pattern : ndarray of shape (n_items, n_factors), optional
        Binary matrix indicating which items load on which factors.
        Required for confirmatory models. 1 = free, 0 = fixed to 0.

    Attributes
    ----------
    slopes : ndarray of shape (n_items, n_factors)
        Factor loading/discrimination parameters.
    intercepts : ndarray of shape (n_items,)
        Item intercept parameters.

    Examples
    --------
    >>> # Exploratory 3-factor model
    >>> model = MultidimensionalModel(n_items=20, n_factors=3)

    >>> # Confirmatory model with specified structure
    >>> pattern = np.array([
    ...     [1, 0, 0],  # Items 0-4 load on factor 1
    ...     [1, 0, 0],
    ...     [0, 1, 0],  # Items 5-9 load on factor 2
    ...     [0, 1, 0],
    ...     [0, 0, 1],  # Items 10-14 load on factor 3
    ... ])
    >>> model = MultidimensionalModel(
    ...     n_items=5, n_factors=3,
    ...     model_type='confirmatory',
    ...     loading_pattern=pattern
    ... )
    """

    model_name = "MIRT"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: Optional[list[str]] = None,
        model_type: Literal["exploratory", "confirmatory"] = "exploratory",
        loading_pattern: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("MultidimensionalModel requires n_factors >= 2")

        self.model_type = model_type

        if model_type == "confirmatory":
            if loading_pattern is None:
                raise ValueError(
                    "loading_pattern required for confirmatory model"
                )
            loading_pattern = np.asarray(loading_pattern)
            if loading_pattern.shape != (n_items, n_factors):
                raise ValueError(
                    f"loading_pattern shape {loading_pattern.shape} doesn't match "
                    f"(n_items={n_items}, n_factors={n_factors})"
                )
            self._loading_pattern = loading_pattern
        else:
            # Exploratory: all items load on all factors
            self._loading_pattern = np.ones((n_items, n_factors))

        super().__init__(n_items, n_factors, item_names)

    def _initialize_parameters(self) -> None:
        """Initialize parameters with default values."""
        # Slopes (factor loadings) - apply loading pattern
        slopes = np.ones((self.n_items, self.n_factors)) * 0.8
        slopes = slopes * self._loading_pattern  # Zero out constrained loadings

        self._parameters["slopes"] = slopes
        self._parameters["intercepts"] = np.zeros(self.n_items)

    @property
    def slopes(self) -> NDArray[np.float64]:
        """Factor loading/slope parameters."""
        return self._parameters["slopes"]

    @property
    def intercepts(self) -> NDArray[np.float64]:
        """Item intercept parameters."""
        return self._parameters["intercepts"]

    @property
    def loading_pattern(self) -> NDArray[np.float64]:
        """Loading pattern matrix (1=free, 0=fixed)."""
        return self._loading_pattern.copy()

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute P(X=1|θ) for multidimensional model.

        Parameters
        ----------
        theta : ndarray of shape (n_persons, n_factors)
            Latent trait values.
        item_idx : int, optional
            If specified, compute for this item only.

        Returns
        -------
        ndarray
            Response probabilities.
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["slopes"]  # (n_items, n_factors)
        d = self._parameters["intercepts"]  # (n_items,)

        if item_idx is not None:
            # Single item: z = a[item] · θ + d[item]
            z = np.dot(theta, a[item_idx]) + d[item_idx]
            return 1.0 / (1.0 + np.exp(-z))

        # All items: z = θ @ a.T + d
        z = np.dot(theta, a.T) + d[None, :]
        return 1.0 / (1.0 + np.exp(-z))

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

        For MIRT, returns the trace of the information matrix
        (sum of diagonal elements across factors).
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["slopes"]

        if item_idx is not None:
            # Sum of squared loadings
            a_sq_sum = np.sum(a[item_idx] ** 2)
            return a_sq_sum * p * q

        # All items: sum of squared loadings for each item
        a_sq_sum = np.sum(a ** 2, axis=1)  # (n_items,)
        return a_sq_sum[None, :] * p * q

    def to_irt_parameterization(self) -> dict[str, NDArray[np.float64]]:
        """Convert slope-intercept to traditional IRT parameterization.

        Returns discrimination (a) and difficulty (b) where:
        - a = slopes (same)
        - b = -intercept / sum(slopes)

        Returns
        -------
        dict
            Dictionary with 'discrimination' and 'difficulty' arrays.
        """
        a = self._parameters["slopes"]
        d = self._parameters["intercepts"]

        # Difficulty: b = -d / Σa (for each item)
        a_sum = np.sum(a, axis=1)
        b = -d / (a_sum + 1e-10)

        return {
            "discrimination": a.copy(),
            "difficulty": b,
        }

    def get_factor_loadings(
        self,
        standardized: bool = True,
    ) -> NDArray[np.float64]:
        """Get factor loadings matrix.

        Parameters
        ----------
        standardized : bool, default=True
            If True, return standardized loadings (like factor analysis).
            If False, return raw slope parameters.

        Returns
        -------
        ndarray of shape (n_items, n_factors)
            Factor loadings.
        """
        a = self._parameters["slopes"]

        if not standardized:
            return a.copy()

        # Standardize: λ = a / sqrt(1 + Σa²)
        a_sq_sum = np.sum(a ** 2, axis=1, keepdims=True)
        denominator = np.sqrt(1 + a_sq_sum)
        return a / denominator

    def communalities(self) -> NDArray[np.float64]:
        """Compute communality (h²) for each item.

        Communality is the proportion of variance explained by the factors.

        Returns
        -------
        ndarray of shape (n_items,)
            Communality for each item.
        """
        loadings = self.get_factor_loadings(standardized=True)
        return np.sum(loadings ** 2, axis=1)

    def set_parameters(self, **params: NDArray[np.float64]) -> "MultidimensionalModel":
        """Set parameters, respecting loading pattern constraints."""
        if "slopes" in params:
            slopes = np.asarray(params["slopes"])
            # Apply loading pattern mask
            slopes = slopes * self._loading_pattern
            params["slopes"] = slopes

        return super().set_parameters(**params)
