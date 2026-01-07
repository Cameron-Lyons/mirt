"""Unfolding (Ideal Point) Models.

This module implements unfolding IRT models where response probability
is a single-peaked function of the distance between person location
and item location (ideal point).

These models are appropriate for attitude/preference data where there
is an "ideal" point and responses decrease as one moves away from it.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel, PolytomousItemModel


class GeneralizedGradedUnfolding(PolytomousItemModel):
    """Generalized Graded Unfolding Model (GGUM).

    The GGUM is an unfolding model for polytomous responses where
    the response probability is a single-peaked function.

    P(Z = z | theta) proportional to:
        exp{alpha * [z * (theta - delta) - sum_{k=0}^{z} tau_k]}
        + exp{alpha * [(M - z) * (theta - delta) - sum_{k=0}^{z} tau_k]}

    where:
    - alpha: discrimination parameter
    - delta: item location (ideal point)
    - tau: threshold parameters (cumulative)
    - M: maximum category (C - 1 where C is number of categories)

    Reference:
    Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000).
    A general item response theory model for unfolding unidimensional
    polytomous responses. Applied Psychological Measurement, 24(1), 3-32.
    """

    model_name = "GGUM"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int] = 5,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        """Initialize GGUM.

        Parameters
        ----------
        n_items : int
            Number of items
        n_categories : int or list
            Number of response categories (same for all or per-item)
        n_factors : int
            Must be 1 (GGUM is unidimensional)
        item_names : list, optional
            Item names
        """
        if n_factors != 1:
            raise ValueError("GGUM only supports unidimensional models (n_factors=1)")

        super().__init__(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )

    def _initialize_parameters(self) -> None:
        """Initialize GGUM parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)

        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)

        max_cats = max(self._n_categories)
        self._parameters["thresholds"] = np.zeros(
            (self.n_items, max_cats), dtype=np.float64
        )

        for j in range(self.n_items):
            n_cat = self._n_categories[j]
            if n_cat > 1:
                self._parameters["thresholds"][j, :n_cat] = np.linspace(-2, 2, n_cat)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Discrimination parameters (alpha)."""
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        """Item location parameters (delta / ideal points)."""
        return self._parameters["location"]

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Threshold parameters (tau)."""
        return self._parameters["thresholds"]

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Compute probability of specific category response.

        Parameters
        ----------
        theta : NDArray
            Ability values (n_persons,) or (n_persons, 1)
        item_idx : int
            Item index
        category : int
            Category (0, 1, ..., C-1)

        Returns
        -------
        NDArray
            Category probabilities (n_persons,)
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()
        n_persons = len(theta_1d)

        n_cat = self._n_categories[item_idx]
        M = n_cat - 1

        alpha = self._parameters["discrimination"][item_idx]
        delta = self._parameters["location"][item_idx]
        tau = self._parameters["thresholds"][item_idx, :n_cat]

        z = category
        tau_sum = np.sum(tau[: z + 1])

        term1 = np.exp(alpha * (z * (theta_1d - delta) - tau_sum))
        term2 = np.exp(alpha * ((M - z) * (theta_1d - delta) - tau_sum))

        numerator = term1 + term2

        denominator = np.zeros(n_persons)
        for k in range(n_cat):
            tau_sum_k = np.sum(tau[: k + 1])
            term1_k = np.exp(alpha * (k * (theta_1d - delta) - tau_sum_k))
            term2_k = np.exp(alpha * ((M - k) * (theta_1d - delta) - tau_sum_k))
            denominator += term1_k + term2_k

        return numerator / (denominator + 1e-10)

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute information for single item."""
        theta = self._ensure_theta_2d(theta)
        n_cat = self._n_categories[item_idx]

        probs = np.zeros((theta.shape[0], n_cat))
        for k in range(n_cat):
            probs[:, k] = self.category_probability(theta, item_idx, k)

        categories = np.arange(n_cat)
        expected = np.sum(probs * categories, axis=1)
        expected_sq = np.sum(probs * categories**2, axis=1)
        variance = expected_sq - expected**2

        alpha = self._parameters["discrimination"][item_idx]
        return alpha**2 * variance

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = GeneralizedGradedUnfolding(
            n_items=self.n_items,
            n_categories=list(self._n_categories),
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class IdealPointModel(DichotomousItemModel):
    """Ideal Point Model for dichotomous unfolding.

    A simpler unfolding model for binary responses:

    P(X = 1 | theta) = exp(-alpha * (theta - delta)^2)

    This creates a single-peaked (Gaussian-like) response function
    centered at the item's ideal point (delta).
    """

    model_name = "IdealPoint"
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)

        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)

        self._parameters["peak_height"] = np.ones(self.n_items, dtype=np.float64)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        return self._parameters["location"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability using Gaussian-like function.

        P(X = 1 | theta) = h * exp(-a * (theta - delta)^2)
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        delta = self._parameters["location"]
        h = self._parameters["peak_height"]

        if item_idx is not None:
            dist_sq = (theta_1d - delta[item_idx]) ** 2
            return h[item_idx] * np.exp(-a[item_idx] * dist_sq)

        dist_sq = (theta_1d[:, None] - delta[None, :]) ** 2
        return h[None, :] * np.exp(-a[None, :] * dist_sq)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1 - p

        a = self._parameters["discrimination"]
        delta = self._parameters["location"]

        if theta.ndim == 1:
            theta_1d = theta
        else:
            theta_1d = theta.ravel()

        if item_idx is not None:
            deriv = -2 * a[item_idx] * (theta_1d - delta[item_idx]) * p
            return deriv**2 / (p * q + 1e-10)

        deriv = -2 * a[None, :] * (theta_1d[:, None] - delta[None, :]) * p
        return deriv**2 / (p * q + 1e-10)

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = IdealPointModel(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class HyperbolicCosineModel(DichotomousItemModel):
    """Hyperbolic Cosine Unfolding Model.

    An alternative ideal point model using:

    P(X = 1 | theta) = 1 / (1 + cosh(alpha * (theta - delta) - gamma))

    This model allows for asymmetric response functions.
    """

    model_name = "HCM"
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["location"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["asymmetry"] = np.zeros(self.n_items, dtype=np.float64)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def location(self) -> NDArray[np.float64]:
        return self._parameters["location"]

    @property
    def asymmetry(self) -> NDArray[np.float64]:
        return self._parameters["asymmetry"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability.

        P(X = 1) = 1 / (1 + cosh(a * (theta - delta) - gamma))
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        delta = self._parameters["location"]
        gamma = self._parameters["asymmetry"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - delta[item_idx]) - gamma[item_idx]
            return 1.0 / (1.0 + np.cosh(z))

        z = a[None, :] * (theta_1d[:, None] - delta[None, :]) - gamma[None, :]
        return 1.0 / (1.0 + np.cosh(z))

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        p = self.probability(theta, item_idx)
        q = 1 - p

        a = self._parameters["discrimination"]
        delta = self._parameters["location"]
        gamma = self._parameters["asymmetry"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - delta[item_idx]) - gamma[item_idx]
            deriv = -a[item_idx] * np.sinh(z) * p**2
            return deriv**2 / (p * q + 1e-10)

        z = a[None, :] * (theta_1d[:, None] - delta[None, :]) - gamma[None, :]
        deriv = -a[None, :] * np.sinh(z) * p**2
        return deriv**2 / (p * q + 1e-10)

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = HyperbolicCosineModel(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model
