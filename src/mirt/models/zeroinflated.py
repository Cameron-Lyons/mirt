"""Zero-Inflated Response Models.

This module implements zero-inflated IRT models that account for
excess zero responses due to non-engagement, guessing, or other factors.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class ZeroInflated2PL(DichotomousItemModel):
    """Zero-Inflated Two-Parameter Logistic model.

    This model assumes that zeros come from two sources:
    1. Regular IRT response process (with probability 1 - pi)
    2. Zero-inflation process (with probability pi)

    P(X = 0 | theta) = pi + (1 - pi) * (1 - P_2PL(X = 1 | theta))
    P(X = 1 | theta) = (1 - pi) * P_2PL(X = 1 | theta)

    where:
    - pi = zero-inflation probability
    - P_2PL = standard 2PL probability

    This is useful for modeling:
    - Non-engagement/rapid guessing
    - Structural zeros
    - Not-reached items
    """

    model_name = "ZI-2PL"
    n_params_per_item = 3
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["zero_inflation"] = np.full(
            self.n_items, 0.1, dtype=np.float64
        )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """Item discriminations."""
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulties."""
        return self._parameters["difficulty"]

    @property
    def zero_inflation(self) -> NDArray[np.float64]:
        """Zero-inflation probabilities."""
        return self._parameters["zero_inflation"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute P(X = 1 | theta).

        P(X = 1) = (1 - pi) * P_2PL(X = 1)
        """
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        pi = self._parameters["zero_inflation"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_2pl = 1.0 / (1.0 + np.exp(-z))
            return (1 - pi[item_idx]) * p_2pl

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_2pl = 1.0 / (1.0 + np.exp(-z))
        return (1 - pi[None, :]) * p_2pl

    def probability_zero(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute P(X = 0 | theta).

        P(X = 0) = pi + (1 - pi) * (1 - P_2PL(X = 1))
        """
        return 1 - self.probability(theta, item_idx)

    def probability_2pl(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute standard 2PL probability (without zero-inflation)."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            return 1.0 / (1.0 + np.exp(-z))

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        return 1.0 / (1.0 + np.exp(-z))

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information.

        For ZI model, information is reduced by zero-inflation.
        """
        theta = self._ensure_theta_2d(theta)

        a = self._parameters["discrimination"]
        pi = self._parameters["zero_inflation"]

        p_2pl = self.probability_2pl(theta, item_idx)
        info_2pl = p_2pl * (1 - p_2pl)

        if item_idx is not None:
            return (1 - pi[item_idx]) * (a[item_idx] ** 2) * info_2pl
        else:
            return (1 - pi[None, :]) * (a[None, :] ** 2) * info_2pl

    def expected_proportion_zeros_from_inflation(
        self,
        theta: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Estimate proportion of zeros due to inflation vs IRT process.

        Parameters
        ----------
        theta : NDArray, optional
            Ability values. If None, uses standard normal.

        Returns
        -------
        NDArray
            Proportion of zeros from inflation for each item
        """
        if theta is None:
            theta = np.linspace(-3, 3, 100).reshape(-1, 1)

        p_0_total = self.probability_zero(theta)
        pi = self._parameters["zero_inflation"]

        mean_p0 = p_0_total.mean(axis=0)
        return pi / (mean_p0 + 1e-10)

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = ZeroInflated2PL(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class ZeroInflated3PL(DichotomousItemModel):
    """Zero-Inflated Three-Parameter Logistic model.

    Combines zero-inflation with guessing parameter:

    P(X = 1 | theta) = (1 - pi) * [c + (1 - c) * P_2PL(X = 1)]
    """

    model_name = "ZI-3PL"
    n_params_per_item = 4
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["guessing"] = np.full(self.n_items, 0.2, dtype=np.float64)
        self._parameters["zero_inflation"] = np.full(
            self.n_items, 0.1, dtype=np.float64
        )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def guessing(self) -> NDArray[np.float64]:
        return self._parameters["guessing"]

    @property
    def zero_inflation(self) -> NDArray[np.float64]:
        return self._parameters["zero_inflation"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute P(X = 1 | theta)."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        pi = self._parameters["zero_inflation"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_2pl = 1.0 / (1.0 + np.exp(-z))
            p_3pl = c[item_idx] + (1 - c[item_idx]) * p_2pl
            return (1 - pi[item_idx]) * p_3pl

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_2pl = 1.0 / (1.0 + np.exp(-z))
        p_3pl = c[None, :] + (1 - c[None, :]) * p_2pl
        return (1 - pi[None, :]) * p_3pl

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["guessing"]
        pi = self._parameters["zero_inflation"]

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_2pl = 1.0 / (1.0 + np.exp(-z))
            p_3pl = c[item_idx] + (1 - c[item_idx]) * p_2pl

            info = (
                (a[item_idx] ** 2)
                * ((1 - c[item_idx]) ** 2)
                * (p_2pl**2)
                * ((1 - p_2pl) ** 2)
            )
            info = info / (p_3pl * (1 - p_3pl) + 1e-10)

            return (1 - pi[item_idx]) * info

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_2pl = 1.0 / (1.0 + np.exp(-z))
        p_3pl = c[None, :] + (1 - c[None, :]) * p_2pl

        info = (
            (a[None, :] ** 2)
            * ((1 - c[None, :]) ** 2)
            * (p_2pl**2)
            * ((1 - p_2pl) ** 2)
        )
        info = info / (p_3pl * (1 - p_3pl) + 1e-10)

        return (1 - pi[None, :]) * info

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = ZeroInflated3PL(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class HurdleIRT(DichotomousItemModel):
    """Hurdle IRT model for zero-inflation.

    Alternative parameterization where zero-inflation probability
    depends on ability (theta-dependent zero-inflation):

    P(engage) = logistic(alpha_0 + alpha_1 * theta)
    P(X = 1 | engaged, theta) = P_2PL(theta)
    P(X = 1 | theta) = P(engage) * P_2PL(theta)
    """

    model_name = "Hurdle"
    n_params_per_item = 4
    supports_multidimensional = False

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)
        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)
        self._parameters["engagement_intercept"] = np.full(
            self.n_items, 2.0, dtype=np.float64
        )
        self._parameters["engagement_slope"] = np.full(
            self.n_items, 0.5, dtype=np.float64
        )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def engagement_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute engagement probability."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        alpha_0 = self._parameters["engagement_intercept"]
        alpha_1 = self._parameters["engagement_slope"]

        if item_idx is not None:
            z = alpha_0[item_idx] + alpha_1[item_idx] * theta_1d
            return 1.0 / (1.0 + np.exp(-z))

        z = alpha_0[None, :] + alpha_1[None, :] * theta_1d[:, None]
        return 1.0 / (1.0 + np.exp(-z))

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute P(X = 1 | theta) = P(engage) * P_2PL."""
        theta = self._ensure_theta_2d(theta)
        theta_1d = theta.ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        p_engage = self.engagement_probability(theta, item_idx)

        if item_idx is not None:
            z = a[item_idx] * (theta_1d - b[item_idx])
            p_2pl = 1.0 / (1.0 + np.exp(-z))
            return p_engage * p_2pl

        z = a[None, :] * (theta_1d[:, None] - b[None, :])
        p_2pl = 1.0 / (1.0 + np.exp(-z))
        return p_engage * p_2pl

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            return (a[item_idx] ** 2) * p * q
        else:
            return (a[None, :] ** 2) * p * q

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = HurdleIRT(
            n_items=self.n_items,
            n_factors=1,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model
