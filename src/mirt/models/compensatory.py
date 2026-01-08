"""Compensatory and Partially Compensatory IRT Models.

Standard MIRT models assume compensatory relationships between dimensions -
high ability on one dimension can compensate for low ability on another.
Partially compensatory models relax this assumption.

References:
    Bolt, D. M., & Lall, V. F. (2003). Estimation of compensatory and
        noncompensatory multidimensional item response models using
        Markov chain Monte Carlo. Applied Psychological Measurement.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class PartiallyCompensatoryModel(DichotomousItemModel):
    """Partially Compensatory Multidimensional IRT Model.

    This model allows non-compensatory relationships where deficiency
    in one dimension cannot be fully compensated by strength in another.

    The model uses a product of logistic functions:

        P(X=1|θ) = prod_k [1 / (1 + exp(-a_k(θ_k - b_k)))]^c_k

    where c_k is the compensation parameter (0 = no compensation, 1 = full).

    Parameters
    ----------
    n_items : int
        Number of items
    n_factors : int
        Number of latent dimensions
    item_names : list of str, optional
        Names for items

    Attributes
    ----------
    discrimination : ndarray of shape (n_items, n_factors)
        Discrimination parameters for each dimension
    difficulty : ndarray of shape (n_items, n_factors)
        Difficulty parameters for each dimension
    compensation : ndarray of shape (n_items, n_factors)
        Compensation parameters (0-1)
    """

    model_name = "PartiallyCompensatory"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Partially compensatory model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))
        self._parameters["difficulty"] = np.zeros((self.n_items, self.n_factors))

        self._parameters["compensation"] = np.full((self.n_items, self.n_factors), 0.5)

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    @property
    def compensation(self) -> NDArray[np.float64]:
        return self._parameters["compensation"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]
        c = self._parameters["compensation"]

        if item_idx is not None:
            prob = np.ones(n_persons)
            for k in range(self.n_factors):
                z_k = a[item_idx, k] * (theta[:, k] - b[item_idx, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))

                prob *= np.power(p_k, c[item_idx, k])

            return prob

        probs = np.ones((n_persons, self.n_items))
        for j in range(self.n_items):
            for k in range(self.n_factors):
                z_k = a[j, k] * (theta[:, k] - b[j, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))
                probs[:, j] *= np.power(p_k, c[j, k])

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 1e-5
        info = np.zeros_like(p)

        for k in range(self.n_factors):
            theta_plus = theta.copy()
            theta_plus[:, k] += h
            theta_minus = theta.copy()
            theta_minus[:, k] -= h

            p_plus = self.probability(theta_plus, item_idx)
            p_minus = self.probability(theta_minus, item_idx)

            dp_k = (p_plus - p_minus) / (2 * h)
            info += (dp_k**2) / (p * (1 - p) + 1e-10)

        return info


class NoncompensatoryModel(DichotomousItemModel):
    """Fully Non-compensatory (Conjunctive) Multidimensional IRT Model.

    In this model, success requires meeting threshold on ALL dimensions.
    High ability on one dimension cannot compensate for low ability on another.

        P(X=1|θ) = prod_k [1 / (1 + exp(-a_k(θ_k - b_k)))]

    This is equivalent to PartiallyCompensatoryModel with compensation = 0.
    """

    model_name = "Noncompensatory"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Non-compensatory model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))
        self._parameters["difficulty"] = np.zeros((self.n_items, self.n_factors))

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            prob = np.ones(n_persons)
            for k in range(self.n_factors):
                z_k = a[item_idx, k] * (theta[:, k] - b[item_idx, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))
                prob *= p_k
            return prob

        probs = np.ones((n_persons, self.n_items))
        for j in range(self.n_items):
            for k in range(self.n_factors):
                z_k = a[j, k] * (theta[:, k] - b[j, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))
                probs[:, j] *= p_k

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 1e-5
        info = np.zeros_like(p)

        for k in range(self.n_factors):
            theta_plus = theta.copy()
            theta_plus[:, k] += h
            theta_minus = theta.copy()
            theta_minus[:, k] -= h

            p_plus = self.probability(theta_plus, item_idx)
            p_minus = self.probability(theta_minus, item_idx)

            dp_k = (p_plus - p_minus) / (2 * h)
            info += (dp_k**2) / (p * (1 - p) + 1e-10)

        return info


class DisjunctiveModel(DichotomousItemModel):
    """Disjunctive Multidimensional IRT Model.

    In this model, success requires meeting threshold on ANY dimension.
    This is the OR logic counterpart to the conjunctive (AND) model.

        P(X=1|θ) = 1 - prod_k [1 - 1/(1 + exp(-a_k(θ_k - b_k)))]
                 = 1 - prod_k P(fail on dimension k)

    Useful for items where multiple paths to success exist.
    """

    model_name = "Disjunctive"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_factors: int = 2,
        item_names: list[str] | None = None,
    ) -> None:
        if n_factors < 2:
            raise ValueError("Disjunctive model requires at least 2 factors")
        super().__init__(n_items, n_factors=n_factors, item_names=item_names)

    def _initialize_parameters(self) -> None:
        self._parameters["discrimination"] = np.ones((self.n_items, self.n_factors))
        self._parameters["difficulty"] = np.zeros((self.n_items, self.n_factors))

    @property
    def discrimination(self) -> NDArray[np.float64]:
        return self._parameters["discrimination"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        if item_idx is not None:
            p_fail_all = np.ones(n_persons)
            for k in range(self.n_factors):
                z_k = a[item_idx, k] * (theta[:, k] - b[item_idx, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))
                p_fail_all *= 1 - p_k
            return 1 - p_fail_all

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            p_fail_all = np.ones(n_persons)
            for k in range(self.n_factors):
                z_k = a[j, k] * (theta[:, k] - b[j, k])
                p_k = 1.0 / (1.0 + np.exp(-z_k))
                p_fail_all *= 1 - p_k
            probs[:, j] = 1 - p_fail_all

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        p = self.probability(theta, item_idx)

        h = 1e-5
        info = np.zeros_like(p)

        for k in range(self.n_factors):
            theta_plus = theta.copy()
            theta_plus[:, k] += h
            theta_minus = theta.copy()
            theta_minus[:, k] -= h

            p_plus = self.probability(theta_plus, item_idx)
            p_minus = self.probability(theta_minus, item_idx)

            dp_k = (p_plus - p_minus) / (2 * h)
            info += (dp_k**2) / (p * (1 - p) + 1e-10)

        return info
