"""Testlet (Two-Tier) Models.

This module implements testlet models that account for local dependence
among items within testlets (e.g., items sharing a common passage).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import DichotomousItemModel


class TestletModel(DichotomousItemModel):
    """Two-tier testlet model for handling local item dependence.

    The testlet model adds testlet-specific random effects to account
    for the common variance among items within a testlet:

    P(X_ij = 1 | theta, gamma_t) = logistic(a_j * theta + d_j * gamma_t - b_j)

    where:
    - theta: General ability factor
    - gamma_t: Testlet-specific random effect (one per testlet)
    - a_j: Item discrimination on general factor
    - d_j: Item loading on testlet factor
    - b_j: Item difficulty

    This is similar to a bifactor model but constrained to the testlet structure.
    """

    model_name = "Testlet"
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        testlet_membership: NDArray[np.int_] | list[int],
        item_names: list[str] | None = None,
    ) -> None:
        """Initialize Testlet model.

        Parameters
        ----------
        n_items : int
            Number of items
        testlet_membership : NDArray or list
            Testlet assignment for each item (0-indexed testlet numbers).
            Items with the same number belong to the same testlet.
            Use -1 for items not in any testlet (standalone items).
        item_names : list of str, optional
            Names for items
        """
        self._testlet_membership = np.asarray(testlet_membership, dtype=np.int_)

        if len(self._testlet_membership) != n_items:
            raise ValueError(
                f"testlet_membership length ({len(self._testlet_membership)}) "
                f"must match n_items ({n_items})"
            )

        unique_testlets = np.unique(self._testlet_membership)
        self._unique_testlets = unique_testlets[unique_testlets >= 0]
        self._n_testlets = len(self._unique_testlets)

        n_factors = 1 + self._n_testlets

        super().__init__(n_items=n_items, n_factors=n_factors, item_names=item_names)

    @property
    def n_testlets(self) -> int:
        """Number of testlets."""
        return self._n_testlets

    @property
    def testlet_membership(self) -> NDArray[np.int_]:
        """Testlet assignment for each item."""
        return self._testlet_membership

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        self._parameters["discrimination"] = np.ones(self.n_items, dtype=np.float64)

        self._parameters["testlet_loadings"] = (
            np.ones(self.n_items, dtype=np.float64) * 0.5
        )

        self._parameters["testlet_loadings"][self._testlet_membership < 0] = 0.0

        self._parameters["difficulty"] = np.zeros(self.n_items, dtype=np.float64)

        self._parameters["testlet_variances"] = np.ones(
            self._n_testlets, dtype=np.float64
        )

    @property
    def discrimination(self) -> NDArray[np.float64]:
        """General factor discrimination."""
        return self._parameters["discrimination"]

    @property
    def testlet_loadings(self) -> NDArray[np.float64]:
        """Testlet-specific factor loadings."""
        return self._parameters["testlet_loadings"]

    @property
    def difficulty(self) -> NDArray[np.float64]:
        """Item difficulties."""
        return self._parameters["difficulty"]

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        For the testlet model, theta should include both general and
        testlet-specific factors: theta = [theta_general, gamma_1, ..., gamma_T]

        If only general theta is provided (1D), testlet effects are
        integrated out using quadrature.

        Parameters
        ----------
        theta : NDArray
            Ability values. Shape (n_persons,) for general only,
            or (n_persons, 1 + n_testlets) for full specification.
        item_idx : int, optional
            Item index

        Returns
        -------
        NDArray
            Response probabilities
        """
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        a = self._parameters["discrimination"]
        d = self._parameters["testlet_loadings"]
        b = self._parameters["difficulty"]

        if theta.shape[1] == 1:
            return self._marginal_probability(theta[:, 0], item_idx)

        theta_general = theta[:, 0]

        if item_idx is not None:
            testlet_idx = self._testlet_membership[item_idx]

            z = a[item_idx] * theta_general - b[item_idx]

            if testlet_idx >= 0:
                testlet_pos = np.where(self._unique_testlets == testlet_idx)[0][0] + 1
                gamma = theta[:, testlet_pos]
                z = z + d[item_idx] * gamma

            return 1.0 / (1.0 + np.exp(-z))

        probs = np.zeros((n_persons, self.n_items))

        for j in range(self.n_items):
            testlet_idx = self._testlet_membership[j]
            z = a[j] * theta_general - b[j]

            if testlet_idx >= 0:
                testlet_pos = np.where(self._unique_testlets == testlet_idx)[0][0] + 1
                gamma = theta[:, testlet_pos]
                z = z + d[j] * gamma

            probs[:, j] = 1.0 / (1.0 + np.exp(-z))

        return probs

    def _marginal_probability(
        self,
        theta_general: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal probability integrating out testlet effects.

        Uses Gauss-Hermite quadrature for numerical integration.
        """
        from scipy.special import roots_hermite

        n_persons = len(theta_general)
        n_quadpts = 11

        nodes, weights = roots_hermite(n_quadpts)
        weights = weights / np.sqrt(np.pi)
        nodes = nodes * np.sqrt(2)

        a = self._parameters["discrimination"]
        d = self._parameters["testlet_loadings"]
        b = self._parameters["difficulty"]
        testlet_vars = self._parameters["testlet_variances"]

        if item_idx is not None:
            testlet_idx = self._testlet_membership[item_idx]

            if testlet_idx < 0:
                z = a[item_idx] * theta_general - b[item_idx]
                return 1.0 / (1.0 + np.exp(-z))

            var_t = testlet_vars[testlet_idx]
            probs = np.zeros(n_persons)

            for q in range(n_quadpts):
                gamma = nodes[q] * np.sqrt(var_t)
                z = a[item_idx] * theta_general + d[item_idx] * gamma - b[item_idx]
                probs += weights[q] * (1.0 / (1.0 + np.exp(-z)))

            return probs

        probs = np.zeros((n_persons, self.n_items))

        for j in range(self.n_items):
            testlet_idx = self._testlet_membership[j]

            if testlet_idx < 0:
                z = a[j] * theta_general - b[j]
                probs[:, j] = 1.0 / (1.0 + np.exp(-z))
            else:
                var_t = testlet_vars[testlet_idx]
                for q in range(n_quadpts):
                    gamma = nodes[q] * np.sqrt(var_t)
                    z = a[j] * theta_general + d[j] * gamma - b[j]
                    probs[:, j] += weights[q] * (1.0 / (1.0 + np.exp(-z)))

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information.

        For testlet model, uses marginal probability for information.
        """
        theta = self._ensure_theta_2d(theta)

        p = self.probability(theta, item_idx)
        q = 1.0 - p

        a = self._parameters["discrimination"]

        if item_idx is not None:
            return (a[item_idx] ** 2) * p * q
        else:
            return (a[None, :] ** 2) * p * q

    def get_testlet_items(self, testlet_idx: int) -> list[int]:
        """Get indices of items belonging to a testlet.

        Parameters
        ----------
        testlet_idx : int
            Testlet index

        Returns
        -------
        list of int
            Item indices in the testlet
        """
        return list(np.where(self._testlet_membership == testlet_idx)[0])

    def testlet_reliability(self) -> dict[int, float]:
        """Compute reliability for each testlet.

        Returns omega-like reliability coefficient for items within each testlet.

        Returns
        -------
        dict
            Testlet index -> reliability coefficient
        """
        reliabilities = {}

        for testlet_idx in self._unique_testlets:
            items = self.get_testlet_items(testlet_idx)

            if len(items) < 2:
                reliabilities[int(testlet_idx)] = np.nan
                continue

            general_loadings = self._parameters["discrimination"][items]
            testlet_loadings = self._parameters["testlet_loadings"][items]

            sum_general = general_loadings.sum()
            sum_testlet = testlet_loadings.sum()

            var_general = sum_general**2
            var_testlet = sum_testlet**2
            var_unique = len(items)

            total_var = var_general + var_testlet + var_unique
            if total_var > 0:
                omega = (var_general + var_testlet) / total_var
            else:
                omega = np.nan

            reliabilities[int(testlet_idx)] = float(omega)

        return reliabilities

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = TestletModel(
            n_items=self.n_items,
            testlet_membership=self._testlet_membership.copy(),
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


def create_testlet_structure(
    n_items: int,
    testlet_sizes: list[int],
) -> NDArray[np.int_]:
    """Create testlet membership array from testlet sizes.

    Parameters
    ----------
    n_items : int
        Total number of items
    testlet_sizes : list of int
        Size of each testlet. Sum should equal n_items.
        Use 1 for standalone items (will be assigned -1).

    Returns
    -------
    NDArray
        Testlet membership array

    Examples
    --------
    >>> create_testlet_structure(10, [3, 3, 1, 3])
    array([0, 0, 0, 1, 1, 1, -1, 2, 2, 2])
    """
    if sum(testlet_sizes) != n_items:
        raise ValueError(
            f"Sum of testlet_sizes ({sum(testlet_sizes)}) must equal n_items ({n_items})"
        )

    membership = np.zeros(n_items, dtype=np.int_)
    current_pos = 0
    testlet_idx = 0

    for size in testlet_sizes:
        if size == 1:
            membership[current_pos] = -1
        else:
            membership[current_pos : current_pos + size] = testlet_idx
            testlet_idx += 1
        current_pos += size

    return membership
