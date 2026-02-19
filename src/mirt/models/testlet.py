"""Testlet (Two-Tier) Models.

This module implements testlet models that account for local dependence
among items within testlets (e.g., items sharing a common passage).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
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
    # Prevent pytest from trying to collect this class as a test case.
    __test__ = False

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

            return sigmoid(z)

        probs = np.zeros((n_persons, self.n_items))

        for j in range(self.n_items):
            testlet_idx = self._testlet_membership[j]
            z = a[j] * theta_general - b[j]

            if testlet_idx >= 0:
                testlet_pos = np.where(self._unique_testlets == testlet_idx)[0][0] + 1
                gamma = theta[:, testlet_pos]
                z = z + d[j] * gamma

            probs[:, j] = sigmoid(z)

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
                return sigmoid(z)

            var_t = testlet_vars[testlet_idx]
            probs = np.zeros(n_persons)

            for q in range(n_quadpts):
                gamma = nodes[q] * np.sqrt(var_t)
                z = a[item_idx] * theta_general + d[item_idx] * gamma - b[item_idx]
                probs += weights[q] * (sigmoid(z))

            return probs

        probs = np.zeros((n_persons, self.n_items))

        for j in range(self.n_items):
            testlet_idx = self._testlet_membership[j]

            if testlet_idx < 0:
                z = a[j] * theta_general - b[j]
                probs[:, j] = sigmoid(z)
            else:
                var_t = testlet_vars[testlet_idx]
                for q in range(n_quadpts):
                    gamma = nodes[q] * np.sqrt(var_t)
                    z = a[j] * theta_general + d[j] * gamma - b[j]
                    probs[:, j] += weights[q] * (sigmoid(z))

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


class BifactorTestletModel(TestletModel):
    """Bifactor testlet model with explicit general + testlet factors.

    This model provides a cleaner bifactor parameterization where:
    - Each item loads on a general factor
    - Each item in a testlet loads on a testlet-specific factor
    - Testlet factors are orthogonal to the general factor

    Parameters
    ----------
    n_items : int
        Number of items.
    testlet_membership : NDArray or list
        Testlet assignment for each item.
    constrain_testlet_loadings : bool, default=False
        If True, constrain testlet loadings to be equal within testlet.
    item_names : list of str, optional
        Names for items.
    """

    model_name = "BifactorTestlet"

    def __init__(
        self,
        n_items: int,
        testlet_membership: NDArray[np.int_] | list[int],
        constrain_testlet_loadings: bool = False,
        item_names: list[str] | None = None,
    ) -> None:
        self._constrain_loadings = constrain_testlet_loadings
        super().__init__(
            n_items=n_items,
            testlet_membership=testlet_membership,
            item_names=item_names,
        )

    @property
    def constrain_testlet_loadings(self) -> bool:
        return self._constrain_loadings

    @property
    def general_loadings(self) -> NDArray[np.float64]:
        """General factor loadings (same as discrimination)."""
        return self._parameters["discrimination"].copy()

    def set_general_loadings(self, loadings: NDArray[np.float64]) -> Self:
        """Set general factor loadings."""
        loadings = np.asarray(loadings, dtype=np.float64)
        if loadings.shape != (self.n_items,):
            raise ValueError(f"loadings shape {loadings.shape} != ({self.n_items},)")
        self._parameters["discrimination"] = loadings
        return self

    def set_testlet_loadings(self, loadings: NDArray[np.float64]) -> Self:
        """Set testlet-specific factor loadings."""
        loadings = np.asarray(loadings, dtype=np.float64)
        if loadings.shape != (self.n_items,):
            raise ValueError(f"loadings shape {loadings.shape} != ({self.n_items},)")

        loadings[self._testlet_membership < 0] = 0.0

        if self._constrain_loadings:
            for testlet_idx in self._unique_testlets:
                items = self.get_testlet_items(testlet_idx)
                mean_loading = loadings[items].mean()
                loadings[items] = mean_loading

        self._parameters["testlet_loadings"] = loadings
        return self

    def explained_variance(self) -> dict:
        """Compute variance explained by general and testlet factors.

        Returns
        -------
        dict
            Contains 'general', 'testlet', and 'unique' variance proportions.
        """
        general = self._parameters["discrimination"]
        testlet = self._parameters["testlet_loadings"]

        var_general = np.mean(general**2)
        var_testlet = np.mean(testlet**2)
        var_unique = 1.0

        total = var_general + var_testlet + var_unique

        return {
            "general": var_general / total,
            "testlet": var_testlet / total,
            "unique": var_unique / total,
            "total_common": (var_general + var_testlet) / total,
        }

    def omega_hierarchical(self) -> float:
        """Compute omega hierarchical (general factor saturation).

        Returns
        -------
        float
            Proportion of reliable variance due to general factor.
        """
        general = self._parameters["discrimination"]
        testlet = self._parameters["testlet_loadings"]

        sum_general = general.sum()
        sum_testlet_sq = sum(
            testlet[self._testlet_membership == t].sum() ** 2
            for t in self._unique_testlets
        )

        numerator = sum_general**2
        denominator = sum_general**2 + sum_testlet_sq + self.n_items

        return numerator / denominator if denominator > 0 else 0.0

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = BifactorTestletModel(
            n_items=self.n_items,
            testlet_membership=self._testlet_membership.copy(),
            constrain_testlet_loadings=self._constrain_loadings,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class RandomTestletEffectsModel(TestletModel):
    """Random effects approach to testlet dependence.

    This model treats testlet effects as random draws from a
    normal distribution with estimated variance. The model integrates
    out the testlet effects in likelihood computations.

    Parameters
    ----------
    n_items : int
        Number of items.
    testlet_membership : NDArray or list
        Testlet assignment for each item.
    n_quadpts : int, default=11
        Number of quadrature points for integration.
    item_names : list of str, optional
        Names for items.
    """

    model_name = "RandomTestletEffects"

    def __init__(
        self,
        n_items: int,
        testlet_membership: NDArray[np.int_] | list[int],
        n_quadpts: int = 11,
        item_names: list[str] | None = None,
    ) -> None:
        self._n_quadpts = n_quadpts
        super().__init__(
            n_items=n_items,
            testlet_membership=testlet_membership,
            item_names=item_names,
        )

    @property
    def n_quadpts(self) -> int:
        return self._n_quadpts

    @property
    def testlet_effect_variance(self) -> NDArray[np.float64]:
        """Variance of testlet random effects."""
        return self._parameters["testlet_variances"].copy()

    def set_testlet_variance(self, testlet_idx: int, variance: float) -> Self:
        """Set variance for a specific testlet."""
        if variance < 0:
            raise ValueError("variance must be non-negative")
        if testlet_idx not in self._unique_testlets:
            raise ValueError(f"Unknown testlet index: {testlet_idx}")

        pos = np.where(self._unique_testlets == testlet_idx)[0][0]
        self._parameters["testlet_variances"][pos] = variance
        return self

    def set_all_testlet_variances(self, variance: float) -> Self:
        """Set same variance for all testlets."""
        if variance < 0:
            raise ValueError("variance must be non-negative")
        self._parameters["testlet_variances"][:] = variance
        return self

    def integrate_out_testlet_effects(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood integrating out testlet effects.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        theta : NDArray
            General ability values (n_persons,).

        Returns
        -------
        NDArray
            Log-likelihood for each person (n_persons,).
        """
        from scipy.special import roots_hermite

        responses = np.asarray(responses)
        theta = np.asarray(theta).ravel()
        n_persons = len(theta)

        nodes, weights = roots_hermite(self._n_quadpts)
        weights = weights / np.sqrt(np.pi)
        nodes = nodes * np.sqrt(2)

        a = self._parameters["discrimination"]
        d = self._parameters["testlet_loadings"]
        b = self._parameters["difficulty"]
        testlet_vars = self._parameters["testlet_variances"]

        standalone_items = np.where(self._testlet_membership < 0)[0]
        ll_standalone = np.zeros(n_persons)

        for j in standalone_items:
            if responses[:, j].min() >= 0:
                z = a[j] * theta - b[j]
                p = sigmoid(z)
                p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                ll_standalone += responses[:, j] * np.log(p) + (
                    1 - responses[:, j]
                ) * np.log(1 - p)

        ll_testlets = np.zeros(n_persons)

        for t_idx in self._unique_testlets:
            items = self.get_testlet_items(t_idx)
            t_pos = np.where(self._unique_testlets == t_idx)[0][0]
            var_t = testlet_vars[t_pos]

            marginal_ll = np.zeros(n_persons)

            for q in range(self._n_quadpts):
                gamma = nodes[q] * np.sqrt(var_t)

                cond_ll = np.zeros(n_persons)
                for j in items:
                    if responses[:, j].min() >= 0:
                        z = a[j] * theta + d[j] * gamma - b[j]
                        p = sigmoid(z)
                        p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                        cond_ll += responses[:, j] * np.log(p) + (
                            1 - responses[:, j]
                        ) * np.log(1 - p)

                marginal_ll += weights[q] * np.exp(cond_ll)

            ll_testlets += np.log(marginal_ll + 1e-300)

        return ll_standalone + ll_testlets

    def estimate_testlet_variances(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Estimate testlet variances using method of moments.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        theta : NDArray
            Estimated general ability values.

        Returns
        -------
        NDArray
            Estimated variance for each testlet.
        """
        responses = np.asarray(responses)
        theta = np.asarray(theta).ravel()

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        estimated_vars = np.zeros(self._n_testlets)

        for t_idx in self._unique_testlets:
            items = self.get_testlet_items(t_idx)
            n_items_t = len(items)

            if n_items_t < 2:
                pos = np.where(self._unique_testlets == t_idx)[0][0]
                estimated_vars[pos] = 0.0
                continue

            residuals = np.zeros((len(theta), n_items_t))
            for i, j in enumerate(items):
                z = a[j] * theta - b[j]
                expected = sigmoid(z)
                residuals[:, i] = responses[:, j] - expected

            corr_matrix = np.corrcoef(residuals.T)
            off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            mean_corr = np.mean(off_diag) if len(off_diag) > 0 else 0.0

            pos = np.where(self._unique_testlets == t_idx)[0][0]
            estimated_vars[pos] = max(0.0, mean_corr)

        return estimated_vars

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = RandomTestletEffectsModel(
            n_items=self.n_items,
            testlet_membership=self._testlet_membership.copy(),
            n_quadpts=self._n_quadpts,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


def compute_testlet_q3(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    testlet_membership: NDArray[np.int_],
) -> dict:
    """Compute Q3 statistic for testlet local dependence.

    The Q3 statistic measures residual correlations between items
    after conditioning on theta. High Q3 values within testlets
    indicate local dependence.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items).
    theta : NDArray
        Ability estimates (n_persons,).
    discrimination : NDArray
        Item discriminations (n_items,).
    difficulty : NDArray
        Item difficulties (n_items,).
    testlet_membership : NDArray
        Testlet assignment for each item.

    Returns
    -------
    dict
        Contains 'q3_matrix', 'within_testlet', 'between_testlet'.
    """
    responses = np.asarray(responses)
    theta = np.asarray(theta).ravel()
    n_items = responses.shape[1]

    residuals = np.zeros_like(responses, dtype=np.float64)
    for j in range(n_items):
        z = discrimination[j] * theta - difficulty[j]
        expected = sigmoid(z)
        residuals[:, j] = responses[:, j] - expected

    q3_matrix = np.corrcoef(residuals.T)

    unique_testlets = np.unique(testlet_membership)
    unique_testlets = unique_testlets[unique_testlets >= 0]

    within_q3 = []
    between_q3 = []

    for i in range(n_items):
        for j in range(i + 1, n_items):
            q3_val = q3_matrix[i, j]
            if (
                testlet_membership[i] >= 0
                and testlet_membership[i] == testlet_membership[j]
            ):
                within_q3.append(q3_val)
            else:
                between_q3.append(q3_val)

    return {
        "q3_matrix": q3_matrix,
        "within_testlet_mean": np.mean(within_q3) if within_q3 else np.nan,
        "between_testlet_mean": np.mean(between_q3) if between_q3 else np.nan,
        "within_testlet_max": np.max(within_q3) if within_q3 else np.nan,
        "between_testlet_max": np.max(between_q3) if between_q3 else np.nan,
    }
