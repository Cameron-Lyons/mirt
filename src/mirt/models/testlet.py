"""Testlet (Two-Tier) Models.

This module implements testlet models that account for local dependence
among items within testlets (e.g., items sharing a common passage).
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import DichotomousItemModel

_PAIRWISE_CORRELATION_TARGET_ELEMENTS = 2_000_000


def _validate_item_index(n_items: int, item_idx: int) -> int:
    if (
        isinstance(item_idx, bool)
        or not isinstance(item_idx, Integral)
        or item_idx < 0
        or item_idx >= n_items
    ):
        raise IndexError(f"Item index {item_idx} out of range [0, {n_items})")
    return int(item_idx)


def _validate_membership(
    n_items: int,
    membership: NDArray[np.int_] | list[int],
) -> NDArray[np.intp]:
    try:
        values = np.asarray(membership)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "testlet_membership must contain integer labels",
            parameter="testlet_membership",
        ) from exc
    if values.ndim != 1:
        raise MirtValidationError(
            "testlet_membership must be one-dimensional",
            parameter="testlet_membership",
            value=values.shape,
        )
    if len(values) != n_items:
        raise MirtValidationError(
            f"testlet_membership length ({len(values)}) must match n_items ({n_items})",
            parameter="testlet_membership",
            value=len(values),
            expected=str(n_items),
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise MirtValidationError(
            "testlet_membership must contain integer labels",
            parameter="testlet_membership",
        )
    normalized = values.astype(np.intp, copy=True)
    if np.any(normalized < -1):
        raise MirtValidationError(
            "testlet labels must be -1 or non-negative integers",
            parameter="testlet_membership",
        )
    return normalized


def _validate_n_quadpts(n_quadpts: int) -> int:
    if (
        isinstance(n_quadpts, bool)
        or not isinstance(n_quadpts, Integral)
        or n_quadpts < 1
    ):
        raise MirtValidationError(
            "n_quadpts must be a positive integer",
            parameter="n_quadpts",
            value=n_quadpts,
            expected=">= 1",
        )
    return int(n_quadpts)


@lru_cache(maxsize=16)
def _normal_quadrature(
    n_quadpts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nodes and weights for a standard-normal expectation."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
    nodes = np.asarray(nodes * np.sqrt(2.0), dtype=np.float64)
    weights = np.asarray(weights / np.sqrt(np.pi), dtype=np.float64)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _logsumexp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def _validate_binary_responses(
    responses: NDArray[np.int_], n_items: int
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    try:
        values = np.asarray(responses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtDataError("responses must contain numeric values") from exc
    if values.ndim != 2:
        raise MirtDataError("responses must be two-dimensional", ndim=values.ndim)
    if values.shape[1] != n_items:
        raise MirtDataError(
            "response item count does not match the model",
            n_items=values.shape[1],
            expected=n_items,
        )

    missing = np.isnan(values) | (np.isfinite(values) & (values < 0.0))
    observed = ~missing
    observed_values = values[observed]
    if not np.all(np.isfinite(observed_values)):
        raise MirtDataError("observed responses must be finite")
    if np.any((observed_values != 0.0) & (observed_values != 1.0)):
        raise MirtDataError("observed responses must be coded 0 or 1")

    codes = np.zeros(values.shape, dtype=np.intp)
    codes[observed] = observed_values.astype(np.intp)
    return codes, observed


def _pairwise_complete_correlations(
    residuals: NDArray[np.float64],
    observed: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Compute pairwise-complete correlations with bounded row chunks."""
    residual_values = np.asarray(residuals, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.bool_)
    if residual_values.ndim != 2 or observed_values.shape != residual_values.shape:
        raise ValueError(
            "residuals and observed must have the same two-dimensional shape"
        )

    n_persons, n_items = residual_values.shape
    pair_counts = np.zeros((n_items, n_items), dtype=np.float64)
    pair_sums = np.zeros_like(pair_counts)
    pair_square_sums = np.zeros_like(pair_counts)
    pair_cross_products = np.zeros_like(pair_counts)
    chunk_size = max(
        1,
        _PAIRWISE_CORRELATION_TARGET_ELEMENTS // max(1, n_items),
    )

    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        valid = observed_values[start:stop] & np.isfinite(residual_values[start:stop])
        valid_float = valid.astype(np.float64)
        values = np.where(valid, residual_values[start:stop], 0.0)
        squares = values * values
        pair_counts += valid_float.T @ valid_float
        pair_sums += values.T @ valid_float
        pair_square_sums += squares.T @ valid_float
        pair_cross_products += values.T @ values

    safe_counts = np.maximum(pair_counts, 1.0)
    covariance = pair_cross_products - pair_sums * pair_sums.T / safe_counts
    variance_rows = pair_square_sums - pair_sums * pair_sums / safe_counts
    np.maximum(variance_rows, 0.0, out=variance_rows)
    denominator = np.sqrt(variance_rows * variance_rows.T)

    correlations = np.full((n_items, n_items), np.nan, dtype=np.float64)
    eligible = (pair_counts >= 2.0) & (denominator > 0.0)
    np.divide(
        covariance,
        denominator,
        out=correlations,
        where=eligible,
    )
    np.clip(correlations, -1.0, 1.0, out=correlations)
    return correlations, pair_counts.astype(np.intp)


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
    __test__ = False

    def __init__(
        self,
        n_items: int,
        testlet_membership: NDArray[np.int_] | list[int],
        item_names: list[str] | None = None,
        n_quadpts: int = 11,
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
        n_quadpts : int, default=11
            Number of Gauss-Hermite points used for marginal probabilities.
        """
        self._testlet_membership = _validate_membership(n_items, testlet_membership)
        unique_testlets = np.unique(self._testlet_membership)
        self._unique_testlets = unique_testlets[unique_testlets >= 0]
        self._n_testlets = len(self._unique_testlets)
        self._testlet_positions = np.full(n_items, -1, dtype=np.intp)
        for position, label in enumerate(self._unique_testlets):
            self._testlet_positions[self._testlet_membership == label] = position
        self._n_quadpts = _validate_n_quadpts(n_quadpts)

        n_factors = 1 + self._n_testlets

        super().__init__(n_items=n_items, n_factors=n_factors, item_names=item_names)

    @property
    def n_testlets(self) -> int:
        """Number of testlets."""
        return self._n_testlets

    @property
    def testlet_membership(self) -> NDArray[np.int_]:
        """Testlet assignment for each item."""
        return self._testlet_membership.copy()

    @property
    def testlet_labels(self) -> NDArray[np.int_]:
        """Sorted labels identifying the modeled testlets."""
        return self._unique_testlets.copy()

    @property
    def testlet_variances(self) -> NDArray[np.float64]:
        """Random-effect variances in ``testlet_labels`` order."""
        return self._parameters["testlet_variances"].copy()

    @property
    def n_quadpts(self) -> int:
        """Number of standard-normal quadrature points."""
        return self._n_quadpts

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

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Atomically set finite parameters while preserving testlet constraints."""
        unknown = set(params) - set(self._parameters)
        if unknown:
            name = sorted(unknown)[0]
            raise MirtValidationError(f"Unknown parameter: {name}", parameter=name)

        updated = {name: values.copy() for name, values in self._parameters.items()}
        for name, value in params.items():
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must contain numeric values", parameter=name
                ) from exc
            if array.shape != updated[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}",
                    parameter=name,
                    value=array.shape,
                    expected=str(updated[name].shape),
                )
            if not np.all(np.isfinite(array)):
                raise MirtValidationError(
                    f"{name} must contain only finite values", parameter=name
                )
            if name == "testlet_variances" and np.any(array < 0.0):
                raise MirtValidationError(
                    "testlet variances must be non-negative",
                    parameter=name,
                    expected=">= 0",
                )
            updated[name] = array.copy()

        standalone = self._testlet_positions < 0
        if np.any(updated["testlet_loadings"][standalone] != 0.0):
            raise MirtValidationError(
                "standalone items must have zero testlet loading",
                parameter="testlet_loadings",
            )
        self._parameters = updated
        return self

    def _prepare_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            values = np.asarray(theta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "theta must contain numeric values", parameter="theta"
            ) from exc
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise MirtValidationError(
                "theta must be one- or two-dimensional",
                parameter="theta",
                value=values.ndim,
                expected="1 or 2",
            )
        if values.shape[1] not in {1, self.n_factors}:
            raise MirtValidationError(
                "theta must contain the general factor alone or every factor",
                parameter="theta",
                value=values.shape[1],
                expected=f"1 or {self.n_factors}",
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "theta must contain only finite values", parameter="theta"
            )
        return values

    def _marginal_components(
        self,
        theta_general: NDArray[np.float64],
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return marginal probabilities and derivatives by general ability."""
        nodes, weights = _normal_quadrature(self._n_quadpts)
        a = self._parameters["discrimination"]
        d = self._parameters["testlet_loadings"]
        b = self._parameters["difficulty"]
        variances = self._parameters["testlet_variances"]

        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            position = self._testlet_positions[item]
            variance = variances[position] if position >= 0 else 0.0
            scale = d[item] * np.sqrt(variance)
            linear = a[item] * theta_general[:, None] - b[item] + scale * nodes[None, :]
            conditional = sigmoid(linear)
            probability = conditional @ weights
            derivative = a[item] * (conditional * (1.0 - conditional)) @ weights
            return probability, derivative

        item_variances = np.zeros(self.n_items, dtype=np.float64)
        in_testlet = self._testlet_positions >= 0
        item_variances[in_testlet] = variances[self._testlet_positions[in_testlet]]
        scale = d * np.sqrt(item_variances)
        linear = (
            a[None, :, None] * theta_general[:, None, None]
            - b[None, :, None]
            + scale[None, :, None] * nodes[None, None, :]
        )
        conditional = sigmoid(linear)
        probability = np.sum(conditional * weights[None, None, :], axis=2)
        derivative = a[None, :] * np.sum(
            conditional * (1.0 - conditional) * weights[None, None, :], axis=2
        )
        return probability, derivative

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
        theta_values = self._prepare_theta(theta)
        if theta_values.shape[1] == 1:
            return self._marginal_components(theta_values[:, 0], item_idx)[0]

        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
        a = self._parameters["discrimination"]
        d = self._parameters["testlet_loadings"]
        b = self._parameters["difficulty"]
        if item_idx is not None:
            position = self._testlet_positions[item]
            linear = a[item] * theta_values[:, 0] - b[item]
            if position >= 0:
                linear += d[item] * theta_values[:, position + 1]
            return sigmoid(linear)

        linear = a[None, :] * theta_values[:, :1] - b[None, :]
        in_testlet = self._testlet_positions >= 0
        if np.any(in_testlet):
            linear[:, in_testlet] += (
                d[None, in_testlet]
                * theta_values[:, self._testlet_positions[in_testlet] + 1]
            )
        return sigmoid(linear)

    def _marginal_probability(
        self,
        theta_general: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal probability integrating out testlet effects."""
        values = self._prepare_theta(theta_general)
        if values.shape[1] != 1:
            raise MirtValidationError(
                "theta_general must contain one general-factor column",
                parameter="theta_general",
                value=values.shape[1],
                expected="1",
            )
        return self._marginal_components(values[:, 0], item_idx)[0]

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute general-factor marginal or full conditional information.

        General-only theta values use the derivative of the quadrature-marginal
        probability. Full theta values use the squared norm of each item's
        active general and testlet loadings.
        """
        theta_values = self._prepare_theta(theta)
        if theta_values.shape[1] == 1:
            probability, derivative = self._marginal_components(
                theta_values[:, 0], item_idx
            )
            denominator = probability * (1.0 - probability)
            return np.divide(
                derivative**2,
                denominator,
                out=np.zeros_like(probability),
                where=denominator > PROB_EPSILON,
            )

        probability = self.probability(theta_values, item_idx)
        if item_idx is not None:
            item = _validate_item_index(self.n_items, item_idx)
            squared_loading = self._parameters["discrimination"][item] ** 2
            if self._testlet_positions[item] >= 0:
                squared_loading += self._parameters["testlet_loadings"][item] ** 2
            return squared_loading * probability * (1.0 - probability)

        squared_loading = self._parameters["discrimination"] ** 2
        in_testlet = self._testlet_positions >= 0
        squared_loading[in_testlet] += (
            self._parameters["testlet_loadings"][in_testlet] ** 2
        )
        return squared_loading[None, :] * probability * (1.0 - probability)

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
        if isinstance(testlet_idx, bool) or not isinstance(testlet_idx, Integral):
            raise MirtValidationError(
                "testlet_idx must be an integer label", parameter="testlet_idx"
            )
        return list(np.where(self._testlet_membership == int(testlet_idx))[0])

    def _testlet_position(self, testlet_idx: int) -> int:
        if isinstance(testlet_idx, bool) or not isinstance(testlet_idx, Integral):
            raise MirtValidationError(
                "testlet_idx must be an integer label", parameter="testlet_idx"
            )
        matches = np.flatnonzero(self._unique_testlets == int(testlet_idx))
        if len(matches) == 0:
            raise MirtValidationError(
                f"Unknown testlet index: {testlet_idx}",
                parameter="testlet_idx",
                value=testlet_idx,
            )
        return int(matches[0])

    def set_testlet_variance(self, testlet_idx: int, variance: float) -> Self:
        """Set one random-effect variance by its external testlet label."""
        position = self._testlet_position(testlet_idx)
        try:
            value = float(variance)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "variance must be numeric", parameter="variance"
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise MirtValidationError(
                "variance must be finite and non-negative",
                parameter="variance",
                value=variance,
                expected=">= 0",
            )
        updated = self._parameters["testlet_variances"].copy()
        updated[position] = value
        return self.set_parameters(testlet_variances=updated)

    def set_all_testlet_variances(self, variance: float) -> Self:
        """Set one finite non-negative variance for every testlet."""
        try:
            value = float(variance)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "variance must be numeric", parameter="variance"
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise MirtValidationError(
                "variance must be finite and non-negative",
                parameter="variance",
                value=variance,
                expected=">= 0",
            )
        return self.set_parameters(
            testlet_variances=np.full(self._n_testlets, value, dtype=np.float64)
        )

    def testlet_reliability(self) -> dict[int, float]:
        """Compute reliability for each testlet.

        Returns omega-like reliability coefficient for items within each testlet.

        Returns
        -------
        dict
            Testlet index -> reliability coefficient
        """
        reliabilities = {}

        for position, testlet_idx in enumerate(self._unique_testlets):
            items = self.get_testlet_items(testlet_idx)

            if len(items) < 2:
                reliabilities[int(testlet_idx)] = np.nan
                continue

            general_loadings = self._parameters["discrimination"][items]
            testlet_loadings = self._parameters["testlet_loadings"][items]

            sum_general = general_loadings.sum()
            sum_testlet = testlet_loadings.sum()

            var_general = sum_general**2
            var_testlet = (
                sum_testlet**2 * self._parameters["testlet_variances"][position]
            )
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
            n_quadpts=self._n_quadpts,
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
    if isinstance(n_items, bool) or not isinstance(n_items, Integral) or n_items < 1:
        raise MirtValidationError(
            "n_items must be a positive integer",
            parameter="n_items",
            value=n_items,
            expected=">= 1",
        )
    if not isinstance(testlet_sizes, list) or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 1
        for size in testlet_sizes
    ):
        raise MirtValidationError(
            "testlet_sizes must contain positive integers",
            parameter="testlet_sizes",
        )
    if sum(testlet_sizes) != n_items:
        raise MirtValidationError(
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
        n_quadpts: int = 11,
    ) -> None:
        self._constrain_loadings = constrain_testlet_loadings
        super().__init__(
            n_items=n_items,
            testlet_membership=testlet_membership,
            item_names=item_names,
            n_quadpts=n_quadpts,
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
        try:
            values = np.asarray(loadings, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "loadings must contain numeric values", parameter="loadings"
            ) from exc
        if values.shape != (self.n_items,):
            raise MirtValidationError(
                f"loadings shape {values.shape} != ({self.n_items},)",
                parameter="loadings",
            )
        return self.set_parameters(discrimination=values)

    def set_testlet_loadings(self, loadings: NDArray[np.float64]) -> Self:
        """Set testlet-specific factor loadings."""
        try:
            values = np.asarray(loadings, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "loadings must contain numeric values", parameter="loadings"
            ) from exc
        if values.shape != (self.n_items,):
            raise MirtValidationError(
                f"loadings shape {values.shape} != ({self.n_items},)",
                parameter="loadings",
            )
        values = values.copy()

        values[self._testlet_membership < 0] = 0.0

        if self._constrain_loadings:
            for testlet_idx in self._unique_testlets:
                items = self.get_testlet_items(testlet_idx)
                mean_loading = values[items].mean()
                values[items] = mean_loading

        return self.set_parameters(testlet_loadings=values)

    def explained_variance(self) -> dict[str, float]:
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
            n_quadpts=self._n_quadpts,
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
        super().__init__(
            n_items=n_items,
            testlet_membership=testlet_membership,
            item_names=item_names,
            n_quadpts=n_quadpts,
        )

    @property
    def n_quadpts(self) -> int:
        return self._n_quadpts

    @property
    def testlet_effect_variance(self) -> NDArray[np.float64]:
        """Variance of testlet random effects."""
        return self.testlet_variances

    def set_testlet_variance(self, testlet_idx: int, variance: float) -> Self:
        """Set variance for a specific testlet."""
        return super().set_testlet_variance(testlet_idx, variance)

    def set_all_testlet_variances(self, variance: float) -> Self:
        """Set same variance for all testlets."""
        return super().set_all_testlet_variances(variance)

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
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        theta_values = self._prepare_theta(theta)
        if theta_values.shape[1] != 1:
            raise MirtValidationError(
                "theta must contain general ability values only",
                parameter="theta",
                value=theta_values.shape[1],
                expected="1",
            )
        if len(theta_values) != len(response_codes):
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(theta_values),
                response_persons=len(response_codes),
            )
        return self._paired_marginal_log_likelihood(
            response_codes, observed, theta_values[:, 0]
        )

    def _paired_marginal_log_likelihood(
        self,
        response_codes: NDArray[np.intp],
        observed: NDArray[np.bool_],
        theta_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute one marginal likelihood at each person's theta value."""
        nodes, weights = _normal_quadrature(self._n_quadpts)
        log_weights = np.log(weights)
        discrimination = self._parameters["discrimination"]
        loading = self._parameters["testlet_loadings"]
        difficulty = self._parameters["difficulty"]
        variances = self._parameters["testlet_variances"]
        likelihood = np.zeros(len(theta_values), dtype=np.float64)

        standalone = self._testlet_positions < 0
        if np.any(standalone):
            linear = (
                theta_values[:, None] * discrimination[None, standalone]
                - difficulty[None, standalone]
            )
            probability = np.clip(sigmoid(linear), PROB_EPSILON, 1.0 - PROB_EPSILON)
            contributions = response_codes[:, standalone] * np.log(probability) + (
                1 - response_codes[:, standalone]
            ) * np.log1p(-probability)
            likelihood += np.sum(
                np.where(observed[:, standalone], contributions, 0.0), axis=1
            )

        for position in range(self._n_testlets):
            items = self._testlet_positions == position
            scale = np.sqrt(variances[position]) * loading[items]
            linear = (
                theta_values[:, None, None] * discrimination[None, items, None]
                - difficulty[None, items, None]
                + scale[None, :, None] * nodes[None, None, :]
            )
            probability = np.clip(sigmoid(linear), PROB_EPSILON, 1.0 - PROB_EPSILON)
            contributions = response_codes[:, items, None] * np.log(probability) + (
                1 - response_codes[:, items, None]
            ) * np.log1p(-probability)
            conditional = np.sum(
                np.where(observed[:, items, None], contributions, 0.0), axis=1
            )
            likelihood += _logsumexp(conditional + log_weights[None, :], axis=1)
        return likelihood

    def _marginal_log_likelihood_batch(
        self,
        response_codes: NDArray[np.intp],
        observed: NDArray[np.bool_],
        theta_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute every response-pattern by general-ability likelihood."""
        nodes, weights = _normal_quadrature(self._n_quadpts)
        log_weights = np.log(weights)
        discrimination = self._parameters["discrimination"]
        loading = self._parameters["testlet_loadings"]
        difficulty = self._parameters["difficulty"]
        variances = self._parameters["testlet_variances"]
        successes = (observed & (response_codes == 1)).astype(np.float64)
        failures = (observed & (response_codes == 0)).astype(np.float64)
        likelihood = np.zeros(
            (len(response_codes), len(theta_values)), dtype=np.float64
        )

        standalone = self._testlet_positions < 0
        if np.any(standalone):
            linear = (
                theta_values[:, None] * discrimination[None, standalone]
                - difficulty[None, standalone]
            )
            probability = np.clip(sigmoid(linear), PROB_EPSILON, 1.0 - PROB_EPSILON)
            likelihood += successes[:, standalone] @ np.log(probability).T
            likelihood += failures[:, standalone] @ np.log1p(-probability).T

        for position in range(self._n_testlets):
            items = self._testlet_positions == position
            scale = np.sqrt(variances[position]) * loading[items]
            linear = (
                theta_values[:, None, None] * discrimination[None, items, None]
                - difficulty[None, items, None]
                + scale[None, :, None] * nodes[None, None, :]
            )
            probability = np.clip(sigmoid(linear), PROB_EPSILON, 1.0 - PROB_EPSILON)
            conditional = np.einsum(
                "pi,tiq->ptq",
                successes[:, items],
                np.log(probability),
                optimize=True,
            )
            conditional += np.einsum(
                "pi,tiq->ptq",
                failures[:, items],
                np.log1p(-probability),
                optimize=True,
            )
            likelihood += _logsumexp(conditional + log_weights[None, None, :], axis=2)
        return likelihood

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute conditional or random-effect-marginal log likelihood."""
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        theta_values = self._prepare_theta(theta)
        if len(theta_values) != len(response_codes):
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(theta_values),
                response_persons=len(response_codes),
            )
        if theta_values.shape[1] == 1:
            return self._paired_marginal_log_likelihood(
                response_codes, observed, theta_values[:, 0]
            )
        normalized = np.where(observed, response_codes, -1)
        return super().log_likelihood(normalized, theta_values)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute conditional or marginal likelihood over a theta grid."""
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        theta_values = self._prepare_theta(theta)
        if theta_values.shape[1] == 1:
            return self._marginal_log_likelihood_batch(
                response_codes, observed, theta_values[:, 0]
            )
        normalized = np.where(observed, response_codes, -1)
        return super().log_likelihood_batch(normalized, theta_values)

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
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        try:
            theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "theta must contain numeric values", parameter="theta"
            ) from exc
        if len(theta_values) != len(response_codes):
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(theta_values),
                response_persons=len(response_codes),
            )
        if not np.all(np.isfinite(theta_values)):
            raise MirtValidationError(
                "theta must contain only finite values", parameter="theta"
            )

        a = self._parameters["discrimination"]
        b = self._parameters["difficulty"]

        expected = sigmoid(theta_values[:, None] * a[None, :] - b[None, :])
        residuals = np.where(observed, response_codes - expected, np.nan)
        estimated_vars = np.zeros(self._n_testlets)

        for position, t_idx in enumerate(self._unique_testlets):
            items = self.get_testlet_items(t_idx)
            n_items_t = len(items)

            if n_items_t < 2:
                continue

            correlations, _ = _pairwise_complete_correlations(
                residuals[:, items],
                observed[:, items],
            )
            values = correlations[np.triu_indices(n_items_t, k=1)]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                estimated_vars[position] = max(0.0, float(np.mean(finite_values)))

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
) -> dict[
    str,
    NDArray[np.float64] | NDArray[np.intp] | float | int,
]:
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
        Contains the Q3 matrix, pairwise-complete sample sizes, within- and
        between-testlet summaries, and the number of usable pairs behind each
        summary.
    """
    response_values = np.asarray(responses)
    if response_values.ndim != 2:
        raise MirtDataError(
            "responses must be two-dimensional", ndim=response_values.ndim
        )
    n_persons, n_items = response_values.shape
    response_codes, observed = _validate_binary_responses(responses, n_items)

    try:
        theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
        discrimination_values = np.asarray(discrimination, dtype=np.float64).reshape(-1)
        difficulty_values = np.asarray(difficulty, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError("Q3 inputs must contain numeric values") from exc
    if len(theta_values) != n_persons:
        raise MirtDataError(
            "theta and responses must contain the same number of persons",
            theta_persons=len(theta_values),
            response_persons=n_persons,
        )
    if discrimination_values.shape != (n_items,) or difficulty_values.shape != (
        n_items,
    ):
        raise MirtValidationError(
            "item parameter lengths must match the response columns",
            parameter="item_parameters",
            expected=str((n_items,)),
        )
    if not (
        np.all(np.isfinite(theta_values))
        and np.all(np.isfinite(discrimination_values))
        and np.all(np.isfinite(difficulty_values))
    ):
        raise MirtValidationError("Q3 inputs must contain only finite values")
    membership = _validate_membership(n_items, testlet_membership)

    expected = sigmoid(
        theta_values[:, None] * discrimination_values[None, :]
        - difficulty_values[None, :]
    )
    residuals = np.where(observed, response_codes - expected, np.nan)
    q3_matrix, pair_counts = _pairwise_complete_correlations(residuals, observed)

    rows, columns = np.triu_indices(n_items, k=1)
    q3_values = q3_matrix[rows, columns]
    finite = np.isfinite(q3_values)
    within = (membership[rows] >= 0) & (membership[rows] == membership[columns])
    within_q3 = q3_values[finite & within]
    between_q3 = q3_values[finite & ~within]

    return {
        "q3_matrix": q3_matrix,
        "pair_counts": pair_counts,
        "within_testlet_mean": (
            float(np.mean(within_q3)) if within_q3.size else np.nan
        ),
        "between_testlet_mean": (
            float(np.mean(between_q3)) if between_q3.size else np.nan
        ),
        "within_testlet_max": (float(np.max(within_q3)) if within_q3.size else np.nan),
        "between_testlet_max": (
            float(np.max(between_q3)) if between_q3.size else np.nan
        ),
        "n_within_pairs": int(within_q3.size),
        "n_between_pairs": int(between_q3.size),
    }
