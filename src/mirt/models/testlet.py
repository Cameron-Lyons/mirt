"""Testlet (Two-Tier) Models.

This module implements testlet models that account for local dependence
among items within testlets (e.g., items sharing a common passage).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from mirt._core import sigmoid
from mirt._correlation import pairwise_correlations
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import DichotomousItemModel
from mirt.models.multidimensional import standardized_loadings

_PAIRWISE_CORRELATION_TARGET_ELEMENTS = 2_000_000
_TESTLET_LIKELIHOOD_TARGET_ELEMENTS = 1_000_000
_TESTLET_PAIR_TARGET_ELEMENTS = 2_000_000


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


def _marginalize_log_likelihood_inplace(
    values: NDArray[np.float64],
    log_weights: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    """Marginalize a temporary likelihood array without another full copy."""
    weight_shape = [1] * values.ndim
    weight_shape[axis] = len(log_weights)
    values += log_weights.reshape(weight_shape)
    maximum = np.max(values, axis=axis, keepdims=True)
    values -= maximum
    np.exp(values, out=values)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(values, axis=axis))


def _log_probability_pair(
    linear: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return clipped log-probability pairs while reusing one input-sized array."""
    probability = np.clip(sigmoid(linear), PROB_EPSILON, 1.0 - PROB_EPSILON)
    log_complement = np.log1p(-probability)
    np.log(probability, out=probability)
    return probability, log_complement


def _requested_likelihood_chunk_size(
    chunk_size: int | None,
    n_rows: int,
) -> int:
    """Validate a requested maximum row count."""
    if chunk_size is None:
        return max(1, n_rows)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size < 1
    ):
        raise MirtValidationError(
            "chunk_size must be a positive integer",
            parameter="chunk_size",
            value=chunk_size,
            expected=">= 1",
        )
    return max(1, min(n_rows, int(chunk_size)))


def _largest_testlet_size(
    positions: NDArray[np.intp],
    n_testlets: int,
) -> int:
    if n_testlets == 0:
        return 0
    assigned = positions[positions >= 0]
    return int(np.bincount(assigned, minlength=n_testlets).max(initial=0))


def _paired_likelihood_chunk_size(
    positions: NDArray[np.intp],
    n_testlets: int,
    n_quadpts: int,
    n_rows: int,
    chunk_size: int | None,
) -> int:
    """Bound the largest person-by-item-by-quadrature temporary."""
    requested = _requested_likelihood_chunk_size(chunk_size, n_rows)
    standalone_items = int(np.count_nonzero(positions < 0))
    testlet_items = _largest_testlet_size(positions, n_testlets)
    values_per_row = max(1, standalone_items, testlet_items * n_quadpts)
    automatic = max(1, _TESTLET_LIKELIHOOD_TARGET_ELEMENTS // values_per_row)
    return min(requested, automatic)


def _batch_likelihood_chunk_sizes(
    positions: NDArray[np.intp],
    n_testlets: int,
    n_quadpts: int,
    n_patterns: int,
    n_theta: int,
    chunk_size: int | None,
) -> tuple[int, int]:
    """Bound response-pattern and theta chunks for a marginal grid."""
    pattern_rows = _requested_likelihood_chunk_size(chunk_size, n_patterns)
    standalone_items = int(np.count_nonzero(positions < 0))
    testlet_items = _largest_testlet_size(positions, n_testlets)
    theta_values = max(1, standalone_items, testlet_items * n_quadpts)
    theta_rows = max(
        1,
        min(
            max(1, n_theta),
            _TESTLET_LIKELIHOOD_TARGET_ELEMENTS
            // max(theta_values, pattern_rows * n_quadpts),
        ),
    )
    pattern_rows = min(
        pattern_rows,
        max(
            1,
            _TESTLET_LIKELIHOOD_TARGET_ELEMENTS // max(1, theta_rows * n_quadpts),
        ),
    )
    return pattern_rows, theta_rows


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


class _TestletPairMoment:
    """Moment condition of one testlet's standard deviation ``sigma``.

    Calling it returns the observed minus the model-implied sum of the
    within-testlet response covariances given the general abilities, over
    the item pairs each person answered. Its root in ``sigma`` is the moment
    estimate. Rows are processed in bounded chunks.
    """

    def __init__(
        self,
        theta: NDArray[np.float64],
        responses: NDArray[np.float64],
        observed: NDArray[np.float64],
        discrimination: NDArray[np.float64],
        difficulty: NDArray[np.float64],
        loadings: NDArray[np.float64],
        nodes: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        self.base = theta[:, None] * discrimination[None, :] - difficulty[None, :]
        self.responses = responses
        self.observed = observed
        self.loadings = loadings
        self.nodes = nodes
        self.weights = weights
        self.rows = max(
            1,
            _TESTLET_LIKELIHOOD_TARGET_ELEMENTS // max(1, loadings.size * nodes.size),
        )

    def __call__(self, sigma: float) -> float:
        total = 0.0
        shift = sigma * self.loadings[:, None] * self.nodes[None, :]
        for start in range(0, self.base.shape[0], self.rows):
            stop = start + self.rows
            observed = self.observed[start:stop]
            curves = sigmoid(self.base[start:stop, :, None] + shift[None, :, :])
            marginal = curves @ self.weights
            residual = observed * (self.responses[start:stop] - marginal)
            observed_pairs = 0.5 * (
                residual.sum(axis=1) ** 2 - np.sum(residual**2, axis=1)
            )
            # E[sum_{j<k} P_j P_k] - sum_{j<k} m_j m_k for the answered items.
            weighted = curves * observed[:, :, None]
            joint = 0.5 * (
                weighted.sum(axis=1) ** 2 - np.sum(weighted * curves, axis=1)
            )
            answered = observed * marginal
            independent = 0.5 * (
                answered.sum(axis=1) ** 2 - np.sum(answered * marginal, axis=1)
            )
            implied_pairs = joint @ self.weights - independent
            total += float(np.sum(observed_pairs - implied_pairs))
        return total


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

    return pairwise_correlations(
        residual_values,
        observed_values,
        chunk_size=max(
            1, _PAIRWISE_CORRELATION_TARGET_ELEMENTS // max(1, residual_values.shape[1])
        ),
    )


class TestletModel(DichotomousItemModel):
    """Two-tier testlet model for handling local item dependence.

    The testlet model adds testlet-specific random effects to account
    for the common variance among items within a testlet:

    P(X_ij = 1 | theta, gamma_t) = logistic(a_j * theta + d_j * gamma_t - b_j)

    where:
    - theta: General ability factor
    - gamma_t: Testlet-specific random effect, ``N(0, sigma_t^2)``
    - a_j: Item discrimination on general factor
    - d_j: Item loading on testlet factor
    - b_j: Item difficulty
    - sigma_t^2: Testlet variance (``testlet_variances``)

    This is a bifactor model whose specific slopes are ``d_j * sigma_t``. A
    full theta holds the standardized testlet effects ``z_t = gamma_t /
    sigma_t``, which are standard normal like the general factor, so
    ``probability`` with every factor evaluates ``logistic(a_j * theta +
    d_j * sigma_t * z_t - b_j)``; general-only theta integrates ``z_t`` out.

    Only the products ``d_j * sigma_t`` are identified. A testlet's variance
    is therefore held fixed, at 1 unless set otherwise, while any loading of
    its testlet is free, as R mirt's ``bfactor`` fixes specific-factor
    variances. Fixing every loading of a testlet with
    ``set_free_parameter_masks``, for example at 1 as in the random
    testlet-effect model of Wang and Wilson (2005), frees its variance, which
    :class:`~mirt.estimation.em.EMEstimator` and
    :class:`~mirt.estimation.bifactor_em.BifactorEMEstimator` then estimate.
    Both fit testlet models by the Gibbons-Hedeker dimension reduction, one
    (general, testlet) grid per testlet; EM falls back to the full product
    grid only for options the reduction lacks.

    References
    ----------
    Wang, W.-C., & Wilson, M. (2005). The Rasch testlet model. *Applied
    Psychological Measurement*, 29(2), 126-149.
    """

    model_name = "Testlet"
    supports_multidimensional = True
    _global_parameters = frozenset({"testlet_variances"})
    _parameter_boxes = {"testlet_variances": (0.0, 25.0)}
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
    def _shared_parameters(self) -> frozenset[str]:
        """Parameters shared by the items of a testlet rather than one item.

        The testlet variances belong to no item; EM's shared-parameter step
        estimates each free one from the expected counts of every item.
        """
        return frozenset({"testlet_variances"})

    def _family_loading_mask(self) -> NDArray[np.bool_]:
        """Return the loadings the family leaves free: those of testlet items."""
        return self._testlet_positions >= 0

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Free coordinates; a testlet's variance only with all loadings fixed.

        Loadings and the variance of a testlet enter the model only through
        their products ``d_j * sigma_t``, so the variance is free only when
        no loading of its testlet is free and at least one is nonzero.
        """
        masks = super().free_parameter_masks
        masks["testlet_loadings"] &= self._family_loading_mask()
        masks = self._apply_free_parameter_restrictions(masks)
        loadings = self._parameters["testlet_loadings"]
        for position in range(self._n_testlets):
            items = self._testlet_positions == position
            if np.any(masks["testlet_loadings"][items]) or not np.any(
                loadings[items] != 0.0
            ):
                masks["testlet_variances"][position] = False
        return masks

    def set_free_parameter_masks(
        self, masks: Mapping[str, NDArray[np.bool_]] | None
    ) -> Self:
        """Restrict free coordinates without changing parameter values.

        Fixing every loading of a testlet frees its variance (see
        :attr:`free_parameter_masks`). A ``testlet_variances`` mask is checked
        against the loadings in the same call, so it may hold such a
        variance fixed again but cannot free the variance of a testlet whose
        loadings are free.

        Raises
        ------
        MirtValidationError
            If a mask is malformed or frees a coordinate the family fixes.
        """
        if (
            masks is None
            or not isinstance(masks, Mapping)
            or "testlet_variances" not in masks
        ):
            return super().set_free_parameter_masks(masks)
        previous = self._free_parameter_restrictions
        super().set_free_parameter_masks(
            {name: mask for name, mask in masks.items() if name != "testlet_variances"}
        )
        allowed = self.free_parameter_masks["testlet_variances"]
        mask = np.asarray(masks["testlet_variances"])
        if mask.dtype != np.bool_ or mask.shape != allowed.shape:
            self._free_parameter_restrictions = previous
            raise MirtValidationError(
                f"Mask for testlet_variances must be Boolean with shape "
                f"{allowed.shape}",
                parameter="masks",
            )
        if np.any(mask & ~allowed):
            self._free_parameter_restrictions = previous
            raise MirtValidationError(
                "Mask for testlet_variances cannot free model-family fixed "
                "parameters: a testlet variance is identified only when every "
                "loading of its testlet is fixed",
                parameter="masks",
            )
        if not np.array_equal(mask, allowed):
            self._free_parameter_restrictions["testlet_variances"] = mask.copy()
        return self

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

    def _testlet_slopes(self) -> NDArray[np.float64]:
        """Return each item's slope ``d_j * sigma_t`` on its standardized effect.

        Standalone items have zero slopes.
        """
        slopes = np.zeros(self.n_items, dtype=np.float64)
        in_testlet = self._testlet_positions >= 0
        slopes[in_testlet] = self._parameters["testlet_loadings"][in_testlet] * np.sqrt(
            self._parameters["testlet_variances"][self._testlet_positions[in_testlet]]
        )
        return slopes

    def get_loading_matrix(self) -> NDArray[np.float64]:
        """Return the bifactor slopes on the general and standardized testlet factors.

        Returns
        -------
        ndarray of shape (n_items, 1 + n_testlets)
            Discriminations in the first column and each testlet item's
            ``d_j * sigma_t`` in the column of its testlet, the slopes of the
            equivalent :class:`~mirt.models.bifactor.BifactorModel`.
        """
        loadings = np.zeros((self.n_items, self.n_factors), dtype=np.float64)
        loadings[:, 0] = self._parameters["discrimination"]
        in_testlet = np.flatnonzero(self._testlet_positions >= 0)
        loadings[in_testlet, 1 + self._testlet_positions[in_testlet]] = (
            self._testlet_slopes()[in_testlet]
        )
        return loadings

    def _standardized_components(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return standardized general and testlet loadings and uniquenesses.

        The slopes of :meth:`get_loading_matrix` are converted with
        ``D = 1.702`` as :class:`~mirt.models.bifactor.BifactorModel` does
        (see :func:`~mirt.models.multidimensional.standardized_loadings`);
        standalone items have zero testlet loadings.
        """
        loadings, communalities = standardized_loadings(self.get_loading_matrix())
        testlet = np.zeros(self.n_items, dtype=np.float64)
        in_testlet = np.flatnonzero(self._testlet_positions >= 0)
        testlet[in_testlet] = loadings[
            in_testlet, 1 + self._testlet_positions[in_testlet]
        ]
        return loadings[:, 0], testlet, 1.0 - communalities

    def _score_variance_components(
        self, items: NDArray[np.intp] | None = None
    ) -> tuple[float, float, float]:
        """Return general, testlet and unique variance of a standardized sum.

        The general and testlet components are squared loading sums, the
        unique component the sum of the uniquenesses ``1 - h2``.
        """
        if items is None:
            items = np.arange(self.n_items)
        general, testlet, unique = self._standardized_components()
        positions = self._testlet_positions[items]
        testlet_variance = sum(
            float(np.sum(testlet[items[positions == position]]) ** 2)
            for position in np.unique(positions[positions >= 0])
        )
        return (
            float(np.sum(general[items]) ** 2),
            testlet_variance,
            float(np.sum(unique[items])),
        )

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
            item = self._validate_item_index(item_idx)
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
        testlet-specific factors: theta = [theta_general, z_1, ..., z_T],
        where ``z_t = gamma_t / sigma_t`` is the standardized effect of
        testlet ``t``, which enters with slope ``d_j * sigma_t``.

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
            item = self._validate_item_index(item_idx)
        a = self._parameters["discrimination"]
        slopes = self._testlet_slopes()
        b = self._parameters["difficulty"]
        if item_idx is not None:
            position = self._testlet_positions[item]
            linear = a[item] * theta_values[:, 0] - b[item]
            if position >= 0:
                linear += slopes[item] * theta_values[:, position + 1]
            return sigmoid(linear)

        linear = a[None, :] * theta_values[:, :1] - b[None, :]
        in_testlet = self._testlet_positions >= 0
        if np.any(in_testlet):
            linear[:, in_testlet] += (
                slopes[None, in_testlet]
                * theta_values[:, self._testlet_positions[in_testlet] + 1]
            )
        return sigmoid(linear)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs without item dispatch."""
        theta_values = self._prepare_theta(theta)
        indices = self._prepare_item_indices(item_indices, theta_values.shape[0])
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)

        if theta_values.shape[1] == 1:
            return self._marginal_probability_pairs(theta_values[:, 0], indices)

        discrimination = self._parameters["discrimination"][indices]
        slopes = self._testlet_slopes()[indices]
        difficulty = self._parameters["difficulty"][indices]
        positions = self._testlet_positions[indices]
        linear = discrimination * theta_values[:, 0] - difficulty
        in_testlet = positions >= 0
        rows = np.flatnonzero(in_testlet)
        linear[rows] += (
            slopes[rows]
            * theta_values[
                rows,
                positions[rows] + 1,
            ]
        )
        return sigmoid(linear)

    def _marginal_probability_pairs(
        self,
        theta_general: NDArray[np.float64],
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Integrate aligned item effects in bounded quadrature blocks."""
        nodes, weights = _normal_quadrature(self._n_quadpts)
        discrimination = self._parameters["discrimination"]
        loadings = self._parameters["testlet_loadings"]
        difficulty = self._parameters["difficulty"]
        variances = self._parameters["testlet_variances"]
        pair_chunk_size = max(
            1,
            _TESTLET_PAIR_TARGET_ELEMENTS // self._n_quadpts,
        )
        result = np.empty(item_indices.size, dtype=np.float64)

        for start in range(0, item_indices.size, pair_chunk_size):
            stop = min(start + pair_chunk_size, item_indices.size)
            chunk_indices = item_indices[start:stop]
            positions = self._testlet_positions[chunk_indices]
            item_variances = np.zeros(chunk_indices.size, dtype=np.float64)
            in_testlet = positions >= 0
            item_variances[in_testlet] = variances[positions[in_testlet]]
            scale = loadings[chunk_indices] * np.sqrt(item_variances)
            linear = (
                discrimination[chunk_indices, None] * theta_general[start:stop, None]
                - difficulty[chunk_indices, None]
                + scale[:, None] * nodes[None, :]
            )
            result[start:stop] = sigmoid(linear) @ weights

        return result

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
        slopes on the general and standardized testlet factors, ``a_j`` and
        ``d_j * sigma_t``.
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
        squared_loading = (
            self._parameters["discrimination"] ** 2 + self._testlet_slopes() ** 2
        )
        if item_idx is not None:
            item = self._validate_item_index(item_idx)
            return squared_loading[item] * probability * (1.0 - probability)
        return squared_loading[None, :] * probability * (1.0 - probability)

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item's discrimination, loading or difficulty.

        Raises
        ------
        MirtValidationError
            For testlet variances, which no single item owns, and for the
            common loadings of a constrained testlet, which its items share;
            set them with ``set_parameters``.
        """
        if param_name in self._global_parameters:
            self._validate_item_index(item_idx)
            raise MirtValidationError(
                f"Parameter {param_name} does not have per-item values; use "
                "set_parameters",
                parameter=param_name,
            )
        if param_name in self._shared_parameters:
            self._validate_item_index(item_idx)
            raise MirtValidationError(
                f"{param_name} are shared by the items of each testlet; use "
                "set_parameters",
                parameter=param_name,
            )
        super().set_item_parameter(item_idx, param_name, value)

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

        McDonald's omega total of each testlet's items from the standardized
        loadings of the equivalent bifactor model (slopes ``a_j`` and
        ``d_j * sigma_t``, ``D = 1.702``), as
        :meth:`BifactorModel.omega_subscale
        <mirt.models.bifactor.BifactorModel.omega_subscale>` computes it: the
        squared general and testlet loading sums over the variance of the
        sum of standardized latent responses, which adds the uniquenesses
        ``1 - h2``.

        Returns
        -------
        dict
            Testlet index -> reliability coefficient, NaN for a testlet with
            fewer than two items
        """
        reliabilities = {}

        for testlet_idx in self._unique_testlets:
            items = np.asarray(self.get_testlet_items(testlet_idx), dtype=np.intp)

            if len(items) < 2:
                reliabilities[int(testlet_idx)] = np.nan
                continue

            general, testlet, unique = self._score_variance_components(items)
            common = general + testlet
            reliabilities[int(testlet_idx)] = float(common / (common + unique))

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

        self._copy_parameter_restrictions_to(new_model)
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
        If True, constrain testlet loadings to be equal within testlet. The
        first item of each testlet stores the free common loading and the
        others copy it. Each common loading is shared by its testlet's
        items, so EM estimates it from all of their expected counts, and
        estimators without a shared-parameter step refuse the model unless
        the loadings are held fixed.
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
    def _shared_parameters(self) -> frozenset[str]:
        """Testlet variances, and the common loadings when they are constrained.

        A common loading belongs to every item of its testlet, so EM's
        shared-parameter step estimates it from their summed expected counts
        rather than an itemwise step from its first item's.
        """
        if self._constrain_loadings:
            return frozenset({"testlet_loadings", "testlet_variances"})
        return frozenset({"testlet_variances"})

    def _family_loading_mask(self) -> NDArray[np.bool_]:
        """Return the stored free loadings: the first item of each constrained testlet."""
        mask = super()._family_loading_mask()
        if self._constrain_loadings:
            mask = np.zeros(self.n_items, dtype=np.bool_)
            for label in self._unique_testlets:
                mask[self.get_testlet_items(int(label))[0]] = True
        return mask

    def _canonical_parameter_values(
        self, name: str, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        canonical = super()._canonical_parameter_values(name, values)
        if name == "testlet_loadings" and self._constrain_loadings:
            for label in self._unique_testlets:
                items = self.get_testlet_items(int(label))
                canonical[items] = canonical[items[0]]
        return canonical

    def _expand_parameter_standard_errors(
        self, name: str, errors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        expanded = super()._expand_parameter_standard_errors(name, errors)
        if name == "testlet_loadings" and self._constrain_loadings:
            for label in self._unique_testlets:
                items = self.get_testlet_items(int(label))
                expanded[items] = expanded[items[0]]
        return expanded

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        if self._constrain_loadings and "testlet_loadings" in params:
            values = np.asarray(params["testlet_loadings"], dtype=np.float64)
            if values.shape == (self.n_items,):
                for label in self._unique_testlets:
                    items = self.get_testlet_items(int(label))
                    if not np.allclose(
                        values[items], values[items[0]], rtol=0.0, atol=1e-12
                    ):
                        raise MirtValidationError(
                            "constrained testlet loadings must use one common value per testlet",
                            parameter="testlet_loadings",
                        )
        return super().set_parameters(**params)

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

        Shares of the items' latent response variance, averaged over items,
        from the standardized loadings of the equivalent bifactor model
        (slopes ``a_j`` and ``d_j * sigma_t`` divided by ``D = 1.702``; see
        :func:`~mirt.models.multidimensional.standardized_loadings`). For
        each item the squared general and testlet loadings and the
        uniqueness ``1 - h2`` sum to one.

        Returns
        -------
        dict
            Contains 'general', 'testlet', and 'unique' variance proportions,
            and their common part 'total_common', the mean communality.
        """
        general, testlet, unique = self._standardized_components()
        return {
            "general": float(np.mean(general**2)),
            "testlet": float(np.mean(testlet**2)),
            "unique": float(np.mean(unique)),
            "total_common": float(np.mean(1.0 - unique)),
        }

    def omega_hierarchical(self) -> float:
        """Compute omega hierarchical (general factor saturation).

        McDonald's omega hierarchical from the standardized loadings of the
        equivalent bifactor model, as
        :meth:`BifactorModel.omega_hierarchical
        <mirt.models.bifactor.BifactorModel.omega_hierarchical>` computes it:
        the squared sum of general loadings over the variance of the sum of
        standardized latent responses, which adds every testlet's squared
        loading sum and the uniquenesses ``1 - h2``. Testlet loadings are
        scaled by the testlet standard deviations ``sigma_t``.

        Returns
        -------
        float
            Proportion of total-score variance due to the general factor.
        """
        general, testlet, unique = self._score_variance_components()
        return float(general / (general + testlet + unique))

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

        self._copy_parameter_restrictions_to(new_model)
        return new_model


class RandomTestletEffectsModel(TestletModel):
    """Random effects approach to testlet dependence.

    This model treats testlet effects as random draws from a normal
    distribution and integrates them out in likelihood computations given
    general abilities alone. As in :class:`TestletModel`, a testlet's
    variance is estimated only when its loadings are held fixed, for example
    at 1 for the random testlet-effect model of Wang and Wilson (2005);
    otherwise it scales the loadings and stays at its value.
    :meth:`estimate_testlet_variances` gives method-of-moments estimates.

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
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute log-likelihood integrating out testlet effects.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        theta : NDArray
            General ability values (n_persons,).
        chunk_size : int, optional
            Maximum number of persons evaluated together. Automatic chunks
            already bound quadrature working memory; use this to set a tighter
            row limit.

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
            response_codes,
            observed,
            theta_values[:, 0],
            chunk_size,
        )

    def _paired_marginal_log_likelihood(
        self,
        response_codes: NDArray[np.intp],
        observed: NDArray[np.bool_],
        theta_values: NDArray[np.float64],
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute bounded one-to-one response-pattern likelihoods."""
        nodes, weights = _normal_quadrature(self._n_quadpts)
        log_weights = np.log(weights)
        discrimination = self._parameters["discrimination"]
        loading = self._parameters["testlet_loadings"]
        difficulty = self._parameters["difficulty"]
        variances = self._parameters["testlet_variances"]
        likelihood = np.zeros(len(theta_values), dtype=np.float64)
        standalone = self._testlet_positions < 0
        testlet_components = []
        for position in range(self._n_testlets):
            items = self._testlet_positions == position
            testlet_components.append(
                (
                    items,
                    discrimination[items],
                    difficulty[items],
                    np.sqrt(variances[position]) * loading[items],
                )
            )
        rows_per_chunk = _paired_likelihood_chunk_size(
            self._testlet_positions,
            self._n_testlets,
            self._n_quadpts,
            len(theta_values),
            chunk_size,
        )

        for start in range(0, len(theta_values), rows_per_chunk):
            stop = min(start + rows_per_chunk, len(theta_values))
            theta_chunk = theta_values[start:stop]
            codes_chunk = response_codes[start:stop]
            observed_chunk = observed[start:stop]
            chunk_likelihood = likelihood[start:stop]

            if np.any(standalone):
                linear = (
                    theta_chunk[:, None] * discrimination[None, standalone]
                    - difficulty[None, standalone]
                )
                log_probability, log_complement = _log_probability_pair(linear)
                successes = observed_chunk[:, standalone] & (
                    codes_chunk[:, standalone] == 1
                )
                failures = observed_chunk[:, standalone] & (
                    codes_chunk[:, standalone] == 0
                )
                chunk_likelihood += np.sum(
                    successes * log_probability + failures * log_complement,
                    axis=1,
                )

            for (
                items,
                item_discrimination,
                item_difficulty,
                scale,
            ) in testlet_components:
                linear = (
                    theta_chunk[:, None, None] * item_discrimination[None, :, None]
                    - item_difficulty[None, :, None]
                    + scale[None, :, None] * nodes[None, None, :]
                )
                log_probability, log_complement = _log_probability_pair(linear)
                successes = observed_chunk[:, items] & (codes_chunk[:, items] == 1)
                failures = observed_chunk[:, items] & (codes_chunk[:, items] == 0)
                conditional = np.einsum(
                    "pi,piq->pq",
                    successes,
                    log_probability,
                    optimize=True,
                )
                conditional += np.einsum(
                    "pi,piq->pq",
                    failures,
                    log_complement,
                    optimize=True,
                )
                chunk_likelihood += _marginalize_log_likelihood_inplace(
                    conditional,
                    log_weights,
                    axis=1,
                )
        return likelihood

    def _marginal_log_likelihood_batch(
        self,
        response_codes: NDArray[np.intp],
        observed: NDArray[np.bool_],
        theta_values: NDArray[np.float64],
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute a bounded response-pattern by general-ability grid."""
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
        testlet_components = []
        for position in range(self._n_testlets):
            items = self._testlet_positions == position
            testlet_components.append(
                (
                    items,
                    discrimination[items],
                    difficulty[items],
                    np.sqrt(variances[position]) * loading[items],
                )
            )
        pattern_rows, theta_rows = _batch_likelihood_chunk_sizes(
            self._testlet_positions,
            self._n_testlets,
            self._n_quadpts,
            len(response_codes),
            len(theta_values),
            chunk_size,
        )

        for theta_start in range(0, len(theta_values), theta_rows):
            theta_stop = min(theta_start + theta_rows, len(theta_values))
            theta_chunk = theta_values[theta_start:theta_stop]
            theta_slice = slice(theta_start, theta_stop)

            if np.any(standalone):
                linear = (
                    theta_chunk[:, None] * discrimination[None, standalone]
                    - difficulty[None, standalone]
                )
                log_probability, log_complement = _log_probability_pair(linear)
                for pattern_start in range(0, len(response_codes), pattern_rows):
                    pattern_stop = min(
                        pattern_start + pattern_rows, len(response_codes)
                    )
                    pattern_slice = slice(pattern_start, pattern_stop)
                    likelihood[pattern_slice, theta_slice] += (
                        successes[pattern_slice, standalone] @ log_probability.T
                        + failures[pattern_slice, standalone] @ log_complement.T
                    )

            for (
                items,
                item_discrimination,
                item_difficulty,
                scale,
            ) in testlet_components:
                linear = (
                    theta_chunk[:, None, None] * item_discrimination[None, :, None]
                    - item_difficulty[None, :, None]
                    + scale[None, :, None] * nodes[None, None, :]
                )
                log_probability, log_complement = _log_probability_pair(linear)
                for pattern_start in range(0, len(response_codes), pattern_rows):
                    pattern_stop = min(
                        pattern_start + pattern_rows, len(response_codes)
                    )
                    pattern_slice = slice(pattern_start, pattern_stop)
                    conditional = np.einsum(
                        "pi,tiq->ptq",
                        successes[pattern_slice, items],
                        log_probability,
                        optimize=True,
                    )
                    conditional += np.einsum(
                        "pi,tiq->ptq",
                        failures[pattern_slice, items],
                        log_complement,
                        optimize=True,
                    )
                    likelihood[pattern_slice, theta_slice] += (
                        _marginalize_log_likelihood_inplace(
                            conditional,
                            log_weights,
                            axis=2,
                        )
                    )
        return likelihood

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute conditional or random-effect-marginal likelihood.

        Marginal evaluation uses automatic bounded chunks. ``chunk_size`` can
        impose a tighter maximum number of response rows per chunk.
        """
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        theta_values = self._prepare_theta(theta)
        _requested_likelihood_chunk_size(chunk_size, len(response_codes))
        if len(theta_values) != len(response_codes):
            raise MirtDataError(
                "theta and responses must contain the same number of persons",
                theta_persons=len(theta_values),
                response_persons=len(response_codes),
            )
        if theta_values.shape[1] == 1:
            return self._paired_marginal_log_likelihood(
                response_codes,
                observed,
                theta_values[:, 0],
                chunk_size,
            )
        normalized = np.where(observed, response_codes, -1)
        return super().log_likelihood(normalized, theta_values)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute a conditional or marginal likelihood grid.

        Marginal grids are chunked across response patterns and theta points.
        ``chunk_size`` can impose a tighter response-pattern row limit.
        """
        response_codes, observed = _validate_binary_responses(responses, self.n_items)
        theta_values = self._prepare_theta(theta)
        _requested_likelihood_chunk_size(chunk_size, len(response_codes))
        if theta_values.shape[1] == 1:
            return self._marginal_log_likelihood_batch(
                response_codes,
                observed,
                theta_values[:, 0],
                chunk_size,
            )
        normalized = np.where(observed, response_codes, -1)
        return super().log_likelihood_batch(normalized, theta_values)

    def estimate_testlet_variances(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Estimate testlet variances by the method of moments.

        Given the general abilities, the responses to two items of a testlet
        covary only through its effect. For each testlet the variance
        ``sigma_t^2`` solves

            sum_p sum_{j<k} (y_pj - m_pj) (y_pk - m_pk) = sum_p sum_{j<k} C_pjk,

        over the pairs of the testlet's items that person ``p`` answered,
        where ``m_pj`` is item ``j``'s testlet-marginal probability at
        ``theta_p`` and ``C_pjk`` the model-implied covariance of the two
        responses given ``theta_p``. Both use the model's discriminations,
        loadings, difficulties and quadrature, with testlet slopes
        ``d_j * sigma_t``. The marginal maximum-likelihood estimates of
        ``EMEstimator`` with the loadings fixed are more efficient; this
        estimate is a starting value or a check.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        theta : NDArray
            General ability values (n_persons,), ideally the true abilities
            or precise estimates: estimation error makes the residuals of
            all items covary and biases the estimates upward.

        Returns
        -------
        NDArray
            Estimated variance for each testlet in ``testlet_labels`` order:
            zero when the residuals show no positive dependence, the upper
            bound 25 when even that variance implies less, and NaN for a
            testlet without two jointly observed items or with only zero
            loadings.
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

        nodes, weights = _normal_quadrature(self._n_quadpts)
        upper = float(np.sqrt(self._parameter_boxes["testlet_variances"][1]))
        estimated_vars = np.full(self._n_testlets, np.nan)
        for position in range(self._n_testlets):
            items = np.flatnonzero(self._testlet_positions == position)
            answered = observed[:, items]
            rows = np.flatnonzero(np.count_nonzero(answered, axis=1) >= 2)
            loadings = self._parameters["testlet_loadings"][items]
            if items.size < 2 or rows.size == 0 or not np.any(loadings != 0.0):
                continue
            moment = _TestletPairMoment(
                theta_values[rows],
                response_codes[np.ix_(rows, items)].astype(np.float64),
                answered[rows].astype(np.float64),
                self._parameters["discrimination"][items],
                self._parameters["difficulty"][items],
                loadings,
                nodes,
                weights,
            )
            if moment(0.0) <= 0.0:
                estimated_vars[position] = 0.0
            elif moment(upper) >= 0.0:
                estimated_vars[position] = upper**2
            else:
                deviation = brentq(moment, 0.0, upper, xtol=1e-8, rtol=1e-10)
                estimated_vars[position] = float(deviation) ** 2

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

        self._copy_parameter_restrictions_to(new_model)
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
