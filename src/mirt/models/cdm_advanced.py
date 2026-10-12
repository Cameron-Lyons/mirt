"""Advanced Cognitive Diagnosis Models.

This module provides:
- GDINA (Generalized DINA with reduced models) and ``fit_gdina``
- HigherOrderCDM (Higher-order model with attribute hierarchy) and
  ``fit_higher_order_cdm``
- AttributeHierarchy (DAG of prerequisite attributes)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, minimize

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtIndexError, MirtValidationError
from mirt.models.cdm import (
    BaseCDM,
    CDMResult,
    _class_posterior,
    _hierarchy_pattern_mask,
    _pattern_log_likelihood_matrix,
    _response_indicators,
    _validate_fit_configuration,
    _validate_fit_inputs,
)
from mirt.utils.numeric import logsumexp

ReducedModelType = Literal["DINA", "DINO", "ACDM", "LLM", "RRUM", "saturated"]

_REDUCED_MODEL_TO_CODE: dict[ReducedModelType, int] = {
    "DINA": 0,
    "DINO": 1,
    "ACDM": 2,
    "LLM": 3,
    "RRUM": 4,
    "saturated": 5,
}
_REDUCED_CODE_TO_MODEL: dict[int, ReducedModelType] = {
    code: name for name, code in _REDUCED_MODEL_TO_CODE.items()
}
_DEFAULT_REDUCED_MODELS: tuple[ReducedModelType, ...] = (
    "DINA",
    "DINO",
    "ACDM",
    "LLM",
    "RRUM",
    "saturated",
)
_MAX_GDINA_PROBABILITY_CHUNK_ENTRIES = 1_000_000
# Implied success probabilities may leave [0, 1] by this much, as converged
# constrained optimizers do, before a coefficient vector is rejected.
_PROBABILITY_RANGE_TOLERANCE = 1e-9
_DEFAULT_HIGHER_ORDER_QUADRATURE = 31
# Box constraints of the higher-order structural M-step, on the loadings and
# on the intercepts -loading * threshold.
_HIGHER_ORDER_LOADING_BOUNDS = (0.1, 5.0)
_HIGHER_ORDER_INTERCEPT_BOUNDS = (-10.0, 10.0)


def _expected_coefficient_count(model_type: ReducedModelType, n_required: int) -> int:
    """Number of stored delta coefficients of one reduced-model item."""
    if model_type == "saturated":
        return 2**n_required
    if model_type in ("DINA", "DINO"):
        return 2
    return n_required + 1


def _reduced_group_probabilities(
    model_type: ReducedModelType,
    group_patterns: NDArray[np.int_],
    delta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unclipped success probability of each latent group of one item.

    ``group_patterns`` holds the required-attribute pattern of each group.
    """
    if model_type == "saturated":
        return np.asarray(delta, dtype=np.float64)
    if model_type in ("DINA", "DINO"):
        if model_type == "DINA":
            ideal = np.all(group_patterns == 1, axis=1)
        else:
            ideal = np.any(group_patterns == 1, axis=1)
        return np.where(ideal, delta[1], delta[0])
    linear = delta[0] + group_patterns @ delta[1:]
    if model_type == "ACDM":
        return np.asarray(linear, dtype=np.float64)
    if model_type == "LLM":
        return np.asarray(sigmoid(linear), dtype=np.float64)
    return delta[0] * np.prod(
        np.where(group_patterns == 1, 1.0, delta[None, 1:]),
        axis=1,
    )


def _higher_order_quadrature(
    n_quad: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equally spaced standard-normal grid on [-4, 4] with normalized weights."""
    points = np.linspace(-4.0, 4.0, n_quad)
    weights = np.exp(-0.5 * points**2)
    return points, weights / np.sum(weights)


def _validate_quadrature_count(n_quad: int) -> int:
    if (
        isinstance(n_quad, (bool, np.bool_))
        or not isinstance(n_quad, (int, np.integer))
        or n_quad <= 0
    ):
        raise MirtValidationError(
            "n_quad must be a positive integer", parameter="n_quad"
        )
    return int(n_quad)


def _gdina_item_chunks(
    item_indices: NDArray[np.intp],
    entries_per_item: int,
) -> Iterator[NDArray[np.intp]]:
    """Split G-DINA items into bounded vectorized probability chunks."""
    chunk_size = max(
        1,
        _MAX_GDINA_PROBABILITY_CHUNK_ENTRIES // max(1, entries_per_item),
    )
    for start in range(0, item_indices.size, chunk_size):
        yield item_indices[start : start + chunk_size]


@dataclass
class AttributeHierarchy:
    """Describes an attribute hierarchy (directed acyclic graph).

    Attributes
    ----------
    adjacency : NDArray
        Adjacency matrix where adjacency[i, j] = 1 means attribute i
        is a prerequisite for attribute j.
    attribute_names : list of str, optional
        Names for attributes.
    """

    adjacency: NDArray[np.int_]
    attribute_names: list[str] | None = None
    _topological_order: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw_adjacency = np.asarray(self.adjacency)
        if raw_adjacency.ndim != 2 or raw_adjacency.shape[0] != raw_adjacency.shape[1]:
            raise MirtValidationError("Adjacency matrix must be square")
        n_attrs = raw_adjacency.shape[0]
        if n_attrs == 0:
            raise MirtValidationError(
                "Adjacency matrix must contain at least one attribute"
            )
        if raw_adjacency.dtype.kind not in "biuf":
            raise MirtValidationError(
                "Adjacency matrix must contain numeric binary values"
            )
        adjacency = np.asarray(raw_adjacency, dtype=np.float64)
        if not np.all(np.isfinite(adjacency)) or np.any(
            (adjacency != 0.0) & (adjacency != 1.0)
        ):
            raise MirtValidationError("Adjacency matrix must contain only 0 and 1")
        if np.any(np.diag(adjacency) != 0.0):
            raise MirtValidationError("An attribute cannot be its own prerequisite")

        self.adjacency = adjacency.astype(np.int_, copy=True)
        self._topological_order = self._compute_topological_order()
        self.adjacency.setflags(write=False)

        if self.attribute_names is None:
            self.attribute_names = [f"A{i}" for i in range(n_attrs)]
        elif len(self.attribute_names) != n_attrs:
            raise MirtValidationError(
                f"attribute_names length ({len(self.attribute_names)}) "
                f"must match n_attributes ({n_attrs})"
            )
        else:
            self.attribute_names = list(self.attribute_names)

    def _validate_attribute_index(self, attribute: int) -> int:
        if (
            isinstance(attribute, (bool, np.bool_))
            or not isinstance(attribute, (int, np.integer))
            or attribute < 0
            or attribute >= self.n_attributes
        ):
            raise MirtIndexError(f"attribute must be in [0, {self.n_attributes})")
        return int(attribute)

    def _compute_topological_order(self) -> list[int]:
        in_degree = np.sum(self.adjacency, axis=0).astype(np.int_, copy=False)
        available = [int(attribute) for attribute in np.flatnonzero(in_degree == 0)]
        order: list[int] = []

        while available:
            attribute = available.pop(0)
            order.append(attribute)
            children = np.flatnonzero(self.adjacency[attribute] == 1)
            in_degree[children] -= 1
            available.extend(int(child) for child in children if in_degree[child] == 0)

        if len(order) != self.n_attributes:
            raise MirtValidationError(
                "Adjacency matrix must describe a directed acyclic graph"
            )
        return order

    @property
    def n_attributes(self) -> int:
        return self.adjacency.shape[0]

    def prerequisites(self, attribute: int) -> list[int]:
        """Get direct prerequisites for an attribute."""
        index = self._validate_attribute_index(attribute)
        return list(np.where(self.adjacency[:, index] == 1)[0])

    def all_prerequisites(self, attribute: int) -> set[int]:
        """Get all prerequisites (direct and indirect) for an attribute."""
        visited: set[int] = set()
        stack = self.prerequisites(self._validate_attribute_index(attribute))

        while stack:
            prereq = stack.pop()
            if prereq not in visited:
                visited.add(prereq)
                stack.extend(self.prerequisites(prereq))

        return visited

    def is_valid_pattern(self, pattern: NDArray[np.int_]) -> bool:
        """Check if an attribute pattern is valid under the hierarchy.

        A pattern is valid if all prerequisites of mastered attributes
        are also mastered.
        """
        raw_pattern = np.asarray(pattern)
        if raw_pattern.shape != (self.n_attributes,):
            raise MirtValidationError(f"pattern must have shape ({self.n_attributes},)")
        if raw_pattern.dtype.kind not in "biuf":
            raise MirtValidationError("pattern must contain numeric binary values")
        values = np.asarray(raw_pattern, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
            raise MirtValidationError("pattern must contain only 0 and 1")

        prerequisites, attributes = np.where(self.adjacency == 1)
        return bool(np.all(values[attributes] <= values[prerequisites]))

    def valid_patterns(self) -> NDArray[np.int_]:
        """Generate all valid attribute patterns under the hierarchy."""
        n_attrs = self.n_attributes
        pattern_indices = np.arange(2**n_attrs, dtype=np.uint64)[:, None]
        bit_indices = np.arange(n_attrs, dtype=np.uint64)[None, :]
        patterns = ((pattern_indices >> bit_indices) & 1).astype(np.int_)
        prerequisites, attributes = np.where(self.adjacency == 1)
        valid = np.all(
            patterns[:, attributes] <= patterns[:, prerequisites],
            axis=1,
        )
        return patterns[valid]

    def topological_order(self) -> list[int]:
        """Return attributes in topological order (prerequisites first)."""
        return list(self._topological_order)


class GDINA(BaseCDM):
    """Generalized DINA model with item-specific reduced models.

    The G-DINA model provides a general framework for CDMs where
    item response probability is modeled using main effects and
    interactions of required attributes.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_attributes : int
        Number of attributes.
    q_matrix : NDArray
        Q-matrix (n_items x n_attributes).
    reduced_models : list of str, optional
        Reduced model type for each item. Options: 'DINA', 'DINO',
        'ACDM', 'LLM', 'RRUM', 'saturated'. Default is 'saturated'.
    item_names : list of str, optional
        Names for items.

    Notes
    -----
    An item requiring ``K_j`` attributes has ``2**K_j`` latent groups, one
    for each mastery pattern of its required attributes. Group ``g`` codes
    those attributes little-endian, in Q-matrix column order: for two
    required attributes the groups are 00, 10, 01 and 11. With three or
    more, this order differs from the one R's GDINA package uses.

    The stored ``delta`` coefficients of an item (``delta_parameters``,
    ``set_delta_parameters`` and the ``delta`` parameter) depend on its
    reduced model:

    - ``saturated`` (default): the success probability of each latent
      group, in group order (R GDINA's ``itemprob``).
    - ``DINA``: the success probabilities of examinees lacking at least one
      required attribute and of examinees mastering all of them, ``g`` and
      ``1 - s``.
    - ``DINO``: the success probabilities of examinees mastering none of
      the required attributes and of examinees mastering at least one.
    - ``ACDM``: an intercept and one main effect per required attribute,
      additive on the probability scale.
    - ``LLM``: an intercept and one main effect per required attribute,
      additive on the logit scale.
    - ``RRUM``: ``pi*``, the success probability when every required
      attribute is mastered, followed by one penalty ``r*_k`` per required
      attribute, with ``P = pi* * prod_k r*_k ** (1 - alpha_k)``.

    The saturated, DINA, DINO and RRUM coefficients are therefore not the
    deltas of de la Torre (2011). :meth:`gdina_deltas` converts any item to
    that parameterization, and :meth:`item_probabilities` returns the
    success probability of each latent group.
    """

    model_name = "GDINA"

    def __init__(
        self,
        n_items: int,
        n_attributes: int,
        q_matrix: NDArray[np.int_],
        reduced_models: list[ReducedModelType] | None = None,
        item_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            item_names=item_names,
        )

        if reduced_models is None:
            self._reduced_models: list[ReducedModelType] = ["saturated"] * n_items
        else:
            if len(reduced_models) != n_items:
                raise MirtValidationError(
                    f"reduced_models length ({len(reduced_models)}) "
                    f"must match n_items ({n_items})"
                )
            invalid_models = [
                model for model in reduced_models if model not in _REDUCED_MODEL_TO_CODE
            ]
            if invalid_models:
                valid = ", ".join(_DEFAULT_REDUCED_MODELS)
                raise MirtValidationError(
                    f"Unknown reduced model: {invalid_models[0]!r}. "
                    f"Valid models: {valid}"
                )
            self._reduced_models = list(reduced_models)

        self._delta_params: list[NDArray[np.float64]] = []
        self._latent_groups: list[NDArray[np.int_]] = []
        self._initialize_gdina_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize model parameters and latent grouping structures."""
        if not hasattr(self, "_reduced_models"):
            self._reduced_models = ["saturated"] * self.n_items
        self._initialize_gdina_parameters()

    def _initialize_gdina_parameters(self) -> None:
        """Initialize delta parameters for each item."""
        self._delta_params = []
        self._latent_groups = []

        for j in range(self.n_items):
            q_j = self._q_matrix[j]
            k_j = np.sum(q_j)
            n_groups = 2**k_j

            latent_groups = np.zeros((n_groups, self._n_attributes), dtype=np.int_)
            required_attrs = np.where(q_j == 1)[0]

            for g in range(n_groups):
                for idx, attr in enumerate(required_attrs):
                    latent_groups[g, attr] = (g >> idx) & 1

            self._latent_groups.append(latent_groups)

            model_type = self._reduced_models[j]
            if model_type == "saturated":
                n_params = n_groups
            elif model_type in ("DINA", "DINO"):
                n_params = 2
            elif model_type in ("ACDM", "LLM", "RRUM"):
                n_params = int(k_j) + 1
            else:
                n_params = n_groups

            delta = np.zeros(n_params)
            delta[0] = 0.2
            if model_type == "ACDM":
                # Additive effects summing to 0.6 keep every group probability
                # in [0.2, 0.8].
                delta[1:] = 0.6 / max(1, n_params - 1)
            elif n_params > 1:
                delta[-1] = 0.6
                if n_params > 2:
                    remaining = 0.3 / (n_params - 2)
                    delta[1:-1] = remaining

            self._delta_params.append(delta)

        self._sync_parameter_cache()

    def _sync_parameter_cache(self) -> None:
        """Synchronize generic parameter cache with item-specific deltas."""
        max_params = max((len(delta) for delta in self._delta_params), default=1)
        delta_matrix = np.full((self.n_items, max_params), np.nan, dtype=np.float64)
        n_delta_params = np.zeros(self.n_items, dtype=np.float64)

        for j, delta in enumerate(self._delta_params):
            n_params = len(delta)
            delta_matrix[j, :n_params] = delta
            n_delta_params[j] = float(n_params)

        reduced_model_code = np.array(
            [_REDUCED_MODEL_TO_CODE[model] for model in self._reduced_models],
            dtype=np.float64,
        )

        self._parameters["delta"] = delta_matrix
        self._parameters["delta_n_params"] = n_delta_params
        self._parameters["reduced_model_code"] = reduced_model_code

    @property
    def reduced_models(self) -> list[ReducedModelType]:
        """Reduced model type for each item."""
        return list(self._reduced_models)

    @property
    def delta_parameters(self) -> list[NDArray[np.float64]]:
        """Delta parameters for each item."""
        return [d.copy() for d in self._delta_params]

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Count item coefficients, excluding padding and model-design metadata."""
        masks = super().free_parameter_masks
        delta_mask = np.zeros_like(self._parameters["delta"], dtype=np.bool_)
        for item, model_type in enumerate(self._reduced_models):
            n_required = int(np.count_nonzero(self._q_matrix[item]))
            if model_type == "saturated":
                n_coefficients = 2**n_required
            elif model_type in {"DINA", "DINO"}:
                n_coefficients = 2
            else:
                n_coefficients = n_required + 1
            delta_mask[item, :n_coefficients] = True
            if n_required == 0 and model_type == "DINA":
                delta_mask[item, 0] = False
            elif n_required == 0 and model_type == "DINO":
                delta_mask[item, 1] = False
        masks["delta"] = delta_mask
        masks["delta_n_params"].fill(False)
        masks["reduced_model_code"].fill(False)
        return self._apply_free_parameter_restrictions(masks)

    def _required_group_patterns(self, item_idx: int) -> NDArray[np.int_]:
        """Required-attribute pattern of each latent group of one item."""
        required = np.flatnonzero(self._q_matrix[item_idx] == 1)
        return self._latent_groups[item_idx][:, required]

    def _validate_item_coefficients(
        self,
        item_idx: int,
        model_type: ReducedModelType,
        delta: NDArray[np.float64],
    ) -> None:
        """Reject coefficients that do not define success probabilities."""
        if not np.all(np.isfinite(delta)):
            raise MirtValidationError(
                f"delta for item {item_idx} must contain only finite values",
                parameter="delta",
            )
        if model_type == "LLM":
            return
        if model_type == "RRUM" and np.any(delta < 0.0):
            raise MirtValidationError(
                f"RRUM delta for item {item_idx} must be non-negative",
                parameter="delta",
            )
        if model_type in ("DINA", "DINO", "saturated"):
            # Gate and saturated coefficients are probabilities themselves,
            # including a coefficient that no latent group uses.
            probabilities = delta
        else:
            probabilities = _reduced_group_probabilities(
                model_type,
                self._required_group_patterns(item_idx),
                delta,
            )
        tolerance = _PROBABILITY_RANGE_TOLERANCE
        if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
            raise MirtValidationError(
                f"delta for item {item_idx} implies success probabilities "
                "outside [0, 1]",
                parameter="delta",
            )

    def set_delta_parameters(self, item_idx: int, delta: ArrayLike) -> Self:
        """Set the stored delta coefficients of one item.

        Parameters
        ----------
        item_idx : int
            Item index in ``[0, n_items)``.
        delta : array_like
            Coefficients in the item's stored parameterization (see the class
            Notes): finite, with success probabilities in ``[0, 1]``.

        Returns
        -------
        GDINA
            The model, for chaining.
        """
        index = self._validate_item_index(item_idx)
        try:
            values = np.array(delta, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "delta must contain numeric values", parameter="delta"
            ) from exc
        expected_len = len(self._delta_params[index])
        if values.ndim != 1 or len(values) != expected_len:
            raise MirtValidationError(
                f"delta length ({values.size}) must be {expected_len} for item {index}",
                parameter="delta",
            )
        self._validate_item_coefficients(index, self._reduced_models[index], values)

        self._delta_params[index] = values
        self._sync_parameter_cache()
        return self

    def item_probabilities(self, item_idx: int) -> NDArray[np.float64]:
        """Return the success probability of each latent group of one item.

        Parameters
        ----------
        item_idx : int
            Item index in ``[0, n_items)``.

        Returns
        -------
        ndarray of shape (2**K_j,)
            Success probabilities in latent-group order, which codes the
            item's required attributes little-endian (see the class Notes).
        """
        index = self._validate_item_index(item_idx)
        return self._item_probability_from_alpha(self._latent_groups[index], index)

    def gdina_deltas(self, item_idx: int) -> NDArray[np.float64]:
        """Return one item's coefficients as G-DINA deltas (de la Torre, 2011).

        The deltas are on the link of the item's reduced model: identity for
        the saturated, DINA, DINO and ACDM models, logit for LLM and log for
        RRUM. They are ordered as the intercept, the main effects of the
        required attributes in Q-matrix column order, then the interactions
        by increasing order, each order in lexicographic attribute order. The
        saturated model returns all ``2**K_j`` deltas, DINA the intercept and
        the highest-order interaction, DINO the intercept and the common
        effect of mastering any required attribute, and the other models the
        intercept and main effects.

        Parameters
        ----------
        item_idx : int
            Item index in ``[0, n_items)``.

        Returns
        -------
        ndarray
            The item's G-DINA deltas. An item without required attributes
            returns its single success probability.
        """
        index = self._validate_item_index(item_idx)
        model_type = self._reduced_models[index]
        delta = self._delta_params[index]
        n_required = int(np.count_nonzero(self._q_matrix[index]))
        if n_required == 0:
            return self.item_probabilities(index)
        if model_type in ("ACDM", "LLM"):
            return delta.copy()
        if model_type in ("DINA", "DINO"):
            return np.array([delta[0], delta[1] - delta[0]], dtype=np.float64)
        if model_type == "RRUM":
            with np.errstate(divide="ignore"):
                log_parameters = np.log(delta)
            return np.concatenate(
                ([log_parameters[0] + np.sum(log_parameters[1:])], -log_parameters[1:])
            )
        # Moebius inversion over the subsets of required attributes turns the
        # latent-group probabilities into main effects and interactions.
        effects = delta.copy()
        for bit in range(n_required):
            has_bit = (np.arange(effects.size) >> bit) & 1 == 1
            effects[has_bit] -= effects[np.flatnonzero(has_bit) ^ (1 << bit)]
        order = sorted(
            range(effects.size),
            key=lambda group: (
                bin(group).count("1"),
                [bit for bit in range(n_required) if (group >> bit) & 1],
            ),
        )
        return effects[order]

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set model parameters using the generic BaseItemModel interface.

        ``delta`` rows hold each item's coefficients in its stored
        parameterization, padded to a common width; ``delta_n_params`` gives
        the number of coefficients of each row and must match the item's
        reduced model, and ``reduced_model_code`` the reduced models. Every
        argument is validated before the model changes.
        """
        allowed = {"delta", "delta_n_params", "reduced_model_code"}
        unknown = set(params) - allowed
        if unknown:
            valid = ", ".join(sorted(allowed))
            unknown_s = ", ".join(sorted(unknown))
            raise MirtValidationError(
                f"Unknown parameter(s): {unknown_s}. Valid parameters: {valid}"
            )

        reduced_models = list(self._reduced_models)
        if "reduced_model_code" in params:
            reduced_codes = np.asarray(params["reduced_model_code"], dtype=np.float64)
            if reduced_codes.shape != (self.n_items,):
                raise MirtValidationError(
                    f"Shape mismatch for reduced_model_code: expected ({self.n_items},), "
                    f"got {reduced_codes.shape}",
                    parameter="reduced_model_code",
                )
            reduced_models = []
            for code in reduced_codes:
                rounded = int(np.rint(code)) if np.isfinite(code) else -1
                if rounded not in _REDUCED_CODE_TO_MODEL:
                    raise MirtValidationError(
                        f"Unknown reduced model code: {code}",
                        parameter="reduced_model_code",
                    )
                reduced_models.append(_REDUCED_CODE_TO_MODEL[rounded])

        if "delta" in params:
            delta_matrix = np.asarray(params["delta"], dtype=np.float64)
            if delta_matrix.ndim != 2 or delta_matrix.shape[0] != self.n_items:
                raise MirtValidationError(
                    f"Shape mismatch for delta: expected ({self.n_items}, n_params), "
                    f"got {delta_matrix.shape}",
                    parameter="delta",
                )
        else:
            delta_matrix = self._parameters["delta"]

        if "delta_n_params" in params:
            n_delta_params = np.asarray(params["delta_n_params"], dtype=np.float64)
            if n_delta_params.shape != (self.n_items,):
                raise MirtValidationError(
                    f"Shape mismatch for delta_n_params: expected ({self.n_items},), "
                    f"got {n_delta_params.shape}",
                    parameter="delta_n_params",
                )
        else:
            n_delta_params = self._parameters["delta_n_params"]

        max_params = delta_matrix.shape[1]
        updated_delta_params: list[NDArray[np.float64]] = []
        for j in range(self.n_items):
            count = n_delta_params[j]
            n_params = int(np.rint(count)) if np.isfinite(count) else 0
            expected = _expected_coefficient_count(
                reduced_models[j], int(np.count_nonzero(self._q_matrix[j]))
            )
            if n_params != expected or n_params > max_params:
                raise MirtValidationError(
                    f"Invalid number of delta parameters for item {j}: {n_params}; "
                    f"its {reduced_models[j]} model has {expected}",
                    parameter="delta_n_params",
                )
            values = delta_matrix[j, :n_params].astype(np.float64).copy()
            self._validate_item_coefficients(j, reduced_models[j], values)
            updated_delta_params.append(values)

        self._reduced_models = reduced_models
        self._delta_params = updated_delta_params
        self._sync_parameter_cache()
        return self

    def _latent_group_idx(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.int_]:
        """Map attribute patterns to latent group indices for an item."""
        alpha_values = self._ensure_alpha_2d(alpha)
        index = self._validate_item_index(item_idx)
        return self._latent_group_idx_from_alpha(alpha_values, index)

    def _latent_group_idx_from_alpha(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Map already-validated patterns to one item's latent groups."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]

        idx = np.zeros(alpha.shape[0], dtype=np.int_)
        for i, attr in enumerate(required_attrs):
            idx += alpha[:, attr] * (2**i)

        return idx

    def _compute_prob_saturated(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for saturated model."""
        group_idx = self._latent_group_idx_from_alpha(alpha, item_idx)
        delta = self._delta_params[item_idx]
        return delta[group_idx]

    def _compute_prob_dina(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for DINA reduced model."""
        q_j = self._q_matrix[item_idx]
        eta = np.all(alpha >= q_j, axis=1).astype(np.float64)
        delta = self._delta_params[item_idx]
        return delta[0] * (1 - eta) + delta[1] * eta

    def _compute_prob_dino(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for DINO reduced model."""
        q_j = self._q_matrix[item_idx]
        required = q_j == 1
        eta = np.any(alpha[:, required], axis=1).astype(np.float64)
        delta = self._delta_params[item_idx]
        return delta[0] * (1 - eta) + delta[1] * eta

    def _compute_prob_acdm(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for additive CDM."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        prob = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            prob += delta[i + 1] * alpha[:, attr]

        return np.clip(prob, 0, 1)

    def _compute_prob_llm(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for linear logistic model."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        logit = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            logit += delta[i + 1] * alpha[:, attr]

        return np.asarray(sigmoid(logit), dtype=np.float64)

    def _compute_prob_rrum(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for RRUM."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        prob = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            penalty = delta[i + 1]
            prob *= penalty ** (1 - alpha[:, attr])

        return np.clip(prob, 0, 1)

    def _item_probability_from_alpha(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Evaluate one item from an already-validated mastery matrix."""
        model_type = self._reduced_models[item_idx]
        if model_type == "saturated":
            return self._compute_prob_saturated(alpha, item_idx)
        if model_type == "DINA":
            return self._compute_prob_dina(alpha, item_idx)
        if model_type == "DINO":
            return self._compute_prob_dino(alpha, item_idx)
        if model_type == "ACDM":
            return self._compute_prob_acdm(alpha, item_idx)
        if model_type == "LLM":
            return self._compute_prob_llm(alpha, item_idx)
        if model_type == "RRUM":
            return self._compute_prob_rrum(alpha, item_idx)
        return self._compute_prob_saturated(alpha, item_idx)

    def _coefficient_matrix(
        self,
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Map family-specific main effects onto their required attributes."""
        coefficients = np.zeros(
            (item_indices.size, self.n_attributes),
            dtype=np.float64,
        )
        for row, item_idx in enumerate(item_indices):
            required = np.flatnonzero(self._q_matrix[item_idx])
            delta = self._delta_params[item_idx]
            for effect_idx, attribute in enumerate(required):
                coefficients[row, attribute] = delta[effect_idx + 1]
        return coefficients

    def _group_reduced_probabilities(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.intp],
        model_type: ReducedModelType,
    ) -> NDArray[np.float64]:
        """Evaluate one reduced family across a bounded item chunk."""
        q_matrix = self._q_matrix[item_indices]
        intercepts = np.fromiter(
            (self._delta_params[item_idx][0] for item_idx in item_indices),
            dtype=np.float64,
            count=item_indices.size,
        )

        if model_type in ("DINA", "DINO"):
            mastery_counts = alpha @ q_matrix.T
            if model_type == "DINA":
                eta = mastery_counts == np.sum(q_matrix, axis=1)
            else:
                eta = mastery_counts > 0
            mastered = np.fromiter(
                (self._delta_params[item_idx][1] for item_idx in item_indices),
                dtype=np.float64,
                count=item_indices.size,
            )
            return intercepts + (mastered - intercepts) * eta

        coefficients = self._coefficient_matrix(item_indices)
        if model_type == "ACDM":
            return np.clip(intercepts + alpha @ coefficients.T, 0.0, 1.0)
        if model_type == "LLM":
            return np.asarray(
                sigmoid(intercepts + alpha @ coefficients.T),
                dtype=np.float64,
            )

        penalties = np.ones_like(coefficients)
        required = q_matrix == 1
        penalties[required] = coefficients[required]
        probability = intercepts * np.prod(
            np.where(
                alpha[:, None, :] == 1,
                1.0,
                penalties[None, :, :],
            ),
            axis=2,
        )
        return np.clip(probability, 0.0, 1.0)

    def _saturated_probabilities(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Evaluate saturated items while sharing equal Q-matrix mappings."""
        result = np.empty((alpha.shape[0], item_indices.size), dtype=np.float64)
        q_patterns, group_codes = np.unique(
            self._q_matrix[item_indices],
            axis=0,
            return_inverse=True,
        )
        for group_code, q_pattern in enumerate(q_patterns):
            positions = np.flatnonzero(group_codes == group_code)
            required = np.flatnonzero(q_pattern)
            group_indices = np.zeros(alpha.shape[0], dtype=np.int_)
            for effect_idx, attribute in enumerate(required):
                group_indices += alpha[:, attribute] * (2**effect_idx)

            entries_per_item = max(alpha.shape[0], 2 ** len(required))
            for chunk_positions in _gdina_item_chunks(
                positions,
                entries_per_item,
            ):
                chunk_items = item_indices[chunk_positions]
                deltas = np.stack(
                    [self._delta_params[item_idx] for item_idx in chunk_items]
                )
                result[:, chunk_positions] = deltas[:, group_indices].T
        return result

    def _all_probabilities(
        self,
        alpha: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate every G-DINA family without repeating input validation."""
        result = np.empty((alpha.shape[0], self.n_items), dtype=np.float64)
        model_types = np.asarray(self._reduced_models, dtype=object)
        for model_type in _DEFAULT_REDUCED_MODELS:
            item_indices = np.flatnonzero(model_types == model_type)
            if item_indices.size == 0:
                continue
            if model_type == "saturated":
                result[:, item_indices] = self._saturated_probabilities(
                    alpha,
                    item_indices,
                )
                continue

            if model_type == "RRUM":
                entries_per_item = max(1, alpha.shape[0]) * self.n_attributes
            else:
                entries_per_item = max(alpha.shape[0], self.n_attributes)
            for chunk in _gdina_item_chunks(
                item_indices,
                entries_per_item,
            ):
                result[:, chunk] = self._group_reduced_probabilities(
                    alpha,
                    chunk,
                    model_type,
                )
        return result

    def probability(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        Parameters
        ----------
        alpha : NDArray
            Attribute patterns (n_patterns,) or (n_patterns, n_attributes)
        item_idx : int, optional
            Item index. If None, compute for all items.

        Returns
        -------
        NDArray
            Probabilities (n_patterns,) or (n_patterns, n_items)
        """
        alpha_values = self._ensure_alpha_2d(alpha)

        if item_idx is not None:
            index = self._validate_item_index(item_idx)
            return self._item_probability_from_alpha(alpha_values, index)
        if self.n_items == 1:
            return self._item_probability_from_alpha(alpha_values, 0)[:, None]
        return self._all_probabilities(alpha_values)

    def probability_pairs(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned mastery-pattern and item pairs."""
        alpha_values = self._ensure_alpha_2d(alpha)
        indices = self._prepare_item_indices(item_indices, alpha_values.shape[0])
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)

        q_matrix = self._q_matrix[indices]
        delta = self._parameters["delta"][indices]
        model_codes = self._parameters["reduced_model_code"][indices].astype(
            np.intp,
            copy=False,
        )
        result = np.empty(indices.size, dtype=np.float64)

        for model_type, model_code in _REDUCED_MODEL_TO_CODE.items():
            selected = model_codes == model_code
            if not np.any(selected):
                continue
            selected_alpha = alpha_values[selected]
            selected_q = q_matrix[selected]
            selected_delta = delta[selected]
            intercept = selected_delta[:, 0]

            if model_type == "saturated":
                effect_positions = np.cumsum(selected_q, axis=1) - 1
                bit_weights = np.zeros_like(selected_q)
                required = selected_q == 1
                bit_weights[required] = np.left_shift(
                    1,
                    effect_positions[required],
                )
                group_indices = np.sum(
                    selected_alpha * bit_weights,
                    axis=1,
                ).astype(np.intp, copy=False)
                result[selected] = selected_delta[
                    np.arange(selected_delta.shape[0]),
                    group_indices,
                ]
                continue

            mastery_counts = np.sum(selected_alpha * selected_q, axis=1)
            if model_type in ("DINA", "DINO"):
                if model_type == "DINA":
                    ideal = mastery_counts == np.sum(selected_q, axis=1)
                else:
                    ideal = mastery_counts > 0
                result[selected] = (
                    intercept + (selected_delta[:, 1] - intercept) * ideal
                )
                continue

            effect_positions = np.cumsum(selected_q, axis=1)
            effects = np.take_along_axis(
                selected_delta,
                effect_positions,
                axis=1,
            )
            required_effects = np.where(selected_q == 1, effects, 0.0)
            if model_type == "ACDM":
                probability = intercept + np.sum(
                    selected_alpha * required_effects,
                    axis=1,
                )
            elif model_type == "LLM":
                probability = sigmoid(
                    intercept + np.sum(selected_alpha * required_effects, axis=1)
                )
            else:
                penalties = np.where(selected_q == 1, effects, 1.0)
                probability = intercept * np.prod(
                    np.where(selected_alpha == 1, 1.0, penalties),
                    axis=1,
                )
            result[selected] = np.clip(probability, 0.0, 1.0)

        return result

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta for G-DINA (returns latent group index)."""
        return self._latent_group_idx(alpha, item_idx)

    def information(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return Bernoulli response variance for each pattern and item."""
        alpha = self._ensure_alpha_2d(alpha)
        probs = self.probability(alpha, item_idx)
        return probs * (1 - probs)

    @staticmethod
    def _selection_log_likelihood(
        probabilities: NDArray[np.float64],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> float:
        probabilities = np.clip(
            np.asarray(probabilities, dtype=np.float64),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        return float(
            np.sum(
                successes * np.log(probabilities)
                + (totals - successes) * np.log1p(-probabilities)
            )
        )

    @staticmethod
    def _selection_group_rates(
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        total = float(np.sum(totals))
        global_rate = float(
            np.clip(
                np.sum(successes) / total,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
        )
        rates = np.full(totals.shape, global_rate, dtype=np.float64)
        np.divide(successes, totals, out=rates, where=totals > PROB_EPSILON)
        return (
            np.clip(rates, PROB_EPSILON, 1.0 - PROB_EPSILON),
            global_rate,
        )

    def _fit_gate_candidate(
        self,
        mastered: NDArray[np.bool_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, int]:
        categories = mastered.astype(np.int_)
        category_successes = np.bincount(
            categories,
            weights=successes,
            minlength=2,
        )
        category_totals = np.bincount(
            categories,
            weights=totals,
            minlength=2,
        )
        rates, global_rate = self._selection_group_rates(
            category_successes,
            category_totals,
        )
        if rates[1] < rates[0]:
            rates[:] = global_rate
        probabilities = rates[categories]
        n_parameters = int(np.unique(categories).size)
        return (
            rates,
            self._selection_log_likelihood(
                probabilities,
                successes,
                totals,
            ),
            n_parameters,
        )

    def _fit_acdm_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        global_rate: float,
        start: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (np.ones(group_patterns.shape[0], dtype=np.float64), group_patterns)
        )
        initial = np.zeros(design.shape[1], dtype=np.float64)
        initial[0] = global_rate
        if start is not None and start.shape == initial.shape:
            # Warm-start EM M-steps from the current estimates when they
            # satisfy the bounds and the probability constraints.
            start_probabilities = design @ start
            if (
                np.all(np.isfinite(start))
                and np.all((start >= 0.0) & (start <= 1.0 - PROB_EPSILON))
                and np.all(start_probabilities >= PROB_EPSILON - 1e-12)
                and np.all(start_probabilities <= 1.0 - PROB_EPSILON + 1e-12)
            ):
                initial = start.copy()

        def objective(parameters: NDArray[np.float64]) -> float:
            probabilities = design @ parameters
            return -self._selection_log_likelihood(
                probabilities,
                successes,
                totals,
            )

        def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.clip(
                design @ parameters,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            derivative = (totals * probabilities - successes) / (
                probabilities * (1.0 - probabilities)
            )
            return design.T @ derivative

        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=gradient,
            bounds=[(0.0, 1.0 - PROB_EPSILON)] * design.shape[1],
            constraints=LinearConstraint(
                design,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            ),
            options={"ftol": 1e-10, "maxiter": 200},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        probabilities = design @ parameters
        feasible = bool(
            np.all(np.isfinite(parameters))
            and np.all(probabilities >= PROB_EPSILON - 1e-12)
            and np.all(probabilities <= 1.0 - PROB_EPSILON + 1e-12)
            and float(result.fun) <= initial_objective + 1e-8
        )
        if not feasible:
            parameters = initial
        return parameters, self._selection_log_likelihood(
            design @ parameters, successes, totals
        )

    def _fit_llm_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        global_rate: float,
        start: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (np.ones(group_patterns.shape[0], dtype=np.float64), group_patterns)
        )
        initial = np.zeros(design.shape[1], dtype=np.float64)
        initial[0] = np.log(global_rate / (1.0 - global_rate))
        bounds = [(-23.0, 23.0)] + [(0.0, 23.0)] * group_patterns.shape[1]
        if start is not None and start.shape == initial.shape:
            if np.all(np.isfinite(start)):
                lower, upper = np.array(bounds).T
                initial = np.clip(start, lower, upper)

        def objective(parameters: NDArray[np.float64]) -> float:
            linear = design @ parameters
            return float(
                np.sum(totals * np.logaddexp(0.0, linear) - successes * linear)
            )

        def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.asarray(sigmoid(design @ parameters), dtype=np.float64)
            return design.T @ (totals * probabilities - successes)

        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 200},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        if (
            not np.all(np.isfinite(parameters))
            or float(result.fun) > initial_objective + 1e-8
        ):
            parameters = initial
        probabilities = np.asarray(sigmoid(design @ parameters), dtype=np.float64)
        return parameters, self._selection_log_likelihood(
            probabilities,
            successes,
            totals,
        )

    def _fit_rrum_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        rates: NDArray[np.float64],
        start: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (
                np.ones(group_patterns.shape[0], dtype=np.float64),
                1.0 - group_patterns,
            )
        )
        initial = np.full(design.shape[1], np.log(0.8), dtype=np.float64)
        initial[0] = np.log(np.clip(np.max(rates), PROB_EPSILON, 1.0 - PROB_EPSILON))

        def objective(log_parameters: NDArray[np.float64]) -> float:
            log_probabilities = design @ log_parameters
            probabilities = np.exp(log_probabilities)
            return float(
                -np.sum(
                    successes * log_probabilities
                    + (totals - successes) * np.log1p(-probabilities)
                )
            )

        def gradient(log_parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.exp(design @ log_parameters)
            derivative = (totals * probabilities - successes) / (1.0 - probabilities)
            return design.T @ derivative

        lower = float(np.log(PROB_EPSILON))
        bounds = [(lower, float(np.log1p(-PROB_EPSILON)))] + [
            (lower, 0.0)
        ] * group_patterns.shape[1]
        if start is not None and start.shape == initial.shape:
            if np.all(np.isfinite(start)) and np.all(start > 0.0):
                lower_bounds, upper_bounds = np.array(bounds).T
                initial = np.clip(np.log(start), lower_bounds, upper_bounds)
        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 200},
        )
        log_parameters = np.asarray(result.x, dtype=np.float64)
        if (
            not np.all(np.isfinite(log_parameters))
            or float(result.fun) > initial_objective + 1e-8
        ):
            log_parameters = initial
        probabilities = np.exp(design @ log_parameters)
        return np.exp(log_parameters), self._selection_log_likelihood(
            probabilities,
            successes,
            totals,
        )

    def _fit_reduced_parameters(
        self,
        model_type: ReducedModelType,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        start: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], float, int]:
        """Fit one item's reduced model to expected latent-group counts.

        ``start``, an item's current coefficients, warm-starts the iterative
        ACDM, LLM and RRUM fits; the closed-form fits ignore it.
        """
        rates, global_rate = self._selection_group_rates(successes, totals)
        n_attributes = group_patterns.shape[1]

        if model_type == "saturated":
            return (
                rates,
                self._selection_log_likelihood(
                    rates,
                    successes,
                    totals,
                ),
                group_patterns.shape[0],
            )
        if model_type == "DINA":
            return self._fit_gate_candidate(
                np.all(group_patterns == 1, axis=1),
                successes,
                totals,
            )
        if model_type == "DINO":
            return self._fit_gate_candidate(
                np.any(group_patterns == 1, axis=1),
                successes,
                totals,
            )
        if model_type == "ACDM":
            parameters, log_likelihood = self._fit_acdm_candidate(
                group_patterns,
                successes,
                totals,
                global_rate,
                start,
            )
        elif model_type == "LLM":
            parameters, log_likelihood = self._fit_llm_candidate(
                group_patterns,
                successes,
                totals,
                global_rate,
                start,
            )
        else:
            parameters, log_likelihood = self._fit_rrum_candidate(
                group_patterns,
                successes,
                totals,
                rates,
                start,
            )
        return parameters, log_likelihood, n_attributes + 1

    def _fit_selection_candidate(
        self,
        model_type: ReducedModelType,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[float, int]:
        _, log_likelihood, n_parameters = self._fit_reduced_parameters(
            model_type,
            group_patterns,
            successes,
            totals,
        )
        return log_likelihood, n_parameters

    @staticmethod
    def _validate_selection_candidates(
        candidate_models: list[ReducedModelType] | None,
    ) -> list[ReducedModelType]:
        if candidate_models is None:
            return list(_DEFAULT_REDUCED_MODELS)
        if len(candidate_models) == 0:
            raise MirtValidationError(
                "candidate_models must contain at least one model"
            )

        validated: list[ReducedModelType] = []
        for model_type in candidate_models:
            if (
                not isinstance(model_type, str)
                or model_type not in _REDUCED_MODEL_TO_CODE
            ):
                valid = ", ".join(_DEFAULT_REDUCED_MODELS)
                raise MirtValidationError(
                    f"Unknown candidate model: {model_type!r}. Valid models: {valid}"
                )
            if model_type in validated:
                raise MirtValidationError(f"Duplicate candidate model: {model_type}")
            validated.append(model_type)
        return validated

    def model_selection(
        self,
        responses: NDArray[np.int_],
        candidate_models: list[ReducedModelType] | None = None,
    ) -> list[ReducedModelType]:
        """Select the best fitted reduced model for each item using BIC.

        Candidate parameters are fitted to expected item-by-pattern counts. Class
        posteriors for each item are computed from all other items, preventing the
        response being scored from also determining its own latent-class weights.
        The selection is read-only and does not alter model parameters.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        candidate_models : list of str, optional
            Models to consider. Default is all available models.

        Returns
        -------
        list
            Best model type for each item.
        """
        candidates = self._validate_selection_candidates(candidate_models)
        response_values = self._validate_responses(responses, require_observed=True)
        patterns = self._attribute_patterns
        probabilities = np.asarray(self.probability(patterns), dtype=np.float64)
        if not np.all(np.isfinite(probabilities)):
            raise MirtValidationError(
                "model probabilities must contain only finite values"
            )
        probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)

        log_correct = np.log(probabilities)
        log_incorrect = np.log1p(-probabilities)
        observed = response_values >= 0
        response_float = np.where(observed, response_values, 0).astype(
            np.float64,
            copy=False,
        )
        observed_float = observed.astype(np.float64)
        total_log_likelihood = (
            response_float @ log_correct.T
            + (observed_float - response_float) @ log_incorrect.T
        )

        best_models: list[ReducedModelType] = []

        for j in range(self.n_items):
            item_observed = observed[:, j]
            item_responses = response_float[:, j]
            item_contribution = observed_float[:, j, None] * (
                item_responses[:, None] * log_correct[None, :, j]
                + (1.0 - item_responses[:, None]) * log_incorrect[None, :, j]
            )
            leave_one_out = total_log_likelihood - item_contribution
            leave_one_out -= np.logaddexp.reduce(
                leave_one_out,
                axis=1,
                keepdims=True,
            )
            posterior = np.exp(leave_one_out)

            observed_posterior = posterior[item_observed]
            pattern_totals = np.sum(observed_posterior, axis=0)
            pattern_successes = np.sum(
                observed_posterior * item_responses[item_observed, None],
                axis=0,
            )

            group_indices = self._latent_group_idx(patterns, j)
            required_attributes = np.flatnonzero(self._q_matrix[j] == 1)
            n_groups = 2 ** len(required_attributes)
            group_totals = np.bincount(
                group_indices,
                weights=pattern_totals,
                minlength=n_groups,
            )
            group_successes = np.bincount(
                group_indices,
                weights=pattern_successes,
                minlength=n_groups,
            )
            group_patterns = self._latent_groups[j][:, required_attributes]
            n_observed = int(np.sum(item_observed))

            best_bic = np.inf
            best_model = candidates[0]
            for model_type in candidates:
                item_log_likelihood, n_parameters = self._fit_selection_candidate(
                    model_type,
                    group_patterns,
                    group_successes,
                    group_totals,
                )
                bic = -2.0 * item_log_likelihood + n_parameters * np.log(n_observed)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model_type

            best_models.append(best_model)

        return best_models

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = GDINA(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            reduced_models=list(self._reduced_models),
            item_names=self.item_names.copy() if self.item_names else None,
        )

        new_model._delta_params = [d.copy() for d in self._delta_params]
        new_model._latent_groups = [g.copy() for g in self._latent_groups]
        new_model._is_fitted = self._is_fitted
        new_model._sync_parameter_cache()

        self._copy_parameter_restrictions_to(new_model)
        self._copy_cdm_state_to(new_model)
        return new_model


class HigherOrderCDM(BaseCDM):
    """Higher-order CDM with attributes loading on a general factor.

    This model assumes that attribute mastery is determined by an
    underlying continuous latent trait (general factor) through a latent
    regression, and that item responses follow a G-DINA item model given
    the mastery pattern (de la Torre & Douglas, 2004).

    Parameters
    ----------
    n_items : int
        Number of items.
    n_attributes : int
        Number of attributes.
    q_matrix : NDArray
        Q-matrix (n_items x n_attributes).
    hierarchy : AttributeHierarchy, optional
        Attribute hierarchy for constrained patterns.
    item_names : list of str, optional
        Names for items.
    reduced_models : list of str, optional
        Reduced G-DINA model of each item, as in :class:`GDINA`. Default is
        'saturated'.

    Notes
    -----
    The higher-order model is the logistic latent regression

        P(alpha_k = 1 | theta) = 1 / (1 + exp(-lambda_k * (theta - tau_k)))

    where theta ~ N(0, 1) is the higher-order trait, lambda_k is the loading,
    and tau_k is the threshold for attribute k. This is the 2PL form of de
    la Torre and Douglas (2004), with slope lambda_k and intercept
    -lambda_k * tau_k. Attributes are independent given theta; under a
    hierarchy, the pattern probabilities are renormalized over
    ``valid_patterns``.

    Items are conditionally independent given the mastery pattern alpha,
    not given theta. The likelihood of a response vector given theta sums
    over the patterns:

        P(X | theta) = sum_alpha P(alpha | theta) prod_j P(X_j | alpha).

    The item model is a :class:`GDINA` (``base_model``). Change its
    coefficients with :meth:`set_delta_parameters` or
    ``set_parameters(delta=...)``; :func:`fit_higher_order_cdm` estimates
    the item and higher-order parameters together.
    """

    model_name = "HigherOrderCDM"
    _global_parameters = frozenset({"loadings", "thresholds"})
    # Itemwise estimators cannot move the structural loadings and thresholds;
    # this names the estimator that integrates the higher-order trait.
    dedicated_estimator = "mirt.fit_higher_order_cdm"
    _item_parameters = frozenset({"delta", "delta_n_params", "reduced_model_code"})

    def __init__(
        self,
        n_items: int,
        n_attributes: int,
        q_matrix: NDArray[np.int_],
        hierarchy: AttributeHierarchy | None = None,
        item_names: list[str] | None = None,
        reduced_models: list[ReducedModelType] | None = None,
    ) -> None:
        super().__init__(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            item_names=item_names,
        )
        if hierarchy is not None:
            if not isinstance(hierarchy, AttributeHierarchy):
                raise MirtValidationError(
                    "hierarchy must be an AttributeHierarchy", parameter="hierarchy"
                )
            if hierarchy.n_attributes != self._n_attributes:
                raise MirtValidationError(
                    f"hierarchy has {hierarchy.n_attributes} attributes, "
                    f"expected {self._n_attributes}",
                    parameter="hierarchy",
                )

        self._hierarchy = hierarchy
        self._loadings = np.ones(n_attributes, dtype=np.float64)
        self._thresholds = np.zeros(n_attributes, dtype=np.float64)
        # Standard-normal grid integrating P(alpha | theta) into the implied
        # class probabilities.
        self._n_quad = _DEFAULT_HIGHER_ORDER_QUADRATURE

        self._base_cdm = GDINA(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            reduced_models=reduced_models,
            item_names=item_names,
        )

        if hierarchy is not None:
            self._valid_patterns = hierarchy.valid_patterns()
        else:
            self._valid_patterns = self._attribute_patterns

        self._sync_parameter_cache()

    def _initialize_parameters(self) -> None:
        """Initialize higher-order parameters."""
        self._loadings = np.ones(self._n_attributes, dtype=np.float64)
        self._thresholds = np.zeros(self._n_attributes, dtype=np.float64)
        self._sync_parameter_cache()

    def _sync_parameter_cache(self) -> None:
        """Mirror the higher-order and item-model parameters in the cache."""
        self._parameters["loadings"] = self._loadings.copy()
        self._parameters["thresholds"] = self._thresholds.copy()
        base_cdm = getattr(self, "_base_cdm", None)
        if base_cdm is not None:
            for name in self._item_parameters:
                self._parameters[name] = base_cdm._parameters[name].copy()

    @property
    def hierarchy(self) -> AttributeHierarchy | None:
        return self._hierarchy

    @property
    def loadings(self) -> NDArray[np.float64]:
        """Higher-order loadings for each attribute."""
        return self._loadings.copy()

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Thresholds for each attribute."""
        return self._thresholds.copy()

    @property
    def valid_patterns(self) -> NDArray[np.int_]:
        """Valid attribute patterns under hierarchy."""
        return self._valid_patterns.copy()

    @property
    def base_model(self) -> GDINA:
        """Return an owned copy of the G-DINA item model."""
        return self._base_cdm.copy()

    @property
    def reduced_models(self) -> list[ReducedModelType]:
        """Reduced G-DINA model of each item."""
        return self._base_cdm.reduced_models

    @property
    def delta_parameters(self) -> list[NDArray[np.float64]]:
        """Stored G-DINA coefficients of each item (see :class:`GDINA`)."""
        return self._base_cdm.delta_parameters

    @property
    def class_probabilities(self) -> NDArray[np.float64]:
        """Marginal probabilities of ``valid_patterns`` implied by the model.

        They integrate ``P(alpha | theta)`` over a standard normal theta, and
        are the default prior of the scoring methods.
        """
        return self._default_pattern_prior()

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Free item coefficients plus every loading and threshold."""
        masks = dict(self._base_cdm.free_parameter_masks)
        masks["loadings"] = np.ones(self._n_attributes, dtype=np.bool_)
        masks["thresholds"] = np.ones(self._n_attributes, dtype=np.bool_)
        return self._apply_free_parameter_restrictions(masks)

    def set_higher_order_params(
        self,
        loadings: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ) -> Self:
        """Set higher-order parameters."""
        loadings = np.asarray(loadings, dtype=np.float64)
        thresholds = np.asarray(thresholds, dtype=np.float64)

        if loadings.shape != (self._n_attributes,):
            raise MirtValidationError(
                f"loadings shape {loadings.shape} != ({self._n_attributes},)",
                parameter="loadings",
            )
        if thresholds.shape != (self._n_attributes,):
            raise MirtValidationError(
                f"thresholds shape {thresholds.shape} != ({self._n_attributes},)",
                parameter="thresholds",
            )
        if not np.all(np.isfinite(loadings)):
            raise MirtValidationError(
                "loadings must contain only finite values", parameter="loadings"
            )
        if not np.all(np.isfinite(thresholds)):
            raise MirtValidationError(
                "thresholds must contain only finite values", parameter="thresholds"
            )

        self._loadings = loadings.copy()
        self._thresholds = thresholds.copy()
        self._sync_parameter_cache()
        return self

    def set_delta_parameters(self, item_idx: int, delta: ArrayLike) -> Self:
        """Set one item's stored G-DINA coefficients.

        See :meth:`GDINA.set_delta_parameters`.
        """
        self._base_cdm.set_delta_parameters(item_idx, delta)
        self._sync_parameter_cache()
        return self

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters using the generic BaseItemModel interface.

        ``loadings`` and ``thresholds`` set the higher-order model, and
        ``delta``, ``delta_n_params`` and ``reduced_model_code`` the G-DINA
        item model, as in :meth:`GDINA.set_parameters`. Every argument is
        validated before the model changes.
        """
        allowed = {"loadings", "thresholds"} | self._item_parameters
        unknown = set(params) - allowed
        if unknown:
            valid = ", ".join(sorted(allowed))
            unknown_s = ", ".join(sorted(unknown))
            raise MirtValidationError(
                f"Unknown parameter(s): {unknown_s}. Valid parameters: {valid}"
            )

        loadings = np.asarray(params.get("loadings", self._loadings), dtype=np.float64)
        thresholds = np.asarray(
            params.get("thresholds", self._thresholds), dtype=np.float64
        )
        if loadings.shape != (self._n_attributes,):
            raise MirtValidationError(
                f"Shape mismatch for loadings: expected ({self._n_attributes},), "
                f"got {loadings.shape}",
                parameter="loadings",
            )
        if thresholds.shape != (self._n_attributes,):
            raise MirtValidationError(
                f"Shape mismatch for thresholds: expected ({self._n_attributes},), "
                f"got {thresholds.shape}",
                parameter="thresholds",
            )
        if not np.all(np.isfinite(loadings)):
            raise MirtValidationError(
                "loadings must contain only finite values", parameter="loadings"
            )
        if not np.all(np.isfinite(thresholds)):
            raise MirtValidationError(
                "thresholds must contain only finite values", parameter="thresholds"
            )

        item_updates = {
            name: value
            for name, value in params.items()
            if name in self._item_parameters
        }
        if item_updates:
            base_cdm = self._base_cdm.copy()
            base_cdm.set_parameters(**item_updates)
            self._base_cdm = base_cdm
        return self.set_higher_order_params(loadings, thresholds)

    def _validate_theta(
        self,
        theta: NDArray[np.float64],
        *,
        allow_empty: bool = False,
    ) -> NDArray[np.float64]:
        raw_theta = np.asarray(theta)
        if raw_theta.ndim > 2 or (raw_theta.ndim == 2 and 1 not in raw_theta.shape):
            raise MirtValidationError(
                "theta must be a scalar, vector, or single-column matrix",
                parameter="theta",
            )
        if raw_theta.dtype.kind not in "biuf":
            raise MirtValidationError(
                "theta must contain numeric values", parameter="theta"
            )
        values = np.asarray(raw_theta, dtype=np.float64).reshape(-1)
        if values.size == 0 and not allow_empty:
            raise MirtValidationError(
                "theta must contain at least one value", parameter="theta"
            )
        if not np.all(np.isfinite(values)):
            raise MirtValidationError(
                "theta must contain only finite values", parameter="theta"
            )
        return values

    def _linear_predictor(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(over="ignore", invalid="ignore"):
            differences = theta[:, np.newaxis] - self._thresholds
            predictor = self._loadings * differences
        predictor[:, self._loadings == 0.0] = 0.0
        finite_limit = np.finfo(np.float64).max / (self._n_attributes + 1)
        return np.clip(predictor, -finite_limit, finite_limit)

    def _conditional_response_probability(
        self, item_idx: int | None = None
    ) -> NDArray[np.float64]:
        return self._base_cdm.probability(self._valid_patterns, item_idx)

    def _profile_patterns(self) -> NDArray[np.int_]:
        return self._valid_patterns

    def _pattern_response_probabilities(
        self,
        alpha: NDArray[np.int_] | None = None,
    ) -> NDArray[np.float64]:
        patterns = self._valid_patterns if alpha is None else alpha
        return np.asarray(self._base_cdm.probability(patterns), dtype=np.float64)

    def _default_pattern_prior(self) -> NDArray[np.float64]:
        points, weights = _higher_order_quadrature(self._n_quad)
        prior = weights @ self.pattern_probability(points)
        return prior / np.sum(prior)

    def attribute_probability(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute attribute mastery probabilities given theta.

        P(alpha_k = 1 | theta) = 1 / (1 + exp(-lambda_k * (theta - tau_k)))

        Parameters
        ----------
        theta : NDArray
            Higher-order ability (n_persons,) or (n_persons, 1).

        Returns
        -------
        NDArray
            Probabilities (n_persons, n_attributes), before any hierarchy
            restriction.
        """
        theta_values = self._validate_theta(theta)
        return np.asarray(sigmoid(self._linear_predictor(theta_values)))

    def _log_pattern_probability(
        self,
        theta_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Log-probability of each valid pattern at validated theta values."""
        predictor = self._linear_predictor(theta_values)
        log_mastery = -np.logaddexp(0.0, -predictor)
        log_nonmastery = -np.logaddexp(0.0, predictor)
        patterns = self._valid_patterns
        log_pattern_probability = (
            log_mastery @ patterns.T + log_nonmastery @ (1 - patterns).T
        )
        # Shift before normalizing: near the float limit, adding log(2) to a
        # huge negative maximum is lost to rounding.
        log_pattern_probability -= np.max(
            log_pattern_probability, axis=1, keepdims=True
        )
        return log_pattern_probability - np.log(
            np.sum(np.exp(log_pattern_probability), axis=1, keepdims=True)
        )

    def pattern_probability(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute probability of each valid attribute pattern given theta.

        Parameters
        ----------
        theta : NDArray
            Higher-order ability (n_persons,).

        Returns
        -------
        NDArray
            Pattern probabilities (n_persons, n_valid_patterns).
        """
        theta_values = self._validate_theta(theta)
        return np.exp(self._log_pattern_probability(theta_values))

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal response probabilities.

        Parameters
        ----------
        theta : NDArray
            Higher-order ability.
        item_idx : int, optional
            Item index.

        Returns
        -------
        NDArray
            Response probabilities ``sum_alpha P(alpha | theta) P_j(alpha)``.
            Items are not independent given theta, so these marginal curves
            do not multiply into the likelihood; see :meth:`log_likelihood`.
        """
        index = None if item_idx is None else self._validate_item_index(item_idx)
        pattern_probability = self.pattern_probability(theta)
        conditional_probability = self._conditional_response_probability(index)
        return pattern_probability @ conditional_probability

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned higher-order ability and item pairs."""
        theta_values = self._validate_theta(theta, allow_empty=True)
        indices = self._prepare_item_indices(item_indices, theta_values.size)
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)

        pattern_probability = self.pattern_probability(theta_values)
        conditional_probability = self._conditional_response_probability()
        selected_probability = conditional_probability[:, indices].T
        return np.einsum(
            "ij,ij->i",
            pattern_probability,
            selected_probability,
        )

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta using base CDM."""
        return self._base_cdm.eta(alpha, item_idx)

    def _theta_grid_log_likelihoods(
        self,
        response_values: NDArray[np.int_],
        theta_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Marginal log-likelihood of every response row at every theta."""
        pattern_log_likelihoods = self.pattern_log_likelihoods(response_values)
        shift = np.max(pattern_log_likelihoods, axis=1, keepdims=True)
        likelihoods = np.exp(pattern_log_likelihoods - shift) @ (
            self.pattern_probability(theta_values).T
        )
        with np.errstate(divide="ignore"):
            return np.log(likelihoods) + shift

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute each person's log-likelihood given theta.

        The likelihood marginalizes the shared mastery pattern,
        ``log sum_alpha P(alpha | theta) prod_j P(X_j | alpha)``.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Binary responses; negative values are missing.
        theta : ndarray
            One higher-order ability per person, or one value for all.

        Returns
        -------
        ndarray of shape (n_persons,)
            Log-likelihood of each response row.
        """
        response_values = self._validate_responses(responses)
        theta_values = self._validate_theta(theta)
        if theta_values.size not in (1, response_values.shape[0]):
            raise MirtValidationError(
                "responses and theta must have compatible row counts"
            )

        pattern_log_likelihoods = self.pattern_log_likelihoods(response_values)
        log_prior = self._log_pattern_probability(theta_values)
        return logsumexp(pattern_log_likelihoods + log_prior, axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Fisher information about theta in each item's marginal response.

        Returns ``(dP_j / dtheta)**2 / (P_j (1 - P_j))`` for the marginal
        curve ``P_j(theta)`` of :meth:`probability`. Because items share the
        mastery pattern, these item informations do not add up to the
        information of the whole response vector.
        """
        index = None if item_idx is None else self._validate_item_index(item_idx)
        theta_values = self._validate_theta(theta)
        pattern_probability = self.pattern_probability(theta_values)
        patterns = self._valid_patterns
        # d log P(alpha | theta) / dtheta = sum_k lambda_k (alpha_k - E[alpha_k]),
        # with the expectation under the (hierarchy-normalized) pattern law.
        expected_mastery = pattern_probability @ patterns
        score = (
            patterns @ self._loadings
            - (expected_mastery @ self._loadings)[:, np.newaxis]
        )
        conditional = self._conditional_response_probability(index)
        marginal = pattern_probability @ conditional
        derivative = (pattern_probability * score) @ conditional
        variance = np.clip(marginal * (1.0 - marginal), PROB_EPSILON, None)
        return derivative**2 / variance

    def estimate_theta(
        self,
        responses: NDArray[np.int_],
        method: Literal["EAP", "MLE"] = "EAP",
        n_quad: int = 31,
    ) -> NDArray[np.float64]:
        """Estimate higher-order ability.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        method : str
            Estimation method ('EAP' or 'MLE').
        n_quad : int
            Number of equally spaced grid points on [-4, 4]. EAP uses a
            standard normal prior; MLE returns the grid point with the
            largest likelihood.

        Returns
        -------
        NDArray
            Estimated theta values (n_persons,).
        """
        response_values = self._validate_responses(responses)
        if method not in ("EAP", "MLE"):
            raise MirtValidationError(
                "method must be 'EAP' or 'MLE'", parameter="method"
            )
        n_quad = _validate_quadrature_count(n_quad)

        quad_points = np.linspace(-4, 4, n_quad)
        log_likes = self._theta_grid_log_likelihoods(response_values, quad_points)

        if method == "MLE":
            best_idx = np.argmax(log_likes, axis=1)
            return quad_points[best_idx]

        log_prior = -0.5 * quad_points**2
        log_posterior = log_likes + log_prior

        posterior = np.exp(
            log_posterior - logsumexp(log_posterior, axis=1, keepdims=True)
        )

        theta_eap = np.sum(posterior * quad_points, axis=1)

        return theta_eap

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = HigherOrderCDM(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            hierarchy=self._hierarchy,
            item_names=self.item_names.copy() if self.item_names else None,
            reduced_models=self._base_cdm.reduced_models,
        )

        new_model._loadings = self._loadings.copy()
        new_model._thresholds = self._thresholds.copy()
        new_model._base_cdm = self._base_cdm.copy()
        new_model._valid_patterns = self._valid_patterns.copy()
        new_model._n_quad = self._n_quad
        new_model._is_fitted = self._is_fitted
        new_model._sync_parameter_cache()

        self._copy_parameter_restrictions_to(new_model)
        self._copy_cdm_state_to(new_model)
        return new_model


def _gdina_item_design(
    model: GDINA,
    patterns: NDArray[np.int_],
) -> tuple[list[NDArray[np.int_]], list[NDArray[np.int_]]]:
    """Latent group of each pattern and required-attribute group patterns."""
    group_indices = [
        model._latent_group_idx_from_alpha(patterns, item_idx)
        for item_idx in range(model.n_items)
    ]
    group_patterns = [
        model._required_group_patterns(item_idx) for item_idx in range(model.n_items)
    ]
    return group_indices, group_patterns


def _gdina_item_m_step(
    model: GDINA,
    posterior: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    group_indices: list[NDArray[np.int_]],
    group_patterns: list[NDArray[np.int_]],
) -> None:
    """Refit every item to the expected latent-group counts of one E-step."""
    # Two GEMMs give every item's expected totals and successes per pattern;
    # each item then pools its patterns into its latent groups.
    pattern_totals = observed.T @ posterior
    pattern_successes = correct.T @ posterior
    for item_idx in range(model.n_items):
        n_groups = group_patterns[item_idx].shape[0]
        item_totals = np.bincount(
            group_indices[item_idx],
            weights=pattern_totals[item_idx],
            minlength=n_groups,
        )
        item_successes = np.bincount(
            group_indices[item_idx],
            weights=pattern_successes[item_idx],
            minlength=n_groups,
        )
        parameters, _, _ = model._fit_reduced_parameters(
            model._reduced_models[item_idx],
            group_patterns[item_idx],
            item_successes,
            item_totals,
            model._delta_params[item_idx],
        )
        model._delta_params[item_idx] = parameters
    model._sync_parameter_cache()


def fit_gdina(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    reduced_models: list[ReducedModelType] | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    *,
    hierarchy: AttributeHierarchy | None = None,
) -> CDMResult[GDINA]:
    """Fit G-DINA model using EM algorithm.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items). Negative values are missing;
        every item needs at least one observed response.
    q_matrix : NDArray
        Q-matrix (n_items, n_attributes).
    reduced_models : list of str, optional
        Reduced model for each item.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the change in marginal log-likelihood.
    verbose : bool
        Whether to print progress.
    hierarchy : AttributeHierarchy, optional
        Attribute hierarchy restricting the latent classes. Patterns in which
        an attribute is mastered without all of its prerequisites keep zero
        class probability.

    Returns
    -------
    CDMResult
        Fitted model, estimated class probabilities, log-likelihood,
        information criteria and convergence information. The fitted model
        stores the class probabilities as the default prior of its scoring
        methods. The result unpacks as ``(fitted_model, class_probabilities)``.
    """
    max_iter, tol = _validate_fit_configuration(max_iter, tol, verbose)
    raw_responses, raw_q_matrix = _validate_fit_inputs(responses, q_matrix)
    n_persons, n_items = raw_responses.shape

    model = GDINA(
        n_items=n_items,
        n_attributes=raw_q_matrix.shape[1],
        q_matrix=raw_q_matrix,
        reduced_models=reduced_models,
    )
    response_values = model._validate_responses(raw_responses, require_observed=True)

    patterns = model._attribute_patterns
    n_patterns = len(patterns)
    permissible = _hierarchy_pattern_mask(hierarchy, model.n_attributes, patterns)
    correct, incorrect, observed = _response_indicators(response_values)
    group_indices, group_patterns = _gdina_item_design(model, patterns)

    if permissible is None:
        class_probs = np.full(n_patterns, 1.0 / n_patterns, dtype=np.float64)
    else:
        class_probs = permissible / np.sum(permissible)
    prev_ll = -np.inf
    converged = False
    n_iterations = max_iter

    for iteration in range(max_iter):
        log_like_matrix = _pattern_log_likelihood_matrix(
            correct, incorrect, model.probability(patterns)
        )
        posterior, current_ll = _class_posterior(log_like_matrix, class_probs)

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            converged = True
            n_iterations = iteration
            break

        class_probs = np.maximum(np.mean(posterior, axis=0), PROB_EPSILON)
        if permissible is not None:
            class_probs[~permissible] = 0.0
        class_probs /= np.sum(class_probs)

        _gdina_item_m_step(
            model, posterior, correct, observed, group_indices, group_patterns
        )
        prev_ll = current_ll
    else:
        # The last M-step followed the last E-step; evaluate its estimates.
        _, current_ll = _class_posterior(
            _pattern_log_likelihood_matrix(
                correct, incorrect, model.probability(patterns)
            ),
            class_probs,
        )

    model._sync_parameter_cache()
    model._class_probabilities = class_probs.copy()
    model._permissible_patterns = None if permissible is None else permissible.copy()
    model._is_fitted = True
    n_classes = n_patterns if permissible is None else int(np.sum(permissible))
    return CDMResult(
        model=model,
        class_probabilities=class_probs.copy(),
        log_likelihood=current_ll,
        n_parameters=model.n_parameters + n_classes - 1,
        n_observations=n_persons,
        n_iterations=n_iterations,
        converged=converged,
    )


def _fit_higher_order_structure(
    expected_counts: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    patterns: NDArray[np.int_],
    loadings: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Maximize the expected latent-regression log-likelihood.

    ``expected_counts[q, a]`` is the expected number of respondents at grid
    node ``q`` with pattern ``a``. The optimizer works on slopes and
    intercepts ``-loading * threshold`` and returns loadings and thresholds.
    """
    n_attributes = patterns.shape[1]
    pattern_values = patterns.astype(np.float64)
    mastered_counts = expected_counts @ pattern_values
    node_totals = np.sum(expected_counts, axis=1)

    def objective(
        parameters: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        slopes = parameters[:n_attributes]
        intercepts = parameters[n_attributes:]
        predictor = quad_points[:, np.newaxis] * slopes + intercepts
        log_mastery = -np.logaddexp(0.0, -predictor)
        log_nonmastery = -np.logaddexp(0.0, predictor)
        unnormalized = (
            log_mastery @ pattern_values.T + log_nonmastery @ (1.0 - pattern_values).T
        )
        # The normalizer is one without a hierarchy; with one, it renormalizes
        # P(alpha | theta) over the valid patterns.
        log_normalizer = logsumexp(unnormalized, axis=1)
        value = float(
            np.sum(expected_counts * unnormalized)
            - np.sum(node_totals * log_normalizer)
        )
        conditional = np.exp(unnormalized - log_normalizer[:, np.newaxis])
        predictor_gradient = mastered_counts - node_totals[:, np.newaxis] * (
            conditional @ pattern_values
        )
        gradient = np.concatenate(
            (quad_points @ predictor_gradient, np.sum(predictor_gradient, axis=0))
        )
        return -value, -gradient

    bounds = [_HIGHER_ORDER_LOADING_BOUNDS] * n_attributes + [
        _HIGHER_ORDER_INTERCEPT_BOUNDS
    ] * n_attributes
    lower, upper = np.array(bounds).T
    start = np.clip(np.concatenate((loadings, -loadings * thresholds)), lower, upper)
    start_value, _ = objective(start)
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 200},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(parameters)) or float(result.fun) > start_value + 1e-10:
        parameters = start
    slopes = parameters[:n_attributes].copy()
    return slopes, -parameters[n_attributes:] / slopes


def fit_higher_order_cdm(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    reduced_models: list[ReducedModelType] | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    *,
    hierarchy: AttributeHierarchy | None = None,
    n_quad: int = _DEFAULT_HIGHER_ORDER_QUADRATURE,
) -> CDMResult[HigherOrderCDM]:
    """Fit a higher-order G-DINA model by marginal maximum likelihood EM.

    The E-step computes each respondent's posterior over the mastery
    patterns. The M-step refits every item's reduced G-DINA model to the
    expected latent-group counts, as :func:`fit_gdina` does, and refits the
    logistic latent regression of the attributes on a standard normal
    higher-order trait to the expected pattern counts at each node of an
    equally spaced grid on [-4, 4] (de la Torre & Douglas, 2004).

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items). Negative values are missing;
        every item needs at least one observed response.
    q_matrix : NDArray
        Q-matrix (n_items, n_attributes).
    reduced_models : list of str, optional
        Reduced G-DINA model of each item. Default is 'saturated'.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the change in marginal log-likelihood.
    verbose : bool
        Whether to print progress.
    hierarchy : AttributeHierarchy, optional
        Attribute hierarchy restricting the mastery patterns to
        ``hierarchy.valid_patterns()``.
    n_quad : int, default=31
        Number of grid nodes integrating the higher-order trait.

    Returns
    -------
    CDMResult
        Fitted :class:`HigherOrderCDM`, the class probabilities of its
        ``valid_patterns`` implied by the fitted latent regression,
        log-likelihood, information criteria and convergence information.

    Notes
    -----
    Loadings are kept in [0.1, 5] and intercepts ``-loading * threshold`` in
    [-10, 10]; an estimate on a bound signals weak identification. With two
    loadings and thresholds per attribute, the higher-order model needs at
    least three attributes to be identified; with fewer it has more
    parameters than the ``2**K - 1`` free class probabilities.
    """
    max_iter, tol = _validate_fit_configuration(max_iter, tol, verbose)
    n_quad = _validate_quadrature_count(n_quad)
    raw_responses, raw_q_matrix = _validate_fit_inputs(responses, q_matrix)
    n_persons, n_items = raw_responses.shape

    model = HigherOrderCDM(
        n_items=n_items,
        n_attributes=raw_q_matrix.shape[1],
        q_matrix=raw_q_matrix,
        hierarchy=hierarchy,
        reduced_models=reduced_models,
    )
    model._n_quad = n_quad
    response_values = model._validate_responses(raw_responses, require_observed=True)

    item_model = model._base_cdm
    patterns = model._valid_patterns
    correct, incorrect, observed = _response_indicators(response_values)
    group_indices, group_patterns = _gdina_item_design(item_model, patterns)
    quad_points, quad_weights = _higher_order_quadrature(n_quad)

    prev_ll = -np.inf
    converged = False
    n_iterations = max_iter

    for iteration in range(max_iter):
        # Joint prior mass w_q P(alpha | theta_q) of each node and pattern.
        joint_prior = quad_weights[:, np.newaxis] * model.pattern_probability(
            quad_points
        )
        class_probs = np.sum(joint_prior, axis=0)
        log_like_matrix = _pattern_log_likelihood_matrix(
            correct, incorrect, item_model.probability(patterns)
        )
        posterior, current_ll = _class_posterior(log_like_matrix, class_probs)

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            converged = True
            n_iterations = iteration
            break

        _gdina_item_m_step(
            item_model, posterior, correct, observed, group_indices, group_patterns
        )
        # Theta is independent of the responses given alpha, so each
        # pattern's posterior mass splits across the nodes in proportion to
        # its joint prior mass.
        pattern_mass = np.sum(posterior, axis=0)
        node_share = np.divide(
            pattern_mass,
            class_probs,
            out=np.zeros_like(pattern_mass),
            where=class_probs > 0.0,
        )
        loadings, thresholds = _fit_higher_order_structure(
            joint_prior * node_share[np.newaxis, :],
            quad_points,
            patterns,
            model._loadings,
            model._thresholds,
        )
        model._loadings = loadings
        model._thresholds = thresholds
        prev_ll = current_ll
    else:
        # The last M-step followed the last E-step; evaluate its estimates.
        class_probs = model._default_pattern_prior()
        _, current_ll = _class_posterior(
            _pattern_log_likelihood_matrix(
                correct, incorrect, item_model.probability(patterns)
            ),
            class_probs,
        )

    model._sync_parameter_cache()
    model._is_fitted = True
    return CDMResult(
        model=model,
        class_probabilities=model._default_pattern_prior(),
        log_likelihood=current_ll,
        n_parameters=model.n_parameters,
        n_observations=n_persons,
        n_iterations=n_iterations,
        converged=converged,
    )
