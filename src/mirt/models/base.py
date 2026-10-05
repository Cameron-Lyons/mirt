import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from mirt._categorical import (
    categorical_log_likelihood_batch,
    category_offsets,
    draw_item_responses,
)
from mirt._model_defaults import record_model_base as _record_model_base
from mirt.constants import PROB_EPSILON
from mirt.exceptions import (
    MirtDataError,
    MirtIndexError,
    MirtModelError,
    MirtValidationError,
)

_DICHOTOMOUS_MAX_PROBABILITY_VALUES = 1_000_000
_POLYTOMOUS_MAX_PROBABILITY_VALUES = 1_000_000
_DICHOTOMOUS_MAX_LIKELIHOOD_VALUES = 131_072
_POLYTOMOUS_MAX_INFORMATION_VALUES = 1_000_000


def _dichotomous_batch_fallback(
    responses: NDArray[np.int_], probabilities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Keep exceptional curves and large response values on bounded item sums."""
    n_persons, n_items = responses.shape
    n_points = len(probabilities)
    result = np.zeros((n_persons, n_points), dtype=np.float64)
    dtype = np.result_type(responses.dtype, probabilities.dtype)
    for item in range(n_items):
        for first in range(0, n_points, _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES):
            last = min(first + _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES, n_points)
            correct = np.log(probabilities[first:last, item])
            incorrect = np.log(1.0 - probabilities[first:last, item])
            row_chunk = max(1, _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES // (last - first))
            for start in range(0, n_persons, row_chunk):
                stop = min(start + row_chunk, n_persons)
                values = np.array(
                    responses[start:stop, item, None], dtype=dtype, copy=True
                )
                observed = values >= 0
                np.copyto(values, 0.0, where=~observed)
                terms = values * correct
                np.subtract(observed, values, out=values)
                terms += values * incorrect
                np.copyto(terms, 0.0, where=~observed)
                result[start:stop, first:last] += terms
    return result


def _simulate_responses(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    seed: int | None,
    chunk_size: int | None,
    default_rows: int,
) -> NDArray[np.int32]:
    """Draw responses in person chunks from one seeded stream."""
    theta_values = model._ensure_theta_2d(theta)
    n_persons = theta_values.shape[0]
    if chunk_size is None:
        chunk_size = max(1, min(n_persons, default_rows))
    elif isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size, (int, np.integer)
    ):
        raise ValueError("chunk_size must be a positive integer")
    elif chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return draw_item_responses(
        model,
        theta_values,
        np.random.default_rng(seed),
        chunk_size=int(chunk_size),
        dtype=np.int32,
    )


@_record_model_base
class BaseItemModel(ABC):
    model_name: str = "BaseModel"
    n_params_per_item: int = 0
    supports_multidimensional: bool = False
    # Stored parameters common to every item rather than indexed by item.
    _shared_parameters: frozenset[str] = frozenset()

    def __init__(
        self,
        n_items: int,
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if n_items <= 0:
            raise MirtValidationError(
                "n_items must be positive",
                parameter="n_items",
                value=n_items,
                expected="> 0",
            )
        if (
            not isinstance(n_factors, (int, float, np.integer, np.floating))
            or n_factors <= 0
        ):
            raise MirtValidationError(
                "n_factors must be positive",
                parameter="n_factors",
                value=n_factors,
                expected="> 0",
            )
        if not self.supports_multidimensional:
            if n_factors != 1:
                raise MirtModelError(
                    f"{self.model_name} only supports unidimensional models",
                    model_type=self.model_name,
                    n_factors=n_factors,
                )
            # Store a plain int even when given 1.0, True or np.int64(1).
            n_factors = 1

        self.n_items = n_items
        self.n_factors = n_factors
        self.item_names = item_names or [f"Item_{i}" for i in range(n_items)]

        if len(self.item_names) != n_items:
            raise MirtValidationError(
                f"Length of item_names ({len(self.item_names)}) must match n_items ({n_items})",
                parameter="item_names",
                value=len(self.item_names),
                expected=str(n_items),
            )

        self._parameters: dict[str, NDArray[np.float64]] = {}
        self._free_parameter_restrictions: dict[str, NDArray[np.bool_]] = {}
        self._is_fitted: bool = False
        self._initialize_parameters()

    @property
    def is_polytomous(self) -> bool:
        return hasattr(self, "_n_categories")

    @abstractmethod
    def _initialize_parameters(self) -> None: ...

    @abstractmethod
    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]: ...

    def _prepare_probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Validate aligned respondent abilities and item indices."""
        theta_2d = self._ensure_theta_2d(theta)
        indices = self._prepare_item_indices(item_indices, theta_2d.shape[0])
        return theta_2d, indices

    def _prepare_item_indices(
        self,
        item_indices: NDArray[np.int_],
        n_rows: int,
    ) -> NDArray[np.intp]:
        """Validate item indices aligned with a known number of input rows."""
        indices = np.asarray(item_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise MirtValidationError(
                "item_indices must be a one-dimensional integer array",
                parameter="item_indices",
                value=indices,
                expected="one-dimensional integer array",
            )
        if indices.shape[0] != n_rows:
            raise MirtValidationError(
                "item_indices must contain one entry per theta row",
                parameter="item_indices",
                value=indices.shape,
                expected=f"({n_rows},)",
            )
        if np.any((indices < 0) | (indices >= self.n_items)):
            raise MirtValidationError(
                "item_indices entries must identify valid model items",
                parameter="item_indices",
                value=indices,
                expected=f"values in [0, {self.n_items})",
            )
        return indices.astype(np.intp, copy=False)

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs without a Cartesian product.

        Each row of ``theta`` is evaluated only for the item at the matching
        position in ``item_indices``. Dichotomous models return one probability
        per pair. Polytomous models return a category matrix padded with zeros
        to the model's maximum category count.
        """
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        if self.is_polytomous:
            category_counts = self._n_categories
            result = np.zeros((indices.size, max(category_counts)), dtype=np.float64)
            for item_idx in np.unique(indices):
                selected = indices == item_idx
                n_categories = category_counts[int(item_idx)]
                result[selected, :n_categories] = self.probability(
                    theta_2d[selected],
                    int(item_idx),
                )
            return result

        result = np.empty(indices.size, dtype=np.float64)
        for item_idx in np.unique(indices):
            selected = indices == item_idx
            result[selected] = self.probability(theta_2d[selected], int(item_idx))
        return result

    @abstractmethod
    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]: ...

    @property
    def parameters(self) -> dict[str, NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._parameters.items()}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_parameters(self) -> int:
        return sum(
            int(np.count_nonzero(mask)) for mask in self.free_parameter_masks.values()
        )

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Boolean masks identifying statistically free stored parameters."""
        return self._apply_free_parameter_restrictions(
            {
                name: np.ones(values.shape, dtype=np.bool_)
                for name, values in self._parameters.items()
            }
        )

    def _apply_free_parameter_restrictions(
        self, masks: dict[str, NDArray[np.bool_]]
    ) -> dict[str, NDArray[np.bool_]]:
        """Apply explicit restrictions after model-family identification masks."""
        for name, restricted in self._free_parameter_restrictions.items():
            masks[name] &= restricted
        return masks

    def _copy_parameter_restrictions_to(self, model: "BaseItemModel") -> None:
        """Preserve explicit masks without sharing mutable arrays with a copy."""
        if not self._free_parameter_restrictions:
            return
        model._free_parameter_restrictions = {
            name: mask.copy()
            for name, mask in self._free_parameter_restrictions.items()
        }

    def set_free_parameter_masks(
        self, masks: Mapping[str, NDArray[np.bool_]] | None
    ) -> Self:
        """Restrict statistically free coordinates without changing parameter values.

        Boolean arrays must match the stored parameter shapes and cannot free
        model-family constraints such as category padding, reference-category
        coefficients, or 1PL discriminations. Unspecified parameters keep their
        family masks. Passing ``None`` clears these additional restrictions.
        Estimators and diagnostics that consume ``free_parameter_masks`` use
        the resulting masks; this method itself does not constrain setters.
        """
        if masks is None:
            self._free_parameter_restrictions = {}
            return self
        if not isinstance(masks, Mapping):
            raise MirtValidationError("masks must be a mapping", parameter="masks")
        previous = self._free_parameter_restrictions
        self._free_parameter_restrictions = {}
        try:
            intrinsic = self.free_parameter_masks
        finally:
            self._free_parameter_restrictions = previous
        validated = {}
        for name, values in masks.items():
            if name not in intrinsic:
                raise MirtValidationError(
                    f"Unknown parameter: {name}", parameter="masks"
                )
            mask = np.asarray(values)
            if mask.dtype != np.bool_ or mask.shape != intrinsic[name].shape:
                raise MirtValidationError(
                    f"Mask for {name} must be Boolean with shape {intrinsic[name].shape}",
                    parameter="masks",
                )
            if np.any(mask & ~intrinsic[name]):
                raise MirtValidationError(
                    f"Mask for {name} cannot free model-family fixed parameters",
                    parameter="masks",
                )
            if not np.array_equal(mask, intrinsic[name]):
                validated[name] = mask.copy()
        self._free_parameter_restrictions = validated
        return self

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return an identified full-storage representation for estimation."""
        return np.asarray(values, dtype=np.float64).copy()

    def _expand_parameter_standard_errors(
        self,
        name: str,
        errors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Expand independent-coordinate errors to dependent stored parameters.

        Input and output have the full parameter-storage shape. The default
        preserves values; families with derived coordinates can propagate
        their uncertainty without treating those coordinates as free.
        """
        return np.asarray(errors, dtype=np.float64).copy()

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        self._parameters.update(self._coerce_parameter_updates(params))
        return self

    def _coerce_parameter_updates(
        self, params: Mapping[str, Any]
    ) -> dict[str, NDArray[np.float64]]:
        """Return owned float arrays for known parameters with stored shapes."""
        validated: dict[str, NDArray[np.float64]] = {}
        for name, value in params.items():
            if name not in self._parameters:
                valid_params = ", ".join(self._parameters.keys())
                raise MirtValidationError(
                    f"Unknown parameter: {name}. Valid parameters: {valid_params}",
                    parameter=name,
                    expected=valid_params,
                )
            try:
                value_arr = np.array(value, dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must contain numeric values",
                    parameter=name,
                    value=value,
                ) from exc
            if value_arr.shape != self._parameters[name].shape:
                raise MirtValidationError(
                    f"Shape mismatch for {name}: expected {self._parameters[name].shape}, "
                    f"got {value_arr.shape}",
                    parameter=name,
                    value=value_arr.shape,
                    expected=str(self._parameters[name].shape),
                )
            validated[name] = value_arr
        return validated

    def _validate_item_index(self, item_idx: int) -> int:
        """Return ``item_idx`` as an ``int`` in ``[0, n_items)``.

        Booleans, non-integers and negative indices raise
        :class:`~mirt.exceptions.MirtIndexError`, which is both an
        ``IndexError`` and a ``ValueError``.
        """
        try:
            if isinstance(item_idx, (bool, np.bool_)):
                raise TypeError
            index = operator.index(item_idx)
        except TypeError:
            raise MirtIndexError(
                "item_idx must be an integer", parameter="item_idx", value=item_idx
            ) from None
        if index < 0 or index >= self.n_items:
            raise MirtIndexError(
                f"item_idx {index} out of range [0, {self.n_items})",
                parameter="item_idx",
            )
        return index

    def _item_indexed(self, name: str) -> bool:
        """Whether stored parameter ``name`` has one leading row per item.

        Parameters listed in ``_shared_parameters`` apply to every item even
        when their length happens to equal ``n_items``.
        """
        values = self._parameters[name]
        return (
            name not in self._shared_parameters
            and values.ndim >= 1
            and values.shape[0] == self.n_items
        )

    def get_item_parameters(
        self, item_idx: int
    ) -> dict[str, float | NDArray[np.float64]]:
        """Return copies of item parameters and shared parameter arrays."""
        item_idx = self._validate_item_index(item_idx)

        result: dict[str, float | NDArray[np.float64]] = {}
        for name, values in self._parameters.items():
            if not self._item_indexed(name):
                result[name] = values.copy()
            elif values.ndim == 1:
                result[name] = float(values[item_idx])
            else:
                result[name] = values[item_idx].copy()
        return result

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set a parameter value for a specific item.

        Args:
            item_idx: Index of the item (0-based).
            param_name: Name of the parameter to set.
            value: New scalar or array value for the item.

        Raises:
            IndexError: If item_idx is out of range.
            MirtValidationError: If param_name is not a valid parameter.
        """
        item_idx = self._validate_item_index(item_idx)
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters.keys())
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        if param_name in self._shared_parameters:
            raise MirtValidationError(
                f"{param_name} is shared by all items; use set_parameters",
                parameter=param_name,
            )
        if not self._item_indexed(param_name):
            raise MirtValidationError(
                f"Parameter {param_name} does not have per-item values",
                parameter=param_name,
            )
        current = self._parameters[param_name]

        updated = current.copy()
        try:
            if updated.ndim == 1:
                updated[item_idx] = float(value)
            else:
                updated[item_idx] = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"Invalid per-item value for {param_name}",
                parameter=param_name,
                value=value,
            ) from exc
        if np.array_equal(updated, current):
            return
        self.set_parameters(**{param_name: updated})

    def _ensure_theta_2d(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2:
            raise MirtValidationError(
                f"theta must be 1D or 2D, got {theta.ndim}D",
                parameter="theta",
                value=theta.ndim,
                expected="1 or 2",
            )
        if theta.shape[1] != self.n_factors:
            raise MirtValidationError(
                f"theta has {theta.shape[1]} factors, expected {self.n_factors}",
                parameter="theta",
                value=theta.shape[1],
                expected=str(self.n_factors),
            )
        return theta

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        self._copy_parameter_restrictions_to(new_model)
        new_model._is_fitted = self._is_fitted
        return new_model

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"n_items={self.n_items}, "
            f"n_factors={self.n_factors}, "
            f"{status})"
        )


@_record_model_base
class _AtomicParameterState(BaseItemModel):
    """Validate a complete candidate state before committing parameter updates.

    Families with cross-parameter domains mix this in ahead of their item
    model base and override :meth:`_validate_parameter_state`.
    """

    def _validate_parameter_state(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        """Raise when a complete candidate parameter state is invalid."""

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters atomically after validating the complete model state."""
        candidate = {**self._parameters, **self._coerce_parameter_updates(params)}
        self._validate_parameter_state(candidate)
        self._parameters = candidate
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item's value, which must match one row of the parameter."""
        item_idx = self._validate_item_index(item_idx)
        if param_name not in self._parameters:
            valid_params = ", ".join(self._parameters)
            raise MirtValidationError(
                f"Unknown parameter: {param_name}. Valid parameters: {valid_params}",
                parameter=param_name,
                expected=valid_params,
            )

        current = self._parameters[param_name]
        try:
            value_array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"Invalid per-item value for {param_name}",
                parameter=param_name,
                value=value,
            ) from exc
        expected_shape = current.shape[1:]
        if value_array.shape != expected_shape:
            message = (
                f"{param_name} must be a scalar for one item"
                if not expected_shape
                else f"{param_name} for one item must have shape {expected_shape}"
            )
            raise MirtValidationError(
                message,
                parameter=param_name,
                value=value_array.shape,
                expected=str(expected_shape) if expected_shape else "scalar",
            )

        updated = current.copy()
        updated[item_idx] = value_array
        self.set_parameters(**{param_name: updated})


@_record_model_base
class DichotomousItemModel(BaseItemModel):
    def _validate_dichotomous_responses(self, responses: NDArray[np.int_]) -> NDArray:
        """Require a numeric response matrix without changing its value semantics."""
        responses = np.asarray(responses)
        if responses.ndim != 2:
            raise MirtDataError(f"responses must be 2D, got {responses.ndim}D")
        if responses.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}",
                n_items=responses.shape[1],
            )
        if responses.dtype.kind not in "biuf":
            raise MirtDataError("responses must contain numeric values")
        return responses

    def icc(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Item characteristic curve (alias for probability)."""
        return self.probability(theta, item_idx)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = self._validate_dichotomous_responses(responses)
        curve_theta = theta
        theta = self._ensure_theta_2d(theta)

        if (
            responses.shape[0] != theta.shape[0]
            and responses.shape[0] != 1
            and theta.shape[0] != 1
        ):
            raise MirtDataError(
                "responses and theta must have matching row counts or a single row"
            )

        p = np.broadcast_to(self.probability(curve_theta), (len(theta), self.n_items))
        p = np.clip(p, PROB_EPSILON, 1.0 - PROB_EPSILON)

        valid = responses >= 0
        values = responses
        if responses.dtype.kind == "u":
            values = responses.astype(np.result_type(responses.dtype, p.dtype))
        ll = values * np.log(p)
        if values.dtype.kind == "b":
            ll = ll.astype(np.result_type(np.int64, p.dtype), copy=False)
        ll += (1 - values) * np.log(1.0 - p)
        np.copyto(ll, 0.0, where=~valid)

        return ll.sum(axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at all theta points.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.
        theta : ndarray of shape (n_theta, n_factors)
            Ability values at which to compute likelihood.

        Returns
        -------
        ndarray of shape (n_persons, n_theta)
            Log-likelihood for each person at each theta point.
        """
        responses = self._validate_dichotomous_responses(responses)
        curve_theta = theta
        theta = self._ensure_theta_2d(theta)

        p = np.broadcast_to(self.probability(curve_theta), (len(theta), self.n_items))
        p = np.clip(p, PROB_EPSILON, 1.0 - PROB_EPSILON)
        log_p = np.log(p)
        log_1_minus_p = np.log1p(-p)

        valid = responses >= 0
        if (
            not np.all(np.isfinite(log_p))
            or not np.all(np.isfinite(log_1_minus_p))
            or (
                responses.dtype.kind == "f"
                and np.max(responses, initial=0.0, where=valid)
                > np.finfo(np.float64).max / (-np.log(PROB_EPSILON) * self.n_items)
            )
        ):
            return _dichotomous_batch_fallback(responses, p)
        response_values = np.array(responses, dtype=np.float64, copy=True)
        np.copyto(response_values, 0.0, where=~valid)
        result = response_values @ log_p.T
        np.subtract(valid, response_values, out=response_values)
        # Bound the second product while accumulating into the owned result.
        for first in range(0, len(theta), _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES):
            last = min(first + _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES, len(theta))
            row_chunk = max(1, _DICHOTOMOUS_MAX_LIKELIHOOD_VALUES // (last - first))
            for start in range(0, len(responses), row_chunk):
                stop = min(start + row_chunk, len(responses))
                result[start:stop, first:last] += (
                    response_values[start:stop] @ log_1_minus_p[first:last].T
                )
        return result

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        probs = self.probability(theta, item_idx)
        if item_idx is None:
            return np.sum(probs, axis=1)
        return probs

    def simulate(
        self,
        theta: NDArray[np.float64],
        seed: int | None = None,
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.int_]:
        """Simulate binary responses conditional on latent trait values.

        Parameters
        ----------
        theta : ndarray
            Latent trait values with shape ``(n_persons, n_factors)``. A
            one-dimensional array is also accepted for unidimensional models.
        seed : int, optional
            Random seed for reproducible response draws.
        chunk_size : int, optional
            Maximum number of persons evaluated at once. By default, a
            memory-bounded chunk size is selected from the model dimensions.

        Returns
        -------
        ndarray
            Binary response matrix with shape ``(n_persons, n_items)``.

        Notes
        -----
        A fixed seed produces identical responses for every valid chunk size.
        """
        return _simulate_responses(
            self,
            theta,
            seed,
            chunk_size,
            _DICHOTOMOUS_MAX_PROBABILITY_VALUES // self.n_items,
        )


@_record_model_base
class PolytomousItemModel(BaseItemModel):
    def __init__(
        self,
        n_items: int,
        n_categories: int | list[int],
        n_factors: int = 1,
        item_names: list[str] | None = None,
    ) -> None:
        if isinstance(n_categories, int):
            self._n_categories = [n_categories] * n_items
        else:
            if len(n_categories) != n_items:
                raise MirtValidationError(
                    f"Length of n_categories ({len(n_categories)}) must match n_items ({n_items})",
                    parameter="n_categories",
                    value=len(n_categories),
                    expected=str(n_items),
                )
            self._n_categories = list(n_categories)

        for i, n_cat in enumerate(self._n_categories):
            if n_cat < 2:
                raise MirtValidationError(
                    f"Item {i} has {n_cat} categories; minimum is 2",
                    parameter="n_categories",
                    value=n_cat,
                    expected=">= 2",
                )

        super().__init__(n_items, n_factors, item_names)

    @property
    def n_categories(self) -> list[int]:
        return self._n_categories.copy()

    @property
    def max_categories(self) -> int:
        return max(self._n_categories)

    def _category_columns(self, width: int, offset: int = 0) -> NDArray[np.bool_]:
        """Return a fresh ``(n_items, width)`` mask of columns below ``count - offset``."""
        counts = np.asarray(self._n_categories, dtype=np.intp)
        return np.arange(width) < (counts - offset)[:, None]

    def copy(self) -> Self:
        new_model = self.__class__(
            n_items=self.n_items,
            n_categories=self._n_categories.copy(),
            n_factors=self.n_factors,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {k: v.copy() for k, v in self._parameters.items()}
        self._copy_parameter_restrictions_to(new_model)
        new_model._is_fitted = self._is_fitted
        return new_model

    def simulate(
        self,
        theta: NDArray[np.float64],
        seed: int | None = None,
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.int_]:
        """Simulate category responses conditional on latent trait values.

        Parameters
        ----------
        theta : ndarray
            Latent trait values with shape ``(n_persons, n_factors)``. A
            one-dimensional array is also accepted for unidimensional models.
        seed : int, optional
            Random seed for reproducible response draws.
        chunk_size : int, optional
            Maximum number of persons evaluated at once. By default, a
            memory-bounded chunk size is selected from the model dimensions.

        Returns
        -------
        ndarray
            Category codes ``0, ..., n_categories[j] - 1`` with shape
            ``(n_persons, n_items)``.

        Notes
        -----
        Each response is drawn by inverse CDF from ``probability(theta)``
        with one uniform per person and item. A fixed seed produces identical
        responses for every valid chunk size.
        """
        return _simulate_responses(
            self,
            theta,
            seed,
            chunk_size,
            _POLYTOMOUS_MAX_PROBABILITY_VALUES // (self.n_items * self.max_categories),
        )

    @abstractmethod
    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]: ...

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Compute all category probabilities for one item."""
        n_cat = self._n_categories[item_idx]
        probabilities = np.empty((theta.shape[0], n_cat), dtype=np.float64)
        for category in range(n_cat):
            probabilities[:, category] = self.category_probability(
                theta, item_idx, category
            )
        return probabilities

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            return self._category_probabilities(
                theta, self._validate_item_index(item_idx)
            )

        max_cat = max(self._n_categories)
        probs = np.zeros((n_persons, self.n_items, max_cat))

        for i in range(self.n_items):
            n_cat = self._n_categories[i]
            probs[:, i, :n_cat] = self._category_probabilities(theta, i)

        return probs

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._item_information(theta, self._validate_item_index(item_idx))
        rows_per_block = max(1, _POLYTOMOUS_MAX_INFORMATION_VALUES // self.n_items)
        if theta.shape[0] <= rows_per_block:
            return self._information_by_item(theta).sum(axis=1)
        # Long batches keep the per-item columns within a bounded block.
        information = np.empty(theta.shape[0], dtype=np.float64)
        for start in range(0, theta.shape[0], rows_per_block):
            columns = self._information_by_item(theta[start : start + rows_per_block])
            information[start : start + rows_per_block] = columns.sum(axis=1)
        return information

    @abstractmethod
    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]: ...

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return every item's information with shape ``(n_theta, n_items)``.

        Column ``j`` equals ``_item_information(theta, j)``. Families with
        closed-form curves override this with an all-item kernel.
        """
        theta = self._ensure_theta_2d(theta)
        information = np.empty((theta.shape[0], self.n_items), dtype=np.float64)
        for item_idx in range(self.n_items):
            information[:, item_idx] = self._item_information(theta, item_idx)
        return information

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        n_persons = theta.shape[0]

        if item_idx is not None:
            item_idx = self._validate_item_index(item_idx)
            n_cat = self._n_categories[item_idx]
            probabilities = self._category_probabilities(theta, item_idx)
            return probabilities @ np.arange(n_cat)

        total_expected = np.zeros(n_persons)
        for i in range(self.n_items):
            total_expected += self.expected_score(theta, i)
        return total_expected

    def category_response_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        return self._category_probabilities(theta, self._validate_item_index(item_idx))

    def _validate_polytomous_responses(
        self,
        responses: NDArray[np.int_],
    ) -> NDArray:
        """Validate a polytomous response matrix for likelihood evaluation."""
        responses = np.asarray(responses)
        if responses.ndim != 2:
            raise MirtDataError(f"responses must be 2D, got {responses.ndim}D")
        if responses.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {responses.shape[1]} items, expected {self.n_items}",
                n_items=responses.shape[1],
            )
        if responses.dtype.kind not in "biuf":
            raise MirtDataError("responses must contain numeric category codes")

        observed = responses >= 0
        if responses.dtype.kind == "f" and np.any(
            observed & (~np.isfinite(responses) | (responses != np.trunc(responses)))
        ):
            raise MirtDataError("responses must contain integer category codes")

        n_categories = np.asarray(self._n_categories)
        invalid = observed & (responses >= n_categories[None, :])
        if np.any(invalid):
            item_idx = int(np.flatnonzero(np.any(invalid, axis=0))[0])
            raise MirtDataError(
                f"responses for item {item_idx} must be below "
                f"{self._n_categories[item_idx]}"
            )
        return responses

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        responses = self._validate_polytomous_responses(responses)
        curve_theta = theta
        theta = self._ensure_theta_2d(theta)
        n_response_rows = responses.shape[0]
        n_theta_rows = theta.shape[0]

        if (
            n_response_rows != n_theta_rows
            and n_response_rows != 1
            and n_theta_rows != 1
        ):
            raise MirtDataError(
                "responses and theta must have matching row counts or a single row"
            )

        n_rows = max(n_response_rows, n_theta_rows)
        ll = np.zeros(n_rows, dtype=np.float64)
        row_indices = np.arange(n_rows)

        for item_idx in range(self.n_items):
            item_responses = np.broadcast_to(responses[:, item_idx], (n_rows,))
            valid = item_responses >= 0
            if not np.any(valid):
                continue

            probabilities = self.probability(curve_theta, item_idx)
            probabilities = np.broadcast_to(
                probabilities, (n_rows, probabilities.shape[1])
            )
            response_indices = np.where(valid, item_responses, 0).astype(
                np.intp, copy=False
            )
            selected = probabilities[row_indices[valid], response_indices[valid]]
            ll[valid] += np.log(np.clip(selected, PROB_EPSILON, 1.0))

        return ll

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for all persons at all theta points.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.
        theta : ndarray of shape (n_theta, n_factors)
            Ability values at which to compute likelihood.

        Returns
        -------
        ndarray of shape (n_persons, n_theta)
            Log-likelihood for each person at each theta point.
        """
        responses = self._validate_polytomous_responses(responses)
        curve_theta = theta
        theta = self._ensure_theta_2d(theta)
        offsets = category_offsets(self._n_categories)
        log_table = np.empty((sum(self._n_categories), theta.shape[0]))
        # Per-item curves keep overridden probability hooks authoritative.
        for item_idx, (offset, n_categories) in enumerate(
            zip(offsets, self._n_categories, strict=True)
        ):
            probabilities = self.probability(curve_theta, item_idx)[:, :n_categories]
            log_table[offset : offset + n_categories] = np.log(
                np.clip(probabilities, PROB_EPSILON, 1 - PROB_EPSILON)
            ).T
        return categorical_log_likelihood_batch(log_table, offsets, responses)
