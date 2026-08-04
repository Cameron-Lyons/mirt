"""Response pattern collapsing for efficient IRT estimation.

This module provides utilities for collapsing identical response patterns
to reduce computational burden during EM estimation, especially for large datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.data import validate_responses


@dataclass
class CollapsedData:
    """Container for collapsed response data.

    Attributes
    ----------
    patterns : ndarray of shape (n_patterns, n_items)
        Unique response patterns.
    frequencies : ndarray of shape (n_patterns,)
        Frequency count for each pattern.
    indices : ndarray of shape (n_persons,)
        Index mapping each original person to their pattern.
    n_persons : int
        Original number of persons.
    n_patterns : int
        Number of unique patterns.
    """

    patterns: NDArray[np.int_]
    frequencies: NDArray[np.int_]
    indices: NDArray[np.int_]
    n_persons: int
    n_patterns: int

    @property
    def compression_ratio(self) -> float:
        """Ratio of patterns to persons (lower = more compression)."""
        return self.n_patterns / self.n_persons

    @property
    def observations_saved(self) -> int:
        """Number of redundant person rows removed by collapsing."""
        return self.n_persons - self.n_patterns

    def _expand(self, pattern_values: ArrayLike) -> NDArray:
        values = np.asarray(pattern_values)
        if values.ndim == 0 or values.shape[0] != self.n_patterns:
            raise MirtValidationError(
                "pattern_values must have one leading entry per pattern",
                parameter="pattern_values",
                value=values.shape,
                expected=f"first dimension of {self.n_patterns}",
            )
        return values[self.indices]

    def expand_weights(
        self, pattern_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Expand pattern-level weights back to person level.

        Parameters
        ----------
        pattern_weights : ndarray of shape (n_patterns, ...)
            Weights computed at the pattern level.

        Returns
        -------
        ndarray of shape (n_persons, ...)
            Weights expanded to person level.
        """
        return self._expand(pattern_weights)

    def expand_scores(self, pattern_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Expand pattern-level scores back to person level.

        Parameters
        ----------
        pattern_scores : ndarray of shape (n_patterns,) or (n_patterns, n_factors)
            Scores computed at the pattern level.

        Returns
        -------
        ndarray
            Scores expanded to person level.
        """
        return self._expand(pattern_scores)


def _normalize_responses(
    responses: ArrayLike,
    missing_code: int,
) -> NDArray[np.int_]:
    """Validate response data and normalize all missing values."""
    if (
        isinstance(missing_code, (bool, np.bool_))
        or not isinstance(missing_code, (int, np.integer))
        or missing_code >= 0
        or missing_code < np.iinfo(np.int_).min
    ):
        raise MirtValidationError(
            "missing_code must be a supported negative integer",
            parameter="missing_code",
            value=missing_code,
        )

    try:
        response_array = np.asarray(responses)
    except (TypeError, ValueError) as exc:
        raise MirtDataError("responses must be a rectangular numeric array") from exc
    if response_array.dtype.kind == "f" and np.any(np.isnan(response_array)):
        response_array = response_array.copy()
        response_array[np.isnan(response_array)] = missing_code

    validated = validate_responses(
        response_array,
        allow_missing=True,
        missing_code=int(missing_code),
    )
    return np.ascontiguousarray(validated)


def _collapse_validated(responses: NDArray[np.int_]) -> CollapsedData:
    """Collapse an already validated, contiguous response matrix."""
    n_persons, n_items = responses.shape

    row_dtype = np.dtype((np.void, responses.itemsize * n_items))
    patterns_flat = responses.view(row_dtype).ravel()
    _unique_patterns, first_indices, inverse, counts = np.unique(
        patterns_flat,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    appearance_order = np.argsort(first_indices, kind="stable")
    first_indices = first_indices[appearance_order]
    patterns = responses[first_indices].copy()
    frequencies = counts[appearance_order].astype(np.int_, copy=False)
    n_patterns = len(first_indices)

    sorted_to_appearance = np.empty(n_patterns, dtype=np.int_)
    sorted_to_appearance[appearance_order] = np.arange(n_patterns, dtype=np.int_)
    indices = sorted_to_appearance[inverse]

    return CollapsedData(
        patterns=patterns,
        frequencies=frequencies,
        indices=indices,
        n_persons=n_persons,
        n_patterns=n_patterns,
    )


def collapse_patterns(
    responses: ArrayLike,
    missing_code: int = -1,
) -> CollapsedData:
    """Collapse identical response patterns for efficient computation.

    This function identifies unique response patterns and their frequencies,
    reducing computational burden for large datasets with many duplicate
    response patterns.

    Parameters
    ----------
    responses : array-like of shape (n_persons, n_items)
        Response matrix. NaN and negative values are treated as missing and
        normalized to ``missing_code``. Patterns retain first-appearance order.
    missing_code : int
        Value used for missing responses.

    Returns
    -------
    CollapsedData
        Container with unique patterns, frequencies, and index mapping.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt.utils.collapse import collapse_patterns
    >>> data = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]])
    >>> collapsed = collapse_patterns(data)
    >>> print(f"Compressed {collapsed.n_persons} to {collapsed.n_patterns} patterns")
    Compressed 4 to 2 patterns
    >>> print(collapsed.frequencies)
    [3 1]
    """
    return _collapse_validated(_normalize_responses(responses, missing_code))


def collapse_with_groups(
    responses: ArrayLike,
    groups: ArrayLike,
    missing_code: int = -1,
) -> tuple[list[CollapsedData], list[NDArray[np.bool_]]]:
    """Collapse patterns separately for each group.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    groups : ndarray of shape (n_persons,)
        Group membership.
    missing_code : int
        Value used for missing responses.

    Returns
    -------
    collapsed_list : list of CollapsedData
        Collapsed data for each group.
    group_masks : list of ndarray
        Boolean masks for each group.
    """
    responses_array = _normalize_responses(responses, missing_code)
    try:
        groups_array = np.asarray(groups)
    except (TypeError, ValueError) as exc:
        raise MirtDataError("groups must be a one-dimensional array") from exc
    if groups_array.ndim != 1:
        raise MirtDataError(f"groups must be a 1D array, got {groups_array.ndim}D")
    if len(groups_array) != len(responses_array):
        raise MirtDataError(
            "groups must contain one value per person",
            n_persons=len(responses_array),
        )
    has_missing_group = (
        np.any(np.isnan(groups_array))
        if groups_array.dtype.kind in "fc"
        else groups_array.dtype.kind == "O"
        and any(
            value is None or isinstance(value, (float, np.floating)) and np.isnan(value)
            for value in groups_array
        )
    )
    if has_missing_group:
        raise MirtDataError("groups must not contain missing values")

    try:
        unique_groups, first_indices = np.unique(groups_array, return_index=True)
    except TypeError as exc:
        raise MirtDataError("group values must be mutually comparable") from exc
    unique_groups = unique_groups[np.argsort(first_indices, kind="stable")]
    collapsed_list = []
    group_masks = []

    for g in unique_groups:
        mask = groups_array == g
        group_masks.append(mask)
        group_data = responses_array[mask]
        collapsed_list.append(_collapse_validated(group_data))

    return collapsed_list, group_masks


def compute_pattern_likelihood(
    collapsed: CollapsedData,
    log_likelihood_func: Callable[
        [NDArray[np.int_], NDArray[np.float64]], NDArray[np.float64]
    ],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log-likelihood for collapsed patterns.

    Parameters
    ----------
    collapsed : CollapsedData
        Collapsed response data.
    log_likelihood_func : callable
        Function that computes log-likelihood: (responses, theta) -> log_lik.
    theta : ndarray
        Theta values at which to compute likelihood.

    Returns
    -------
    ndarray of shape (n_patterns, ...)
        Log-likelihood values with one leading entry per pattern.
    """
    values = np.asarray(log_likelihood_func(collapsed.patterns, theta))
    if values.ndim == 0 or values.shape[0] != collapsed.n_patterns:
        raise MirtValidationError(
            "log_likelihood_func must return one leading entry per pattern",
            parameter="log_likelihood_func",
            value=values.shape,
            expected=f"first dimension of {collapsed.n_patterns}",
        )
    return values


def weighted_sum_from_collapsed(
    collapsed: CollapsedData,
    pattern_values: ArrayLike,
) -> float:
    """Compute frequency-weighted sum of pattern-level values.

    Parameters
    ----------
    collapsed : CollapsedData
        Collapsed response data.
    pattern_values : ndarray of shape (n_patterns,)
        Values computed at pattern level.

    Returns
    -------
    float
        Weighted sum.
    """
    values = np.asarray(pattern_values, dtype=np.float64)
    if values.shape != (collapsed.n_patterns,):
        raise MirtValidationError(
            "pattern_values must contain one value per pattern",
            parameter="pattern_values",
            value=values.shape,
            expected=f"({collapsed.n_patterns},)",
        )
    return float(np.dot(collapsed.frequencies, values))
