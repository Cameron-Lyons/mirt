"""Response pattern collapsing for efficient IRT estimation.

This module provides utilities for collapsing identical response patterns
to reduce computational burden during EM estimation, especially for large datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


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
        return pattern_weights[self.indices]

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
        return pattern_scores[self.indices]


def collapse_patterns(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> CollapsedData:
    """Collapse identical response patterns for efficient computation.

    This function identifies unique response patterns and their frequencies,
    reducing computational burden for large datasets with many duplicate
    response patterns.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix with missing data coded as missing_code.
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
    responses = np.asarray(responses, dtype=np.int_)
    n_persons, n_items = responses.shape

    patterns_view = responses.view(dtype=f"S{responses.itemsize * n_items}")
    patterns_flat = patterns_view.ravel()

    unique_patterns, indices, counts = np.unique(
        patterns_flat,
        return_inverse=True,
        return_counts=True,
    )

    n_patterns = len(unique_patterns)
    patterns = unique_patterns.view(responses.dtype).reshape(n_patterns, n_items)

    return CollapsedData(
        patterns=patterns,
        frequencies=counts,
        indices=indices,
        n_persons=n_persons,
        n_patterns=n_patterns,
    )


def collapse_with_groups(
    responses: NDArray[np.int_],
    groups: NDArray,
    missing_code: int = -1,
) -> tuple[list[CollapsedData], list[NDArray]]:
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
    unique_groups = np.unique(groups)
    collapsed_list = []
    group_masks = []

    for g in unique_groups:
        mask = groups == g
        group_masks.append(mask)
        group_data = responses[mask]
        collapsed_list.append(collapse_patterns(group_data, missing_code))

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
    ndarray of shape (n_patterns,)
        Log-likelihood for each pattern.
    """
    return log_likelihood_func(collapsed.patterns, theta)


def weighted_sum_from_collapsed(
    collapsed: CollapsedData,
    pattern_values: NDArray[np.float64],
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
    return float(np.sum(collapsed.frequencies * pattern_values))
