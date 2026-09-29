"""Shared array kernels for classical test statistics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_ALPHA_DELETED_CHUNK_ELEMENTS = 2_000_000


def _sample_variance(
    values: NDArray[np.float64],
    *,
    axis: int | None = None,
) -> NDArray[np.float64] | float:
    """Compute sample variance without warnings for sparse columns."""
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=axis)
    sums = np.nansum(values, axis=axis)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    squared_deviations = np.array(values, dtype=np.float64, copy=True)
    squared_deviations -= means if axis is None else np.expand_dims(means, axis=axis)
    np.square(squared_deviations, out=squared_deviations)
    np.copyto(squared_deviations, 0.0, where=~valid)
    squared_sum = np.sum(squared_deviations, axis=axis)
    return np.divide(
        squared_sum,
        counts - 1,
        out=np.zeros_like(squared_sum, dtype=np.float64),
        where=counts > 1,
    )


def _alpha_if_deleted_numpy(
    responses: NDArray[np.float64],
    *,
    total_scores: NDArray[np.float64] | None = None,
    observed_counts: NDArray[np.intp] | None = None,
    item_variances: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute deletion coefficients, optionally reusing prepared statistics."""
    n_persons, n_items = responses.shape
    remaining_items = n_items - 1
    alpha = np.zeros(n_items, dtype=np.float64)
    if remaining_items < 2:
        return alpha

    if observed_counts is None:
        observed_counts = np.sum(np.isfinite(responses), axis=1, dtype=np.intp)
    if total_scores is None:
        total_scores = np.nansum(responses, axis=1)
    if item_variances is None:
        item_variances = np.asarray(_sample_variance(responses, axis=0))
    remaining_variance_sums = np.sum(item_variances) - item_variances
    chunk_size = min(
        n_items,
        max(1, _ALPHA_DELETED_CHUNK_ELEMENTS // max(1, n_persons)),
    )

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        valid_chunk = np.isfinite(responses[:, start:stop])
        deleted_scores = total_scores[:, None] - np.where(
            valid_chunk,
            responses[:, start:stop],
            0.0,
        )
        deleted_scores[observed_counts[:, None] == valid_chunk] = np.nan
        total_variances = np.asarray(_sample_variance(deleted_scores, axis=0))
        usable = total_variances > 0.0
        alpha[start:stop] = np.divide(
            remaining_variance_sums[start:stop],
            total_variances,
            out=np.zeros(stop - start, dtype=np.float64),
            where=usable,
        )
        alpha[start:stop] = np.where(
            usable,
            (remaining_items / (remaining_items - 1)) * (1.0 - alpha[start:stop]),
            0.0,
        )

    alpha[np.abs(alpha) <= 4.0 * np.finfo(np.float64).eps] = 0.0
    return alpha
