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
    if axis is None:
        squared_deviations = np.where(valid, (values - means) ** 2, 0.0)
    else:
        squared_deviations = np.where(
            valid,
            (values - np.expand_dims(means, axis=axis)) ** 2,
            0.0,
        )
    squared_sum = np.sum(squared_deviations, axis=axis)
    return np.divide(
        squared_sum,
        counts - 1,
        out=np.zeros_like(squared_sum, dtype=np.float64),
        where=counts > 1,
    )


def _alpha_if_deleted_numpy(
    responses: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute every deletion coefficient in bounded matrix chunks."""
    n_persons, n_items = responses.shape
    remaining_items = n_items - 1
    alpha = np.zeros(n_items, dtype=np.float64)
    if remaining_items < 2:
        return alpha

    valid = np.isfinite(responses)
    observed_per_person = np.sum(valid, axis=1, dtype=np.intp)
    total_scores = np.nansum(responses, axis=1)
    item_variances = np.asarray(_sample_variance(responses, axis=0))
    remaining_variance_sums = np.sum(item_variances) - item_variances
    chunk_size = min(
        n_items,
        max(1, _ALPHA_DELETED_CHUNK_ELEMENTS // max(1, n_persons)),
    )

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        valid_chunk = valid[:, start:stop]
        deleted_scores = total_scores[:, None] - np.where(
            valid_chunk,
            responses[:, start:stop],
            0.0,
        )
        deleted_scores[observed_per_person[:, None] == valid_chunk] = np.nan
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
