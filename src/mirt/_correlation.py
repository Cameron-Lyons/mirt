"""Shared, bounded-memory pairwise-complete correlation kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_CORRELATION_CHUNK_ELEMENTS = 1_000_000


def pairwise_correlations(
    values: NDArray[np.float64],
    observed: NDArray[np.bool_] | None = None,
    *,
    min_count: int = 2,
    min_denominator: float = 0.0,
    chunk_size: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return correlations and counts, excluding missing/nonfinite entries.

    Undefined correlations are NaN. Callers own their diagonal and insufficient
    sample conventions. Complete blocks need only one matrix product; incomplete
    blocks accumulate moments on each pair's shared observations. Fixed column
    offsets reduce cancellation without changing pairwise centering.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or (observed is not None and observed.shape != values.shape):
        raise ValueError("values and observed must have the same two-dimensional shape")
    n_persons, n_items = values.shape
    if chunk_size is None:
        chunk_size = max(1, _CORRELATION_CHUNK_ELEMENTS // max(1, n_items))

    counts = np.zeros((n_items, n_items), dtype=np.float64)
    sums = np.zeros_like(counts)
    square_sums = np.zeros_like(counts)
    products = np.zeros_like(counts)
    offsets = np.zeros(n_items, dtype=np.float64)
    have_offsets = np.zeros(n_items, dtype=np.bool_)

    for start in range(0, n_persons, chunk_size):
        block = values[start : start + chunk_size]
        valid = np.isfinite(block)
        if observed is not None:
            valid &= observed[start : start + chunk_size]
        complete = np.all(valid)
        if not np.all(have_offsets):
            first = np.argmax(valid, axis=0)
            columns = np.flatnonzero(~have_offsets & valid[first, np.arange(n_items)])
            offsets[columns] = block[first[columns], columns]
            have_offsets[columns] = True

        centered = np.subtract(block, offsets, out=np.zeros_like(block), where=valid)
        products += centered.T @ centered
        squares = centered * centered
        if complete:
            counts += block.shape[0]
            sums += centered.sum(axis=0)[:, None]
            square_sums += squares.sum(axis=0)[:, None]
        else:
            mask = valid.astype(np.float64)
            counts += mask.T @ mask
            sums += centered.T @ mask
            square_sums += squares.T @ mask

    safe_counts = np.maximum(counts, 1.0)
    covariance = products - sums * sums.T / safe_counts
    variances = square_sums - sums * sums / safe_counts
    np.maximum(variances, 0.0, out=variances)
    denominator = np.sqrt(variances * variances.T)
    correlations = np.full_like(counts, np.nan)
    np.divide(
        covariance,
        denominator,
        out=correlations,
        where=(counts >= min_count) & (denominator > min_denominator),
    )
    np.clip(correlations, -1.0, 1.0, out=correlations)
    return correlations, counts.astype(np.intp)


def q3_correlations(
    residuals: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Apply diagnostic Q3's zero diagonal and sparse-pair conventions."""
    correlations, counts = pairwise_correlations(
        residuals, observed, min_count=3, chunk_size=chunk_size
    )
    correlations[counts < 3] = 0.0
    np.fill_diagonal(correlations, 0.0)
    return correlations
