"""Internal categorical counting and sampling helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_FREQUENCY_CHUNK_ELEMENTS = 1_000_000
_MAX_FREQUENCY_ENTRIES = 5_000_000


def item_category_frequencies(
    responses: NDArray,
    valid: NDArray[np.bool_],
    *,
    max_categories: int = 32,
) -> NDArray[np.intp] | None:
    """Count validated nonnegative category codes in bounded row blocks.

    Return None for sparse or empty category ranges so callers can use their
    per-item fallback without allocating a large dense frequency table.
    """
    n_persons, n_items = responses.shape
    n_categories = int(np.max(responses, where=valid, initial=-1)) + 1
    if (
        not 0 < n_categories <= max_categories
        or n_categories * n_items > _MAX_FREQUENCY_ENTRIES
    ):
        return None

    frequencies = np.zeros((n_categories, n_items), dtype=np.intp)
    rows_per_block = max(1, _FREQUENCY_CHUNK_ELEMENTS // max(n_items, 1))
    for start in range(0, n_persons, rows_per_block):
        stop = start + rows_per_block
        block = responses[start:stop]
        block_valid = valid[start:stop]
        for category in range(n_categories):
            frequencies[category] += np.count_nonzero(
                (block == category) & block_valid, axis=0
            )
    return frequencies


def sample_categorical_rows(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Draw one category from every row of a probability matrix."""
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(probabilities.shape[0])
    return np.sum(uniforms[:, None] >= cumulative, axis=1).astype(np.int_)
