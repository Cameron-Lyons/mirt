"""Response-pattern grouping shared by scoring and data utilities.

Fallback mode: numpy. Rows retain first-appearance order on both backends.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import mirt_rs, rust_enabled

FALLBACK_MODE = "numpy"


def response_pattern_indices(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Return first row indices, inverse indices, and counts for integer rows.

    Callers normalize missing responses before grouping. Returning indices
    lets them select patterns once, without copying response rows in Rust.
    """
    values = np.asarray(responses)
    if values.ndim != 2 or values.dtype.kind not in "bi":
        raise ValueError("responses must be a two-dimensional signed integer array")
    n_persons, n_items = values.shape
    if n_persons == 0:
        return tuple(np.empty(0, dtype=np.intp) for _ in range(3))
    if n_items == 0:
        return (
            np.array([0], dtype=np.intp),
            np.zeros(n_persons, dtype=np.intp),
            np.array([n_persons], dtype=np.intp),
        )

    # Small category codes need fewer bytes to hash or sort for wide tests.
    minimum, maximum = int(np.min(values)), int(np.max(values))
    key_dtype = values.dtype.newbyteorder("=")
    for dtype in (np.int8, np.int16, np.int32):
        bounds = np.iinfo(dtype)
        if bounds.min <= minimum and maximum <= bounds.max:
            key_dtype = np.dtype(dtype)
            break
    # C-contiguity alone permits unaligned NumPy buffers. Native typed slices
    # require alignment as well; np.require preserves safe arrays without copies.
    keys = np.require(values, dtype=key_dtype, requirements=["C", "A"])
    if rust_enabled():
        return mirt_rs.response_pattern_indices(keys)

    row_dtype = np.dtype((np.void, keys.itemsize * n_items))
    _, first, inverse, counts = np.unique(
        keys.view(row_dtype).ravel(),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    # Recover encounter order in linear time instead of sorting unique rows again.
    first_mask = np.zeros(n_persons, dtype=np.bool_)
    first_mask[first] = True
    ordered_first = np.flatnonzero(first_mask)
    appearance_order = inverse[ordered_first]
    sorted_to_appearance = np.empty(first.size, dtype=np.intp)
    sorted_to_appearance[appearance_order] = np.arange(first.size, dtype=np.intp)
    return (
        ordered_first,
        sorted_to_appearance[inverse],
        counts[appearance_order],
    )
