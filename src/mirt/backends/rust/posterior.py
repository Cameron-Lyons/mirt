"""Independent row searches for posterior quantiles, intervals, and sampling.

Fallback mode: numpy. Search results match NumPy's insertion-point semantics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import mirt_rs, rust_enabled

FALLBACK_MODE = "numpy"


def row_searchsorted(
    cumulative: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    side: Literal["left", "right"] = "left",
) -> NDArray[np.intp]:
    """Search sorted rows using shared 1D targets or row-specific 2D targets.

    Each row must be sorted as for ``np.searchsorted``. Rows are independent:
    no arithmetic offsets are added to the CDF or targets, so searches retain
    precision even when a target is adjacent to a CDF boundary.
    """
    cumulative = np.asarray(cumulative, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if cumulative.ndim != 2:
        raise ValueError("cumulative must be a two-dimensional array")
    if targets.ndim not in (1, 2):
        raise ValueError("targets must be a one- or two-dimensional array")
    if targets.ndim == 2 and targets.shape[0] != cumulative.shape[0]:
        raise ValueError("targets must have one row per cumulative row")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    shape = (cumulative.shape[0], targets.shape[-1])
    if rust_enabled():
        return mirt_rs.row_searchsorted(
            cumulative, np.broadcast_to(targets, shape), side == "right"
        )

    indices = np.empty(shape, dtype=np.intp)
    if targets.ndim == 1:
        # Quantiles share a few thresholds across many respondents. Reducing
        # comparisons by column avoids a Python call for every respondent.
        compare = np.less if side == "left" else np.less_equal
        for column, target in enumerate(targets):
            if np.isnan(target):
                indices[:, column] = (
                    np.sum(~np.isnan(cumulative), axis=1)
                    if side == "left"
                    else cumulative.shape[1]
                )
            else:
                indices[:, column] = np.sum(compare(cumulative, target), axis=1)
    else:
        for row in range(shape[0]):
            indices[row] = np.searchsorted(cumulative[row], targets[row], side=side)
    return indices


def shortest_mass_intervals(
    coordinates: NDArray[np.float64],
    weights: NDArray[np.float64],
    level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find shortest contiguous intervals, breaking ties by mass then position."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        coordinates.ndim != 1
        or coordinates.size == 0
        or not np.all(np.isfinite(coordinates))
        or np.any(np.diff(coordinates) <= 0.0)
    ):
        raise ValueError(
            "coordinates must be a non-empty strictly increasing finite vector"
        )
    if weights.ndim != 2 or weights.shape[1] != coordinates.size:
        raise ValueError("weights must have one column per coordinate")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be finite and strictly between zero and one")
    totals = np.sum(weights, axis=1, keepdims=True)
    if np.any(weights < 0.0) or np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError(
            "weights must be nonnegative and finite with positive row sums"
        )
    cumulative = np.empty((weights.shape[0], coordinates.size + 1), dtype=np.float64)
    cumulative[:, 0] = 0.0
    np.divide(weights, totals, out=cumulative[:, 1:])
    np.cumsum(cumulative[:, 1:], axis=1, out=cumulative[:, 1:])
    cumulative[:, -1] = 1.0
    if rust_enabled():
        bounds = mirt_rs.shortest_mass_intervals(coordinates, cumulative, level)
        return bounds[:, 0], bounds[:, 1]

    targets = cumulative[:, :-1] + level
    end_boundaries = row_searchsorted(cumulative, targets)
    # Tiny positive levels can round away when added to a nonzero prefix mass.
    # Every interval must still include its starting coordinate.
    np.maximum(end_boundaries, np.arange(1, coordinates.size + 1), out=end_boundaries)
    valid = (targets <= 1.0) & (end_boundaries <= coordinates.size)
    safe_end_boundaries = np.clip(end_boundaries, 1, coordinates.size)

    widths = coordinates[safe_end_boundaries - 1] - coordinates[None, :]
    widths[~valid] = np.inf
    minimum_width = np.min(widths, axis=1)
    width_tolerance = (
        16.0 * np.finfo(np.float64).eps * max(1.0, float(np.ptp(coordinates)))
    )
    shortest = widths <= minimum_width[:, None] + width_tolerance
    rows = np.arange(weights.shape[0], dtype=np.intp)[:, None]
    enclosed_mass = cumulative[rows, safe_end_boundaries] - cumulative[:, :-1]
    greatest_mass = np.max(np.where(shortest, enclosed_mass, -np.inf), axis=1)
    mass_tolerance = 16.0 * np.finfo(np.float64).eps
    preferred = shortest & (enclosed_mass >= greatest_mass[:, None] - mass_tolerance)
    best_start = np.argmax(preferred, axis=1)
    best_end = safe_end_boundaries[np.arange(weights.shape[0]), best_start] - 1
    return coordinates[best_start], coordinates[best_end]
