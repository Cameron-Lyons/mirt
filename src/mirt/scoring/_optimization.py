"""Configuration helpers for optimizer-based ability scoring."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def validate_theta_bounds(bounds: object) -> tuple[float, float]:
    """Return finite, increasing theta bounds as native floats."""
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Iterable):
        raise ValueError("bounds must contain exactly two finite values")
    values: tuple[object, ...] = tuple(bounds)
    if len(values) != 2:
        raise ValueError("bounds must contain exactly two finite values")
    try:
        lower, upper = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must contain exactly two finite values") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must contain finite values with lower < upper")
    return lower, upper
