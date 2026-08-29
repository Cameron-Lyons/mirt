"""Dependency-free multiple-testing adjustments for diagnostic results."""

from __future__ import annotations

from numbers import Integral
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

PValueAdjustment: TypeAlias = Literal["bonferroni", "holm", "fdr_bh", "none"]
_P_VALUE_ADJUSTMENTS = frozenset({"bonferroni", "holm", "fdr_bh", "none"})


def _validate_p_value_adjustment(
    method: object,
    *,
    name: str = "method",
) -> PValueAdjustment:
    """Return a supported adjustment name."""
    if not isinstance(method, str) or method not in _P_VALUE_ADJUSTMENTS:
        raise ValueError(f"{name} must be 'bonferroni', 'holm', 'fdr_bh', or 'none'")
    return cast(PValueAdjustment, method)


def _coerce_p_values(p_values: ArrayLike) -> NDArray[np.float64]:
    """Return validated probabilities while preserving the input shape."""
    if np.iscomplexobj(p_values):
        raise ValueError("p_values must contain real probabilities or NaN")
    try:
        values = np.asarray(p_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_values must contain real probabilities or NaN") from exc

    invalid = (~np.isnan(values)) & (
        (~np.isfinite(values)) | (values < 0.0) | (values > 1.0)
    )
    if np.any(invalid):
        raise ValueError("p_values must contain probabilities in [0, 1] or NaN")
    return values


def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize an axis with clear public input errors."""
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
        raise TypeError("axis must be an integer or None")
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for an array with {ndim} dimensions"
        )
    return normalized


def _adjust_rows(
    rows: NDArray[np.float64],
    method: PValueAdjustment,
) -> NDArray[np.float64]:
    """Adjust independent p-value families stored one per row."""
    if method == "none" or rows.shape[1] == 0:
        return rows.copy()

    finite = np.isfinite(rows)
    family_sizes = np.count_nonzero(finite, axis=1)
    if method == "bonferroni":
        return np.clip(rows * family_sizes[:, None], 0.0, 1.0)

    sortable = np.where(finite, rows, np.inf)
    order = np.argsort(sortable, axis=1, kind="stable")
    ordered = np.take_along_axis(sortable, order, axis=1)
    positions = np.arange(rows.shape[1])[None, :]
    ordered_finite = positions < family_sizes[:, None]
    finite_ordered = np.where(ordered_finite, ordered, 0.0)

    if method == "holm":
        factors = family_sizes[:, None] - positions
        candidates = np.where(ordered_finite, finite_ordered * factors, -np.inf)
        ordered_adjusted = np.maximum.accumulate(candidates, axis=1)
    else:
        ranks = positions + 1
        candidates = np.where(
            ordered_finite,
            finite_ordered * family_sizes[:, None] / ranks,
            np.inf,
        )
        ordered_adjusted = np.minimum.accumulate(
            candidates[:, ::-1],
            axis=1,
        )[:, ::-1]

    ordered_adjusted = np.where(
        ordered_finite,
        np.clip(ordered_adjusted, 0.0, 1.0),
        np.nan,
    )
    adjusted = np.full_like(rows, np.nan)
    np.put_along_axis(adjusted, order, ordered_adjusted, axis=1)
    return adjusted


def adjust_p_values(
    p_values: ArrayLike,
    method: PValueAdjustment = "holm",
    *,
    axis: int | None = None,
) -> NDArray[np.float64]:
    """Adjust one or more families of p-values for multiple testing.

    Parameters
    ----------
    p_values : array-like
        Probabilities in ``[0, 1]``. ``NaN`` entries are preserved and excluded
        from the corresponding family size.
    method : {"bonferroni", "holm", "fdr_bh", "none"}, default="holm"
        Bonferroni family-wise control, Holm step-down family-wise control,
        Benjamini-Hochberg false-discovery control, or no adjustment.
    axis : int or None, default=None
        Axis containing each testing family. If ``None``, all entries form one
        family. Otherwise, every slice along ``axis`` is adjusted independently.

    Returns
    -------
    NDArray
        Adjusted p-values with the same shape as the input.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer or ``None``.
    ValueError
        If the method, axis, or p-values are invalid.
    """
    validated_method = _validate_p_value_adjustment(method)
    values = _coerce_p_values(p_values)

    if axis is None:
        adjusted = _adjust_rows(values.reshape(1, -1), validated_method)
        return adjusted.reshape(values.shape)

    normalized_axis = _normalize_axis(axis, values.ndim)
    moved = np.moveaxis(values, normalized_axis, -1)
    n_families = int(np.prod(moved.shape[:-1], dtype=np.intp))
    adjusted_rows = _adjust_rows(
        moved.reshape(n_families, moved.shape[-1]),
        validated_method,
    )
    adjusted = adjusted_rows.reshape(moved.shape)
    return np.moveaxis(adjusted, -1, normalized_axis)
