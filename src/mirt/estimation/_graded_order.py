"""Ordered graded-response thresholds in the itemwise free-parameter layout.

Graded response items store increasing thresholds; graded bifactor items
store R's decreasing intercepts, whose negations are the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

# Smallest gap kept between adjacent movable graded thresholds. It keeps
# roundoff and numerical derivative probes from creating negative category
# probabilities; the native graded M-step uses the same gap.
THRESHOLD_GAP = 1e-6


@dataclass(frozen=True)
class GradedOrder:
    """Ordering constraints on one graded item's thresholds.

    Attributes
    ----------
    constraint : LinearConstraint
        Adjacent thresholds at least ``THRESHOLD_GAP`` apart, in the item's
        free-coordinate layout.
    positions : ndarray of int
        Layout index of each threshold, or ``-1`` for a fixed threshold.
    values : ndarray
        Current threshold values; fixed thresholds keep theirs.
    item : int
        Item index, for error messages.
    sign : float, default=1.0
        The thresholds are ``sign`` times the stored coordinates: ``-1`` for
        the decreasing intercepts of graded bifactor items.
    """

    constraint: LinearConstraint
    positions: NDArray[np.intp]
    values: NDArray[np.float64]
    item: int
    sign: float = 1.0

    def satisfied(self, params: NDArray[np.float64]) -> bool:
        """Return whether ``params`` keep the thresholds ordered."""
        constraint = self.constraint
        return bool(np.all(constraint.A @ params >= constraint.lb))

    def project(
        self,
        params: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the nearest point of the box whose thresholds are ordered.

        Points whose clipped values are ordered are only clipped to the box.
        Otherwise each run of adjacent free thresholds is projected onto the
        ordered values between its box and the gaps to fixed neighbours:
        thresholds less their minimum offsets are fitted by an isotonic
        regression, which is clipped to that interval.

        Raises
        ------
        MirtValidationError
            If fixed thresholds or the box leave a run of free thresholds no
            ordered values.
        """
        if self.sign < 0:
            # Project the negated intercepts, which are increasing thresholds.
            flip = np.ones(len(params))
            flip[self.positions[self.positions >= 0]] = -1.0
            thresholds = GradedOrder(
                LinearConstraint(
                    self.constraint.A * flip, self.constraint.lb, self.constraint.ub
                ),
                self.positions,
                self.values,
                self.item,
            )
            low = np.where(flip > 0, lower, -np.asarray(upper))
            high = np.where(flip > 0, upper, -np.asarray(lower))
            return flip * thresholds.project(flip * np.asarray(params), low, high)
        raw = np.asarray(params, dtype=np.float64)
        x = np.clip(raw, lower, upper)
        if self.satisfied(x):
            return x
        positions = self.positions
        n_thresholds = len(positions)
        start = 0
        while start < n_thresholds:
            if positions[start] < 0:
                start += 1
                continue
            end = start
            while end + 1 < n_thresholds and positions[end + 1] >= 0:
                end += 1
            index = positions[start : end + 1]
            low = float(lower[index[0]])
            if start > 0:
                low = max(low, float(self.values[start - 1]) + THRESHOLD_GAP)
            high = float(upper[index[-1]])
            if end + 1 < n_thresholds:
                high = min(high, float(self.values[end + 1]) - THRESHOLD_GAP)
            offsets = THRESHOLD_GAP * np.arange(len(index))
            # Beyond roundoff, no ordered values fit between the bounds.
            if high - offsets[-1] < low - THRESHOLD_GAP * 1e-6:
                raise MirtValidationError(
                    f"fixed thresholds of GRM item {self.item} leave no room for "
                    "ordered free thresholds within their bounds",
                    parameter="thresholds",
                )
            ceiling = max(low, high - offsets[-1])
            levels = np.clip(
                _pool_adjacent_violators(raw[index] - offsets), low, ceiling
            )
            # Count down from a clipped ceiling so that the last threshold
            # lands exactly on its bound.
            x[index] = np.where(
                levels == ceiling, high - offsets[::-1], levels + offsets
            )
            start = end + 1
        return x


def _pool_adjacent_violators(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the least-squares non-decreasing fit to ``values``."""
    totals: list[float] = []
    counts: list[int] = []
    for value in values:
        total, count = float(value), 1
        while totals and totals[-1] / counts[-1] > total / count:
            total += totals.pop()
            count += counts.pop()
        totals.append(total)
        counts.append(count)
    return np.repeat(np.divide(totals, counts), counts)


def graded_order(
    model: BaseItemModel, item_idx: int, n_parameters: int
) -> GradedOrder | None:
    """Return the threshold ordering of a graded item in the estimator layout.

    Parameters
    ----------
    model : BaseItemModel
        Model owning the item. Graded response models order their thresholds
        and graded bifactor models their decreasing intercepts; other models
        have no ordering.
    item_idx : int
        Item index.
    n_parameters : int
        Length of the item's free-coordinate vector, as returned by
        ``BaseEstimator._get_item_params_and_bounds``.

    Returns
    -------
    GradedOrder or None
        None when no pair of adjacent thresholds has a movable member. Fixed
        thresholds contribute constants and padded storage is excluded.

    Raises
    ------
    MirtValidationError
        If two adjacent fixed thresholds are out of order.
    """
    from mirt.models.bifactor import GradedBifactorModel
    from mirt.models.polytomous import GradedResponseModel

    if isinstance(model, GradedResponseModel):
        stored, orientation = "thresholds", 1.0
    elif isinstance(model, GradedBifactorModel):
        stored, orientation = "intercepts", -1.0
    else:
        return None
    n_thresholds = model.n_categories[item_idx] - 1
    if n_thresholds < 2:
        return None
    free_masks = model.free_parameter_masks
    offset = 0
    positions = np.full(n_thresholds, -1, dtype=np.intp)
    values = None
    for name, array in model.parameters.items():
        if not model._item_indexed(name):
            continue
        indices = np.flatnonzero(np.asarray(free_masks[name][item_idx]).reshape(-1))
        if name == stored:
            values = np.asarray(array[item_idx], dtype=np.float64).reshape(-1)
            values = orientation * values
            values = values[:n_thresholds].copy()
            for column, index in enumerate(indices):
                if index < n_thresholds:
                    positions[index] = offset + column
        offset += len(indices)
    if values is None:
        return None
    rows = []
    lower = []
    for first in range(n_thresholds - 1):
        row = np.zeros(n_parameters)
        fixed_difference = 0.0
        for index, sign in ((first, -1.0), (first + 1, 1.0)):
            if positions[index] >= 0:
                row[positions[index]] = sign
            else:
                fixed_difference += sign * values[index]
        if not np.any(row):
            if fixed_difference < 0:
                raise MirtValidationError(
                    "fixed GRM thresholds must be ordered"
                    if orientation > 0
                    else f"fixed intercepts of graded item {item_idx} must decrease",
                    parameter=stored,
                )
            continue
        rows.append(row)
        lower.append(THRESHOLD_GAP - fixed_difference)
    if not rows:
        return None
    # Rows act on the thresholds, which are orientation times the coordinates.
    constraint = LinearConstraint(
        orientation * np.asarray(rows), np.asarray(lower), np.inf
    )
    return GradedOrder(constraint, positions, values, item_idx, orientation)


def graded_threshold_constraint(
    model: BaseItemModel, item_idx: int, n_parameters: int
) -> LinearConstraint | None:
    """Return the linear threshold-ordering constraint of a graded item, if any."""
    order = graded_order(model, item_idx, n_parameters)
    return None if order is None else order.constraint
