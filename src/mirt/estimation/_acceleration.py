"""SQUAREM extrapolation helpers for EM item parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.estimation._graded_order import THRESHOLD_GAP
from mirt.estimation.base import _parameter_bounds

if TYPE_CHECKING:
    from mirt.estimation._shared_step import TiedCoordinates
    from mirt.models.base import BaseItemModel


def squarem_step_length(
    start: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    step_max: float,
) -> float:
    """Return the SqS3 step length ``|r| / |v|`` clamped to ``[1, step_max]``.

    ``r`` is the first EM step from ``start`` and ``v`` is the change between
    the two consecutive EM steps.
    """
    r = first - start
    v = second - first - r
    v_norm = float(np.sqrt(v @ v))
    if not np.isfinite(v_norm) or v_norm == 0.0:
        return 1.0
    alpha = float(np.sqrt(r @ r)) / v_norm
    if not np.isfinite(alpha):
        return 1.0
    return float(min(max(alpha, 1.0), step_max))


def squarem_point(
    start: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    alpha: float,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extrapolate two EM steps and project the result onto the bounds.

    A unit step length returns ``second``, the plain second EM iterate.
    """
    r = first - start
    v = second - first - r
    return np.clip(start + 2.0 * alpha * r + alpha**2 * v, lower, upper)


class FreeItemParameters:
    """Pack a model's free item and shared parameters into one bounded vector.

    Coordinates follow model parameter order and C order within each array,
    as the M-step uses them. Parameters that are neither indexed by item nor
    shared by all items are never moved by the M-step and are left out. A
    group of coordinates tied by equality constraints is one coordinate, at
    the position of its first member.
    """

    def __init__(
        self, model: BaseItemModel, tied: TiedCoordinates | None = None
    ) -> None:
        masks = model.free_parameter_masks
        self._masks: dict[str, NDArray[np.bool_]] = {}
        lower: list[NDArray[np.float64]] = []
        upper: list[NDArray[np.float64]] = []
        for name in model.parameters:
            if name not in model._shared_parameters and not model._item_indexed(name):
                continue
            mask = np.asarray(masks[name], dtype=np.bool_)
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            self._masks[name] = mask
            low, high = _parameter_bounds(model, name)
            lower.append(np.full(count, low))
            upper.append(np.full(count, high))
        self.lower = np.concatenate(lower) if lower else np.empty(0)
        self.upper = np.concatenate(upper) if upper else np.empty(0)
        self._expand: NDArray[np.intp] | None = None
        self._keep: NDArray[np.intp] | None = None
        if tied is not None:
            self._expand = tied.tying(
                {
                    name: np.flatnonzero(mask.ravel())
                    for name, mask in self._masks.items()
                }
            )
            # Every member of a group holds the same value and bounds.
            _, self._keep = np.unique(self._expand, return_index=True)
            self.lower = self.lower[self._keep]
            self.upper = self.upper[self._keep]

    def get(self, model: BaseItemModel) -> NDArray[np.float64]:
        """Return the current free coordinates."""
        params = model.parameters
        chunks = [
            model._canonical_parameter_values(name, params[name])[mask]
            for name, mask in self._masks.items()
        ]
        vector = np.concatenate(chunks) if chunks else np.empty(0)
        return vector if self._keep is None else vector[self._keep]

    def set(
        self,
        model: BaseItemModel,
        vector: NDArray[np.float64],
        *,
        check_order: bool = False,
    ) -> bool:
        """Install free coordinates, returning ``False`` for invalid points.

        Invalid points leave the model unchanged. With ``check_order``, graded
        thresholds must keep the ordering gap of the current parameters.
        """
        if self._expand is not None:
            vector = vector[self._expand]
        params = model.parameters
        updates = {}
        offset = 0
        for name, mask in self._masks.items():
            values = model._canonical_parameter_values(name, params[name])
            count = int(np.count_nonzero(mask))
            values[mask] = vector[offset : offset + count]
            updates[name] = model._canonical_parameter_values(name, values)
            offset += count
        if check_order and not _keeps_threshold_order(model, updates):
            return False
        try:
            model.set_parameters(**updates)
        except ValueError:
            model.set_parameters(**{name: params[name] for name in updates})
            return False
        return True


def _keeps_threshold_order(
    model: BaseItemModel, updates: dict[str, NDArray[np.float64]]
) -> bool:
    """Require graded thresholds to stay ordered like the current iterate."""
    from mirt.models.polytomous import GradedResponseModel

    if not isinstance(model, GradedResponseModel) or "thresholds" not in updates:
        return True
    n_gaps = np.asarray(model.n_categories) - 2
    current = np.diff(model.parameters["thresholds"], axis=1)
    proposed = np.diff(updates["thresholds"], axis=1)
    used = np.arange(current.shape[1]) < n_gaps[:, None]
    required = np.minimum(THRESHOLD_GAP, current)
    return bool(np.all(proposed[used] >= required[used]))
