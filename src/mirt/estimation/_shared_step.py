"""EM M-steps for item parameters shared by several items.

Rating-scale families give all items common category thresholds, and the
graded rating scale model also a common slope. After the itemwise M-step, the
expected complete-data log-likelihood ``sum_j sum_q sum_c r_jqc log P_jc(q)``
is maximized over the free shared coordinates with the item parameters held
fixed. Each EM iteration is then a conditional maximization, which keeps the
marginal likelihood nondecreasing (Meng and Rubin, 1993).

User equality constraints, such as equal slopes for a set of items, tie
coordinates of different items to one value. The items they link are
optimized jointly over their free coordinates with each tied group as a
single coordinate, which is the exact M-step of the constrained model.

References
----------
Meng, X.-L., & Rubin, D. B. (1993). Maximum likelihood estimation via the
    ECM algorithm: A general framework. Biometrika, 80(2), 267-278.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize
from scipy.special import xlogy

from mirt._core import sigmoid
from mirt.estimation._graded_order import THRESHOLD_GAP as _THRESHOLD_GAP
from mirt.estimation.base import _free_shared_parameters, _parameter_bounds
from mirt.exceptions import MirtEstimationError, MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.constraints import EqualityConstraint
    from mirt.models.base import BaseItemModel

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
# Relative step of the central differences that give numerical item
# objectives a gradient in the joint tied-item step.
_DIFFERENCE_STEP = 1e-5
_GROUP_KEYS = frozenset({"parameter", "items", "column"})
# Extra gaps tried, largest first, when disordered starting thresholds are
# moved to ordered ones; collapsed thresholds stall the optimizer.
_ORDER_MARGINS = (0.2, 0.0)


class SharedParameters:
    """Free shared coordinates of a model packed into one bounded vector.

    Coordinates follow model parameter order and C order within each array.
    """

    def __init__(self, model: BaseItemModel) -> None:
        masks = model.free_parameter_masks
        self.masks = {
            name: np.asarray(masks[name], dtype=np.bool_)
            for name in _free_shared_parameters(model)
        }
        self.bounds: list[tuple[float, float]] = []
        for name, mask in self.masks.items():
            self.bounds.extend(
                [_parameter_bounds(model, name)] * int(np.count_nonzero(mask))
            )
        self.size = len(self.bounds)

    def get(self, model: BaseItemModel) -> NDArray[np.float64]:
        """Return the current free shared coordinates."""
        params = model._parameters
        chunks = [
            model._canonical_parameter_values(name, params[name])[mask]
            for name, mask in self.masks.items()
        ]
        return np.concatenate(chunks) if chunks else np.empty(0)

    def values(
        self, model: BaseItemModel, vector: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Return full canonical shared arrays holding ``vector``."""
        result = {}
        offset = 0
        for name, mask in self.masks.items():
            values = model._canonical_parameter_values(name, model._parameters[name])
            count = int(np.count_nonzero(mask))
            values[mask] = vector[offset : offset + count]
            result[name] = model._canonical_parameter_values(name, values)
            offset += count
        return result

    def free(self, gradients: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Select the free coordinates of full-shape shared gradients."""
        chunks = [gradients[name][mask] for name, mask in self.masks.items()]
        return np.concatenate(chunks) if chunks else np.empty(0)


def binary_category_counts(
    correct: NDArray[np.float64], observed: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """Return each binary item's (incorrect, correct) expected counts."""
    return [
        np.column_stack((n_k - r_k, r_k))
        for r_k, n_k in zip(correct, observed, strict=True)
    ]


def _rating_scale_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> _Objective:
    """Exact objective of RSM thresholds, a common partial-credit step."""
    from mirt.estimation._polytomous_objective import _softmax_loss_and_residual
    from mirt.models.polytomous import _partial_credit_probabilities

    location = theta[:, None, None] - model._parameters["difficulty"][None, :, None]

    def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        thresholds = layout.values(model, vector)["thresholds"]
        probabilities = _partial_credit_probabilities(location - thresholds)
        loss, residual = _softmax_loss_and_residual(
            probabilities, counts, epsilon, 1.0 - epsilon
        )
        # Threshold v lowers the logit of every category above v.
        tails = np.cumsum(residual[..., :0:-1], axis=-1)[..., ::-1]
        gradient = {"thresholds": -tails.sum(axis=(0, 1))}
        return loss, layout.free(gradient)

    return objective


def _graded_rating_scale_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> _Objective:
    """Exact objective of the common GRSM slope and thresholds."""
    from mirt.estimation._polytomous_objective import _clipped_loss_and_counts
    from mirt.models.polytomous import _graded_probabilities

    location = theta[:, None, None] - model._parameters["difficulty"][None, :, None]

    def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        values = {**model._parameters, **layout.values(model, vector)}
        slope = float(values["discrimination"][0])
        centered = location - values["thresholds"]
        logits = slope * centered
        probabilities = _graded_probabilities(logits)
        loss, effective = _clipped_loss_and_counts(
            probabilities, counts, epsilon, 1.0 - epsilon
        )
        score = np.divide(
            effective,
            probabilities,
            out=np.zeros_like(probabilities),
            where=effective != 0.0,
        )
        cumulative = sigmoid(logits)
        common = (score[..., :-1] - score[..., 1:]) * cumulative * (1.0 - cumulative)
        gradient = {
            "discrimination": np.array([np.sum(common * centered)]),
            "thresholds": -slope * common.sum(axis=(0, 1)),
        }
        return loss, layout.free(gradient)

    return objective


def _numerical_objective(
    model: BaseItemModel,
    layout: SharedParameters,
    nodes: NDArray[np.float64],
    counts: Sequence[NDArray[np.float64]],
    epsilon: float,
) -> Callable[[NDArray[np.float64]], float]:
    """Objective of any item model from its all-item probabilities.

    Trial values are written to the model's storage; the caller restores it.
    """

    def objective(vector: NDArray[np.float64]) -> float:
        model._parameters.update(layout.values(model, vector))
        probabilities = np.asarray(model.probability(nodes), dtype=np.float64)
        if probabilities.ndim == 2:
            probabilities = np.stack((1.0 - probabilities, probabilities), axis=-1)
        loss = 0.0
        for item, item_counts in enumerate(counts):
            item_probabilities = probabilities[:, item, : item_counts.shape[1]]
            loss -= float(
                np.sum(
                    xlogy(
                        item_counts, np.clip(item_probabilities, epsilon, 1 - epsilon)
                    )
                )
            )
        return loss

    return objective


def _ordered_threshold_constraint(
    model: BaseItemModel, layout: SharedParameters
) -> LinearConstraint | None:
    """Keep GRSM thresholds ordered in the free shared layout."""
    from mirt.models.polytomous import GradedRatingScaleModel

    mask = layout.masks.get("thresholds")
    if not isinstance(model, GradedRatingScaleModel) or mask is None:
        return None
    offset = 0
    for name, other in layout.masks.items():
        if name == "thresholds":
            break
        offset += int(np.count_nonzero(other))
    positions = {
        int(index): offset + column for column, index in enumerate(np.flatnonzero(mask))
    }
    values = model._parameters["thresholds"]
    rows = []
    lower = []
    for first in range(values.size - 1):
        row = np.zeros(layout.size)
        fixed_difference = 0.0
        for index, sign in ((first, -1.0), (first + 1, 1.0)):
            if index in positions:
                row[positions[index]] = sign
            else:
                fixed_difference += sign * values[index]
        if np.any(row):
            rows.append(row)
            lower.append(_THRESHOLD_GAP - fixed_difference)
    if not rows:
        return None
    return LinearConstraint(np.asarray(rows), np.asarray(lower), np.inf)


def optimize_shared_parameters(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    counts: Sequence[NDArray[np.float64]],
    *,
    epsilon: float,
    max_iter: int,
    ftol: float,
) -> None:
    """Maximize the expected complete-data log-likelihood over shared coordinates.

    Parameters
    ----------
    model : BaseItemModel
        Model updated in place. Item parameters stay fixed.
    nodes : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    counts : sequence of ndarray
        Expected category counts of each item, shape ``(n_points, C_j)``. A
        dichotomous item has columns for incorrect and correct responses.
    epsilon : float
        Probability clipping bound of the item objectives.
    max_iter : int
        Iteration limit of the optimizer.
    ftol : float
        Relative function-change tolerance of the optimizer.
    """
    from mirt.models.polytomous import (
        GradedRatingScaleModel,
        RatingScaleModel,
        _uses_authored_rating_scale_hooks,
    )

    layout = SharedParameters(model)
    if not layout.size:
        return
    exact = {
        RatingScaleModel: _rating_scale_objective,
        GradedRatingScaleModel: _graded_rating_scale_objective,
    }.get(type(model))
    analytic = exact is not None and _uses_authored_rating_scale_hooks(model)
    objective: _Objective | Callable[[NDArray[np.float64]], float]
    if exact is not None and analytic:
        # Rating-scale items share one category count.
        stacked = np.stack(counts, axis=1)
        objective = exact(model, layout, nodes[:, 0], stacked, epsilon)
    else:
        objective = _numerical_objective(model, layout, nodes, counts, epsilon)

    def value(vector: NDArray[np.float64]) -> float:
        result = objective(vector)
        return float(result[0] if isinstance(result, tuple) else result)

    start = layout.get(model)
    constraint = _ordered_threshold_constraint(model, layout)
    original = {name: model._parameters[name] for name in layout.masks}
    try:
        candidate = minimize(
            objective,
            x0=start,
            method="SLSQP" if constraint is not None else "L-BFGS-B",
            jac=analytic,
            bounds=layout.bounds,
            options={"maxiter": max_iter, "ftol": ftol},
            constraints=() if constraint is None else (constraint,),
        ).x
        # A conditional maximization step must not decrease the objective.
        accepted = (
            np.all(np.isfinite(candidate))
            and (
                constraint is None
                or np.all(constraint.A @ candidate >= constraint.lb - 1e-8)
            )
            and value(candidate) <= value(start)
        )
    finally:
        model._parameters.update(original)
    if accepted:
        model.set_parameters(**layout.values(model, candidate))


@dataclass(frozen=True)
class EqualityGroup:
    """Items whose coordinates of one stored parameter are held equal.

    Attributes
    ----------
    parameter : str
        Stored per-item parameter, such as ``"discrimination"``.
    items : tuple of int or str, optional
        Zero-based item positions or item names; ``None`` selects every item.
    column : int, optional
        Coordinate within each item's row of an array parameter, such as one
        threshold or the slope on one factor. Without it, whole rows are tied
        column by column.
    """

    parameter: str
    items: tuple[int | str, ...] | None = None
    column: int | None = None


EqualityConstraints: TypeAlias = Sequence[
    "Mapping[str, Any] | Sequence[Any] | EqualityGroup | EqualityConstraint"
]


def _constraint_error(message: str, index: int | None = None) -> MirtValidationError:
    prefix = "constraints" if index is None else f"constraints[{index}]"
    return MirtValidationError(f"{prefix}: {message}", parameter="constraints")


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _equality_group(entry: object, index: int) -> EqualityGroup:
    """Convert one user constraint into an :class:`EqualityGroup`."""
    from mirt.estimation.constraints import EqualityConstraint, ParameterConstraint

    parameter: Any
    items: Any
    column: Any
    if isinstance(entry, EqualityGroup):
        parameter, items, column = entry.parameter, entry.items, entry.column
    elif isinstance(entry, EqualityConstraint):
        # Like EqualityConstraint.apply, an empty item list means every item.
        parameter, items, column = entry.param_name, entry.item_indices or None, None
    elif isinstance(entry, ParameterConstraint):
        raise _constraint_error(
            f"{type(entry).__name__} is not an equality constraint; hold "
            "coordinates at a value with fixed= and start_values=",
            index,
        )
    elif isinstance(entry, Mapping):
        unknown = sorted(set(map(str, entry)) - _GROUP_KEYS)
        if unknown or "parameter" not in entry:
            raise _constraint_error(
                "a mapping needs 'parameter' and may give 'items' and 'column'"
                + (f"; unknown keys {', '.join(unknown)}" if unknown else ""),
                index,
            )
        parameter = entry["parameter"]
        items, column = entry.get("items"), entry.get("column")
    elif isinstance(entry, (tuple, list)) and len(entry) in (2, 3):
        parameter, items, column = (*entry, None) if len(entry) == 2 else entry
    else:
        raise _constraint_error(
            "write each constraint as {'parameter': name, 'items': [...]} "
            "or (parameter, items[, column])",
            index,
        )
    if not isinstance(parameter, str) or not parameter:
        raise _constraint_error("the parameter must be a stored parameter name", index)
    positions: tuple[int | str, ...] | None = None
    if items is not None:
        try:
            if isinstance(items, (str, bytes, Mapping)):
                raise TypeError
            values = tuple(items)
        except TypeError:
            raise _constraint_error(
                "items must be a sequence of item positions or names", index
            ) from None
        positions = tuple(int(item) if _is_integer(item) else item for item in values)
        if not all(
            (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
            or (isinstance(item, str) and item)
            for item in positions
        ):
            raise _constraint_error(
                "items must be zero-based item positions or item names", index
            )
        if len(positions) < 2:
            raise _constraint_error("a constraint must tie at least two items", index)
        if len(set(positions)) != len(positions):
            raise _constraint_error("items lists an item more than once", index)
    if column is not None and (not _is_integer(column) or column < 0):
        raise _constraint_error("column must be a non-negative integer", index)
    return EqualityGroup(parameter, positions, None if column is None else int(column))


def validate_equality_constraints(
    constraints: EqualityConstraints | None,
) -> tuple[EqualityGroup, ...]:
    """Validate the structure of user equality constraints.

    Parameters
    ----------
    constraints : sequence, optional
        Constraint groups, each a mapping ``{"parameter": name, "items":
        [...], "column": c}`` (``items`` and ``column`` optional), a tuple
        ``(parameter, items)`` or ``(parameter, items, column)``, or an
        :class:`~mirt.estimation.constraints.EqualityConstraint`.

    Returns
    -------
    tuple of EqualityGroup
        The groups, empty without constraints. Items and parameters are
        checked against a model by :func:`resolve_equality_constraints`.

    Raises
    ------
    MirtValidationError
        If a group is malformed.
    """
    if constraints is None:
        return ()
    if isinstance(constraints, (str, bytes, Mapping)) or not isinstance(
        constraints, Sequence
    ):
        raise _constraint_error(
            "constraints must be a sequence of constraint groups, for example "
            "[{'parameter': 'discrimination', 'items': [0, 1, 2]}]"
        )
    return tuple(
        _equality_group(entry, index) for index, entry in enumerate(constraints)
    )


class TiedCoordinates:
    """Free stored coordinates that estimation keeps equal, in groups.

    Each group is a stored parameter name with flat indices into its storage,
    one coordinate per item. The coordinates of a group always hold the same
    value and act as one parameter.

    Parameters
    ----------
    groups : sequence of (str, ndarray)
        Parameter name and flat storage indices of each group.
    members : sequence of sequence of int
        Items of each group, aligned with ``groups``.

    Attributes
    ----------
    groups : tuple of (str, ndarray)
        Parameter name and flat storage indices of each group.
    items : tuple of int
        Items with a tied coordinate, in increasing order.
    components : tuple of tuple of int
        Items linked through shared groups, directly or through other items.
        Each component's M-step is independent of the others'.
    """

    def __init__(
        self,
        groups: Sequence[tuple[str, NDArray[np.intp]]],
        members: Sequence[Sequence[int]],
    ) -> None:
        self.groups = tuple(groups)
        components: list[set[int]] = []
        for group in members:
            linked = {int(item) for item in group}
            separate = []
            for component in components:
                if component & linked:
                    linked |= component
                else:
                    separate.append(component)
            components = [*separate, linked]
        self.components = tuple(sorted(tuple(sorted(c)) for c in components if c))
        self.items = tuple(sorted(item for c in self.components for item in c))
        self._group_of = {
            (name, int(flat)): index
            for index, (name, flats) in enumerate(self.groups)
            for flat in flats
        }

    @property
    def n_redundant(self) -> int:
        """Number of stored free coordinates that duplicate another."""
        return sum(flats.size - 1 for _, flats in self.groups)

    def group_of(self, name: str, flat: int) -> int | None:
        """Return the group holding a stored coordinate, or None."""
        return self._group_of.get((name, int(flat)))

    def equalize(self, model: BaseItemModel) -> None:
        """Set each group's coordinates to their mean value."""
        updates: dict[str, NDArray[np.float64]] = {}
        for name, flats in self.groups:
            values = updates.get(name)
            if values is None:
                values = model._canonical_parameter_values(
                    name, model._parameters[name]
                ).reshape(-1)
            values[flats] = values[flats].mean()
            updates[name] = values
        model.set_parameters(
            **{
                name: values.reshape(model._parameters[name].shape)
                for name, values in updates.items()
            }
        )

    def tying(self, free_indices: Mapping[str, NDArray[np.intp]]) -> NDArray[np.intp]:
        """Return the reduced coordinate of each packed free coordinate.

        Parameters
        ----------
        free_indices : mapping of str to ndarray
            Flat free indices of each packed parameter, in packing order.

        Returns
        -------
        ndarray of int
            For every packed coordinate, its column in the reduced vector,
            which keeps the first coordinate of each group in packing order.
        """
        positions: dict[tuple[str, int], int] = {}
        offset = 0
        for name, flats in free_indices.items():
            for rank, flat in enumerate(np.asarray(flats).ravel()):
                positions[(name, int(flat))] = offset + rank
            offset += np.asarray(flats).size
        representative = np.arange(offset)
        for name, flats in self.groups:
            members = [positions[(name, int(flat))] for flat in flats]
            representative[members] = min(members)
        return np.unique(representative, return_inverse=True)[1].astype(np.intp)

    def combine_standard_errors(
        self, errors: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Give each group the error of its diagonal curvature.

        A group's curvature is the sum of its members', which belong to
        different items, so ``se = (sum se_c ** -2) ** -0.5``. Members without
        curvature contribute nothing.
        """
        result = {
            name: np.array(values, dtype=np.float64, copy=True)
            for name, values in errors.items()
        }
        for name, flats in self.groups:
            if name not in result:
                continue
            values = result[name].reshape(-1)
            members = values[flats]
            usable = np.isfinite(members) & (members > 0.0)
            information = float(np.sum(members[usable] ** -2.0))
            values[flats] = information**-0.5 if information > 0.0 else np.nan
        return result


def resolve_equality_constraints(
    constraints: EqualityConstraints | None, model: BaseItemModel
) -> TiedCoordinates | None:
    """Resolve equality constraints to the stored coordinates of ``model``.

    Parameters
    ----------
    constraints : sequence, optional
        Groups accepted by :func:`validate_equality_constraints`.
    model : BaseItemModel
        Model to be fitted, with its final free-parameter masks.

    Returns
    -------
    TiedCoordinates or None
        The tied coordinates, or None without constraints.

    Raises
    ------
    MirtValidationError
        If a group names an unknown or shared parameter, an unknown item or
        column, a fixed coordinate, or a coordinate already tied by another
        group.
    MirtModelError
        If ``model`` is a mixed-format model.
    """
    from mirt.models.mixed_format import MixedItemModel

    groups = validate_equality_constraints(constraints)
    if not groups:
        return None
    if isinstance(model, MixedItemModel):
        raise MirtModelError(
            "equality constraints are not supported for mixed-format models",
            model_type=model.model_name,
        )
    names = list(model.item_names)
    lookup = {name: index for index, name in enumerate(names)}
    masks = model.free_parameter_masks
    tied: list[tuple[str, NDArray[np.intp]]] = []
    members: list[list[int]] = []
    owners: dict[tuple[str, int], int] = {}
    for index, group in enumerate(groups):
        name = group.parameter
        if name not in model._parameters:
            raise _constraint_error(
                f"unknown parameter {name!r}; use one of "
                + ", ".join(model._parameters),
                index,
            )
        if name in model._shared_parameters:
            raise _constraint_error(
                f"{name} is already shared by every item of {model.model_name}",
                index,
            )
        if not model._item_indexed(name):
            raise _constraint_error(f"{name} has no per-item values", index)
        positions: list[int] = []
        for item in range(model.n_items) if group.items is None else group.items:
            if isinstance(item, str):
                if item not in lookup or names.count(item) > 1:
                    raise _constraint_error(
                        f"item {item!r} does not name exactly one item", index
                    )
                item = lookup[item]
            if item >= model.n_items:
                raise _constraint_error(
                    f"item {item} is out of range for {model.n_items} items", index
                )
            positions.append(item)
        if len(set(positions)) != len(positions):
            raise _constraint_error("items lists an item more than once", index)
        shape = model._parameters[name].shape
        row_size = int(np.prod(shape[1:], dtype=np.intp))
        if group.column is not None and len(shape) == 1:
            raise _constraint_error(
                f"{name} has one value per item; omit column", index
            )
        if group.column is not None and group.column >= row_size:
            raise _constraint_error(
                f"{name} has {row_size} columns per item; column {group.column} "
                "does not exist",
                index,
            )
        free = masks[name].reshape(-1)
        hint = (
            "; give a column to tie one coordinate per item"
            if group.column is None and len(shape) > 1
            else ""
        )
        for column in range(row_size) if group.column is None else (group.column,):
            flats = np.array(
                [item * row_size + column for item in positions], dtype=np.intp
            )
            where = "" if len(shape) == 1 else f" (column {column})"
            for item, flat in zip(positions, flats, strict=True):
                if not free[flat]:
                    raise _constraint_error(
                        f"{name} of {names[item]}{where} is fixed; constraints tie "
                        f"free coordinates only{hint}",
                        index,
                    )
                other = owners.setdefault((name, int(flat)), index)
                if other != index:
                    raise _constraint_error(
                        f"{name} of {names[item]}{where} is also tied by "
                        f"constraints[{other}]; merge the two groups",
                        index,
                    )
            tied.append((name, flats))
            members.append(positions)
    return TiedCoordinates(tied, members)


def item_coordinate_keys(model: BaseItemModel, item: int) -> list[tuple[str, int]]:
    """Return ``(parameter, flat storage index)`` of an item's free coordinates.

    The order is that of
    :meth:`~mirt.estimation.base.BaseEstimator._get_item_params_and_bounds`.
    """
    masks = model.free_parameter_masks
    keys: list[tuple[str, int]] = []
    for name, values in model._parameters.items():
        if not model._item_indexed(name):
            continue
        row_size = int(np.prod(values.shape[1:], dtype=np.intp))
        columns = np.flatnonzero(np.asarray(masks[name][item]).reshape(-1))
        keys.extend((name, item * row_size + int(column)) for column in columns)
    return keys


@dataclass(frozen=True)
class TiedItemObjective:
    """One tied item's M-step terms in its free-coordinate layout.

    ``objective`` returns the negative expected log-likelihood (plus any
    negative log-prior) with its gradient, or is None for an item without
    expected responses, which then contributes nothing.
    """

    item: int
    start: NDArray[np.float64]
    bounds: list[tuple[float, float]]
    objective: _Objective | None
    constraint: LinearConstraint | None = None


def with_differenced_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    bounds: Sequence[tuple[float, float]],
) -> _Objective:
    """Add a central-difference gradient to a value-only item objective.

    Differences are one-sided at a bound, so probes stay inside the box.
    """
    lower = np.array([low for low, _ in bounds], dtype=np.float64)
    upper = np.array([high for _, high in bounds], dtype=np.float64)

    def differenced(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value = float(objective(vector))
        gradient = np.zeros(vector.size)
        steps = _DIFFERENCE_STEP * np.maximum(1.0, np.abs(vector))
        for index in range(vector.size):
            high = min(vector[index] + steps[index], upper[index])
            low = max(vector[index] - steps[index], lower[index])
            if high <= low:
                continue
            probe = vector.copy()
            probe[index] = high
            above = float(objective(probe))
            probe[index] = low
            gradient[index] = (above - float(objective(probe))) / (high - low)
        return value, gradient

    return differenced


def _diagonal_curvature(
    parts: Sequence[TiedItemObjective],
    indices: Sequence[NDArray[np.intp]],
    vector: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return positive diagonal curvatures of the summed item objectives.

    Each item's gradient is differenced along its own coordinates only, so
    this costs a few evaluations of every item objective.
    """
    curvature = np.zeros(vector.size)
    for part, index in zip(parts, indices, strict=True):
        if part.objective is None:
            continue
        center = vector[index]
        steps = 1e-4 * np.maximum(1.0, np.abs(center))
        for position, slot in enumerate(index):
            high = min(center[position] + steps[position], upper[slot])
            low = max(center[position] - steps[position], lower[slot])
            if high <= low:
                continue
            probe = center.copy()
            probe[position] = high
            above = part.objective(probe)[1][position]
            probe[position] = low
            below = part.objective(probe)[1][position]
            curvature[slot] += (above - below) / (high - low)
    positive = curvature[curvature > 0.0]
    floor = 1e-8 * positive.max() if positive.size else 1.0
    return np.where(curvature > floor, curvature, max(floor, 1e-12))


def optimize_tied_items(
    model: BaseItemModel,
    tied: TiedCoordinates,
    parts: Sequence[TiedItemObjective],
    *,
    max_iter: int,
    ftol: float,
) -> list[NDArray[np.float64]] | None:
    """Jointly optimize the items linked by equality constraints.

    The free coordinates of every tied item form one vector in which each
    tied group is a single coordinate. Item objectives are summed, so this
    is the exact M-step of the constrained items.

    Parameters
    ----------
    model : BaseItemModel
        Model whose items ``parts`` describe; it is not modified.
    tied : TiedCoordinates
        Groups of tied coordinates.
    parts : sequence of TiedItemObjective
        Objective, start and box of every item of a component in
        ``tied.components``, so that every member of a group is included.
    max_iter : int
        Iteration limit of the optimizer.
    ftol : float
        Relative function-change tolerance of the optimizer.

    Returns
    -------
    list of ndarray or None
        Each part's new free coordinates, or None when the step is infeasible
        or would raise the summed objective. Starting values that break the
        order of graded thresholds are first moved to the nearest ordered
        point, which is returned if the step fails from there.

    Raises
    ------
    MirtModelError
        If an item's free coordinates do not follow the standard layout.
    MirtEstimationError
        If no ordered thresholds lie within the parameter bounds.
    """
    slots: dict[tuple[object, ...], int] = {}
    lower: list[float] = []
    upper: list[float] = []
    start: list[float] = []
    indices: list[NDArray[np.intp]] = []
    for part in parts:
        keys = item_coordinate_keys(model, part.item)
        if len(keys) != part.start.size or len(part.bounds) != part.start.size:
            raise MirtModelError(
                "equality constraints need the standard free-coordinate layout "
                f"of item parameters, which item {part.item} does not have",
                model_type=model.model_name,
            )
        index = np.empty(len(keys), dtype=np.intp)
        for position, key in enumerate(keys):
            group = tied.group_of(*key)
            slot_key = ("coordinate", *key) if group is None else ("group", group)
            low, high = part.bounds[position]
            slot = slots.get(slot_key)
            if slot is None:
                slot = slots[slot_key] = len(start)
                start.append(float(part.start[position]))
                lower.append(low)
                upper.append(high)
            else:
                lower[slot] = max(lower[slot], low)
                upper[slot] = min(upper[slot], high)
            index[position] = slot
        indices.append(index)
    size = len(start)
    if not size:
        return None

    def objective(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        total = 0.0
        gradient = np.zeros(size)
        for part, index in zip(parts, indices, strict=True):
            if part.objective is None:
                continue
            value, item_gradient = part.objective(vector[index])
            total += float(value)
            gradient[index] += item_gradient
        return total, gradient

    rows: list[NDArray[np.float64]] = []
    limits: list[NDArray[np.float64]] = []
    for part, index in zip(parts, indices, strict=True):
        if part.constraint is None:
            continue
        matrix = np.atleast_2d(np.asarray(part.constraint.A, dtype=np.float64))
        mapped = np.zeros((matrix.shape[0], size))
        mapped[:, index] = matrix
        rows.append(mapped)
        limits.append(np.broadcast_to(part.constraint.lb, matrix.shape[0]))
    constraint = (
        LinearConstraint(np.vstack(rows), np.concatenate(limits), np.inf)
        if rows
        else None
    )
    initial = np.asarray(start)
    low, high = np.asarray(lower), np.asarray(upper)
    projected = False
    if constraint is not None and not _satisfies(constraint, initial):
        # SLSQP often fails from disordered thresholds, such as starting values
        # whose tied column breaks an item's order, so the step starts from the
        # nearest ordered point.
        initial = _nearest_feasible(initial, constraint, low, high)
        projected = True
    reference = objective(initial)[0]
    # SLSQP starts from a unit Hessian, so coordinates are rescaled by the
    # square root of their curvature; L-BFGS-B adapts its own scaling.
    scale = (
        np.ones(size)
        if constraint is None
        else np.sqrt(_diagonal_curvature(parts, indices, initial, low, high))
    )

    def optimize(scaling: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
        def scaled(vector: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            value, gradient = objective(vector / scaling)
            return value, gradient / scaling

        result = minimize(
            scaled,
            x0=initial * scaling,
            method="SLSQP" if constraint is not None else "L-BFGS-B",
            jac=True,
            bounds=list(zip(low * scaling, high * scaling, strict=True)),
            options={"maxiter": max_iter, "ftol": ftol},
            constraints=()
            if constraint is None
            else (LinearConstraint(constraint.A / scaling, constraint.lb, np.inf),),
        )
        return result.x / scaling, bool(result.success)

    def feasible_value(candidate: NDArray[np.float64]) -> float:
        if not (
            np.all(np.isfinite(candidate))
            and (constraint is None or _satisfies(constraint, candidate))
        ):
            return np.inf
        value = objective(candidate)[0]
        return value if np.isfinite(value) else np.inf

    candidate, success = optimize(scale)
    candidate_value = feasible_value(candidate)
    if constraint is not None and (
        not success
        or not np.isfinite(candidate_value)
        or np.any(constraint.A @ candidate - constraint.lb <= 1e-8)
    ):
        # SLSQP can collapse a category and report convergence after its
        # clipped probability loses its gradient. Tiny rounding differences
        # determine whether it finds this plateau. Restart from the original
        # ordered point with one common curvature scale, keeping whichever
        # feasible endpoint has the better objective. Empty categories can
        # still attain their minimum gap when that is the actual optimum.
        retry, _ = optimize(np.full(size, np.max(scale)))
        retry_value = feasible_value(retry)
        if retry_value < candidate_value:
            candidate, candidate_value = retry, retry_value
    # The M-step must not lower the expected complete-data log-likelihood.
    if not np.isfinite(candidate_value) or candidate_value > reference:
        if not projected:
            return None
        candidate = initial
    return [candidate[index] for index in indices]


def _satisfies(constraint: LinearConstraint, vector: NDArray[np.float64]) -> bool:
    """Return whether ``vector`` meets the lower limits of ``constraint``."""
    return bool(np.all(constraint.A @ vector >= constraint.lb - 1e-8))


def _nearest_feasible(
    vector: NDArray[np.float64],
    constraint: LinearConstraint,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project ``vector`` onto the box and the lower limits of ``constraint``.

    Raises
    ------
    MirtEstimationError
        If no point of the box meets the constraint.
    """
    for margin in _ORDER_MARGINS:
        result = minimize(
            lambda point: (
                float(np.sum((point - vector) ** 2)),
                2.0 * (point - vector),
            ),
            x0=np.clip(vector, lower, upper),
            method="SLSQP",
            jac=True,
            bounds=list(zip(lower, upper, strict=True)),
            constraints=(
                LinearConstraint(constraint.A, constraint.lb + margin, np.inf),
            ),
        )
        if np.all(np.isfinite(result.x)) and _satisfies(constraint, result.x):
            return np.asarray(result.x, dtype=np.float64)
    raise MirtEstimationError(
        "starting values of the items linked by equality constraints leave "
        "graded thresholds out of order, and no ordered values were found"
    )
