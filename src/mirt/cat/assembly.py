"""Constrained fixed-form assembly from an IRT item pool.

The optimizer selects an exact item set while honoring content, security, and
cost constraints.  It can maximize weighted Fisher information or minimize
absolute deviation from a target information curve.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.cat.content import ContentBlueprint

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


AssemblyMethod = Literal["maximize_information", "target_information"]


@dataclass(frozen=True)
class FormAssemblyResult:
    """Optimal fixed-form assembly and its diagnostics.

    Attributes
    ----------
    selected_items : NDArray[np.intp]
        Sorted zero-based indices of selected items.
    theta : NDArray[np.float64]
        Ability grid used by the objective.
    information : NDArray[np.float64]
        Assembled test-information curve on ``theta``.
    objective_value : float
        Weighted mean information when maximizing information, or weighted
        mean absolute deviation when matching a target curve.
    method : {"maximize_information", "target_information"}
        Objective used for assembly.
    content_counts : dict[str, int]
        Selected-item counts for each requested content area.
    total_cost : float | None
        Sum of selected item costs when costs were supplied.
    solver_message : str
        Completion detail reported by the mixed-integer optimizer.
    """

    selected_items: NDArray[np.intp]
    theta: NDArray[np.float64]
    information: NDArray[np.float64]
    objective_value: float
    method: AssemblyMethod
    content_counts: dict[str, int]
    total_cost: float | None
    solver_message: str

    @property
    def n_items(self) -> int:
        """Number of items in the assembled form."""
        return int(self.selected_items.size)

    def summary(self) -> str:
        """Return a compact human-readable assembly summary."""
        objective_name = (
            "weighted information"
            if self.method == "maximize_information"
            else "weighted target deviation"
        )
        lines = [
            "Fixed-form assembly",
            f"Items: {self.n_items}",
            f"Objective: {objective_name} = {self.objective_value:.6g}",
        ]
        if self.total_cost is not None:
            lines.append(f"Total cost: {self.total_cost:.6g}")
        if self.content_counts:
            counts = ", ".join(
                f"{name}={count}" for name, count in self.content_counts.items()
            )
            lines.append(f"Content: {counts}")
        return "\n".join(lines)


def _validate_form_size(value: int, pool_size: int) -> int:
    """Return a positive fixed-form length."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("form_size must be an integer")
    size = int(value)
    if size < 1:
        raise ValueError("form_size must be positive")
    if size > pool_size:
        raise ValueError("form_size cannot exceed the number of candidate items")
    return size


def _validate_item_index(value: Any, pool_size: int, name: str) -> int:
    """Return a valid zero-based pool index."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must contain integer item indices")
    index = int(value)
    if index < 0 or index >= pool_size:
        raise ValueError(f"{name} contains an item index outside the model")
    return index


def _validate_item_set(
    values: Collection[int] | None,
    pool_size: int,
    name: str,
) -> set[int]:
    """Validate a collection of unique pool indices."""
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a collection of item indices")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection of item indices") from exc
    indices = {_validate_item_index(value, pool_size, name) for value in raw_values}
    if len(indices) != len(raw_values):
        raise ValueError(f"{name} must not contain duplicate item indices")
    return indices


def _validate_theta(theta: ArrayLike) -> NDArray[np.float64]:
    """Return a non-empty finite unidimensional ability grid."""
    try:
        values = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise ValueError("theta must contain numeric values") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain only finite values")
    return values


def _validate_weights(
    theta_weights: ArrayLike | None,
    n_theta: int,
) -> NDArray[np.float64]:
    """Return normalized non-negative objective weights."""
    if theta_weights is None:
        return np.full(n_theta, 1.0 / n_theta, dtype=np.float64)
    try:
        weights = np.asarray(theta_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta_weights must contain numeric values") from exc
    if weights.shape != (n_theta,):
        raise ValueError(f"theta_weights must have shape {(n_theta,)}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("theta_weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("theta_weights must have positive mass")
    return np.asarray(weights / total, dtype=np.float64)


def _validate_target(
    target_information: ArrayLike | None,
    n_theta: int,
) -> NDArray[np.float64] | None:
    """Broadcast and validate an optional information target."""
    if target_information is None:
        return None
    try:
        raw = np.asarray(target_information, dtype=np.float64)
        target = np.broadcast_to(raw, (n_theta,)).astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"target_information must be scalar or have shape {(n_theta,)}"
        ) from exc
    if not np.all(np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("target_information must be finite and non-negative")
    return target


def _validate_costs(
    item_costs: ArrayLike | None,
    pool_size: int,
    max_cost: float | None,
) -> tuple[NDArray[np.float64] | None, float | None]:
    """Validate optional pool costs and a form budget."""
    if item_costs is None:
        if max_cost is not None:
            raise ValueError("item_costs are required when max_cost is supplied")
        return None, None
    try:
        costs = np.asarray(item_costs, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("item_costs must contain numeric values") from exc
    if costs.shape != (pool_size,):
        raise ValueError(f"item_costs must have shape {(pool_size,)}")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("item_costs must be finite and non-negative")
    if max_cost is None:
        return costs, None
    if isinstance(max_cost, (bool, np.bool_)) or not isinstance(max_cost, Real):
        raise ValueError("max_cost must be a finite non-negative number")
    budget = float(max_cost)
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError("max_cost must be a finite non-negative number")
    return costs, budget


def _validate_enemy_pairs(
    enemy_pairs: Collection[tuple[int, int]] | None,
    pool_size: int,
) -> list[tuple[int, int]]:
    """Return unique unordered pairs that may not appear together."""
    if enemy_pairs is None:
        return []
    normalized: set[tuple[int, int]] = set()
    for pair in enemy_pairs:
        try:
            members = tuple(pair)
        except TypeError as exc:
            raise ValueError("enemy_pairs must contain two-item pairs") from exc
        if len(members) != 2:
            raise ValueError("enemy_pairs must contain two-item pairs")
        first = _validate_item_index(members[0], pool_size, "enemy_pairs")
        second = _validate_item_index(members[1], pool_size, "enemy_pairs")
        if first == second:
            raise ValueError("an enemy pair must contain two distinct items")
        normalized.add((min(first, second), max(first, second)))
    return sorted(normalized)


def _information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    pool_size: int,
) -> NDArray[np.float64]:
    """Evaluate a validated theta-by-item information matrix."""
    theta_matrix = theta[:, None]
    raw = np.asarray(model.information(theta_matrix), dtype=np.float64)
    expected_shape = (theta.size, pool_size)
    if raw.shape != expected_shape:
        raw = np.column_stack(
            [
                np.asarray(model.information(theta_matrix, item_idx=item_idx))
                for item_idx in range(pool_size)
            ]
        ).astype(np.float64, copy=False)
    if raw.shape != expected_shape:
        raise ValueError(
            f"model information has shape {raw.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("model information must contain only finite values")
    if np.any(raw < -1e-12):
        raise ValueError("model information must be non-negative")
    return np.clip(raw, 0.0, None)


def _validate_blueprint(
    blueprint: ContentBlueprint | None,
    pool_size: int,
) -> ContentBlueprint | None:
    """Validate that content definitions refer to this item pool."""
    if blueprint is None:
        return None
    if not isinstance(blueprint, ContentBlueprint):
        raise TypeError("blueprint must be a ContentBlueprint")
    for area in blueprint.areas:
        if any(item_idx >= pool_size for item_idx in area.items):
            raise ValueError(
                f"content area {area.name!r} contains an item outside the model"
            )
    return blueprint


def assemble_form(
    model: BaseItemModel,
    form_size: int,
    theta: ArrayLike,
    *,
    theta_weights: ArrayLike | None = None,
    target_information: ArrayLike | None = None,
    blueprint: ContentBlueprint | None = None,
    candidate_items: Collection[int] | None = None,
    required_items: Collection[int] | None = None,
    excluded_items: Collection[int] | None = None,
    enemy_pairs: Collection[tuple[int, int]] | None = None,
    item_costs: ArrayLike | None = None,
    max_cost: float | None = None,
    solver_options: Mapping[str, bool | int | float] | None = None,
) -> FormAssemblyResult:
    """Assemble an optimal constrained fixed form.

    The default objective maximizes a weighted mean of test information over
    ``theta``.  Supplying ``target_information`` instead minimizes weighted
    absolute deviation between the assembled and requested curves.  Content
    area minima and maxima are always hard constraints; ``target_items`` on a
    :class:`~mirt.cat.content.ContentArea` remains descriptive.

    Parameters
    ----------
    model : BaseItemModel
        Unidimensional model defining the item pool.
    form_size : int
        Exact number of items to select.
    theta : array-like
        Ability grid for evaluating item information.
    theta_weights : array-like, optional
        Non-negative objective weights. Values are normalized to sum to one.
    target_information : array-like, optional
        Desired test-information curve. A scalar is broadcast over ``theta``.
    blueprint : ContentBlueprint, optional
        Content-area minimum and maximum constraints.
    candidate_items : collection of int, optional
        Pool subset eligible for selection. Defaults to every item.
    required_items : collection of int, optional
        Items that must appear in the form.
    excluded_items : collection of int, optional
        Items removed from the candidate pool.
    enemy_pairs : collection of tuple[int, int], optional
        Item pairs that may not appear in the same form.
    item_costs : array-like, optional
        Non-negative cost for every model item.
    max_cost : float, optional
        Maximum total selected-item cost.
    solver_options : mapping, optional
        Options forwarded to :func:`scipy.optimize.milp`, such as
        ``time_limit`` or ``mip_rel_gap``.

    Returns
    -------
    FormAssemblyResult
        Optimal selected items and assembly diagnostics.

    Raises
    ------
    RuntimeError
        If the requested constraints have no optimal feasible form.

    Examples
    --------
    >>> theta = np.linspace(-2.0, 2.0, 21)
    >>> result = assemble_form(model, 20, theta)
    >>> result.selected_items
    array([...])
    """
    pool_size = getattr(model, "n_items", None)
    if (
        isinstance(pool_size, (bool, np.bool_))
        or not isinstance(pool_size, Integral)
        or int(pool_size) < 1
    ):
        raise ValueError("model must expose a positive integer n_items")
    pool_size = int(pool_size)
    if getattr(model, "n_factors", None) != 1:
        raise ValueError("assemble_form requires a unidimensional model")

    theta_values = _validate_theta(theta)
    weights = _validate_weights(theta_weights, theta_values.size)
    target = _validate_target(target_information, theta_values.size)
    content = _validate_blueprint(blueprint, pool_size)
    costs, budget = _validate_costs(item_costs, pool_size, max_cost)
    pairs = _validate_enemy_pairs(enemy_pairs, pool_size)

    if candidate_items is None:
        candidates = set(range(pool_size))
    else:
        candidates = _validate_item_set(candidate_items, pool_size, "candidate_items")
    required = _validate_item_set(required_items, pool_size, "required_items")
    excluded = _validate_item_set(excluded_items, pool_size, "excluded_items")
    if required & excluded:
        raise ValueError("required_items and excluded_items must be disjoint")
    if not required <= candidates:
        raise ValueError("required_items must be included in candidate_items")
    candidates.difference_update(excluded)
    size = _validate_form_size(form_size, len(candidates))
    if len(required) > size:
        raise ValueError("form_size cannot be smaller than required_items")

    candidate_indices = np.asarray(sorted(candidates), dtype=np.intp)
    item_positions = {
        int(item_idx): position
        for position, item_idx in enumerate(candidate_indices.tolist())
    }
    information = _information_matrix(model, theta_values, pool_size)
    candidate_information = information[:, candidate_indices]
    n_candidates = candidate_indices.size
    n_deviations = theta_values.size if target is not None else 0
    n_variables = n_candidates + n_deviations

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(
        entries: Collection[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(lower_bounds)
        for column, coefficient in entries:
            if coefficient != 0.0:
                row_indices.append(row)
                column_indices.append(column)
                coefficients.append(float(coefficient))
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))

    add_constraint([(position, 1.0) for position in range(n_candidates)], size, size)

    if content is not None:
        for area in content.areas:
            entries = [
                (item_positions[item_idx], 1.0)
                for item_idx in area.items
                if item_idx in item_positions
            ]
            add_constraint(entries, area.min_items, area.max_items)

    for first, second in pairs:
        if first in item_positions and second in item_positions:
            add_constraint(
                [(item_positions[first], 1.0), (item_positions[second], 1.0)],
                -np.inf,
                1.0,
            )

    if budget is not None:
        assert costs is not None
        add_constraint(
            [
                (position, float(costs[item_idx]))
                for position, item_idx in enumerate(candidate_indices)
            ],
            -np.inf,
            budget,
        )

    base_matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(lower_bounds), n_variables),
        dtype=np.float64,
    ).tocsr()
    constraint_matrix = base_matrix
    constraint_lower = np.asarray(lower_bounds, dtype=np.float64)
    constraint_upper = np.asarray(upper_bounds, dtype=np.float64)

    if target is None:
        item_utility = weights @ candidate_information
        objective = -np.asarray(item_utility, dtype=np.float64)
        method: AssemblyMethod = "maximize_information"
    else:
        information_block = csr_matrix(candidate_information)
        negative_identity = -eye(theta_values.size, format="csr")
        positive_deviation = hstack(
            (information_block, negative_identity), format="csr"
        )
        negative_deviation = hstack(
            (-information_block, negative_identity), format="csr"
        )
        constraint_matrix = vstack(
            (base_matrix, positive_deviation, negative_deviation), format="csr"
        )
        constraint_lower = np.concatenate(
            (
                constraint_lower,
                np.full(2 * theta_values.size, -np.inf),
            )
        )
        constraint_upper = np.concatenate((constraint_upper, target, -target))
        objective = np.concatenate((np.zeros(n_candidates), weights))
        method = "target_information"

    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.concatenate(
        (
            np.ones(n_candidates, dtype=np.float64),
            np.full(n_deviations, np.inf, dtype=np.float64),
        )
    )
    for item_idx in required:
        position = item_positions[item_idx]
        variable_lower[position] = 1.0
        variable_upper[position] = 1.0

    if solver_options is not None and not isinstance(solver_options, Mapping):
        raise TypeError("solver_options must be a mapping")
    options = dict(solver_options) if solver_options is not None else {"presolve": True}
    result = milp(
        c=objective,
        integrality=np.concatenate(
            (
                np.ones(n_candidates, dtype=np.int8),
                np.zeros(n_deviations, dtype=np.int8),
            )
        ),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            constraint_matrix,
            constraint_lower,
            constraint_upper,
        ),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"form assembly failed: {result.message}")

    selected_positions = np.flatnonzero(result.x[:n_candidates] > 0.5)
    selected_items = candidate_indices[selected_positions]
    if selected_items.size != size:
        raise RuntimeError("form assembly returned an invalid item count")
    assembled_information = np.sum(information[:, selected_items], axis=1)
    if target is None:
        objective_value = float(weights @ assembled_information)
    else:
        objective_value = float(weights @ np.abs(assembled_information - target))

    content_counts = (
        {
            area.name: len(set(selected_items.tolist()) & area.items)
            for area in content.areas
        }
        if content is not None
        else {}
    )
    total_cost = float(np.sum(costs[selected_items])) if costs is not None else None

    return FormAssemblyResult(
        selected_items=np.asarray(selected_items, dtype=np.intp),
        theta=theta_values.copy(),
        information=np.asarray(assembled_information, dtype=np.float64),
        objective_value=objective_value,
        method=method,
        content_counts=content_counts,
        total_cost=total_cost,
        solver_message=str(result.message),
    )
