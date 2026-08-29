"""Exact joint assembly of parallel fixed forms.

The optimizer selects every form simultaneously so item reuse, pairwise
overlap, content, security, and cost constraints are enforced globally rather
than through order-dependent sequential assembly.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.cat.assembly import (
    FormAssemblyResult,
    _information_matrix,
    _validate_blueprint,
    _validate_costs,
    _validate_enemy_pairs,
    _validate_form_size,
    _validate_item_set,
    _validate_theta,
    _validate_weights,
)
from mirt.cat.content import ContentBlueprint

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


ParallelAssemblyMethod = Literal[
    "maximize_minimum_information",
    "target_information",
]


@dataclass(frozen=True)
class ParallelFormAssemblyResult:
    """Jointly optimized fixed forms and cross-form diagnostics.

    Attributes
    ----------
    forms : tuple[FormAssemblyResult, ...]
        Per-form selections and information diagnostics.
    objective_value : float
        Minimum weighted information across forms for the max-min objective,
        or mean weighted absolute target deviation across forms.
    method : {"maximize_minimum_information", "target_information"}
        Joint objective used for assembly.
    item_usage : dict[int, int]
        Number of forms containing each selected pool item.
    overlap_matrix : NDArray[np.intp]
        Pairwise shared-item counts. Diagonal entries equal the form size.
    solver_message : str
        Completion detail reported by the mixed-integer optimizer.
    """

    forms: tuple[FormAssemblyResult, ...]
    objective_value: float
    method: ParallelAssemblyMethod
    item_usage: dict[int, int]
    overlap_matrix: NDArray[np.intp]
    solver_message: str

    @property
    def n_forms(self) -> int:
        """Number of assembled forms."""
        return len(self.forms)

    @property
    def form_size(self) -> int:
        """Number of items on each form."""
        return self.forms[0].n_items

    def summary(self) -> str:
        """Return a compact human-readable parallel-assembly summary."""
        objective_name = (
            "minimum weighted information"
            if self.method == "maximize_minimum_information"
            else "mean weighted target deviation"
        )
        upper = self.overlap_matrix[np.triu_indices(self.n_forms, k=1)]
        max_overlap = int(np.max(upper)) if upper.size else 0
        return "\n".join(
            [
                "Parallel-form assembly",
                f"Forms: {self.n_forms}",
                f"Items per form: {self.form_size}",
                f"Objective: {objective_name} = {self.objective_value:.6g}",
                f"Maximum pairwise overlap: {max_overlap}",
            ]
        )


def _validate_positive_count(value: Any, name: str, *, minimum: int = 1) -> int:
    """Return an integer count at or above ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    count = int(value)
    if count < minimum:
        qualifier = "at least two" if minimum == 2 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _validate_target_matrix(
    target_information: ArrayLike | None,
    n_forms: int,
    n_theta: int,
) -> NDArray[np.float64] | None:
    """Return a finite non-negative form-by-theta target matrix."""
    if target_information is None:
        return None
    try:
        raw = np.asarray(target_information, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_information must contain numeric values") from exc
    if raw.ndim == 0:
        target = np.full((n_forms, n_theta), float(raw), dtype=np.float64)
    elif raw.shape == (n_theta,):
        target = np.broadcast_to(raw, (n_forms, n_theta)).copy()
    elif raw.shape == (n_forms, n_theta):
        target = raw.copy()
    else:
        raise ValueError(
            "target_information must be scalar or have shape "
            f"{(n_theta,)} or {(n_forms, n_theta)}"
        )
    if not np.all(np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("target_information must be finite and non-negative")
    return target


def _validate_max_item_usage(value: int, n_forms: int) -> int:
    """Return a valid non-anchor cross-form item-use limit."""
    usage = _validate_positive_count(value, "max_item_usage")
    if usage > n_forms:
        raise ValueError("max_item_usage cannot exceed n_forms")
    return usage


def _validate_max_overlap(
    value: int | None,
    form_size: int,
    n_required: int,
) -> int | None:
    """Return an optional pairwise shared-item limit."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("max_pairwise_overlap must be an integer")
    overlap = int(value)
    if overlap < 0:
        raise ValueError("max_pairwise_overlap must be non-negative")
    if overlap > form_size:
        raise ValueError("max_pairwise_overlap cannot exceed form_size")
    if overlap < n_required:
        raise ValueError("max_pairwise_overlap cannot be smaller than required_items")
    return overlap


def assemble_parallel_forms(
    model: BaseItemModel,
    n_forms: int,
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
    max_item_usage: int = 1,
    max_pairwise_overlap: int | None = None,
    solver_options: Mapping[str, bool | int | float] | None = None,
) -> ParallelFormAssemblyResult:
    """Jointly assemble exact parallel forms from one calibrated item pool.

    Without an information target, the optimizer maximizes the minimum
    weighted information across forms. This max-min objective prevents one
    form from receiving all of the strongest items. Supplying a common or
    form-specific target instead minimizes mean weighted absolute deviation.

    Non-required items may appear on at most ``max_item_usage`` forms; the
    default therefore produces disjoint non-anchor sets. ``required_items``
    are common anchors placed on every form and are exempt from that use limit.
    ``max_pairwise_overlap`` counts anchors and all other shared items.

    Parameters
    ----------
    model : BaseItemModel
        Unidimensional model defining the item pool.
    n_forms : int
        Number of forms to assemble; must be at least two.
    form_size : int
        Exact number of items on every form.
    theta : array-like
        Ability grid for evaluating item information.
    theta_weights : array-like, optional
        Non-negative objective weights, normalized to sum to one.
    target_information : array-like, optional
        Scalar, common ``(n_theta,)`` curve, or form-specific
        ``(n_forms, n_theta)`` target matrix.
    blueprint : ContentBlueprint, optional
        Content-area bounds applied independently to every form.
    candidate_items : collection of int, optional
        Pool subset eligible for selection. Defaults to every item.
    required_items : collection of int, optional
        Common anchor items placed on every form.
    excluded_items : collection of int, optional
        Items removed from the candidate pool.
    enemy_pairs : collection of tuple[int, int], optional
        Item pairs that may not appear together on any form.
    item_costs : array-like, optional
        Non-negative cost for every model item.
    max_cost : float, optional
        Per-form maximum total cost.
    max_item_usage : int, default=1
        Maximum forms containing any non-required item.
    max_pairwise_overlap : int, optional
        Maximum shared items between every pair of forms.
    solver_options : mapping, optional
        Options forwarded to :func:`scipy.optimize.milp`.

    Returns
    -------
    ParallelFormAssemblyResult
        Joint optimum and per-form/cross-form diagnostics.

    Raises
    ------
    RuntimeError
        If the joint constraints have no optimal feasible solution.
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
        raise ValueError("assemble_parallel_forms requires a unidimensional model")

    form_count = _validate_positive_count(n_forms, "n_forms", minimum=2)
    theta_values = _validate_theta(theta)
    weights = _validate_weights(theta_weights, theta_values.size)
    targets = _validate_target_matrix(
        target_information,
        form_count,
        theta_values.size,
    )
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
    usage_limit = _validate_max_item_usage(max_item_usage, form_count)
    overlap_limit = _validate_max_overlap(
        max_pairwise_overlap,
        size,
        len(required),
    )
    needed_nonanchors = form_count * (size - len(required))
    available_nonanchors = (len(candidates) - len(required)) * usage_limit
    if available_nonanchors < needed_nonanchors:
        raise ValueError(
            "max_item_usage leaves insufficient candidate capacity for all forms"
        )

    candidate_indices = np.asarray(sorted(candidates), dtype=np.intp)
    item_positions = {
        int(item_idx): position
        for position, item_idx in enumerate(candidate_indices.tolist())
    }
    information = _information_matrix(model, theta_values, pool_size)
    candidate_information = information[:, candidate_indices]
    n_candidates = int(candidate_indices.size)
    n_selection = form_count * n_candidates
    form_pairs = list(combinations(range(form_count), 2))

    n_objective = 1 if targets is None else form_count * theta_values.size
    objective_offset = n_selection
    overlap_offset = objective_offset + n_objective
    n_overlap = len(form_pairs) * n_candidates if overlap_limit is not None else 0
    n_variables = n_selection + n_objective + n_overlap

    def selection_column(form_idx: int, item_position: int) -> int:
        return form_idx * n_candidates + item_position

    def deviation_column(form_idx: int, theta_idx: int) -> int:
        return objective_offset + form_idx * theta_values.size + theta_idx

    def overlap_column(pair_idx: int, item_position: int) -> int:
        return overlap_offset + pair_idx * n_candidates + item_position

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

    for form_idx in range(form_count):
        add_constraint(
            [
                (selection_column(form_idx, position), 1.0)
                for position in range(n_candidates)
            ],
            size,
            size,
        )
        if content is not None:
            for area in content.areas:
                add_constraint(
                    [
                        (
                            selection_column(form_idx, item_positions[item_idx]),
                            1.0,
                        )
                        for item_idx in area.items
                        if item_idx in item_positions
                    ],
                    area.min_items,
                    area.max_items,
                )
        for first, second in pairs:
            if first in item_positions and second in item_positions:
                add_constraint(
                    [
                        (
                            selection_column(form_idx, item_positions[first]),
                            1.0,
                        ),
                        (
                            selection_column(form_idx, item_positions[second]),
                            1.0,
                        ),
                    ],
                    -np.inf,
                    1.0,
                )
        if budget is not None:
            assert costs is not None
            add_constraint(
                [
                    (
                        selection_column(form_idx, position),
                        float(costs[item_idx]),
                    )
                    for position, item_idx in enumerate(candidate_indices)
                ],
                -np.inf,
                budget,
            )

    for item_position, item_idx in enumerate(candidate_indices):
        if int(item_idx) not in required:
            add_constraint(
                [
                    (selection_column(form_idx, item_position), 1.0)
                    for form_idx in range(form_count)
                ],
                -np.inf,
                usage_limit,
            )

    if overlap_limit is not None:
        for pair_idx, (first_form, second_form) in enumerate(form_pairs):
            overlap_entries: list[tuple[int, float]] = []
            for item_position in range(n_candidates):
                shared_column = overlap_column(pair_idx, item_position)
                add_constraint(
                    [
                        (selection_column(first_form, item_position), 1.0),
                        (selection_column(second_form, item_position), 1.0),
                        (shared_column, -1.0),
                    ],
                    -np.inf,
                    1.0,
                )
                overlap_entries.append((shared_column, 1.0))
            add_constraint(overlap_entries, -np.inf, overlap_limit)

    objective = np.zeros(n_variables, dtype=np.float64)
    if targets is None:
        item_utility = np.asarray(weights @ candidate_information, dtype=np.float64)
        minimum_column = objective_offset
        objective[minimum_column] = -1.0
        for form_idx in range(form_count):
            add_constraint(
                [(minimum_column, 1.0)]
                + [
                    (selection_column(form_idx, position), -float(utility))
                    for position, utility in enumerate(item_utility)
                ],
                -np.inf,
                0.0,
            )
        method: ParallelAssemblyMethod = "maximize_minimum_information"
    else:
        for form_idx in range(form_count):
            for theta_idx in range(theta_values.size):
                deviation = deviation_column(form_idx, theta_idx)
                objective[deviation] = weights[theta_idx] / form_count
                information_entries = [
                    (
                        selection_column(form_idx, position),
                        float(candidate_information[theta_idx, position]),
                    )
                    for position in range(n_candidates)
                ]
                add_constraint(
                    information_entries + [(deviation, -1.0)],
                    -np.inf,
                    float(targets[form_idx, theta_idx]),
                )
                add_constraint(
                    [
                        (column, -coefficient)
                        for column, coefficient in information_entries
                    ]
                    + [(deviation, -1.0)],
                    -np.inf,
                    -float(targets[form_idx, theta_idx]),
                )
        method = "target_information"

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    constraint_matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(lower_bounds), n_variables),
        dtype=np.float64,
    ).tocsr()
    variable_lower = np.zeros(n_variables, dtype=np.float64)
    variable_upper = np.concatenate(
        (
            np.ones(n_selection, dtype=np.float64),
            np.full(n_objective, np.inf, dtype=np.float64),
            np.ones(n_overlap, dtype=np.float64),
        )
    )
    for item_idx in required:
        item_position = item_positions[item_idx]
        for form_idx in range(form_count):
            column = selection_column(form_idx, item_position)
            variable_lower[column] = 1.0
            variable_upper[column] = 1.0

    if solver_options is not None and not isinstance(solver_options, Mapping):
        raise TypeError("solver_options must be a mapping")
    options = dict(solver_options) if solver_options is not None else {"presolve": True}
    result = milp(
        c=objective,
        integrality=np.concatenate(
            (
                np.ones(n_selection, dtype=np.int8),
                np.zeros(n_objective + n_overlap, dtype=np.int8),
            )
        ),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            constraint_matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"parallel form assembly failed: {result.message}")

    selected_mask = result.x[:n_selection].reshape(form_count, n_candidates) > 0.5
    selected_by_form = tuple(
        candidate_indices[np.flatnonzero(selected_mask[form_idx])]
        for form_idx in range(form_count)
    )
    if any(selected.size != size for selected in selected_by_form):
        raise RuntimeError("parallel form assembly returned an invalid item count")

    information_by_form = np.stack(
        [np.sum(information[:, selected], axis=1) for selected in selected_by_form]
    )
    weighted_information = information_by_form @ weights
    if targets is None:
        joint_objective = float(np.min(weighted_information))
    else:
        deviations = np.abs(information_by_form - targets)
        joint_objective = float(np.mean(deviations @ weights))

    per_form_results: list[FormAssemblyResult] = []
    for form_idx, selected in enumerate(selected_by_form):
        form_information = information_by_form[form_idx]
        if targets is None:
            form_objective = float(weighted_information[form_idx])
            form_method: Literal["maximize_information", "target_information"] = (
                "maximize_information"
            )
        else:
            form_objective = float(
                weights @ np.abs(form_information - targets[form_idx])
            )
            form_method = "target_information"
        content_counts = (
            {
                area.name: len(set(selected.tolist()) & area.items)
                for area in content.areas
            }
            if content is not None
            else {}
        )
        total_cost = float(np.sum(costs[selected])) if costs is not None else None
        per_form_results.append(
            FormAssemblyResult(
                selected_items=np.asarray(selected, dtype=np.intp),
                theta=theta_values.copy(),
                information=np.asarray(form_information, dtype=np.float64),
                objective_value=form_objective,
                method=form_method,
                content_counts=content_counts,
                total_cost=total_cost,
                solver_message=str(result.message),
            )
        )

    usage_counts = np.sum(selected_mask, axis=0, dtype=np.intp)
    item_usage = {
        int(candidate_indices[position]): int(count)
        for position, count in enumerate(usage_counts)
        if count > 0
    }
    overlap_matrix = np.asarray(
        selected_mask.astype(np.intp) @ selected_mask.astype(np.intp).T,
        dtype=np.intp,
    )
    return ParallelFormAssemblyResult(
        forms=tuple(per_form_results),
        objective_value=joint_objective,
        method=method,
        item_usage=item_usage,
        overlap_matrix=overlap_matrix,
        solver_message=str(result.message),
    )
