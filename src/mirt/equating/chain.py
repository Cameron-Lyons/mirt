"""Chain linking across multiple time points.

This module provides functions for linking IRT models across
multiple time points or administrations using pairwise linking.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.equating.linking import LinkingResult, link

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10
_CONCURRENT_METHODS = frozenset({"stocking_lord", "tcc", "haebara"})


@dataclass
class ChainLinkingResult:
    """Result of chain linking across multiple time points.

    Attributes
    ----------
    cumulative_A : list[float]
        Cumulative slope transformations to reference scale.
    cumulative_B : list[float]
        Cumulative intercept transformations to reference scale.
    pairwise_results : list[LinkingResult]
        Results from each pairwise linking.
    drift_accumulation : NDArray[np.float64] | None
        Accumulated drift statistics.
    reference_index : int
        Index of reference time point.
    """

    cumulative_A: list[float]
    cumulative_B: list[float]
    pairwise_results: list[LinkingResult]
    drift_accumulation: NDArray[np.float64] | None
    reference_index: int


@dataclass
class TimePointModel:
    """Model and anchor information for a time point.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model for this time point.
    anchor_items : list[int]
        Indices of anchor items in this model.
    time_label : str
        Label for this time point.
    """

    model: "BaseItemModel"
    anchor_items: list[int]
    time_label: str = ""


@dataclass
class _ItemCurveSet:
    """Expected-score and category curves for selected items."""

    expected: NDArray[np.float64]
    categories: list[NDArray[np.float64]] | None


def _validate_grid(theta_range: tuple[float, float], n_theta: int) -> None:
    """Validate a unidimensional evaluation grid."""
    if len(theta_range) != 2:
        raise ValueError("theta_range must contain exactly two bounds")
    lower, upper = theta_range
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite increasing bounds")
    if isinstance(n_theta, (bool, np.bool_)) or not isinstance(
        n_theta, (int, np.integer)
    ):
        raise ValueError("n_theta must be an integer")
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")


def _validate_anchor_pair(
    anchors_left: list[int],
    anchors_right: list[int],
    n_items_left: int,
    n_items_right: int,
    *,
    name: str,
) -> None:
    """Validate paired item indices for two forms."""
    if len(anchors_left) != len(anchors_right):
        raise ValueError(f"{name} must contain equally sized anchor lists")
    if len(anchors_left) < 2:
        raise ValueError(f"{name} must contain at least two anchor pairs")

    for side, anchors, n_items in (
        ("left", anchors_left, n_items_left),
        ("right", anchors_right, n_items_right),
    ):
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in anchors
        ):
            raise ValueError(f"{name} {side} indices must be integers")
        if any(index < 0 or index >= n_items for index in anchors):
            raise ValueError(f"{name} contains an out-of-range {side} item index")
        if len(set(int(index) for index in anchors)) != len(anchors):
            raise ValueError(f"{name} contains duplicate {side} item indices")


def _time_point_constants(
    result: ChainLinkingResult,
    time_index: int,
) -> tuple[float, float]:
    """Return validated cumulative constants for one time point."""
    if len(result.cumulative_A) != len(result.cumulative_B):
        raise ValueError("cumulative_A and cumulative_B must have the same length")
    if not result.cumulative_A:
        raise ValueError("chain result contains no time points")
    if isinstance(time_index, (bool, np.bool_)) or not isinstance(
        time_index, (int, np.integer)
    ):
        raise ValueError("time_index must be an integer")
    if time_index < 0 or time_index >= len(result.cumulative_A):
        raise ValueError(f"Invalid time_index: {time_index}")

    slope = float(result.cumulative_A[time_index])
    intercept = float(result.cumulative_B[time_index])
    if not np.isfinite(slope) or slope <= 0.0 or not np.isfinite(intercept):
        raise ValueError("chain result contains invalid transformation constants")
    return slope, intercept


def chain_link(
    models: list["BaseItemModel"],
    anchor_item_pairs: list[tuple[list[int], list[int]]],
    method: str = "stocking_lord",
    reference_index: int = 0,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    compute_drift: bool = True,
) -> ChainLinkingResult:
    """Perform chain linking across multiple time points.

    Links a sequence of models to a common reference scale by
    accumulating pairwise transformations.

    Parameters
    ----------
    models : list[BaseItemModel]
        List of models in temporal order.
    anchor_item_pairs : list[tuple[list[int], list[int]]]
        List of (anchors_t, anchors_t+1) pairs for consecutive models.
        Length should be len(models) - 1.
    method : str
        Linking method for pairwise linking.
    reference_index : int
        Index of reference model (default 0 = first).
    theta_range : tuple[float, float]
        Range for curve matching methods.
    n_theta : int
        Number of theta points.
    compute_drift : bool
        Whether to track drift accumulation.

    Returns
    -------
    ChainLinkingResult
        Cumulative transformations and pairwise results.

    Examples
    --------
    >>> result = chain_link(
    ...     models=[model_t1, model_t2, model_t3],
    ...     anchor_item_pairs=[
    ...         ([0, 1, 2], [0, 1, 2]),  # t1 -> t2
    ...         ([0, 1, 2], [0, 1, 2]),  # t2 -> t3
    ...     ],
    ...     reference_index=0,
    ... )
    >>> # Transform t3 parameters to t1 scale
    >>> A_t3_to_t1 = result.cumulative_A[2]
    >>> B_t3_to_t1 = result.cumulative_B[2]
    """
    n_models = len(models)

    if n_models == 0:
        raise ValueError("models must contain at least one model")

    if len(anchor_item_pairs) != n_models - 1:
        raise ValueError(
            f"Expected {n_models - 1} anchor pairs, got {len(anchor_item_pairs)}"
        )

    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(
        reference_index, (int, np.integer)
    ):
        raise ValueError("reference_index must be an integer")
    if reference_index < 0 or reference_index >= n_models:
        raise ValueError(f"Invalid reference_index: {reference_index}")

    _validate_grid(theta_range, n_theta)

    for pair_index, (anchors_left, anchors_right) in enumerate(anchor_item_pairs):
        _validate_anchor_pair(
            anchors_left,
            anchors_right,
            models[pair_index].n_items,
            models[pair_index + 1].n_items,
            name=f"anchor_item_pairs[{pair_index}]",
        )

    pairwise_results: list[LinkingResult] = []

    for t in range(n_models - 1):
        anchors_t, anchors_t1 = anchor_item_pairs[t]

        result = link(
            models[t],
            models[t + 1],
            anchors_t,
            anchors_t1,
            method=method,
            theta_range=theta_range,
            n_theta=n_theta,
            compute_diagnostics=True,
        )
        pairwise_results.append(result)

    pairwise_A = [r.constants.A for r in pairwise_results]
    pairwise_B = [r.constants.B for r in pairwise_results]

    cumulative_A, cumulative_B = accumulate_constants(
        pairwise_A, pairwise_B, reference_index
    )

    drift_accumulation = None
    if compute_drift:
        drift_accumulation = _compute_drift_accumulation(pairwise_results)

    return ChainLinkingResult(
        cumulative_A=cumulative_A,
        cumulative_B=cumulative_B,
        pairwise_results=pairwise_results,
        drift_accumulation=drift_accumulation,
        reference_index=reference_index,
    )


def accumulate_constants(
    pairwise_A: list[float],
    pairwise_B: list[float],
    reference_index: int = 0,
) -> tuple[list[float], list[float]]:
    """Accumulate pairwise constants to reference scale.

    For transformations A_t, B_t from time t to t+1:
        theta_ref = A_cum * theta_t + B_cum

    Parameters
    ----------
    pairwise_A : list[float]
        Pairwise slope constants.
    pairwise_B : list[float]
        Pairwise intercept constants.
    reference_index : int
        Index of reference time point.

    Returns
    -------
    tuple[list[float], list[float]]
        Cumulative A and B for each time point.
    """
    slopes = np.asarray(pairwise_A, dtype=np.float64)
    intercepts = np.asarray(pairwise_B, dtype=np.float64)
    if slopes.ndim != 1 or intercepts.ndim != 1:
        raise ValueError("pairwise constants must be one-dimensional")
    if slopes.shape != intercepts.shape:
        raise ValueError("pairwise_A and pairwise_B must have the same length")
    if not np.all(np.isfinite(slopes)) or np.any(slopes <= 0.0):
        raise ValueError("pairwise_A must contain finite positive values")
    if not np.all(np.isfinite(intercepts)):
        raise ValueError("pairwise_B must contain only finite values")

    n_models = len(slopes) + 1
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(
        reference_index, (int, np.integer)
    ):
        raise ValueError("reference_index must be an integer")
    if reference_index < 0 or reference_index >= n_models:
        raise ValueError(f"Invalid reference_index: {reference_index}")

    cumulative_A = [1.0] * n_models
    cumulative_B = [0.0] * n_models

    for t in range(reference_index - 1, -1, -1):
        next_A = cumulative_A[t + 1]
        next_B = cumulative_B[t + 1]
        cumulative_A[t] = float(next_A * slopes[t])
        cumulative_B[t] = float(next_A * intercepts[t] + next_B)

    for t in range(reference_index + 1, n_models):
        previous_A = cumulative_A[t - 1]
        previous_B = cumulative_B[t - 1]
        cumulative_A[t] = float(previous_A / slopes[t - 1])
        cumulative_B[t] = float(
            previous_B - previous_A * intercepts[t - 1] / slopes[t - 1]
        )

    return cumulative_A, cumulative_B


def _compute_drift_accumulation(
    pairwise_results: list[LinkingResult],
) -> NDArray[np.float64]:
    """Compute accumulated drift across time points."""
    n_pairs = len(pairwise_results)

    max_anchors = max(
        (
            len(result.anchor_diagnostics.robust_z)
            for result in pairwise_results
            if result.anchor_diagnostics is not None
        ),
        default=0,
    )

    drift_matrix = np.full((n_pairs, max_anchors), np.nan, dtype=np.float64)

    for t, result in enumerate(pairwise_results):
        if result.anchor_diagnostics is not None:
            n_anchors = len(result.anchor_diagnostics.robust_z)
            drift_matrix[t, :n_anchors] = result.anchor_diagnostics.robust_z

    return drift_matrix


def transform_to_reference(
    model: "BaseItemModel",
    chain_result: ChainLinkingResult,
    time_index: int,
    in_place: bool = False,
) -> "BaseItemModel":
    """Transform model parameters to reference scale.

    Parameters
    ----------
    model : BaseItemModel
        Model to transform.
    chain_result : ChainLinkingResult
        Chain linking result.
    time_index : int
        Index of model's time point.
    in_place : bool
        Modify in place or return copy.

    Returns
    -------
    BaseItemModel
        Model on reference scale.
    """
    from mirt.equating.linking import transform_parameters

    A, B = _time_point_constants(chain_result, time_index)

    return transform_parameters(model, A, B, in_place=in_place)


def transform_theta_to_reference(
    theta: NDArray[np.float64],
    chain_result: ChainLinkingResult,
    time_index: int,
) -> NDArray[np.float64]:
    """Transform theta estimates to reference scale.

    Parameters
    ----------
    theta : NDArray
        Theta estimates from time point.
    chain_result : ChainLinkingResult
        Chain linking result.
    time_index : int
        Index of time point.

    Returns
    -------
    NDArray
        Theta on reference scale.
    """
    A, B = _time_point_constants(chain_result, time_index)

    theta_array = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values")

    return A * theta_array + B


def _prepare_concurrent_relations(
    models: list["BaseItemModel"],
    anchor_matrices: list[list[list[tuple[int, int]]]],
    method: str,
) -> tuple[
    list[tuple[int, int, NDArray[np.int_], NDArray[np.int_]]],
    list[NDArray[np.int_]],
]:
    """Validate and compact the sparse concurrent-link design."""
    n_models = len(models)
    if len(anchor_matrices) > n_models - 1:
        raise ValueError("anchor_matrices contains too many model rows")

    relations: list[tuple[int, int, NDArray[np.int_], NDArray[np.int_]]] = []
    adjacency: list[set[int]] = [set() for _ in models]
    selected_items: list[set[int]] = [set() for _ in models]

    for left_model, row in enumerate(anchor_matrices):
        max_relations = n_models - left_model - 1
        if len(row) > max_relations:
            raise ValueError(
                f"anchor_matrices[{left_model}] contains too many model relations"
            )

        for offset, pairs in enumerate(row):
            if not pairs:
                continue
            right_model = left_model + offset + 1
            if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
                raise ValueError("each concurrent anchor pair must contain two indices")

            anchors_left = [pair[0] for pair in pairs]
            anchors_right = [pair[1] for pair in pairs]
            relation_name = f"anchor_matrices[{left_model}][{offset}]"
            _validate_anchor_pair(
                anchors_left,
                anchors_right,
                models[left_model].n_items,
                models[right_model].n_items,
                name=relation_name,
            )
            if method == "haebara":
                left_polytomous = models[left_model].is_polytomous
                right_polytomous = models[right_model].is_polytomous
                if left_polytomous != right_polytomous:
                    raise ValueError(
                        "Haebara relations cannot mix dichotomous and polytomous items"
                    )
                if left_polytomous:
                    left_categories = models[left_model].n_categories
                    right_categories = models[right_model].n_categories
                    if any(
                        left_categories[left_index] != right_categories[right_index]
                        for left_index, right_index in pairs
                    ):
                        raise ValueError(
                            "paired polytomous items must have matching category counts"
                        )
            relations.append(
                (
                    left_model,
                    right_model,
                    np.asarray(anchors_left, dtype=np.intp),
                    np.asarray(anchors_right, dtype=np.intp),
                )
            )
            adjacency[left_model].add(right_model)
            adjacency[right_model].add(left_model)
            selected_items[left_model].update(int(index) for index in anchors_left)
            selected_items[right_model].update(int(index) for index in anchors_right)

    if not relations:
        raise ValueError("anchor_matrices must contain at least one anchor relation")

    reachable = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reachable:
            reachable.add(neighbor)
            frontier.append(neighbor)
    if len(reachable) != n_models:
        raise ValueError("anchor_matrices must connect every model to the reference")

    selected_arrays = [
        np.asarray(sorted(indices), dtype=np.intp) for indices in selected_items
    ]
    item_positions = [
        {int(item): position for position, item in enumerate(indices)}
        for indices in selected_arrays
    ]
    compact_relations = [
        (
            left_model,
            right_model,
            np.asarray(
                [item_positions[left_model][int(index)] for index in anchors_left],
                dtype=np.intp,
            ),
            np.asarray(
                [item_positions[right_model][int(index)] for index in anchors_right],
                dtype=np.intp,
            ),
        )
        for left_model, right_model, anchors_left, anchors_right in relations
    ]

    return compact_relations, selected_arrays


def _validated_probabilities(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate probability bounds and absorb harmless roundoff."""
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -_PROBABILITY_TOLERANCE)
        | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")
    return np.clip(probabilities, 0.0, 1.0, out=probabilities)


def _expected_item_score_curves(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.int_],
) -> _ItemCurveSet:
    """Evaluate selected items without scanning unused portions of large banks."""
    n_theta = theta.shape[0]
    all_items_selected = len(item_indices) == model.n_items and np.array_equal(
        item_indices, np.arange(model.n_items)
    )

    if all_items_selected:
        probabilities = np.array(model.probability(theta), dtype=np.float64, copy=True)
    else:
        probabilities = None

    if model.is_polytomous and probabilities is not None:
        if probabilities.ndim != 3 or probabilities.shape[:2] != (
            n_theta,
            model.n_items,
        ):
            raise ValueError("model returned invalid polytomous probabilities")
        probabilities = _validated_probabilities(probabilities)
        probability_mass = probabilities.sum(axis=2, keepdims=True)
        if np.any(np.abs(probability_mass - 1.0) > _PROBABILITY_TOLERANCE):
            raise ValueError("model category probabilities must sum to one")
        probabilities /= probability_mass
        categories = np.arange(probabilities.shape[2], dtype=np.float64)
        category_counts = model.n_categories
        return _ItemCurveSet(
            expected=probabilities @ categories,
            categories=[
                probabilities[:, int(index), : category_counts[int(index)]]
                for index in item_indices
            ],
        )

    if model.is_polytomous:
        expected = np.empty((n_theta, len(item_indices)), dtype=np.float64)
        category_curves: list[NDArray[np.float64]] = []
        category_counts = model.n_categories
        for column, item_index in enumerate(item_indices):
            n_categories = category_counts[int(item_index)]
            item_probabilities = np.array(
                model.probability(theta, int(item_index)),
                dtype=np.float64,
                copy=True,
            )
            if item_probabilities.shape != (n_theta, n_categories):
                raise ValueError("model returned invalid polytomous probabilities")
            item_probabilities = _validated_probabilities(item_probabilities)
            probability_mass = item_probabilities.sum(axis=1, keepdims=True)
            if np.any(np.abs(probability_mass - 1.0) > _PROBABILITY_TOLERANCE):
                raise ValueError("model category probabilities must sum to one")
            item_probabilities /= probability_mass
            category_curves.append(item_probabilities)
            expected[:, column] = item_probabilities @ np.arange(
                n_categories, dtype=np.float64
            )
        return _ItemCurveSet(expected=expected, categories=category_curves)

    if probabilities is None:
        item_curves = []
        for item_index in item_indices:
            item_probabilities = np.array(
                model.probability(theta, int(item_index)),
                dtype=np.float64,
                copy=True,
            )
            if item_probabilities.shape != (n_theta,):
                raise ValueError("model returned invalid dichotomous probabilities")
            item_curves.append(_validated_probabilities(item_probabilities))
        return _ItemCurveSet(expected=np.column_stack(item_curves), categories=None)

    if probabilities.shape != (n_theta, model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    return _ItemCurveSet(
        expected=_validated_probabilities(probabilities),
        categories=None,
    )


def concurrent_link(
    models: list["BaseItemModel"],
    anchor_matrices: list[list[list[tuple[int, int]]]],
    method: str = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> list[tuple[float, float]]:
    """Perform concurrent (simultaneous) linking of multiple forms.

    Unlike chain linking, this optimizes all transformations
    simultaneously to minimize total discrepancy.

    Parameters
    ----------
    models : list[BaseItemModel]
        List of models to link.
    anchor_matrices : list[list[list[tuple[int, int]]]]
        For each model i, for each subsequent model j, list of (item_i, item_j) anchor pairs.
    method : str
        Curve-matching criterion: ``"stocking_lord"``/``"tcc"`` compares
        summed expected-score curves, while ``"haebara"`` compares paired
        item curves.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    list[tuple[float, float]]
        (A, B) transformation for each model to common scale.
    """
    from scipy import optimize

    n_models = len(models)
    if n_models < 2:
        raise ValueError("concurrent linking requires at least two models")
    if method not in _CONCURRENT_METHODS:
        supported = ", ".join(sorted(_CONCURRENT_METHODS))
        raise ValueError(
            f"Unknown concurrent linking method: {method}; use {supported}"
        )
    if any(model.n_factors != 1 for model in models):
        raise ValueError("concurrent linking currently requires unidimensional models")
    _validate_grid(theta_range, n_theta)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    relations, selected_items = _prepare_concurrent_relations(
        models, anchor_matrices, method
    )

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = np.exp(-0.5 * theta_grid**2)
    weights = weights / np.sum(weights)
    n_free_models = n_models - 1

    def criterion(params: NDArray[np.float64]) -> float:
        slopes = np.concatenate(([1.0], np.exp(params[:n_free_models])))
        intercepts = np.concatenate(([0.0], params[n_free_models:]))

        total_loss = 0.0
        score_curves = [
            _expected_item_score_curves(
                model,
                ((theta_grid - intercept) / slope)[:, None],
                indices,
            )
            for model, slope, intercept, indices in zip(
                models,
                slopes,
                intercepts,
                selected_items,
                strict=True,
            )
        ]

        for left_model, right_model, anchors_left, anchors_right in relations:
            left_curves = score_curves[left_model].expected[:, anchors_left]
            right_curves = score_curves[right_model].expected[:, anchors_right]
            if method == "haebara":
                left_categories = score_curves[left_model].categories
                right_categories = score_curves[right_model].categories
                if left_categories is None or right_categories is None:
                    curve_difference = left_curves - right_curves
                    total_loss += float(np.sum(weights[:, None] * curve_difference**2))
                else:
                    for left_index, right_index in zip(
                        anchors_left, anchors_right, strict=True
                    ):
                        category_difference = (
                            left_categories[int(left_index)]
                            - right_categories[int(right_index)]
                        )
                        total_loss += float(
                            np.sum(weights[:, None] * category_difference**2)
                        )
            else:
                curve_difference = left_curves.sum(axis=1) - right_curves.sum(axis=1)
                total_loss += float(weights @ curve_difference**2)

        return total_loss

    x0 = np.zeros(2 * n_free_models, dtype=np.float64)
    max_log_slope = float(np.log(100.0))
    bounds = [(-max_log_slope, max_log_slope)] * n_free_models + [
        (None, None)
    ] * n_free_models

    result = optimize.minimize(
        criterion,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol},
    )

    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError("concurrent linking failed to find a finite solution")

    slopes = [1.0, *np.exp(result.x[:n_free_models]).tolist()]
    intercepts = [0.0, *result.x[n_free_models:].tolist()]

    return [
        (float(slope), float(intercept))
        for slope, intercept in zip(slopes, intercepts, strict=True)
    ]


def chain_linking_summary(result: ChainLinkingResult) -> str:
    """Generate summary of chain linking results.

    Parameters
    ----------
    result : ChainLinkingResult
        Chain linking result.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Chain Linking Summary")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Number of time points: {len(result.cumulative_A)}")
    lines.append(f"Reference index: {result.reference_index}")
    lines.append("")

    lines.append("Cumulative Transformations to Reference Scale")
    lines.append("-" * 50)
    lines.append(f"{'Time':>6} {'A':>12} {'B':>12}")
    lines.append("-" * 50)

    for t, (A, B) in enumerate(
        zip(result.cumulative_A, result.cumulative_B, strict=True)
    ):
        marker = " *" if t == result.reference_index else ""
        lines.append(f"{t:>6} {A:>12.4f} {B:>12.4f}{marker}")

    lines.append("-" * 50)
    lines.append("* = reference point")
    lines.append("")

    lines.append("Pairwise Linking Results")
    lines.append("-" * 50)

    for t, pr in enumerate(result.pairwise_results):
        lines.append(f"Time {t} -> {t + 1}:")
        lines.append(f"  A = {pr.constants.A:.4f}, B = {pr.constants.B:.4f}")
        lines.append(f"  Method: {pr.constants.method}")

        if pr.fit_statistics is not None:
            lines.append(f"  TCC RMSE: {pr.fit_statistics.tcc_rmse:.4f}")

        if pr.anchor_diagnostics is not None:
            n_flagged = int(np.sum(pr.anchor_diagnostics.flagged))
            lines.append(f"  Flagged items: {n_flagged}")

        lines.append("")

    if result.drift_accumulation is not None:
        lines.append("Drift Accumulation")
        lines.append("-" * 50)

        valid_mask = ~np.isnan(result.drift_accumulation)
        if np.any(valid_mask):
            mean_drift = np.nanmean(np.abs(result.drift_accumulation))
            max_drift = np.nanmax(np.abs(result.drift_accumulation))
            lines.append(f"Mean |z| across all pairs: {mean_drift:.3f}")
            lines.append(f"Max |z| across all pairs: {max_drift:.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def detect_longitudinal_drift(
    chain_result: ChainLinkingResult,
    threshold: float = 2.5,
) -> dict[str, list]:
    """Detect items with consistent drift across time points.

    Parameters
    ----------
    chain_result : ChainLinkingResult
        Chain linking result with drift accumulation.
    threshold : float
        Z-score threshold for flagging.

    Returns
    -------
    dict[str, list]
        Dictionary with consistently drifting items and patterns.
    """
    if chain_result.drift_accumulation is None:
        return {"consistently_flagged": [], "drift_direction": []}

    drift = chain_result.drift_accumulation
    n_pairs, n_items = drift.shape

    consistently_flagged = []
    drift_direction = []

    for j in range(n_items):
        item_drift = drift[:, j]
        valid = ~np.isnan(item_drift)

        if np.sum(valid) < 2:
            continue

        valid_drift = item_drift[valid]
        n_flagged = np.sum(np.abs(valid_drift) > threshold)

        if n_flagged >= np.sum(valid) / 2:
            consistently_flagged.append(j)

            mean_dir = np.mean(valid_drift)
            if mean_dir > 0.5:
                drift_direction.append("increasing")
            elif mean_dir < -0.5:
                drift_direction.append("decreasing")
            else:
                drift_direction.append("variable")

    return {
        "consistently_flagged": consistently_flagged,
        "drift_direction": drift_direction,
    }
