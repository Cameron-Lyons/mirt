"""Chain linking across multiple time points.

This module provides functions for linking IRT models across
multiple time points or administrations using pairwise linking.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.equating._curves import _expected_item_score_curves, _ItemCurveSet
from mirt.equating.linking import LinkingResult, link, transform_parameters
from mirt.equating.polytomous import (
    _NRM_METHODS,
    _ORDERED_METHODS,
    _linker_for,
    _polytomous_model_type,
    transform_polytomous_parameters,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_CONCURRENT_METHODS = frozenset({"stocking_lord", "tcc", "haebara"})
# Central-difference step of the curves' theta slopes in the concurrent
# Jacobian; truncation and rounding errors are both near 1e-10.
_THETA_DIFFERENCE_STEP = 1e-5
# Concurrent Jacobians with more entries are stored sparse: each residual
# depends on the constants of at most two forms.
_DENSE_JACOBIAN_ELEMENTS = 4_000_000
# Termination on relative criterion reduction and on the scaled gradient,
# as in :func:`mirt.equating.link`; ``tol`` bounds the parameter change.
_CONCURRENT_FTOL = 1e-15
_CONCURRENT_GTOL = 1e-12


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
        Pairwise robust z-statistics, with one column per physical anchor item
        and NaN where that item is absent from an adjacent link.
    reference_index : int
        Index of reference time point.
    drift_item_ids : list[tuple[int, int]] | None
        Identity of each drift column as its earliest (time index, item index).
        Identities follow the supplied anchor correspondences across forms.
    drift_difficulty_changes : NDArray[np.float64] | None
        Later-minus-earlier difficulty changes on the reference metric,
        aligned with ``drift_accumulation``. These describe residual change
        after linking, rather than unadjusted differences between forms.
    cumulative_A_se, cumulative_B_se : list[float] | None
        Delta-method standard errors of the cumulative constants, zero at
        the reference, when :func:`chain_link` received ``vcovs``.
    cumulative_covariance : NDArray[np.float64] | None
        Covariance of each time point's cumulative ``(A, B)``, shape
        ``(n_models, 2, 2)``, when :func:`chain_link` received ``vcovs``.
    """

    cumulative_A: list[float]
    cumulative_B: list[float]
    pairwise_results: list[LinkingResult]
    drift_accumulation: NDArray[np.float64] | None
    reference_index: int
    drift_item_ids: list[tuple[int, int]] | None = None
    drift_difficulty_changes: NDArray[np.float64] | None = None
    cumulative_A_se: list[float] | None = None
    cumulative_B_se: list[float] | None = None
    cumulative_covariance: NDArray[np.float64] | None = None


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
    vcovs: "list[NDArray[np.float64] | FitResult] | None" = None,
) -> ChainLinkingResult:
    """Perform chain linking across multiple time points.

    Links a sequence of models to a common reference scale by
    accumulating pairwise transformations.

    Parameters
    ----------
    models : list[BaseItemModel]
        List of models in temporal order. Adjacent models must share a
        response family: dichotomous forms use :func:`link`, GRM forms use
        :func:`link_grm`, GPCM/PCM forms use :func:`link_gpcm`, and NRM forms
        use :func:`link_nrm`.
    anchor_item_pairs : list[tuple[list[int], list[int]]]
        List of (anchors_t, anchors_t+1) pairs for consecutive models.
        Length should be len(models) - 1.
    method : str
        Linking method for pairwise linking. Polytomous pairs support the
        methods of their family's linker; ``"tcc"`` is their Stocking-Lord
        expected-score criterion.
    reference_index : int
        Index of reference model (default 0 = first).
    theta_range : tuple[float, float]
        Range for curve matching methods.
    n_theta : int
        Number of theta points.
    compute_drift : bool
        Whether to track drift accumulation. NRM anchors have no single
        location parameter, so their ``drift_difficulty_changes`` are NaN;
        their robust z-statistics are still reported.
    vcovs : list of ndarray or FitResult, optional
        Parameter covariance of each calibration, in any form accepted by
        :func:`~mirt.equating.delta_method_se`. When given, each pairwise
        result receives delta-method standard errors, and
        ``cumulative_A_se``, ``cumulative_B_se``, and
        ``cumulative_covariance`` propagate them through the composition.
        Calibrations are independent, so parameter covariances are block
        diagonal across models; a model in two adjacent links contributes to
        both, and the covariance this induces between them is included.

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

    pair_linkers = []
    for pair_index, (anchors_left, anchors_right) in enumerate(anchor_item_pairs):
        _validate_anchor_pair(
            anchors_left,
            anchors_right,
            models[pair_index].n_items,
            models[pair_index + 1].n_items,
            name=f"anchor_item_pairs[{pair_index}]",
        )
        pair_linkers.append(
            _pair_linker(models[pair_index], models[pair_index + 1], method, pair_index)
        )

    pairwise_results: list[LinkingResult] = []

    for t, (linker, pair_method, _) in enumerate(pair_linkers):
        anchors_t, anchors_t1 = anchor_item_pairs[t]

        result = linker(
            models[t],
            models[t + 1],
            anchors_t,
            anchors_t1,
            method=pair_method,
            theta_range=theta_range,
            n_theta=n_theta,
            compute_diagnostics=True,
        )
        pairwise_results.append(result)

    # link(left, right) maps right coordinates onto the left metric.
    # accumulate_constants accepts the forward (left -> right) maps.
    pairwise_A = [1.0 / r.constants.A for r in pairwise_results]
    pairwise_B = [-r.constants.B / r.constants.A for r in pairwise_results]

    cumulative_A, cumulative_B = accumulate_constants(
        pairwise_A, pairwise_B, reference_index
    )

    cumulative_A_se = None
    cumulative_B_se = None
    cumulative_covariance = None
    if vcovs is not None:
        pairwise_covariance, cumulative_covariance = _chain_delta_covariance(
            models,
            anchor_item_pairs,
            [pair_method for _, pair_method, _ in pair_linkers],
            [(r.constants.A, r.constants.B) for r in pairwise_results],
            vcovs,
            reference_index,
            theta_range,
            n_theta,
        )
        pairwise_se = np.sqrt(np.diag(pairwise_covariance)).reshape(-1, 2)
        for result, (A_se, B_se) in zip(pairwise_results, pairwise_se, strict=True):
            result.constants.A_se, result.constants.B_se = float(A_se), float(B_se)
        cumulative_A_se = [float(np.sqrt(c[0, 0])) for c in cumulative_covariance]
        cumulative_B_se = [float(np.sqrt(c[1, 1])) for c in cumulative_covariance]

    drift_accumulation = None
    drift_item_ids = None
    drift_difficulty_changes = None
    if compute_drift:
        drift_accumulation, drift_item_ids, drift_difficulty_changes = (
            _compute_drift_accumulation(
                pairwise_results,
                anchor_item_pairs,
                cumulative_A,
                [has_location for _, _, has_location in pair_linkers],
            )
        )

    return ChainLinkingResult(
        cumulative_A=cumulative_A,
        cumulative_B=cumulative_B,
        pairwise_results=pairwise_results,
        drift_accumulation=drift_accumulation,
        reference_index=reference_index,
        drift_item_ids=drift_item_ids,
        drift_difficulty_changes=drift_difficulty_changes,
        cumulative_A_se=cumulative_A_se,
        cumulative_B_se=cumulative_B_se,
        cumulative_covariance=cumulative_covariance,
    )


def _chain_delta_covariance(
    models: list["BaseItemModel"],
    anchor_item_pairs: list[tuple[list[int], list[int]]],
    pair_methods: list[str],
    pairwise_constants: list[tuple[float, float]],
    vcovs: "list[NDArray[np.float64] | FitResult]",
    reference_index: int,
    theta_range: tuple[float, float],
    n_theta: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Delta-method covariance of the pairwise and cumulative chain constants.

    Each pairwise ``(A_t, B_t)`` is differentiated with respect to the
    anchor parameters of both its forms. A form's coordinates from its two
    links share one covariance block, so the pairwise covariance includes
    the dependence that shared forms induce; the cumulative covariance
    follows from the Jacobian of the composition.
    """
    from scipy import linalg

    from mirt.equating.diagnostics import (
        _anchor_covariance,
        _constants_jacobian,
        _delta_side,
        _DeltaComponent,
        _validate_linking_pair,
    )
    from mirt.equating.linking import _validate_curve_grid

    n_models = len(models)
    if len(vcovs) != n_models:
        raise ValueError(
            f"vcovs must contain one covariance per model; expected {n_models}, "
            f"got {len(vcovs)}"
        )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    if not anchor_item_pairs:
        return np.zeros((0, 0)), np.zeros((n_models, 2, 2))
    sides = []
    for t, (anchors_left, anchors_right) in enumerate(anchor_item_pairs):
        left, right = _validate_linking_pair(
            models[t], models[t + 1], anchors_left, anchors_right, pair_methods[t]
        )
        sides.append(
            (
                _delta_side(models[t], left, "old"),
                _delta_side(models[t + 1], right, "new"),
            )
        )

    # Each model's coordinates from both of its links, in one shared order.
    positions: list[dict[tuple[str, int], int]] = []
    covariances = []
    for index, model in enumerate(models):
        used = [sides[index][0]] if index < len(sides) else []
        used += [sides[index - 1][1]] if index > 0 else []
        varied: dict[str, set[int]] = {}
        for side in used:
            for component in side.components:
                varied.setdefault(component.name, set()).update(
                    int(flat) for flat in component.indices
                )
        components = []
        for name, values in model.parameters.items():
            if name in varied:
                flat = np.array(sorted(varied[name]), dtype=np.intp)
                stored = np.asarray(values, dtype=np.float64).ravel()
                components.append(_DeltaComponent(name, stored[flat], flat))
        coordinates: dict[tuple[str, int], int] = {}
        for component in components:
            for flat in component.indices:
                coordinates[(component.name, int(flat))] = len(coordinates)
        positions.append(coordinates)
        covariances.append(
            _anchor_covariance(
                vcovs[index],
                model,
                components,
                str(index),
                names=(f"vcovs[{index}]", f"models[{index}]"),
            )
        )

    offsets = np.cumsum([0] + [covariance.shape[0] for covariance in covariances])
    jacobian = np.zeros((2 * len(sides), int(offsets[-1])), dtype=np.float64)
    for t, (old, new) in enumerate(sides):
        local = [
            np.array(
                [
                    positions[index][(component.name, int(flat))]
                    for component in side.components
                    for flat in component.indices
                ],
                dtype=np.intp,
            )
            for index, side in ((t, old), (t + 1, new))
        ]
        # Coordinates without covariance cannot move the constants.
        active = np.flatnonzero(
            np.concatenate(
                [
                    np.any(covariances[index][rows] != 0.0, axis=1)
                    for index, rows in zip((t, t + 1), local, strict=True)
                ]
            )
        )
        if active.size:
            columns = np.concatenate([offsets[t] + local[0], offsets[t + 1] + local[1]])
            jacobian[2 * t : 2 * t + 2, columns[active]] = _constants_jacobian(
                pair_methods[t], old, new, active, theta_grid, weights
            )

    pairwise = jacobian @ linalg.block_diag(*covariances) @ jacobian.T
    pairwise = 0.5 * (pairwise + pairwise.T)
    composition = _cumulative_constant_jacobians(pairwise_constants, reference_index)
    cumulative = composition @ pairwise @ composition.transpose(0, 2, 1)
    for covariance in (pairwise, *cumulative):
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(covariance))))
        if np.any(np.diag(covariance) < -tolerance):
            raise ValueError("Propagated covariance produced a negative variance")
        # Rounding can leave exactly known variances slightly negative.
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 0.0))
    return pairwise, cumulative


def _cumulative_constant_jacobians(
    pairwise_constants: list[tuple[float, float]],
    reference_index: int,
) -> NDArray[np.float64]:
    """Jacobians of each cumulative ``(A, B)`` with respect to the pairwise ones.

    Pairwise constants map model ``t + 1`` onto model ``t``:
    ``theta_t = A_t * theta_{t+1} + B_t``. Returns an array of shape
    ``(n_models, 2, 2 * n_pairs)`` whose columns are ``(A_0, B_0, A_1, ...)``.
    """
    n_pairs = len(pairwise_constants)
    slopes = np.ones(n_pairs + 1)
    jacobians = np.zeros((n_pairs + 1, 2, 2 * n_pairs), dtype=np.float64)
    # Below the reference: C_t = C_{t+1} / A_t, D_t = D_{t+1} - C_{t+1} B_t / A_t.
    for t in range(reference_index - 1, -1, -1):
        A, B = pairwise_constants[t]
        following = jacobians[t + 1]
        jacobians[t, 0] = following[0] / A
        jacobians[t, 0, 2 * t] -= slopes[t + 1] / A**2
        jacobians[t, 1] = following[1] - (B / A) * following[0]
        jacobians[t, 1, 2 * t] += slopes[t + 1] * B / A**2
        jacobians[t, 1, 2 * t + 1] -= slopes[t + 1] / A
        slopes[t] = slopes[t + 1] / A
    # Above it: C_t = C_{t-1} A_{t-1}, D_t = C_{t-1} B_{t-1} + D_{t-1}.
    for t in range(reference_index + 1, n_pairs + 1):
        A, B = pairwise_constants[t - 1]
        previous = jacobians[t - 1]
        jacobians[t, 0] = A * previous[0]
        jacobians[t, 0, 2 * (t - 1)] += slopes[t - 1]
        jacobians[t, 1] = B * previous[0] + previous[1]
        jacobians[t, 1, 2 * (t - 1) + 1] += slopes[t - 1]
        slopes[t] = slopes[t - 1] * A
    return jacobians


def _pair_linker(
    model_left: "BaseItemModel",
    model_right: "BaseItemModel",
    method: str,
    pair_index: int,
) -> tuple[Callable[..., LinkingResult], str, bool]:
    """Choose one adjacent pair's linker, its method, and location units.

    The flag reports whether the anchors' signed location differences are in
    theta units, which NRM category-intercept contrasts are not.
    """
    family = _polytomous_model_type(model_left)
    if family != _polytomous_model_type(model_right):
        raise ValueError(
            f"anchor_item_pairs[{pair_index}] links different response families: "
            f"{model_left.model_name} and {model_right.model_name}"
        )
    if family is None:
        return link, method, True
    pair_method = "stocking_lord" if method == "tcc" else method
    supported = _NRM_METHODS if family == "nrm" else _ORDERED_METHODS
    if pair_method not in supported:
        raise ValueError(
            f"{method} linking is not available for {model_left.model_name} "
            f"models; use one of {', '.join(sorted(supported))}"
        )
    return _linker_for(model_left), pair_method, family != "nrm"


def accumulate_constants(
    pairwise_A: list[float],
    pairwise_B: list[float],
    reference_index: int = 0,
) -> tuple[list[float], list[float]]:
    """Accumulate pairwise constants to reference scale.

    The inputs describe forward coordinate maps:
    ``theta_{t+1} = pairwise_A[t] * theta_t + pairwise_B[t]``.
    The returned constants map each time point onto the reference metric:
    ``theta_ref = cumulative_A[t] * theta_t + cumulative_B[t]``.
    Constants returned by :func:`link` describe the opposite direction and
    must be inverted before passing them to this function.

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
    anchor_item_pairs: list[tuple[list[int], list[int]]],
    cumulative_A: list[float],
    location_units: list[bool] | None = None,
) -> tuple[NDArray[np.float64], list[tuple[int, int]], NDArray[np.float64]]:
    """Align adjacent-link diagnostics using physical anchor correspondences.

    Difficulty changes stay NaN for links whose signed location differences
    are not in theta units (``location_units[t]`` is False).
    """
    parents: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    for t, (left_items, right_items) in enumerate(anchor_item_pairs):
        for left_item, right_item in zip(left_items, right_items, strict=True):
            left_root = find((t, int(left_item)))
            right_root = find((t + 1, int(right_item)))
            # The earliest occurrence supplies a stable, order-independent ID.
            parents[max(left_root, right_root)] = min(left_root, right_root)

    item_ids = sorted({find(node) for node in parents})
    columns = {identity: column for column, identity in enumerate(item_ids)}
    shape = (len(pairwise_results), len(item_ids))
    drift_matrix = np.full(shape, np.nan, dtype=np.float64)
    difficulty_changes = np.full(shape, np.nan, dtype=np.float64)

    for t, result in enumerate(pairwise_results):
        diagnostics = result.anchor_diagnostics
        if diagnostics is None:
            continue
        indices = diagnostics.item_indices
        z_scores = np.asarray(diagnostics.robust_z, dtype=np.float64)
        signed_b = np.asarray(diagnostics.signed_diff_b, dtype=np.float64)
        if (
            len(set(indices)) != len(indices)
            or not set(indices).issubset(anchor_item_pairs[t][0])
            or z_scores.shape != (len(indices),)
            or signed_b.shape != (len(indices),)
        ):
            raise ValueError("Pairwise anchor diagnostics do not match anchor items")
        has_location = location_units is None or location_units[t]
        for index, z_score, difference in zip(indices, z_scores, signed_b, strict=True):
            column = columns[find((t, int(index)))]
            drift_matrix[t, column] = z_score
            if has_location:
                # Pair diagnostics use earlier - transformed later. Multiplying
                # by the earlier slope puts the opposite change on reference.
                difficulty_changes[t, column] = -cumulative_A[t] * difference

    return drift_matrix, item_ids, difficulty_changes


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
        Model to transform. GRM, GPCM, and NRM parameters are transformed
        with :func:`transform_polytomous_parameters`. PCM discriminations are
        fixed at 1 and cannot be rescaled, so PCM models raise ValueError.
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
    A, B = _time_point_constants(chain_result, time_index)

    model_type = _polytomous_model_type(model)
    if model_type is not None:
        return transform_polytomous_parameters(
            model, A, B, model_type=model_type, in_place=in_place
        )
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


def _scaled_difference(
    upper: _ItemCurveSet, lower: _ItemCurveSet, factor: NDArray[np.float64]
) -> _ItemCurveSet:
    """Expected and category curves of ``factor * (upper - lower)`` by theta row."""
    categories = None
    if upper.categories is not None and lower.categories is not None:
        categories = [
            factor[:, None] * (above - below)
            for above, below in zip(upper.categories, lower.categories, strict=True)
        ]
    return _ItemCurveSet(
        factor[:, None] * (upper.expected - lower.expected), categories
    )


def concurrent_link(
    models: list["BaseItemModel"],
    anchor_matrices: list[list[list[tuple[int, int]]]],
    method: str = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    max_iter: int = 100,
    tol: float = 1e-10,
    reference_index: int = 0,
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
        Maximum number of criterion evaluations of the trust-region
        least-squares solver. Default 100.
    tol : float
        Relative tolerance on the change in the free forms' ``(log A, B)``
        between iterations (``xtol`` of :func:`scipy.optimize.least_squares`).
        The solver also stops once the criterion's relative reduction or its
        gradient vanishes to rounding accuracy. The default ``1e-10``
        recovers exact transformations to rounding accuracy.
    reference_index : int
        Model that defines the common metric and theta-weighting grid.
        Its transformation is fixed at ``(1, 0)``.

    Returns
    -------
    list[tuple[float, float]]
        (A, B) transformation for each model to common scale.

    Raises
    ------
    RuntimeError
        If the solver exhausts ``max_iter`` or fails before converging.

    Notes
    -----
    The criterion is the sum of the relations' squared, theta-weighted
    curve differences, minimized as a nonlinear least-squares problem in
    each free form's ``(log A, B)``, with ``A`` bounded in ``[0.01, 100]``.
    A form's curves depend on its constants only through the grid
    ``u = (theta - B) / A``, so the Jacobian follows from each curve's theta
    slope by the chain rule; the slopes are central differences, accurate
    to about ``1e-10``.
    """
    from scipy import optimize, sparse

    n_models = len(models)
    if n_models < 2:
        raise ValueError("concurrent linking requires at least two models")
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(
        reference_index, (int, np.integer)
    ):
        raise ValueError("reference_index must be an integer")
    if reference_index < 0 or reference_index >= n_models:
        raise ValueError(f"Invalid reference_index: {reference_index}")
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
    root_weights = np.sqrt(weights)
    n_free_models = n_models - 1
    free_indices = [index for index in range(n_models) if index != reference_index]
    # The reference metric never changes during optimization.
    reference_curves = _expected_item_score_curves(
        models[reference_index], theta_grid[:, None], selected_items[reference_index]
    )
    # Moving one form's (log A, B) only changes the relations that touch it.
    incident_relations = [
        [
            position
            for position, (left_model, right_model, _, _) in enumerate(relations)
            if index in (left_model, right_model)
        ]
        for index in range(n_models)
    ]
    max_log_slope = float(np.log(100.0))

    def form_grid(params: NDArray[np.float64], position: int) -> NDArray[np.float64]:
        """Form coordinates ``(theta - B) / A`` of the reference grid."""
        slope = np.exp(params[position])
        return (theta_grid - params[n_free_models + position]) / slope

    def form_curves(index: int, grid: NDArray[np.float64]) -> _ItemCurveSet:
        return _expected_item_score_curves(
            models[index], grid[:, None], selected_items[index]
        )

    def side_vector(
        relation: tuple[int, int, NDArray[np.int_], NDArray[np.int_]],
        side: int,
        curves: _ItemCurveSet,
    ) -> NDArray[np.float64]:
        """One side's square-root-weighted curves in the relation's layout."""
        anchors = relation[2 + side]
        if method != "haebara":
            return root_weights * curves.expected[:, anchors].sum(axis=1)
        if curves.categories is None:
            return (root_weights[:, None] * curves.expected[:, anchors]).ravel()
        return np.concatenate(
            [
                (root_weights[:, None] * curves.categories[int(anchor)]).ravel()
                for anchor in anchors
            ]
        )

    def relation_residuals(
        relation: tuple[int, int, NDArray[np.int_], NDArray[np.int_]],
        score_curves: list[_ItemCurveSet],
    ) -> NDArray[np.float64]:
        left_model, right_model, _, _ = relation
        return side_vector(relation, 0, score_curves[left_model]) - side_vector(
            relation, 1, score_curves[right_model]
        )

    cache: dict[bytes, NDArray[np.float64]] = {}

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        key = params.tobytes()
        if key not in cache:
            score_curves = [reference_curves] * n_models
            for position, index in enumerate(free_indices):
                score_curves[index] = form_curves(index, form_grid(params, position))
            cache.clear()
            cache[key] = np.concatenate(
                [relation_residuals(relation, score_curves) for relation in relations]
            )
        return cache[key]

    def relation_rows(
        relation: tuple[int, int, NDArray[np.int_], NDArray[np.int_]],
    ) -> int:
        left_model, _, anchors_left, _ = relation
        if method != "haebara":
            return n_theta
        if not models[left_model].is_polytomous:
            return n_theta * anchors_left.size
        category_counts = models[left_model].n_categories
        items = selected_items[left_model][anchors_left]
        return n_theta * sum(category_counts[int(item)] for item in items)

    # First residual row of each relation, and the total, in stacking order.
    row_offsets = np.cumsum([0] + [relation_rows(relation) for relation in relations])

    def jacobian(
        params: NDArray[np.float64],
    ) -> NDArray[np.float64] | sparse.csr_matrix:
        """Chain-rule derivatives that re-evaluate only each form's own curves.

        ``du/dlog(A) = -u`` and ``du/dB = -1 / A`` scale the theta slopes of
        the form's curves at its grid ``u``.
        """
        rows, columns, values = [], [], []
        step = _THETA_DIFFERENCE_STEP
        for position, index in enumerate(free_indices):
            grid = form_grid(params, position)
            upper = form_curves(index, grid + step)
            lower = form_curves(index, grid - step)
            factors = (-grid, np.full(n_theta, -np.exp(-params[position])))
            for column, factor in zip(
                (position, n_free_models + position), factors, strict=True
            ):
                derivative = _scaled_difference(upper, lower, factor / (2.0 * step))
                for relation_position in incident_relations[index]:
                    relation = relations[relation_position]
                    side = 0 if relation[0] == index else 1
                    entries = side_vector(relation, side, derivative)
                    start = int(row_offsets[relation_position])
                    rows.append(np.arange(start, start + entries.size))
                    columns.append(np.full(entries.size, column))
                    values.append(entries if side == 0 else -entries)
        shape = (int(row_offsets[-1]), 2 * n_free_models)
        if shape[0] * shape[1] > _DENSE_JACOBIAN_ELEMENTS:
            return sparse.csr_matrix(
                (
                    np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=shape,
            )
        matrix = np.zeros(shape, dtype=np.float64)
        for row, column, entries in zip(rows, columns, values, strict=True):
            matrix[row, column] = entries
        return matrix

    x0 = np.zeros(2 * n_free_models, dtype=np.float64)
    unbounded = np.full(n_free_models, np.inf)
    log_slope_bound = np.full(n_free_models, max_log_slope)
    bounds = (np.r_[-log_slope_bound, -unbounded], np.r_[log_slope_bound, unbounded])

    result = optimize.least_squares(
        residuals,
        x0,
        jac=jacobian,
        bounds=bounds,
        method="trf",
        x_scale="jac",
        ftol=_CONCURRENT_FTOL,
        xtol=float(tol),
        gtol=_CONCURRENT_GTOL,
        max_nfev=int(max_iter),
    )

    if not np.all(np.isfinite(result.x)) or not np.all(np.isfinite(result.fun)):
        raise RuntimeError("concurrent linking failed to find a finite solution")
    if not result.success:
        raise RuntimeError(f"concurrent linking failed to converge: {result.message}")

    slopes = np.ones(n_models, dtype=np.float64)
    intercepts = np.zeros(n_models, dtype=np.float64)
    slopes[free_indices] = np.exp(result.x[:n_free_models])
    intercepts[free_indices] = result.x[n_free_models:]

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

    standard_errors = (
        None
        if result.cumulative_A_se is None or result.cumulative_B_se is None
        else list(zip(result.cumulative_A_se, result.cumulative_B_se, strict=True))
    )
    lines.append("Cumulative Transformations to Reference Scale")
    lines.append("-" * 50)
    header = f"{'Time':>6} {'A':>12} {'B':>12}"
    if standard_errors is not None:
        header += f" {'SE(A)':>10} {'SE(B)':>10}"
    lines.append(header)
    lines.append("-" * 50)

    for t, (A, B) in enumerate(
        zip(result.cumulative_A, result.cumulative_B, strict=True)
    ):
        marker = " *" if t == result.reference_index else ""
        row = f"{t:>6} {A:>12.4f} {B:>12.4f}"
        if standard_errors is not None:
            row += f" {standard_errors[t][0]:>10.4f} {standard_errors[t][1]:>10.4f}"
        lines.append(row + marker)

    lines.append("-" * 50)
    lines.append("* = reference point")
    lines.append("")

    lines.append("Pairwise Linking Results")
    lines.append("-" * 50)

    for t, pr in enumerate(result.pairwise_results):
        lines.append(f"Time {t + 1} -> {t}:")
        lines.append(f"  A = {pr.constants.A:.4f}, B = {pr.constants.B:.4f}")
        if pr.constants.A_se is not None and pr.constants.B_se is not None:
            lines.append(
                f"  SE(A) = {pr.constants.A_se:.4f}, SE(B) = {pr.constants.B_se:.4f}"
            )
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
    """Detect repeatedly flagged physical anchors and signed difficulty change.

    An item must occur in at least two adjacent links and exceed the threshold
    in at least half its observed links. Direction describes its residual
    difficulty change after placing both forms on the reference metric.
    Robust z-statistics quantify unusual discrepancy magnitude; their signs
    do not describe a parameter increasing or decreasing.

    Parameters
    ----------
    chain_result : ChainLinkingResult
        Chain linking result with drift accumulation.
    threshold : float
        Z-score threshold for flagging.

    Returns
    -------
    dict[str, list]
        ``consistently_flagged`` contains drift-matrix column indices.
        ``flagged_item_ids`` gives the corresponding earliest (time, item)
        identities, or None for legacy results without identity metadata.
        ``drift_direction`` contains "increasing" for nonnegative difficulty
        changes with at least one positive change, "decreasing" for the
        converse, and "variable" for mixed signs or zero change. Results
        without at least two signed changes have direction "unknown".
    """
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("threshold must be finite and positive")

    if chain_result.drift_accumulation is None:
        return {
            "consistently_flagged": [],
            "drift_direction": [],
            "flagged_item_ids": [],
        }

    drift = np.asarray(chain_result.drift_accumulation, dtype=np.float64)
    if drift.ndim != 2:
        raise ValueError("drift_accumulation must be a two-dimensional matrix")
    n_items = drift.shape[1]
    item_ids = chain_result.drift_item_ids
    if item_ids is not None and len(item_ids) != n_items:
        raise ValueError("drift_item_ids must match drift_accumulation columns")
    changes = chain_result.drift_difficulty_changes
    if changes is not None:
        changes = np.asarray(changes, dtype=np.float64)
        if changes.shape != drift.shape or np.any(np.isinf(changes)):
            raise ValueError(
                "drift_difficulty_changes must match drift_accumulation and "
                "contain finite values or NaN"
            )

    consistently_flagged = []
    drift_direction = []
    flagged_item_ids = []

    for j in range(n_items):
        item_drift = drift[:, j]
        valid = ~np.isnan(item_drift)

        if np.sum(valid) < 2:
            continue

        valid_drift = item_drift[valid]
        n_flagged = np.sum(np.abs(valid_drift) > threshold)

        if n_flagged >= np.sum(valid) / 2:
            consistently_flagged.append(j)
            flagged_item_ids.append(None if item_ids is None else item_ids[j])

            signed_changes = (
                np.array([], dtype=np.float64)
                if changes is None
                else changes[valid & np.isfinite(changes[:, j]), j]
            )
            positive = signed_changes > 1e-10
            negative = signed_changes < -1e-10
            if signed_changes.size < 2:
                drift_direction.append("unknown")
            elif np.any(positive) and not np.any(negative):
                drift_direction.append("increasing")
            elif np.any(negative) and not np.any(positive):
                drift_direction.append("decreasing")
            else:
                drift_direction.append("variable")

    return {
        "consistently_flagged": consistently_flagged,
        "drift_direction": drift_direction,
        "flagged_item_ids": flagged_item_ids,
    }
