"""Item parameter drift detection for IRT linking.

This module provides methods for detecting item parameter drift
between test administrations, including robust z-statistics,
area-based methods, and Wald tests.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.equating._curves import _expected_item_score_curves
from mirt.equating.linking import (
    _LINKING_METHODS,
    _fit_curve_link,
    _link_form,
    _normalize_anchor_indices,
    _robust_z_scores,
    _validate_anchor_pairs,
    _validate_curve_grid,
    _validate_linking_constants,
    _validate_theta_grid,
    _validate_transform_constants,
    link,
)
from mirt.equating.polytomous import (
    _linker_for,
    _minimize_positive_scale,
    _polytomous_model_type,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class DriftResult:
    """Result of item parameter drift analysis.

    Attributes
    ----------
    drift_statistics : NDArray[np.float64]
        Drift statistic for each anchor item.
    flagged_items : list[int]
        Indices of items flagged for drift.
    p_values : NDArray[np.float64] | None
        P-values for drift statistics if available.
    effect_sizes : NDArray[np.float64] | None
        Effect size measures for drift.
    method : str
        Detection method used.
    threshold : float
        Threshold used for flagging.
    """

    drift_statistics: NDArray[np.float64]
    flagged_items: list[int]
    p_values: NDArray[np.float64] | None
    effect_sizes: NDArray[np.float64] | None
    method: str
    threshold: float


_DRIFT_METHODS = frozenset({"robust_z", "3sigma", "area", "wald"})
_DEFAULT_THRESHOLDS = {
    "robust_z": 2.5,
    "3sigma": 3.0,
    "area": 0.2,
    "wald": 0.05,
}


def detect_drift(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    method: Literal["robust_z", "3sigma", "area", "wald"] = "robust_z",
    threshold: float | None = None,
    A: float | None = None,
    B: float | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    se_old: dict[str, NDArray[np.float64]] | None = None,
    se_new: dict[str, NDArray[np.float64]] | None = None,
) -> DriftResult:
    """Detect item parameter drift between two calibrations.

    Parameters
    ----------
    model_old : BaseItemModel
        Model from first calibration (reference).
    model_new : BaseItemModel
        Model from second calibration.
    anchor_items_old : list[int]
        Indices of anchor items in old model.
    anchor_items_new : list[int]
        Indices of anchor items in new model.
    method : str
        Detection method:
        - "robust_z": Robust z-statistic using MAD
        - "3sigma": Three-sigma rule
        - "area": Area between ICC method
        - "wald": Wald test using standard errors
    threshold : float | None
        Threshold for flagging. Defaults depend on method:
        - robust_z: 2.5
        - 3sigma: 3.0
        - area: 0.2 (UADS)
        - wald: 0.05 (p-value)
    A : float | None
        Linking slope. If None, computed using Stocking-Lord.
    B : float | None
        Linking intercept. If None, computed using Stocking-Lord.
    theta_range : tuple[float, float]
        Range of theta for area calculation.
    n_theta : int
        Number of theta points.
    se_old : dict | None
        Standard errors for old model parameters.
    se_new : dict | None
        Standard errors for new model parameters.

    Returns
    -------
    DriftResult
        Drift analysis results.
    """
    if method not in _DRIFT_METHODS:
        raise ValueError(f"Unknown drift detection method: {method}")

    items_old, items_new = _validate_drift_inputs(
        model_old,
        model_new,
        anchor_items_old,
        anchor_items_new,
        theta_range,
        n_theta,
    )
    if method in {"robust_z", "3sigma"} and len(items_old) < 2:
        raise ValueError(f"{method} drift requires at least 2 anchor items")
    threshold = _validate_threshold(method, threshold)
    A, B = _resolve_linking_constants(
        model_old,
        model_new,
        items_old,
        items_new,
        A,
        B,
        theta_range,
        n_theta,
    )

    if method == "area":
        theta_grid = _validate_theta_grid(theta_range, n_theta)
        drift_stats = _area_drift(
            model_old, model_new, items_old, items_new, theta_grid, A, B
        )
        p_values = None
        flagged = drift_stats > threshold
        effect_sizes = drift_stats.copy()
        return DriftResult(
            drift_statistics=drift_stats,
            flagged_items=[items_old[i] for i, flag in enumerate(flagged) if flag],
            p_values=p_values,
            effect_sizes=effect_sizes,
            method=method,
            threshold=threshold,
        )

    disc_old, diff_old = _extract_parameter_arrays(model_old, items_old, method)
    disc_new, diff_new = _extract_parameter_arrays(model_new, items_new, method)
    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    if method == "robust_z":
        drift_stats, p_values = _robust_z_drift(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )
        flagged = np.abs(drift_stats) > threshold
        effect_sizes = _compute_drift_effect_sizes(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )

    elif method == "3sigma":
        drift_stats = _three_sigma_drift(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )
        p_values = None
        flagged = np.abs(drift_stats) > threshold
        effect_sizes = _compute_drift_effect_sizes(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )

    elif method == "wald":
        if se_old is None or se_new is None:
            raise ValueError("Standard errors required for Wald test")

        drift_stats, p_values = _wald_drift(
            disc_old,
            diff_old,
            disc_new_trans,
            diff_new_trans,
            se_old,
            se_new,
            items_old,
            items_new,
            A,
            model_old.n_items,
            model_new.n_items,
        )
        flagged = p_values < threshold
        effect_sizes = _compute_drift_effect_sizes(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )

    flagged_items = [items_old[i] for i, flag in enumerate(flagged) if flag]

    return DriftResult(
        drift_statistics=drift_stats,
        flagged_items=flagged_items,
        p_values=p_values,
        effect_sizes=effect_sizes,
        method=method,
        threshold=threshold,
    )


def _validate_drift_inputs(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
    theta_range: tuple[float, float],
    n_theta: int,
) -> tuple[list[int], list[int]]:
    """Validate shared drift-analysis inputs and normalize item indices."""
    if len(items_old) != len(items_new):
        raise ValueError("Anchor item lists must have same length")
    if not items_old:
        raise ValueError("At least one anchor item is required")
    if model_old.n_factors != 1 or model_new.n_factors != 1:
        raise ValueError("Drift analysis requires unidimensional models")

    normalized = _normalize_anchor_indices(
        items_old, items_new, model_old.n_items, model_new.n_items
    )
    # Population weights are validated only where constants are estimated.
    _validate_theta_grid(theta_range, n_theta)
    return normalized


def _validate_threshold(method: str, threshold: float | None) -> float:
    """Resolve and validate the flagging threshold for a method."""
    value = _DEFAULT_THRESHOLDS[method] if threshold is None else float(threshold)
    if not np.isfinite(value):
        raise ValueError("threshold must be finite")
    if method == "wald":
        if not 0.0 < value < 1.0:
            raise ValueError("Wald threshold must be between 0 and 1")
    elif value <= 0.0:
        raise ValueError("threshold must be positive")
    return value


def _resolve_linking_constants(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
    A: float | None,
    B: float | None,
    theta_range: tuple[float, float],
    n_theta: int,
) -> tuple[float, float]:
    """Validate explicit constants or estimate them from model score curves."""
    if (A is None) != (B is None):
        raise ValueError("A and B must be provided together")
    if A is None:
        if len(items_old) < 2:
            raise ValueError("At least 2 anchor items are required to estimate A and B")
        return _estimate_curve_link(
            model_old,
            model_new,
            items_old,
            items_new,
            theta_range,
            n_theta,
        )
    return _validate_transform_constants(A, B)


def _estimate_curve_link(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
    theta_range: tuple[float, float],
    n_theta: int,
) -> tuple[float, float]:
    """Estimate Stocking-Lord constants from model-native curves.

    Dichotomous forms with slope and location parameters use the estimator
    of ``link(method="stocking_lord")``. Built-in GRM, GPCM/PCM, and NRM
    forms of one family whose anchors have matching category counts use
    the same estimator of ``link_grm``, ``link_gpcm``, or ``link_nrm``.
    Other forms match summed expected item scores of ``model.probability``
    on the same grid.
    """
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    if _has_shared_polytomous_linker(model_old, model_new, items_old, items_new):
        linker = _linker_for(model_old)
        constants = linker(
            model_old,
            model_new,
            items_old,
            items_new,
            method="stocking_lord",
            theta_range=theta_range,
            n_theta=n_theta,
            compute_diagnostics=False,
        ).constants
        return float(constants.A), float(constants.B)

    forms = None
    if not (model_old.is_polytomous or model_new.is_polytomous):
        try:
            forms = (
                _link_form(model_old, items_old, "old"),
                _link_form(model_new, items_new, "new"),
            )
        except ValueError:
            forms = None
    if forms is not None:
        A, B, _ = _fit_curve_link(*forms, theta_grid, weights, "stocking_lord")
        return _validate_linking_constants(A, B, "Drift")

    old_tcc = _expected_item_score_curves(model_old, theta_grid, items_old)
    target = old_tcc.expected.sum(axis=1)

    def criterion(params: NDArray[np.float64]) -> float:
        log_scale, shift = float(params[0]), float(params[1])
        if not np.isfinite(log_scale) or not np.isfinite(shift) or abs(log_scale) > 20:
            return float("inf")
        theta_new = (theta_grid - shift) / np.exp(log_scale)
        new_tcc = _expected_item_score_curves(model_new, theta_new, items_new)
        return float(np.sum(weights * (target - new_tcc.expected.sum(axis=1)) ** 2))

    A, B, _ = _minimize_positive_scale(criterion, 1.0, 0.0, "Drift")
    return A, B


def _has_shared_polytomous_linker(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
) -> bool:
    """Whether a polytomous pairwise linker reproduces both forms' curves.

    The linkers evaluate closed-form category curves, so both models must
    be unmodified built-ins of one family with the same category counts on
    corresponding anchors.
    """
    if not (model_old.is_polytomous and model_new.is_polytomous):
        return False
    family = _polytomous_model_type(model_old)
    if family is None or family != _polytomous_model_type(model_new):
        return False
    if not (
        uses_builtin_model_hooks(model_old) and uses_builtin_model_hooks(model_new)
    ):
        return False
    counts_old = np.asarray(model_old.n_categories)[items_old]
    counts_new = np.asarray(model_new.n_categories)[items_new]
    return bool(np.array_equal(counts_old, counts_new))


def _extract_parameter_arrays(
    model: "BaseItemModel", items: list[int], method: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract unidimensional slope and location parameters."""
    try:
        discrimination = np.asarray(model.discrimination, dtype=np.float64)
        difficulty = np.asarray(model.difficulty, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(
            f"{method} drift requires discrimination and difficulty parameters; "
            "use area drift for this model"
        ) from exc

    if discrimination.ndim == 2 and discrimination.shape[1] == 1:
        discrimination = discrimination[:, 0]
    if discrimination.ndim != 1 or difficulty.ndim != 1:
        raise ValueError(
            f"{method} drift requires one-dimensional discrimination and difficulty arrays"
        )
    selected_disc = discrimination[items]
    selected_diff = difficulty[items]
    if not np.all(np.isfinite(selected_disc)) or not np.all(np.isfinite(selected_diff)):
        raise ValueError("Item parameters must be finite")
    return selected_disc, selected_diff


def _robust_z_drift(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Robust z-statistic for drift detection using MAD."""
    combined = np.sqrt((disc_old - disc_new) ** 2 + (diff_old - diff_new) ** 2)
    z_scores = _robust_z_scores(combined)
    p_values = 2 * stats.norm.sf(np.abs(z_scores))

    return z_scores, p_values


def _three_sigma_drift(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Three-sigma rule for drift detection."""
    diff_a = disc_old - disc_new
    diff_b = diff_old - diff_new

    combined = np.sqrt(diff_a**2 + diff_b**2)

    mean_diff = np.mean(combined)
    std_diff = np.std(combined, ddof=1)

    if std_diff < 1e-10:
        return np.zeros_like(combined)

    return (combined - mean_diff) / std_diff


def _area_drift(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
    theta_grid: NDArray[np.float64],
    A: float,
    B: float,
) -> NDArray[np.float64]:
    """Unsigned area between model-native expected item-score curves."""
    curves_old = _expected_item_score_curves(model_old, theta_grid, items_old).expected
    theta_new = (theta_grid - B) / A
    curves_new = _expected_item_score_curves(model_new, theta_new, items_new).expected
    return np.asarray(
        np.trapezoid(np.abs(curves_old - curves_new), theta_grid, axis=0),
        dtype=np.float64,
    )


def _wald_drift(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    se_old: dict[str, NDArray[np.float64]],
    se_new: dict[str, NDArray[np.float64]],
    items_old: list[int],
    items_new: list[int],
    A: float,
    n_items_old: int,
    n_items_new: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wald test for parameter drift."""
    se_disc_old = _select_standard_errors(
        se_old, "discrimination", items_old, n_items_old, "old"
    )
    se_diff_old = _select_standard_errors(
        se_old, "difficulty", items_old, n_items_old, "old"
    )
    se_disc_new = _select_standard_errors(
        se_new, "discrimination", items_new, n_items_new, "new"
    )
    se_diff_new = _select_standard_errors(
        se_new, "difficulty", items_new, n_items_new, "new"
    )

    diff_a = disc_old - disc_new
    diff_b = diff_old - diff_new
    var_a = se_disc_old**2 + (se_disc_new / A) ** 2
    var_b = se_diff_old**2 + (A * se_diff_new) ** 2
    wald_stats = diff_a**2 / var_a + diff_b**2 / var_b
    return wald_stats, stats.chi2.sf(wald_stats, df=2)


def _select_standard_errors(
    errors: dict[str, NDArray[np.float64]],
    parameter: str,
    items: list[int],
    n_model_items: int,
    label: str,
) -> NDArray[np.float64]:
    """Validate and select full-model or anchor-aligned standard errors."""
    if parameter not in errors:
        raise ValueError(f"Missing {parameter} standard errors for the {label} model")
    values = np.asarray(errors[parameter], dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            f"{parameter} standard errors for the {label} model must be one-dimensional"
        )
    if len(values) == n_model_items:
        selected = values[items]
    elif len(values) == len(items):
        selected = values
    else:
        raise ValueError(
            f"{parameter} standard errors for the {label} model must have length "
            f"{n_model_items} (all items) or {len(items)} (anchors)"
        )
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError(
            f"{parameter} standard errors for the {label} model must be finite and positive"
        )
    return selected


def _compute_drift_effect_sizes(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute standardized effect sizes for drift."""
    diff_a = disc_old - disc_new
    diff_b = diff_old - diff_new

    pooled_sd_a = np.sqrt((np.var(disc_old) + np.var(disc_new)) / 2)
    pooled_sd_b = np.sqrt((np.var(diff_old) + np.var(diff_new)) / 2)

    effect_a = _standardized_difference(diff_a, pooled_sd_a)
    effect_b = _standardized_difference(diff_b, pooled_sd_b)

    return np.sqrt(effect_a**2 + effect_b**2)


def _standardized_difference(
    difference: NDArray[np.float64], scale: float
) -> NDArray[np.float64]:
    """Standardize differences without manufacturing huge finite effects."""
    if scale >= 1e-10:
        return difference / scale
    return np.where(np.isclose(difference, 0.0, atol=1e-12), 0.0, np.inf)


def purify_anchors(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    method: str = "stocking_lord",
    threshold: float = 2.5,
    min_anchors: int = 3,
    max_iterations: int = 10,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    robust: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Iteratively remove drifting items from anchor set.

    Parameters
    ----------
    model_old : BaseItemModel
        Model from first calibration.
    model_new : BaseItemModel
        Model from second calibration.
    anchors_old : list[int]
        Initial anchor items in old model.
    anchors_new : list[int]
        Initial anchor items in new model.
    method : str
        Linking method for iteration.
    threshold : float
        Z-score threshold for removal.
    min_anchors : int
        Minimum number of anchors to retain.
    max_iterations : int
        Maximum purification iterations.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    weights : NDArray[np.float64] | None
        Population weights used in every curve-matching iteration.
    robust : bool
        Use medians and robust scale estimates for moment linking.

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        Purified anchors (old, new) and list of removed indices.
    """
    if method not in _LINKING_METHODS:
        raise ValueError(f"Unknown linking method: {method}")
    current_old, current_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    _, weights = _validate_curve_grid(theta_range, n_theta, weights)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    for name, value, minimum in (
        ("min_anchors", min_anchors, 2),
        ("max_iterations", max_iterations, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}")

    removed = []

    for _ in range(max_iterations):
        if len(current_old) <= min_anchors:
            break

        result = link(
            model_old,
            model_new,
            current_old,
            current_new,
            method=method,
            theta_range=theta_range,
            n_theta=n_theta,
            weights=weights,
            robust=robust,
            compute_diagnostics=True,
        )

        if result.anchor_diagnostics is None:
            break

        z_scores = result.anchor_diagnostics.robust_z
        max_z_idx = int(np.argmax(np.abs(z_scores)))
        max_z = float(np.abs(z_scores[max_z_idx]))

        if max_z <= threshold:
            break

        removed.append(current_old[max_z_idx])
        del current_old[max_z_idx]
        del current_new[max_z_idx]

    return current_old, current_new, removed


def signed_area_difference(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    item_old: int,
    item_new: int,
    A: float = 1.0,
    B: float = 0.0,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 101,
) -> tuple[float, float]:
    """Compute signed and unsigned area between two ICCs.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.
    item_old : int
        Item index in old model.
    item_new : int
        Item index in new model.
    A : float
        Linking slope.
    B : float
        Linking intercept.
    theta_range : tuple[float, float]
        Range of theta values.
    n_theta : int
        Number of theta points.

    Returns
    -------
    tuple[float, float]
        Signed area and unsigned area.
    """
    items_old, items_new = _validate_drift_inputs(
        model_old,
        model_new,
        [item_old],
        [item_new],
        theta_range,
        n_theta,
    )
    scale, shift = _resolve_linking_constants(
        model_old,
        model_new,
        items_old,
        items_new,
        A,
        B,
        theta_range,
        n_theta,
    )
    theta_grid = _validate_theta_grid(theta_range, n_theta)
    curve_old = _expected_item_score_curves(model_old, theta_grid, items_old).expected
    curve_new = _expected_item_score_curves(
        model_new, (theta_grid - shift) / scale, items_new
    ).expected
    difference = curve_old[:, 0] - curve_new[:, 0]

    signed_area = float(np.trapezoid(difference, theta_grid))
    unsigned_area = float(np.trapezoid(np.abs(difference), theta_grid))

    return signed_area, unsigned_area
