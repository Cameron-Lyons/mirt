"""Item parameter drift detection for IRT linking.

This module provides methods for detecting item parameter drift
between test administrations, including robust z-statistics,
area-based methods, and Wald tests.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

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
        theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
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

    normalized: list[list[int]] = []
    for label, items, n_items in (
        ("old", items_old, model_old.n_items),
        ("new", items_new, model_new.n_items),
    ):
        current: list[int] = []
        for item in items:
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)
            ):
                raise ValueError(
                    f"Anchor indices for the {label} model must be integers"
                )
            index = int(item)
            if index < 0 or index >= n_items:
                raise ValueError(
                    f"Anchor index {index} out of range for the {label} model "
                    f"with {n_items} items"
                )
            current.append(index)
        if len(set(current)) != len(current):
            raise ValueError(f"Anchor indices for the {label} model must be unique")
        normalized.append(current)

    _validate_theta_grid(theta_range, n_theta)
    return normalized[0], normalized[1]


def _validate_theta_grid(
    theta_range: tuple[float, float], n_theta: int
) -> tuple[float, float]:
    """Validate a one-dimensional integration grid."""
    if not isinstance(theta_range, (tuple, list)) or len(theta_range) != 2:
        raise ValueError("theta_range must contain exactly two values")
    lower, upper = float(theta_range[0]), float(theta_range[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite, increasing values")
    if isinstance(n_theta, (bool, np.bool_)) or not isinstance(
        n_theta, (int, np.integer)
    ):
        raise ValueError("n_theta must be an integer")
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")
    return lower, upper


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

    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A must be finite and positive")
    if not np.isfinite(shift):
        raise ValueError("B must be finite")
    return scale, shift


def _estimate_curve_link(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    items_old: list[int],
    items_new: list[int],
    theta_range: tuple[float, float],
    n_theta: int,
) -> tuple[float, float]:
    """Estimate positive linking constants from model-native expected scores."""
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights /= weights.sum()
    old_tcc = _expected_item_score_curves(model_old, theta_grid, items_old).sum(axis=1)

    def criterion(params: NDArray[np.float64]) -> float:
        scale = float(np.exp(params[0]))
        shift = float(params[1])
        theta_new = (theta_grid - shift) / scale
        new_tcc = _expected_item_score_curves(model_new, theta_new, items_new).sum(
            axis=1
        )
        return float(np.sum(weights * (old_tcc - new_tcc) ** 2))

    initial = np.array([0.0, 0.0])
    if criterion(initial) <= np.finfo(np.float64).eps:
        return 1.0, 0.0

    result = optimize.minimize(
        criterion,
        initial,
        method="Powell",
        bounds=[(np.log(0.05), np.log(20.0)), (-20.0, 20.0)],
        options={"xtol": 1e-8, "ftol": 1e-12, "maxiter": 500},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Drift linking failed to converge: {result.message}")
    return float(np.exp(result.x[0])), float(result.x[1])


def _expected_item_score_curves(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    items: list[int],
) -> NDArray[np.float64]:
    """Evaluate expected item scores, batching dense selections."""
    use_full_batch = len(items) * 4 >= model.n_items or model.n_items <= 256
    if use_full_batch:
        probabilities = np.asarray(model.probability(theta), dtype=np.float64)
        if model.is_polytomous:
            categories = np.arange(probabilities.shape[2], dtype=np.float64)
            curves = np.sum(probabilities * categories[None, None, :], axis=2)
        else:
            curves = probabilities
        selected = curves[:, items]
    else:
        columns = []
        for item in items:
            probabilities = np.asarray(
                model.probability(theta, item_idx=item), dtype=np.float64
            )
            if model.is_polytomous:
                categories = np.arange(probabilities.shape[1], dtype=np.float64)
                columns.append(np.sum(probabilities * categories[None, :], axis=1))
            else:
                columns.append(probabilities)
        selected = np.column_stack(columns)

    expected_shape = (len(theta), len(items))
    if selected.shape != expected_shape:
        raise ValueError(
            f"Model returned expected-score curves with shape {selected.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(selected)):
        raise ValueError("Model returned non-finite expected-score curves")
    return selected


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
    diff_a = disc_old - disc_new
    diff_b = diff_old - diff_new

    combined = np.sqrt(diff_a**2 + diff_b**2)

    median_diff = np.median(combined)
    mad = np.median(np.abs(combined - median_diff)) * 1.4826

    if mad < 1e-10:
        matches_median = np.isclose(combined, median_diff, rtol=1e-8, atol=1e-10)
        z_scores = np.where(matches_median, 0.0, np.inf)
    else:
        z_scores = (combined - median_diff) / mad

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
    curves_old = _expected_item_score_curves(model_old, theta_grid, items_old)
    theta_new = (theta_grid - B) / A
    curves_new = _expected_item_score_curves(model_new, theta_new, items_new)
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

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        Purified anchors (old, new) and list of removed indices.
    """
    from mirt.equating.linking import link

    current_old = list(anchors_old)
    current_new = list(anchors_new)
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
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    curve_old = _expected_item_score_curves(model_old, theta_grid, items_old)[:, 0]
    curve_new = _expected_item_score_curves(
        model_new, (theta_grid - shift) / scale, items_new
    )[:, 0]
    difference = curve_old - curve_new

    signed_area = float(np.trapezoid(difference, theta_grid))
    unsigned_area = float(np.trapezoid(np.abs(difference), theta_grid))

    return signed_area, unsigned_area
