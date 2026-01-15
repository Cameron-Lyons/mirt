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
    if len(anchor_items_old) != len(anchor_items_new):
        raise ValueError("Anchor item lists must have same length")

    disc_old = np.asarray(model_old.discrimination)[anchor_items_old]
    diff_old = np.asarray(model_old.difficulty)[anchor_items_old]
    disc_new = np.asarray(model_new.discrimination)[anchor_items_new]
    diff_new = np.asarray(model_new.difficulty)[anchor_items_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    if A is None or B is None:
        from mirt.equating.linking import link

        result = link(
            model_old,
            model_new,
            anchor_items_old,
            anchor_items_new,
            method="stocking_lord",
            compute_diagnostics=False,
        )
        A = result.constants.A
        B = result.constants.B

    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    if threshold is None:
        default_thresholds = {"robust_z": 2.5, "3sigma": 3.0, "area": 0.2, "wald": 0.05}
        threshold = default_thresholds.get(method, 2.5)

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

    elif method == "area":
        theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
        drift_stats = _area_drift(
            disc_old, diff_old, disc_new_trans, diff_new_trans, theta_grid
        )
        p_values = None
        flagged = drift_stats > threshold
        effect_sizes = drift_stats.copy()

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
            anchor_items_old,
            anchor_items_new,
            A,
        )
        flagged = p_values < threshold
        effect_sizes = _compute_drift_effect_sizes(
            disc_old, diff_old, disc_new_trans, diff_new_trans
        )

    else:
        raise ValueError(f"Unknown drift detection method: {method}")

    flagged_items = [anchor_items_old[i] for i, f in enumerate(flagged) if f]

    return DriftResult(
        drift_statistics=drift_stats,
        flagged_items=flagged_items,
        p_values=p_values,
        effect_sizes=effect_sizes,
        method=method,
        threshold=float(threshold),
    )


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
        z_scores = np.zeros_like(combined)
    else:
        z_scores = (combined - median_diff) / mad

    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

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
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unsigned area difference between ICCs."""
    n_items = len(disc_old)
    area_diff = np.zeros(n_items)

    for j in range(n_items):
        p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta_grid - diff_old[j])))
        p_new = 1.0 / (1.0 + np.exp(-disc_new[j] * (theta_grid - diff_new[j])))
        area_diff[j] = np.trapezoid(np.abs(p_old - p_new), theta_grid)

    return area_diff


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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wald test for parameter drift."""
    n_items = len(disc_old)
    wald_stats = np.zeros(n_items)
    p_values = np.zeros(n_items)

    se_disc_old = np.asarray(se_old.get("discrimination", np.zeros(len(disc_old))))
    se_diff_old = np.asarray(se_old.get("difficulty", np.zeros(len(diff_old))))
    se_disc_new = np.asarray(se_new.get("discrimination", np.zeros(len(disc_new))))
    se_diff_new = np.asarray(se_new.get("difficulty", np.zeros(len(diff_new))))

    if se_disc_old.ndim > 1:
        se_disc_old = se_disc_old[:, 0]
    if se_disc_new.ndim > 1:
        se_disc_new = se_disc_new[:, 0]

    for i, (idx_old, idx_new) in enumerate(zip(items_old, items_new)):
        diff_a = disc_old[i] - disc_new[i]
        diff_b = diff_old[i] - diff_new[i]

        var_a = se_disc_old[idx_old] ** 2 + (se_disc_new[idx_new] / A) ** 2
        var_b = se_diff_old[idx_old] ** 2 + (A * se_diff_new[idx_new]) ** 2

        var_a = max(var_a, 1e-10)
        var_b = max(var_b, 1e-10)

        wald_a = diff_a**2 / var_a
        wald_b = diff_b**2 / var_b

        wald_stats[i] = wald_a + wald_b

        p_values[i] = 1 - stats.chi2.cdf(wald_stats[i], df=2)

    return wald_stats, p_values


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

    pooled_sd_a = max(pooled_sd_a, 1e-10)
    pooled_sd_b = max(pooled_sd_b, 1e-10)

    effect_a = diff_a / pooled_sd_a
    effect_b = diff_b / pooled_sd_b

    return np.sqrt(effect_a**2 + effect_b**2)


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
    disc_old = float(np.asarray(model_old.discrimination)[item_old].ravel()[0])
    diff_old = float(np.asarray(model_old.difficulty)[item_old])
    disc_new = float(np.asarray(model_new.discrimination)[item_new].ravel()[0])
    diff_new = float(np.asarray(model_new.difficulty)[item_new])

    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)

    p_old = 1.0 / (1.0 + np.exp(-disc_old * (theta_grid - diff_old)))
    p_new = 1.0 / (1.0 + np.exp(-disc_new_trans * (theta_grid - diff_new_trans)))

    signed_area = float(np.trapezoid(p_old - p_new, theta_grid))
    unsigned_area = float(np.trapezoid(np.abs(p_old - p_new), theta_grid))

    return signed_area, unsigned_area
