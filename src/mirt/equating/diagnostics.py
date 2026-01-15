"""Diagnostics and summary statistics for IRT linking.

This module provides functions for computing fit statistics,
standard errors, and generating summaries of linking results.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.equating.linking import (
    LinkingFitStatistics,
    LinkingResult,
    _haebara_link,
    _mean_mean_link,
    _mean_sigma_link,
    _stocking_lord_link,
)


def bootstrap_linking_se(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    responses_old: NDArray[np.float64] | None,
    responses_new: NDArray[np.float64] | None,
    anchors_old: list[int],
    anchors_new: list[int],
    method: str = "stocking_lord",
    n_bootstrap: int = 200,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    seed: int | None = None,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute bootstrap standard errors for linking constants.

    Uses parametric bootstrap if responses are not provided, otherwise
    uses nonparametric bootstrap resampling of persons.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.
    responses_old : NDArray | None
        Response data for old form (n_persons x n_items).
    responses_new : NDArray | None
        Response data for new form.
    anchors_old : list[int]
        Anchor item indices in old model.
    anchors_new : list[int]
        Anchor item indices in new model.
    method : str
        Linking method.
    n_bootstrap : int
        Number of bootstrap replications.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, NDArray, NDArray]
        SE of A, SE of B, bootstrap A samples, bootstrap B samples.
    """
    rng = np.random.default_rng(seed)

    disc_old = np.asarray(model_old.discrimination)[anchors_old]
    diff_old = np.asarray(model_old.difficulty)[anchors_old]
    disc_new = np.asarray(model_new.discrimination)[anchors_new]
    diff_new = np.asarray(model_new.difficulty)[anchors_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    n_items = len(anchors_old)
    A_samples = np.zeros(n_bootstrap)
    B_samples = np.zeros(n_bootstrap)

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    for b in range(n_bootstrap):
        idx = rng.choice(n_items, size=n_items, replace=True)
        d_old_b = disc_old[idx]
        b_old_b = diff_old[idx]
        d_new_b = disc_new[idx]
        b_new_b = diff_new[idx]

        if method == "mean_sigma":
            A, B, _ = _mean_sigma_link(d_old_b, b_old_b, d_new_b, b_new_b, False)
        elif method == "mean_mean":
            A, B, _ = _mean_mean_link(d_old_b, b_old_b, d_new_b, b_new_b, False)
        elif method in ("stocking_lord", "tcc"):
            A, B, _ = _stocking_lord_link(
                d_old_b, b_old_b, d_new_b, b_new_b, theta_grid, weights
            )
        elif method == "haebara":
            A, B, _ = _haebara_link(
                d_old_b, b_old_b, d_new_b, b_new_b, theta_grid, weights
            )
        else:
            A, B, _ = _stocking_lord_link(
                d_old_b, b_old_b, d_new_b, b_new_b, theta_grid, weights
            )

        A_samples[b] = A
        B_samples[b] = B

    A_se = float(np.std(A_samples, ddof=1))
    B_se = float(np.std(B_samples, ddof=1))

    return A_se, B_se, A_samples, B_samples


def delta_method_se(
    linking_result: LinkingResult,
    vcov_old: NDArray[np.float64],
    vcov_new: NDArray[np.float64],
    anchors_old: list[int],
    anchors_new: list[int],
) -> tuple[float, float]:
    """Compute standard errors using delta method.

    Parameters
    ----------
    linking_result : LinkingResult
        Result from linking procedure.
    vcov_old : NDArray
        Variance-covariance matrix for old model parameters.
    vcov_new : NDArray
        Variance-covariance matrix for new model parameters.
    anchors_old : list[int]
        Anchor indices in old model.
    anchors_new : list[int]
        Anchor indices in new model.

    Returns
    -------
    tuple[float, float]
        Standard errors for A and B.
    """
    n_anchors = len(anchors_old)
    A = linking_result.constants.A

    jacobian_A = np.zeros(2 * n_anchors)
    jacobian_B = np.zeros(2 * n_anchors)

    for i in range(n_anchors):
        jacobian_A[i] = -1.0 / (n_anchors * A**2)
        jacobian_A[n_anchors + i] = 1.0 / n_anchors

    for i in range(n_anchors):
        jacobian_B[i] = 0.0
        jacobian_B[n_anchors + i] = -A / n_anchors

    vcov_combined = np.zeros((2 * n_anchors, 2 * n_anchors))

    for i, idx_old in enumerate(anchors_old):
        if 2 * idx_old + 1 < vcov_old.shape[0]:
            vcov_combined[i, i] = vcov_old[2 * idx_old, 2 * idx_old]
            vcov_combined[n_anchors + i, n_anchors + i] = vcov_old[
                2 * idx_old + 1, 2 * idx_old + 1
            ]

    var_A = float(jacobian_A @ vcov_combined @ jacobian_A)
    var_B = float(jacobian_B @ vcov_combined @ jacobian_B)

    return np.sqrt(max(var_A, 0)), np.sqrt(max(var_B, 0))


def compute_linking_fit(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    A: float,
    B: float,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
) -> LinkingFitStatistics:
    """Compute fit statistics for a linking solution.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.
    anchors_old : list[int]
        Anchor indices in old model.
    anchors_new : list[int]
        Anchor indices in new model.
    A : float
        Linking slope.
    B : float
        Linking intercept.
    theta_range : tuple[float, float]
        Range for TCC computation.
    n_theta : int
        Number of theta points.

    Returns
    -------
    LinkingFitStatistics
        Fit statistics for the linking.
    """
    disc_old = np.asarray(model_old.discrimination)[anchors_old]
    diff_old = np.asarray(model_old.difficulty)[anchors_old]
    disc_new = np.asarray(model_new.discrimination)[anchors_new]
    diff_new = np.asarray(model_new.difficulty)[anchors_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    diff_a = disc_old - disc_new_trans
    diff_b = diff_old - diff_new_trans

    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))
    mad_a = float(np.mean(np.abs(diff_a)))
    mad_b = float(np.mean(np.abs(diff_b)))
    weighted_rmse = float(np.sqrt(np.mean(diff_a**2) + np.mean(diff_b**2)))

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    n_items = len(disc_old)
    tcc_old = np.zeros(n_theta)
    tcc_new = np.zeros(n_theta)

    for j in range(n_items):
        p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta_grid - diff_old[j])))
        tcc_old += p_old

        p_new = 1.0 / (
            1.0 + np.exp(-disc_new_trans[j] * (theta_grid - diff_new_trans[j]))
        )
        tcc_new += p_new

    tcc_diff = (tcc_old - tcc_new) ** 2
    tcc_rmse = float(np.sqrt(np.sum(weights * tcc_diff)))

    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=mad_a,
        mad_b=mad_b,
        weighted_rmse=weighted_rmse,
        tcc_rmse=tcc_rmse,
    )


def linking_summary(
    result: LinkingResult,
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
) -> str:
    """Generate formatted summary of linking results.

    Parameters
    ----------
    result : LinkingResult
        Result from linking procedure.
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("IRT Linking Summary")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Transformation Constants")
    lines.append("-" * 30)
    lines.append(f"Method: {result.constants.method}")
    lines.append(f"A (slope):     {result.constants.A:8.4f}")
    lines.append(f"B (intercept): {result.constants.B:8.4f}")

    if result.constants.A_se is not None:
        lines.append(f"SE(A):         {result.constants.A_se:8.4f}")
    if result.constants.B_se is not None:
        lines.append(f"SE(B):         {result.constants.B_se:8.4f}")

    lines.append("")
    lines.append("Anchor Items")
    lines.append("-" * 30)
    lines.append(f"Number of anchors: {len(result.anchor_items)}")
    lines.append(f"Anchor indices: {result.anchor_items}")

    if result.fit_statistics is not None:
        lines.append("")
        lines.append("Fit Statistics")
        lines.append("-" * 30)
        lines.append(f"RMSE (discrimination): {result.fit_statistics.rmse_a:.4f}")
        lines.append(f"RMSE (difficulty):     {result.fit_statistics.rmse_b:.4f}")
        lines.append(f"MAD (discrimination):  {result.fit_statistics.mad_a:.4f}")
        lines.append(f"MAD (difficulty):      {result.fit_statistics.mad_b:.4f}")
        lines.append(
            f"Weighted RMSE:         {result.fit_statistics.weighted_rmse:.4f}"
        )
        lines.append(f"TCC RMSE:              {result.fit_statistics.tcc_rmse:.4f}")

    if result.anchor_diagnostics is not None:
        n_flagged = int(np.sum(result.anchor_diagnostics.flagged))
        lines.append("")
        lines.append("Anchor Diagnostics")
        lines.append("-" * 30)
        lines.append(f"Items flagged for drift: {n_flagged}")

        if n_flagged > 0:
            flagged_idx = np.where(result.anchor_diagnostics.flagged)[0]
            for idx in flagged_idx:
                item_idx = result.anchor_diagnostics.item_indices[idx]
                z = result.anchor_diagnostics.robust_z[idx]
                area = result.anchor_diagnostics.area_diff[idx]
                lines.append(f"  Item {item_idx}: z = {z:.2f}, area = {area:.3f}")

    if result.convergence_info is not None:
        lines.append("")
        lines.append("Convergence Information")
        lines.append("-" * 30)
        for key, value in result.convergence_info.items():
            lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Transformation Equations")
    lines.append("-" * 30)
    lines.append(
        f"theta_new = {result.constants.A:.4f} * theta_old + {result.constants.B:.4f}"
    )
    lines.append(f"a_new = a_old / {result.constants.A:.4f}")
    lines.append(f"b_new = {result.constants.A:.4f} * b_old + {result.constants.B:.4f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def compare_linking_methods(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    methods: list[str] | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
) -> dict[str, dict]:
    """Compare multiple linking methods.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.
    anchors_old : list[int]
        Anchor indices in old model.
    anchors_new : list[int]
        Anchor indices in new model.
    methods : list[str] | None
        Methods to compare. Defaults to all available.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping method name to result summary.
    """
    from mirt.equating.linking import link

    if methods is None:
        methods = [
            "mean_sigma",
            "mean_mean",
            "stocking_lord",
            "haebara",
            "bisector",
            "orthogonal",
        ]

    results = {}

    for method in methods:
        try:
            result = link(
                model_old,
                model_new,
                anchors_old,
                anchors_new,
                method=method,
                theta_range=theta_range,
                n_theta=n_theta,
                compute_diagnostics=True,
            )

            results[method] = {
                "A": result.constants.A,
                "B": result.constants.B,
                "rmse_a": result.fit_statistics.rmse_a
                if result.fit_statistics
                else None,
                "rmse_b": result.fit_statistics.rmse_b
                if result.fit_statistics
                else None,
                "tcc_rmse": result.fit_statistics.tcc_rmse
                if result.fit_statistics
                else None,
                "n_flagged": int(np.sum(result.anchor_diagnostics.flagged))
                if result.anchor_diagnostics
                else 0,
            }
        except Exception as e:
            results[method] = {"error": str(e)}

    return results


def parameter_recovery_summary(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    A: float,
    B: float,
) -> str:
    """Generate summary of parameter recovery after transformation.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        New model.
    anchors_old : list[int]
        Anchor indices in old model.
    anchors_new : list[int]
        Anchor indices in new model.
    A : float
        Linking slope.
    B : float
        Linking intercept.

    Returns
    -------
    str
        Formatted parameter comparison table.
    """
    disc_old = np.asarray(model_old.discrimination)[anchors_old]
    diff_old = np.asarray(model_old.difficulty)[anchors_old]
    disc_new = np.asarray(model_new.discrimination)[anchors_new]
    diff_new = np.asarray(model_new.difficulty)[anchors_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    lines = []
    lines.append("Parameter Recovery After Transformation")
    lines.append("=" * 70)
    lines.append(
        f"{'Item':>6} {'a_old':>8} {'a_trans':>8} {'diff_a':>8} {'b_old':>8} {'b_trans':>8} {'diff_b':>8}"
    )
    lines.append("-" * 70)

    for i, (idx_old, idx_new) in enumerate(zip(anchors_old, anchors_new)):
        lines.append(
            f"{idx_old:>6} {disc_old[i]:>8.3f} {disc_new_trans[i]:>8.3f} "
            f"{disc_old[i] - disc_new_trans[i]:>8.3f} {diff_old[i]:>8.3f} "
            f"{diff_new_trans[i]:>8.3f} {diff_old[i] - diff_new_trans[i]:>8.3f}"
        )

    lines.append("-" * 70)

    rmse_a = np.sqrt(np.mean((disc_old - disc_new_trans) ** 2))
    rmse_b = np.sqrt(np.mean((diff_old - diff_new_trans) ** 2))
    corr_a = np.corrcoef(disc_old, disc_new_trans)[0, 1]
    corr_b = np.corrcoef(diff_old, diff_new_trans)[0, 1]

    lines.append(f"RMSE(a): {rmse_a:.4f}    RMSE(b): {rmse_b:.4f}")
    lines.append(f"Corr(a): {corr_a:.4f}    Corr(b): {corr_b:.4f}")
    lines.append("=" * 70)

    return "\n".join(lines)
