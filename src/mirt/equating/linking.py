"""Core IRT linking methods for test equating.

This module provides comprehensive IRT linking functionality including
mean/sigma, mean/mean, Stocking-Lord, Haebara, TCC, bisector, and
orthogonal regression methods.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

try:
    from mirt._rust_backend import RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class LinkingConstants:
    """Linear transformation constants for IRT linking.

    Attributes
    ----------
    A : float
        Slope of linear transformation.
    B : float
        Intercept of linear transformation.
    A_se : float | None
        Bootstrap standard error of A.
    B_se : float | None
        Bootstrap standard error of B.
    method : str
        Linking method used.
    """

    A: float
    B: float
    A_se: float | None = None
    B_se: float | None = None
    method: str = ""


@dataclass
class AnchorDiagnostics:
    """Diagnostics for anchor item quality assessment.

    Attributes
    ----------
    item_indices : list[int]
        Indices of anchor items.
    signed_diff_a : NDArray[np.float64]
        Signed differences in discrimination after transformation.
    signed_diff_b : NDArray[np.float64]
        Signed differences in difficulty after transformation.
    area_diff : NDArray[np.float64]
        Unsigned area between item characteristic curves.
    robust_z : NDArray[np.float64]
        Robust z-statistics for drift detection.
    flagged : NDArray[np.bool_]
        Boolean array indicating flagged items.
    """

    item_indices: list[int]
    signed_diff_a: NDArray[np.float64]
    signed_diff_b: NDArray[np.float64]
    area_diff: NDArray[np.float64]
    robust_z: NDArray[np.float64]
    flagged: NDArray[np.bool_]


@dataclass
class LinkingFitStatistics:
    """Fit statistics for linking quality assessment.

    Attributes
    ----------
    rmse_a : float
        Root mean square error for discrimination.
    rmse_b : float
        Root mean square error for difficulty.
    mad_a : float
        Mean absolute deviation for discrimination.
    mad_b : float
        Mean absolute deviation for difficulty.
    weighted_rmse : float
        Weighted RMSE combining a and b.
    tcc_rmse : float
        RMSE of test characteristic curves.
    """

    rmse_a: float
    rmse_b: float
    mad_a: float
    mad_b: float
    weighted_rmse: float
    tcc_rmse: float


@dataclass
class LinkingResult:
    """Result of IRT linking procedure.

    Attributes
    ----------
    constants : LinkingConstants
        Transformation constants A and B.
    anchor_items : list[int]
        Indices of anchor items used.
    anchor_diagnostics : AnchorDiagnostics | None
        Diagnostics for anchor item quality.
    fit_statistics : LinkingFitStatistics | None
        Fit statistics for linking quality.
    transformed_parameters : dict[str, NDArray] | None
        Transformed parameters if requested.
    convergence_info : dict | None
        Optimization convergence information.
    """

    constants: LinkingConstants
    anchor_items: list[int]
    anchor_diagnostics: AnchorDiagnostics | None = None
    fit_statistics: LinkingFitStatistics | None = None
    transformed_parameters: dict[str, NDArray] | None = None
    convergence_info: dict | None = None


def link(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    method: Literal[
        "mean_sigma",
        "mean_mean",
        "stocking_lord",
        "haebara",
        "tcc",
        "bisector",
        "orthogonal",
    ] = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    compute_se: bool = False,
    n_bootstrap: int = 200,
    robust: bool = False,
    purify_anchors: bool = False,
    purify_threshold: float = 2.5,
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link two IRT models using anchor items.

    Finds transformation constants A and B such that:
        theta_new_scale = A * theta_old_scale + B
        a_new_scale = a_old / A
        b_new_scale = A * b_old + B

    Parameters
    ----------
    model_old : BaseItemModel
        Model calibrated on old/reference scale.
    model_new : BaseItemModel
        Model calibrated on new scale.
    anchor_items_old : list[int]
        Indices of anchor items in old model.
    anchor_items_new : list[int]
        Indices of anchor items in new model.
    method : str
        Linking method:
        - "mean_sigma": Mean/sigma method (moment matching)
        - "mean_mean": Mean/mean method
        - "stocking_lord": Stocking-Lord TCC method
        - "haebara": Haebara item-level curve matching
        - "tcc": Full test characteristic curve matching
        - "bisector": Robust bisector regression
        - "orthogonal": Orthogonal/Deming regression
    theta_range : tuple[float, float]
        Range of theta values for curve matching methods.
    n_theta : int
        Number of theta points for curve matching.
    weights : NDArray[np.float64] | None
        Weights for theta points (default: normal density).
    compute_se : bool
        Whether to compute bootstrap standard errors.
    n_bootstrap : int
        Number of bootstrap replications for SE.
    robust : bool
        Use robust estimation (median instead of mean).
    purify_anchors : bool
        Whether to iteratively purify anchor set.
    purify_threshold : float
        Z-score threshold for anchor purification.
    compute_diagnostics : bool
        Whether to compute anchor diagnostics and fit statistics.

    Returns
    -------
    LinkingResult
        Transformation constants and diagnostics.

    Examples
    --------
    >>> result = link(old_model, new_model, [0, 1, 2], [0, 1, 2])
    >>> A, B = result.constants.A, result.constants.B
    >>> theta_equated = A * theta_new + B
    """
    if len(anchor_items_old) != len(anchor_items_new):
        raise ValueError(
            f"Anchor item lists must have same length: "
            f"{len(anchor_items_old)} vs {len(anchor_items_new)}"
        )

    if len(anchor_items_old) < 2:
        raise ValueError("At least 2 anchor items required for linking")

    disc_old = np.asarray(model_old.discrimination)[anchor_items_old]
    diff_old = np.asarray(model_old.difficulty)[anchor_items_old]
    disc_new = np.asarray(model_new.discrimination)[anchor_items_new]
    diff_new = np.asarray(model_new.difficulty)[anchor_items_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    working_anchors_old = list(anchor_items_old)
    working_anchors_new = list(anchor_items_new)
    working_disc_old = disc_old.copy()
    working_diff_old = diff_old.copy()
    working_disc_new = disc_new.copy()
    working_diff_new = diff_new.copy()

    if purify_anchors:
        working_anchors_old, working_anchors_new, _ = _purify_anchors_iterative(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            working_anchors_old,
            working_anchors_new,
            method=method,
            threshold=purify_threshold,
            theta_range=theta_range,
            n_theta=n_theta,
        )
        mask = [i in working_anchors_old for i in anchor_items_old]
        working_disc_old = disc_old[mask]
        working_diff_old = diff_old[mask]
        working_disc_new = disc_new[mask]
        working_diff_new = diff_new[mask]

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    if weights is None:
        weights = stats.norm.pdf(theta_grid)
        weights = weights / np.sum(weights)

    if method == "mean_sigma":
        A, B, conv_info = _mean_sigma_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            robust,
        )
    elif method == "mean_mean":
        A, B, conv_info = _mean_mean_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            robust,
        )
    elif method == "stocking_lord":
        A, B, conv_info = _stocking_lord_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            theta_grid,
            weights,
        )
    elif method == "haebara":
        A, B, conv_info = _haebara_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            theta_grid,
            weights,
        )
    elif method == "tcc":
        A, B, conv_info = _tcc_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            theta_grid,
            weights,
        )
    elif method == "bisector":
        A, B, conv_info = _bisector_link(
            working_disc_old, working_diff_old, working_disc_new, working_diff_new
        )
    elif method == "orthogonal":
        A, B, conv_info = _orthogonal_link(
            working_disc_old, working_diff_old, working_disc_new, working_diff_new
        )
    else:
        raise ValueError(f"Unknown linking method: {method}")

    A_se: float | None = None
    B_se: float | None = None
    if compute_se:
        A_se, B_se = _bootstrap_linking_se(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            method,
            n_bootstrap,
            theta_range,
            n_theta,
            robust,
        )

    anchor_diagnostics: AnchorDiagnostics | None = None
    fit_statistics: LinkingFitStatistics | None = None

    if compute_diagnostics:
        anchor_diagnostics = _compute_anchor_diagnostics(
            disc_old,
            diff_old,
            disc_new,
            diff_new,
            A,
            B,
            anchor_items_old,
            theta_grid,
        )
        fit_statistics = _compute_fit_statistics(
            disc_old, diff_old, disc_new, diff_new, A, B, theta_grid, weights
        )

    constants = LinkingConstants(
        A=float(A), B=float(B), A_se=A_se, B_se=B_se, method=method
    )

    return LinkingResult(
        constants=constants,
        anchor_items=working_anchors_old,
        anchor_diagnostics=anchor_diagnostics,
        fit_statistics=fit_statistics,
        convergence_info=conv_info,
    )


def transform_parameters(
    model: "BaseItemModel",
    A: float,
    B: float,
    in_place: bool = False,
) -> "BaseItemModel":
    """Apply linear transformation to model parameters.

    Transforms parameters so scores on the new scale equal A * old_scale + B.
    For discrimination: a_new = a / A
    For difficulty: b_new = A * b + B

    Parameters
    ----------
    model : BaseItemModel
        Model to transform.
    A : float
        Slope of transformation.
    B : float
        Intercept of transformation.
    in_place : bool
        If True, modify model in place. Otherwise return copy.

    Returns
    -------
    BaseItemModel
        Model with transformed parameters.
    """
    if not in_place:
        model = model.copy()

    disc = np.asarray(model.discrimination)
    diff = np.asarray(model.difficulty)

    new_disc = disc / A
    new_diff = A * diff + B

    model.set_parameters(discrimination=new_disc, difficulty=new_diff)

    return model


def _mean_sigma_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    robust: bool = False,
) -> tuple[float, float, dict]:
    """Mean/sigma linking method."""
    if robust:
        loc_old_a = np.median(disc_old)
        loc_new_a = np.median(disc_new)
        scale_old_a = np.median(np.abs(disc_old - loc_old_a)) * 1.4826
        scale_new_a = np.median(np.abs(disc_new - loc_new_a)) * 1.4826
        loc_old_b = np.median(diff_old)
        loc_new_b = np.median(diff_new)
    else:
        loc_old_a = np.mean(disc_old)
        loc_new_a = np.mean(disc_new)
        scale_old_a = np.std(disc_old, ddof=1)
        scale_new_a = np.std(disc_new, ddof=1)
        loc_old_b = np.mean(diff_old)
        loc_new_b = np.mean(diff_new)

    if scale_new_a < 1e-10:
        A = 1.0
    else:
        A = scale_old_a / scale_new_a

    B = loc_old_b - A * loc_new_b

    return A, B, {"method": "mean_sigma", "robust": robust}


def _mean_mean_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    robust: bool = False,
) -> tuple[float, float, dict]:
    """Mean/mean linking method."""
    if robust:
        mean_disc_old = np.median(disc_old)
        mean_disc_new = np.median(disc_new)
        mean_diff_old = np.median(diff_old)
        mean_diff_new = np.median(diff_new)
    else:
        mean_disc_old = np.mean(disc_old)
        mean_disc_new = np.mean(disc_new)
        mean_diff_old = np.mean(diff_old)
        mean_diff_new = np.mean(diff_new)

    if mean_disc_new < 1e-10:
        A = 1.0
    else:
        A = mean_disc_old / mean_disc_new

    B = mean_diff_old - A * mean_diff_new

    return A, B, {"method": "mean_mean", "robust": robust}


def _stocking_lord_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord test characteristic curve method."""
    n_items = len(disc_old)
    n_theta = len(theta_grid)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params
        tcc_old = np.zeros(n_theta)
        tcc_new = np.zeros(n_theta)

        for j in range(n_items):
            p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta_grid - diff_old[j])))
            tcc_old += p_old

            disc_trans = disc_new[j] / A
            diff_trans = A * diff_new[j] + B
            p_new = 1.0 / (1.0 + np.exp(-disc_trans * (theta_grid - diff_trans)))
            tcc_new += p_new

        diff_sq = (tcc_old - tcc_new) ** 2
        return float(np.sum(weights * diff_sq))

    result = optimize.minimize(
        criterion,
        np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )

    A, B = result.x
    return (
        float(A),
        float(B),
        {
            "method": "stocking_lord",
            "success": result.success,
            "fun": result.fun,
            "nit": result.nit,
        },
    )


def _haebara_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Haebara item-level curve matching method."""
    n_items = len(disc_old)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params
        total = 0.0

        for j in range(n_items):
            p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta_grid - diff_old[j])))

            disc_trans = disc_new[j] / A
            diff_trans = A * diff_new[j] + B
            p_new = 1.0 / (1.0 + np.exp(-disc_trans * (theta_grid - diff_trans)))

            diff_sq = (p_old - p_new) ** 2
            total += np.sum(weights * diff_sq)

        return float(total)

    result = optimize.minimize(
        criterion,
        np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )

    A, B = result.x
    return (
        float(A),
        float(B),
        {
            "method": "haebara",
            "success": result.success,
            "fun": result.fun,
            "nit": result.nit,
        },
    )


def _tcc_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Full TCC matching (equivalent to Stocking-Lord for dichotomous)."""
    return _stocking_lord_link(
        disc_old, diff_old, disc_new, diff_new, theta_grid, weights
    )


def _bisector_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Bisector regression for robust linking."""
    slope_a_on_b = _ols_slope(disc_new, disc_old)
    slope_b_on_a = _ols_slope(disc_old, disc_new)

    if abs(slope_b_on_a) < 1e-10:
        A_disc = 1.0
    else:
        bisector_slope = np.sign(slope_a_on_b) * np.sqrt(
            abs(slope_a_on_b / slope_b_on_a)
        )
        A_disc = 1.0 / bisector_slope if abs(bisector_slope) > 1e-10 else 1.0

    mean_diff_old = np.mean(diff_old)
    mean_diff_new = np.mean(diff_new)

    slope_b_y_on_x = _ols_slope(diff_new, diff_old)
    slope_b_x_on_y = _ols_slope(diff_old, diff_new)

    if abs(slope_b_x_on_y) < 1e-10:
        A_diff = 1.0
    else:
        bisector_slope_b = np.sign(slope_b_y_on_x) * np.sqrt(
            abs(slope_b_y_on_x / slope_b_x_on_y)
        )
        A_diff = bisector_slope_b

    A = (A_disc + A_diff) / 2
    B = mean_diff_old - A * mean_diff_new

    return A, B, {"method": "bisector", "A_disc": A_disc, "A_diff": A_diff}


def _orthogonal_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Orthogonal/Deming regression for linking."""
    var_disc_old = np.var(disc_old, ddof=1)
    var_disc_new = np.var(disc_new, ddof=1)
    cov_disc = np.cov(disc_old, disc_new, ddof=1)[0, 1]

    delta = 1.0
    diff_var = var_disc_old - delta * var_disc_new
    discriminant = diff_var**2 + 4 * delta * cov_disc**2

    if discriminant < 0:
        A_disc = 1.0
    else:
        A_disc = (
            (diff_var + np.sqrt(discriminant)) / (2 * cov_disc)
            if abs(cov_disc) > 1e-10
            else 1.0
        )

    mean_diff_old = np.mean(diff_old)
    mean_diff_new = np.mean(diff_new)
    var_diff_old = np.var(diff_old, ddof=1)
    var_diff_new = np.var(diff_new, ddof=1)
    cov_diff = np.cov(diff_old, diff_new, ddof=1)[0, 1]

    diff_var_b = var_diff_old - delta * var_diff_new
    discriminant_b = diff_var_b**2 + 4 * delta * cov_diff**2

    if discriminant_b < 0:
        A_diff = 1.0
    else:
        A_diff = (
            (diff_var_b + np.sqrt(discriminant_b)) / (2 * cov_diff)
            if abs(cov_diff) > 1e-10
            else 1.0
        )

    A = (1.0 / A_disc + A_diff) / 2
    B = mean_diff_old - A * mean_diff_new

    return A, B, {"method": "orthogonal", "A_disc": 1.0 / A_disc, "A_diff": A_diff}


def _ols_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute OLS slope of y on x."""
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    if var_x < 1e-10:
        return 0.0
    return float(cov_xy / var_x)


def _compute_anchor_diagnostics(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
) -> AnchorDiagnostics:
    """Compute diagnostics for anchor item quality."""
    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    signed_diff_a = disc_old - disc_new_trans
    signed_diff_b = diff_old - diff_new_trans

    n_items = len(disc_old)
    area_diff = np.zeros(n_items)
    for j in range(n_items):
        p_old = 1.0 / (1.0 + np.exp(-disc_old[j] * (theta_grid - diff_old[j])))
        p_new = 1.0 / (
            1.0 + np.exp(-disc_new_trans[j] * (theta_grid - diff_new_trans[j]))
        )
        area_diff[j] = np.trapezoid(np.abs(p_old - p_new), theta_grid)

    combined_diff = np.sqrt(signed_diff_a**2 + signed_diff_b**2)
    median_diff = np.median(combined_diff)
    mad = np.median(np.abs(combined_diff - median_diff)) * 1.4826

    if mad < 1e-10:
        robust_z = np.zeros(n_items)
    else:
        robust_z = (combined_diff - median_diff) / mad

    flagged = np.abs(robust_z) > 2.5

    return AnchorDiagnostics(
        item_indices=anchor_indices,
        signed_diff_a=signed_diff_a,
        signed_diff_b=signed_diff_b,
        area_diff=area_diff,
        robust_z=robust_z,
        flagged=flagged,
    )


def _compute_fit_statistics(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute fit statistics for linking quality."""
    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    diff_a = disc_old - disc_new_trans
    diff_b = diff_old - diff_new_trans

    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))
    mad_a = float(np.mean(np.abs(diff_a)))
    mad_b = float(np.mean(np.abs(diff_b)))

    weighted_rmse = float(np.sqrt(np.mean(diff_a**2) + np.mean(diff_b**2)))

    n_items = len(disc_old)
    tcc_old = np.zeros(len(theta_grid))
    tcc_new = np.zeros(len(theta_grid))

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


def _bootstrap_linking_se(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    method: str,
    n_bootstrap: int,
    theta_range: tuple[float, float],
    n_theta: int,
    robust: bool,
) -> tuple[float, float]:
    """Compute bootstrap standard errors for linking constants."""
    rng = np.random.default_rng()
    n_items = len(disc_old)

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
            A, B, _ = _mean_sigma_link(d_old_b, b_old_b, d_new_b, b_new_b, robust)
        elif method == "mean_mean":
            A, B, _ = _mean_mean_link(d_old_b, b_old_b, d_new_b, b_new_b, robust)
        elif method in ("stocking_lord", "tcc"):
            A, B, _ = _stocking_lord_link(
                d_old_b, b_old_b, d_new_b, b_new_b, theta_grid, weights
            )
        elif method == "haebara":
            A, B, _ = _haebara_link(
                d_old_b, b_old_b, d_new_b, b_new_b, theta_grid, weights
            )
        elif method == "bisector":
            A, B, _ = _bisector_link(d_old_b, b_old_b, d_new_b, b_new_b)
        elif method == "orthogonal":
            A, B, _ = _orthogonal_link(d_old_b, b_old_b, d_new_b, b_new_b)
        else:
            A, B = 1.0, 0.0

        A_samples[b] = A
        B_samples[b] = B

    return float(np.std(A_samples, ddof=1)), float(np.std(B_samples, ddof=1))


def _purify_anchors_iterative(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    anchors_old: list[int],
    anchors_new: list[int],
    method: str,
    threshold: float,
    theta_range: tuple[float, float],
    n_theta: int,
    min_anchors: int = 3,
    max_iterations: int = 10,
) -> tuple[list[int], list[int], list[int]]:
    """Iteratively purify anchor set by removing drifting items."""
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    current_old = list(anchors_old)
    current_new = list(anchors_new)
    removed = []

    for _ in range(max_iterations):
        if len(current_old) <= min_anchors:
            break

        mask = [i in current_old for i in anchors_old]
        d_old = disc_old[mask]
        b_old = diff_old[mask]
        d_new = disc_new[mask]
        b_new = diff_new[mask]

        if method == "mean_sigma":
            A, B, _ = _mean_sigma_link(d_old, b_old, d_new, b_new, False)
        elif method == "mean_mean":
            A, B, _ = _mean_mean_link(d_old, b_old, d_new, b_new, False)
        elif method in ("stocking_lord", "tcc"):
            A, B, _ = _stocking_lord_link(
                d_old, b_old, d_new, b_new, theta_grid, weights
            )
        elif method == "haebara":
            A, B, _ = _haebara_link(d_old, b_old, d_new, b_new, theta_grid, weights)
        else:
            A, B, _ = _mean_sigma_link(d_old, b_old, d_new, b_new, False)

        d_new_trans = d_new / A
        b_new_trans = A * b_new + B

        diff_a = d_old - d_new_trans
        diff_b = b_old - b_new_trans
        combined = np.sqrt(diff_a**2 + diff_b**2)

        median_diff = np.median(combined)
        mad = np.median(np.abs(combined - median_diff)) * 1.4826

        if mad < 1e-10:
            break

        z_scores = (combined - median_diff) / mad
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z = np.abs(z_scores[max_z_idx])

        if max_z <= threshold:
            break

        remove_idx = current_old[max_z_idx]
        removed.append(remove_idx)
        del current_old[max_z_idx]
        del current_new[max_z_idx]

    return current_old, current_new, removed
