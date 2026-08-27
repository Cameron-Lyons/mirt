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
from scipy.special import expit

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_LINKING_METHODS = frozenset(
    {
        "mean_sigma",
        "mean_mean",
        "stocking_lord",
        "haebara",
        "tcc",
        "bisector",
        "orthogonal",
    }
)
_CLOSED_FORM_LINKING_METHODS = frozenset(
    {"mean_sigma", "mean_mean", "bisector", "orthogonal"}
)
_BOOTSTRAP_CHUNK_ELEMENTS = 1_000_000


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
    random_state: int | np.random.Generator | None = None,
) -> LinkingResult:
    """Link two IRT models using anchor items.

    Finds transformation constants A and B that place the new calibration
    onto the old/reference scale:
        theta_old_scale = A * theta_new_scale + B
        a_new_on_old_scale = a_new / A
        b_new_on_old_scale = A * b_new + B

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
    random_state : int | numpy.random.Generator | None
        Seed or generator for reproducible bootstrap standard errors.

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
    if method not in _LINKING_METHODS:
        raise ValueError(f"Unknown linking method: {method}")
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old,
        model_new,
        anchor_items_old,
        anchor_items_new,
    )
    disc_old, diff_old, lower_old, upper_old = _extract_link_parameters(
        model_old, anchors_old, "old"
    )
    disc_new, diff_new, lower_new, upper_new = _extract_link_parameters(
        model_new, anchors_new, "new"
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, weights)

    if compute_se:
        if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
            n_bootstrap, (int, np.integer)
        ):
            raise ValueError("n_bootstrap must be an integer")
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
    if purify_anchors and (
        not np.isfinite(purify_threshold) or purify_threshold <= 0.0
    ):
        raise ValueError("purify_threshold must be finite and positive")

    working_anchors_old = anchors_old.copy()
    working_anchors_new = anchors_new.copy()
    working_disc_old = disc_old.copy()
    working_diff_old = diff_old.copy()
    working_disc_new = disc_new.copy()
    working_diff_new = diff_new.copy()
    working_lower_old = lower_old.copy()
    working_upper_old = upper_old.copy()
    working_lower_new = lower_new.copy()
    working_upper_new = upper_new.copy()

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
            lower_old=working_lower_old,
            upper_old=working_upper_old,
            lower_new=working_lower_new,
            upper_new=working_upper_new,
        )
        mask = np.array([i in working_anchors_old for i in anchors_old])
        working_disc_old = disc_old[mask]
        working_diff_old = diff_old[mask]
        working_disc_new = disc_new[mask]
        working_diff_new = diff_new[mask]
        working_lower_old = lower_old[mask]
        working_upper_old = upper_old[mask]
        working_lower_new = lower_new[mask]
        working_upper_new = upper_new[mask]

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
            working_lower_old,
            working_upper_old,
            working_lower_new,
            working_upper_new,
        )
    elif method == "haebara":
        A, B, conv_info = _haebara_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            theta_grid,
            weights,
            working_lower_old,
            working_upper_old,
            working_lower_new,
            working_upper_new,
        )
    elif method == "tcc":
        A, B, conv_info = _tcc_link(
            working_disc_old,
            working_diff_old,
            working_disc_new,
            working_diff_new,
            theta_grid,
            weights,
            working_lower_old,
            working_upper_old,
            working_lower_new,
            working_upper_new,
        )
    elif method == "bisector":
        A, B, conv_info = _bisector_link(
            working_disc_old, working_diff_old, working_disc_new, working_diff_new
        )
    elif method == "orthogonal":
        A, B, conv_info = _orthogonal_link(
            working_disc_old, working_diff_old, working_disc_new, working_diff_new
        )
    A, B = _validate_linking_constants(A, B, method)

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
            working_lower_old,
            working_upper_old,
            working_lower_new,
            working_upper_new,
            random_state,
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
            anchors_old,
            theta_grid,
            lower_old,
            upper_old,
            lower_new,
            upper_new,
        )
        fit_statistics = _compute_fit_statistics(
            disc_old,
            diff_old,
            disc_new,
            diff_new,
            A,
            B,
            theta_grid,
            weights,
            lower_old,
            upper_old,
            lower_new,
            upper_new,
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


def _validate_anchor_pairs(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
) -> tuple[list[int], list[int]]:
    """Validate and normalize corresponding anchor indices."""
    if len(anchors_old) != len(anchors_new):
        raise ValueError(
            f"Anchor item lists must have same length: "
            f"{len(anchors_old)} vs {len(anchors_new)}"
        )
    if len(anchors_old) < 2:
        raise ValueError("At least 2 anchor items required for linking")

    normalized: list[list[int]] = []
    for label, anchors, n_items in (
        ("old", anchors_old, model_old.n_items),
        ("new", anchors_new, model_new.n_items),
    ):
        current: list[int] = []
        for anchor in anchors:
            if isinstance(anchor, (bool, np.bool_)) or not isinstance(
                anchor, (int, np.integer)
            ):
                raise ValueError(
                    f"Anchor indices for the {label} model must be integers"
                )
            index = int(anchor)
            if index < 0 or index >= n_items:
                raise ValueError(
                    f"Anchor index {index} out of range for the {label} model "
                    f"with {n_items} items"
                )
            current.append(index)
        if len(set(current)) != len(current):
            raise ValueError(f"Anchor indices for the {label} model must be unique")
        normalized.append(current)
    return normalized[0], normalized[1]


def _extract_link_parameters(
    model: "BaseItemModel", anchors: list[int], label: str
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extract validated unidimensional dichotomous item parameters."""
    if model.n_factors != 1:
        raise ValueError("Core linking requires unidimensional models")
    try:
        discrimination = np.asarray(model.discrimination, dtype=np.float64)
        difficulty = np.asarray(model.difficulty, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(
            "Core linking requires discrimination and difficulty parameters; "
            "use the model-specific polytomous linker"
        ) from exc

    if discrimination.ndim == 2 and discrimination.shape == (model.n_items, 1):
        discrimination = discrimination[:, 0]
    if discrimination.shape != (model.n_items,) or difficulty.shape != (model.n_items,):
        raise ValueError(
            f"Parameters for the {label} model must contain one value per item"
        )

    lower = np.asarray(
        getattr(model, "guessing", np.zeros(model.n_items)), dtype=np.float64
    )
    upper = np.asarray(
        getattr(model, "upper", np.ones(model.n_items)), dtype=np.float64
    )
    if lower.shape != (model.n_items,) or upper.shape != (model.n_items,):
        raise ValueError(
            f"Asymptotes for the {label} model must contain one value per item"
        )

    selected_disc = discrimination[anchors]
    selected_diff = difficulty[anchors]
    selected_lower = lower[anchors]
    selected_upper = upper[anchors]
    arrays = (selected_disc, selected_diff, selected_lower, selected_upper)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"Item parameters for the {label} model must be finite")
    if np.any(selected_disc <= 0.0):
        raise ValueError(
            f"Discrimination parameters for the {label} model must be positive"
        )
    if np.any(selected_lower < 0.0) or np.any(selected_upper > 1.0):
        raise ValueError(f"Asymptotes for the {label} model must lie in [0, 1]")
    if np.any(selected_lower >= selected_upper):
        raise ValueError(
            f"Lower asymptotes for the {label} model must be below upper asymptotes"
        )
    return arrays


def _validate_curve_grid(
    theta_range: tuple[float, float],
    n_theta: int,
    weights: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and construct the shared curve-matching grid."""
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

    theta_grid = np.linspace(lower, upper, int(n_theta))
    if weights is None:
        normalized_weights = stats.norm.pdf(theta_grid)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (n_theta,):
            raise ValueError(f"weights must have shape ({n_theta},)")
        if not np.all(np.isfinite(normalized_weights)):
            raise ValueError("weights must be finite")
        if np.any(normalized_weights < 0.0):
            raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(normalized_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("weights must have a positive sum")
    return theta_grid, normalized_weights / weight_sum


def _validate_linking_constants(A: float, B: float, method: str) -> tuple[float, float]:
    """Require a finite, orientation-preserving linear transformation."""
    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"{method} linking did not produce a positive finite slope")
    if not np.isfinite(shift):
        raise RuntimeError(f"{method} linking did not produce a finite intercept")
    return scale, shift


def _icc_matrix(
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    theta: NDArray[np.float64],
    lower: NDArray[np.float64] | None = None,
    upper: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate every dichotomous item curve in one stable batch."""
    if lower is None:
        lower = np.zeros_like(discrimination)
    if upper is None:
        upper = np.ones_like(discrimination)
    logistic = expit(discrimination[None, :] * (theta[:, None] - difficulty[None, :]))
    return lower[None, :] + (upper - lower)[None, :] * logistic


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
    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A must be finite and positive")
    if not np.isfinite(shift):
        raise ValueError("B must be finite")

    if not in_place:
        model = model.copy()

    disc = np.asarray(model.discrimination)
    diff = np.asarray(model.difficulty)

    new_disc = disc / scale
    new_diff = scale * diff + shift

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
        loc_old_b = np.median(diff_old)
        loc_new_b = np.median(diff_new)
        scale_old_b = np.median(np.abs(diff_old - loc_old_b)) * 1.4826
        scale_new_b = np.median(np.abs(diff_new - loc_new_b)) * 1.4826
    else:
        loc_old_b = np.mean(diff_old)
        loc_new_b = np.mean(diff_new)
        scale_old_b = np.std(diff_old, ddof=1)
        scale_new_b = np.std(diff_new, ddof=1)

    if scale_new_b < 1e-10:
        mean_disc_old = np.median(disc_old) if robust else np.mean(disc_old)
        mean_disc_new = np.median(disc_new) if robust else np.mean(disc_new)
        A = mean_disc_new / mean_disc_old
    else:
        A = scale_old_b / scale_new_b

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

    A = mean_disc_new / mean_disc_old

    B = mean_diff_old - A * mean_diff_new

    return A, B, {"method": "mean_mean", "robust": robust}


def _stocking_lord_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
) -> tuple[float, float, dict]:
    """Stocking-Lord test characteristic curve method."""
    curves_old = _icc_matrix(disc_old, diff_old, theta_grid, lower_old, upper_old)
    tcc_old = curves_old.sum(axis=1)
    initial_A, initial_B, _ = _mean_sigma_link(disc_old, diff_old, disc_new, diff_new)
    initial_A, initial_B = _validate_linking_constants(initial_A, initial_B, "initial")
    initial = np.array([np.log(initial_A), initial_B])

    def criterion(params: NDArray[np.float64]) -> float:
        log_A, B = float(params[0]), float(params[1])
        if not np.isfinite(log_A) or not np.isfinite(B) or abs(log_A) > 20.0:
            return float("inf")
        A = float(np.exp(log_A))
        curves_new = _icc_matrix(
            disc_new / A,
            A * diff_new + B,
            theta_grid,
            lower_new,
            upper_new,
        )
        return float(np.sum(weights * (tcc_old - curves_new.sum(axis=1)) ** 2))

    initial_value = criterion(initial)
    if initial_value <= np.finfo(np.float64).eps:
        return (
            initial_A,
            initial_B,
            {
                "method": "stocking_lord",
                "success": True,
                "fun": initial_value,
                "nit": 0,
            },
        )

    result = optimize.minimize(
        criterion,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Stocking-Lord linking failed to converge: {result.message}"
        )

    A, B = float(np.exp(result.x[0])), float(result.x[1])
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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
) -> tuple[float, float, dict]:
    """Haebara item-level curve matching method."""
    curves_old = _icc_matrix(disc_old, diff_old, theta_grid, lower_old, upper_old)
    initial_A, initial_B, _ = _mean_sigma_link(disc_old, diff_old, disc_new, diff_new)
    initial_A, initial_B = _validate_linking_constants(initial_A, initial_B, "initial")
    initial = np.array([np.log(initial_A), initial_B])

    def criterion(params: NDArray[np.float64]) -> float:
        log_A, B = float(params[0]), float(params[1])
        if not np.isfinite(log_A) or not np.isfinite(B) or abs(log_A) > 20.0:
            return float("inf")
        A = float(np.exp(log_A))
        curves_new = _icc_matrix(
            disc_new / A,
            A * diff_new + B,
            theta_grid,
            lower_new,
            upper_new,
        )
        return float(np.sum(weights[:, None] * (curves_old - curves_new) ** 2))

    initial_value = criterion(initial)
    if initial_value <= np.finfo(np.float64).eps:
        return (
            initial_A,
            initial_B,
            {
                "method": "haebara",
                "success": True,
                "fun": initial_value,
                "nit": 0,
            },
        )

    result = optimize.minimize(
        criterion,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Haebara linking failed to converge: {result.message}")

    A, B = float(np.exp(result.x[0])), float(result.x[1])
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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
) -> tuple[float, float, dict]:
    """Full TCC matching (equivalent to Stocking-Lord for dichotomous)."""
    A, B, info = _stocking_lord_link(
        disc_old,
        diff_old,
        disc_new,
        diff_new,
        theta_grid,
        weights,
        lower_old,
        upper_old,
        lower_new,
        upper_new,
    )
    info["method"] = "tcc"
    return A, B, info


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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
) -> AnchorDiagnostics:
    """Compute diagnostics for anchor item quality."""
    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B

    signed_diff_a = disc_old - disc_new_trans
    signed_diff_b = diff_old - diff_new_trans

    curves_old = _icc_matrix(disc_old, diff_old, theta_grid, lower_old, upper_old)
    curves_new = _icc_matrix(
        disc_new_trans, diff_new_trans, theta_grid, lower_new, upper_new
    )
    area_diff = np.asarray(
        np.trapezoid(np.abs(curves_old - curves_new), theta_grid, axis=0),
        dtype=np.float64,
    )

    combined_diff = np.sqrt(signed_diff_a**2 + signed_diff_b**2 + area_diff**2)
    robust_z = _robust_z_scores(combined_diff)
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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
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

    tcc_old = _icc_matrix(disc_old, diff_old, theta_grid, lower_old, upper_old).sum(
        axis=1
    )
    tcc_new = _icc_matrix(
        disc_new_trans,
        diff_new_trans,
        theta_grid,
        lower_new,
        upper_new,
    ).sum(axis=1)

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


def _robust_z_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute MAD-based z-scores without hiding isolated tied outliers."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    if mad >= 1e-10:
        return (values - median) / mad
    matches_median = np.isclose(values, median, rtol=1e-8, atol=1e-10)
    return np.where(matches_median, 0.0, np.inf)


def _batch_ols_slope(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute one OLS slope per row without constructing covariance matrices."""
    centered_x = x - np.mean(x, axis=1, keepdims=True)
    centered_y = y - np.mean(y, axis=1, keepdims=True)
    degrees_of_freedom = x.shape[1] - 1
    variance = np.sum(centered_x * centered_x, axis=1) / degrees_of_freedom
    covariance = np.sum(centered_x * centered_y, axis=1) / degrees_of_freedom
    slopes = np.zeros(x.shape[0], dtype=np.float64)
    np.divide(covariance, variance, out=slopes, where=variance >= 1e-10)
    return slopes


def _batch_bisector_links(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute bisector links for a batch of paired anchor samples."""
    slope_a_on_b = _batch_ols_slope(disc_new, disc_old)
    slope_b_on_a = _batch_ols_slope(disc_old, disc_new)
    bisector_slope = np.zeros_like(slope_a_on_b)
    valid_disc = np.abs(slope_b_on_a) >= 1e-10
    np.divide(
        slope_a_on_b,
        slope_b_on_a,
        out=bisector_slope,
        where=valid_disc,
    )
    bisector_slope = np.sign(slope_a_on_b) * np.sqrt(np.abs(bisector_slope))
    A_disc = np.ones_like(bisector_slope)
    invertible_disc = valid_disc & (np.abs(bisector_slope) >= 1e-10)
    np.divide(1.0, bisector_slope, out=A_disc, where=invertible_disc)

    slope_b_y_on_x = _batch_ols_slope(diff_new, diff_old)
    slope_b_x_on_y = _batch_ols_slope(diff_old, diff_new)
    bisector_slope_b = np.zeros_like(slope_b_y_on_x)
    valid_diff = np.abs(slope_b_x_on_y) >= 1e-10
    np.divide(
        slope_b_y_on_x,
        slope_b_x_on_y,
        out=bisector_slope_b,
        where=valid_diff,
    )
    bisector_slope_b = np.sign(slope_b_y_on_x) * np.sqrt(np.abs(bisector_slope_b))
    A_diff = np.where(valid_diff, bisector_slope_b, 1.0)

    A = (A_disc + A_diff) / 2.0
    B = np.mean(diff_old, axis=1) - A * np.mean(diff_new, axis=1)
    return A, B


def _batch_orthogonal_component(
    old: NDArray[np.float64], new: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute one Deming-regression slope per paired sample."""
    var_old = np.var(old, axis=1, ddof=1)
    var_new = np.var(new, axis=1, ddof=1)
    centered_old = old - np.mean(old, axis=1, keepdims=True)
    centered_new = new - np.mean(new, axis=1, keepdims=True)
    covariance = np.sum(centered_old * centered_new, axis=1) / (old.shape[1] - 1)
    variance_difference = var_old - var_new
    discriminant = variance_difference**2 + 4.0 * covariance**2
    slopes = np.ones(old.shape[0], dtype=np.float64)
    valid = np.abs(covariance) >= 1e-10
    np.divide(
        variance_difference + np.sqrt(discriminant),
        2.0 * covariance,
        out=slopes,
        where=valid,
    )
    return slopes


def _batch_orthogonal_links(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute orthogonal links for a batch of paired anchor samples."""
    A_disc = _batch_orthogonal_component(disc_old, disc_new)
    A_diff = _batch_orthogonal_component(diff_old, diff_new)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (1.0 / A_disc + A_diff) / 2.0
    B = np.mean(diff_old, axis=1) - A * np.mean(diff_new, axis=1)
    return A, B


def _closed_form_bootstrap_samples(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    robust: bool = False,
    fallback_on_either_scale: bool = False,
    chunk_size: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate closed-form paired-anchor bootstrap replicates in chunks."""
    if method not in _CLOSED_FORM_LINKING_METHODS:
        raise ValueError(f"{method} is not a closed-form linking method")
    n_items = disc_old.size
    if chunk_size is None:
        chunk_size = max(
            1,
            min(n_bootstrap, _BOOTSTRAP_CHUNK_ELEMENTS // max(1, n_items)),
        )
    elif isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("chunk_size must be a positive integer")
    elif chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    A_samples = np.empty(n_bootstrap, dtype=np.float64)
    B_samples = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, int(chunk_size)):
        stop = min(start + int(chunk_size), n_bootstrap)
        sampled = rng.integers(0, n_items, size=(stop - start, n_items))
        sampled_disc_old = disc_old[sampled]
        sampled_diff_old = diff_old[sampled]
        sampled_disc_new = disc_new[sampled]
        sampled_diff_new = diff_new[sampled]

        if method in {"mean_sigma", "mean_mean"}:
            location = np.median if robust else np.mean
            disc_location_old = location(sampled_disc_old, axis=1)
            disc_location_new = location(sampled_disc_new, axis=1)
            diff_location_old = location(sampled_diff_old, axis=1)
            diff_location_new = location(sampled_diff_new, axis=1)
            if method == "mean_mean":
                A = disc_location_new / disc_location_old
            else:
                if robust:
                    scale_old = (
                        np.median(
                            np.abs(sampled_diff_old - diff_location_old[:, None]),
                            axis=1,
                        )
                        * 1.4826
                    )
                    scale_new = (
                        np.median(
                            np.abs(sampled_diff_new - diff_location_new[:, None]),
                            axis=1,
                        )
                        * 1.4826
                    )
                else:
                    scale_old = np.std(sampled_diff_old, axis=1, ddof=1)
                    scale_new = np.std(sampled_diff_new, axis=1, ddof=1)
                fallback = scale_new < 1e-10
                if fallback_on_either_scale:
                    fallback |= scale_old < 1e-10
                A = np.empty(stop - start, dtype=np.float64)
                np.divide(scale_old, scale_new, out=A, where=~fallback)
                A[fallback] = disc_location_new[fallback] / disc_location_old[fallback]
            B = diff_location_old - A * diff_location_new
        elif method == "bisector":
            A, B = _batch_bisector_links(
                sampled_disc_old,
                sampled_diff_old,
                sampled_disc_new,
                sampled_diff_new,
            )
        else:
            A, B = _batch_orthogonal_links(
                sampled_disc_old,
                sampled_diff_old,
                sampled_disc_new,
                sampled_diff_new,
            )

        A_samples[start:stop] = A
        B_samples[start:stop] = B
    return A_samples, B_samples


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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute bootstrap standard errors for linking constants."""
    rng = np.random.default_rng(random_state)
    n_items = len(disc_old)

    if method in _CLOSED_FORM_LINKING_METHODS:
        A_samples, B_samples = _closed_form_bootstrap_samples(
            disc_old,
            diff_old,
            disc_new,
            diff_new,
            method,
            n_bootstrap,
            rng,
            robust=robust,
        )
        invalid = np.flatnonzero(
            (~np.isfinite(A_samples)) | (A_samples <= 0.0) | (~np.isfinite(B_samples))
        )
        if invalid.size:
            index = int(invalid[0])
            _validate_linking_constants(A_samples[index], B_samples[index], method)
        return float(np.std(A_samples, ddof=1)), float(np.std(B_samples, ddof=1))

    if lower_old is None:
        lower_old = np.zeros(n_items)
    if upper_old is None:
        upper_old = np.ones(n_items)
    if lower_new is None:
        lower_new = np.zeros(n_items)
    if upper_new is None:
        upper_new = np.ones(n_items)

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
        l_old_b = lower_old[idx]
        u_old_b = upper_old[idx]
        l_new_b = lower_new[idx]
        u_new_b = upper_new[idx]

        if method in ("stocking_lord", "tcc"):
            A, B, _ = _stocking_lord_link(
                d_old_b,
                b_old_b,
                d_new_b,
                b_new_b,
                theta_grid,
                weights,
                l_old_b,
                u_old_b,
                l_new_b,
                u_new_b,
            )
        elif method == "haebara":
            A, B, _ = _haebara_link(
                d_old_b,
                b_old_b,
                d_new_b,
                b_new_b,
                theta_grid,
                weights,
                l_old_b,
                u_old_b,
                l_new_b,
                u_new_b,
            )
        else:
            raise ValueError(f"Unknown linking method: {method}")

        A_samples[b], B_samples[b] = _validate_linking_constants(A, B, method)

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
    lower_old: NDArray[np.float64] | None = None,
    upper_old: NDArray[np.float64] | None = None,
    lower_new: NDArray[np.float64] | None = None,
    upper_new: NDArray[np.float64] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Iteratively purify anchor set by removing drifting items."""
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    n_items = len(disc_old)
    if lower_old is None:
        lower_old = np.zeros(n_items)
    if upper_old is None:
        upper_old = np.ones(n_items)
    if lower_new is None:
        lower_new = np.zeros(n_items)
    if upper_new is None:
        upper_new = np.ones(n_items)

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
        l_old = lower_old[mask]
        u_old = upper_old[mask]
        l_new = lower_new[mask]
        u_new = upper_new[mask]

        if method == "mean_sigma":
            A, B, _ = _mean_sigma_link(d_old, b_old, d_new, b_new, False)
        elif method == "mean_mean":
            A, B, _ = _mean_mean_link(d_old, b_old, d_new, b_new, False)
        elif method in ("stocking_lord", "tcc"):
            A, B, _ = _stocking_lord_link(
                d_old,
                b_old,
                d_new,
                b_new,
                theta_grid,
                weights,
                l_old,
                u_old,
                l_new,
                u_new,
            )
        elif method == "haebara":
            A, B, _ = _haebara_link(
                d_old,
                b_old,
                d_new,
                b_new,
                theta_grid,
                weights,
                l_old,
                u_old,
                l_new,
                u_new,
            )
        else:
            A, B, _ = _mean_sigma_link(d_old, b_old, d_new, b_new, False)

        d_new_trans = d_new / A
        b_new_trans = A * b_new + B

        diff_a = d_old - d_new_trans
        diff_b = b_old - b_new_trans
        curves_old = _icc_matrix(d_old, b_old, theta_grid, l_old, u_old)
        curves_new = _icc_matrix(d_new_trans, b_new_trans, theta_grid, l_new, u_new)
        areas = np.trapezoid(np.abs(curves_old - curves_new), theta_grid, axis=0)
        combined = np.sqrt(diff_a**2 + diff_b**2 + areas**2)
        z_scores = _robust_z_scores(combined)
        max_z_idx = np.argmax(np.abs(z_scores))
        max_z = np.abs(z_scores[max_z_idx])

        if max_z <= threshold:
            break

        remove_idx = current_old[max_z_idx]
        removed.append(remove_idx)
        del current_old[max_z_idx]
        del current_new[max_z_idx]

    return current_old, current_new, removed
