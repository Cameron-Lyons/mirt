"""Linking methods for polytomous IRT models.

This module provides linking functions for Graded Response Model (GRM),
Generalized Partial Credit Model (GPCM), and Nominal Response Model (NRM).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.equating.linking import (
    AnchorDiagnostics,
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
)


@dataclass
class PolytomousLinkingResult(LinkingResult):
    """Extended linking result for polytomous models.

    Attributes
    ----------
    category_fit : dict[int, float] | None
        Category-level fit statistics.
    """

    category_fit: dict[int, float] | None = None


def link_grm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link two Graded Response Models.

    For GRM, the transformation is:
        a_new = a_old / A
        b_jk_new = A * b_jk_old + B

    Parameters
    ----------
    model_old : BaseItemModel
        Reference GRM model.
    model_new : BaseItemModel
        New GRM model.
    anchors_old : list[int]
        Anchor item indices in old model.
    anchors_new : list[int]
        Anchor item indices in new model.
    method : str
        Linking method.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    weights : NDArray | None
        Theta point weights.
    compute_diagnostics : bool
        Whether to compute diagnostics.

    Returns
    -------
    LinkingResult
        Linking constants and diagnostics.
    """
    if len(anchors_old) != len(anchors_new):
        raise ValueError("Anchor lists must have same length")

    disc_old = np.asarray(model_old.discrimination)[anchors_old]
    disc_new = np.asarray(model_new.discrimination)[anchors_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    thresholds_old = _extract_thresholds(model_old, anchors_old)
    thresholds_new = _extract_thresholds(model_new, anchors_new)

    diff_old = np.array([np.mean(t) for t in thresholds_old])
    diff_new = np.array([np.mean(t) for t in thresholds_new])

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    if weights is None:
        weights = stats.norm.pdf(theta_grid)
        weights = weights / np.sum(weights)

    if method == "mean_sigma":
        A, B, conv_info = _mean_sigma_polytomous(disc_old, diff_old, disc_new, diff_new)
    elif method == "mean_mean":
        A, B, conv_info = _mean_mean_polytomous(disc_old, diff_old, disc_new, diff_new)
    elif method == "stocking_lord":
        A, B, conv_info = _stocking_lord_grm(
            disc_old, thresholds_old, disc_new, thresholds_new, theta_grid, weights
        )
    elif method == "haebara":
        A, B, conv_info = _haebara_grm(
            disc_old, thresholds_old, disc_new, thresholds_new, theta_grid, weights
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    fit_statistics = None
    anchor_diagnostics = None

    if compute_diagnostics:
        fit_statistics = _compute_grm_fit(
            disc_old,
            thresholds_old,
            disc_new,
            thresholds_new,
            A,
            B,
            theta_grid,
            weights,
        )
        anchor_diagnostics = _compute_grm_diagnostics(
            disc_old,
            thresholds_old,
            disc_new,
            thresholds_new,
            A,
            B,
            anchors_old,
            theta_grid,
        )

    return LinkingResult(
        constants=LinkingConstants(A=float(A), B=float(B), method=method),
        anchor_items=anchors_old,
        fit_statistics=fit_statistics,
        anchor_diagnostics=anchor_diagnostics,
        convergence_info=conv_info,
    )


def link_gpcm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link two Generalized Partial Credit Models.

    For GPCM, the transformation is:
        a_new = a_old / A
        d_jk_new = d_jk_old / A (step parameters)
        b_j_new = A * b_j_old + B (location)

    Parameters
    ----------
    model_old : BaseItemModel
        Reference GPCM model.
    model_new : BaseItemModel
        New GPCM model.
    anchors_old : list[int]
        Anchor item indices in old model.
    anchors_new : list[int]
        Anchor item indices in new model.
    method : str
        Linking method.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    weights : NDArray | None
        Theta point weights.
    compute_diagnostics : bool
        Whether to compute diagnostics.

    Returns
    -------
    LinkingResult
        Linking constants and diagnostics.
    """
    if len(anchors_old) != len(anchors_new):
        raise ValueError("Anchor lists must have same length")

    disc_old = np.asarray(model_old.discrimination)[anchors_old]
    disc_new = np.asarray(model_new.discrimination)[anchors_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    steps_old = _extract_steps(model_old, anchors_old)
    steps_new = _extract_steps(model_new, anchors_new)

    diff_old = np.array([np.mean(s) for s in steps_old])
    diff_new = np.array([np.mean(s) for s in steps_new])

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    if weights is None:
        weights = stats.norm.pdf(theta_grid)
        weights = weights / np.sum(weights)

    if method == "mean_sigma":
        A, B, conv_info = _mean_sigma_polytomous(disc_old, diff_old, disc_new, diff_new)
    elif method == "mean_mean":
        A, B, conv_info = _mean_mean_polytomous(disc_old, diff_old, disc_new, diff_new)
    elif method == "stocking_lord":
        A, B, conv_info = _stocking_lord_gpcm(
            disc_old, steps_old, disc_new, steps_new, theta_grid, weights
        )
    elif method == "haebara":
        A, B, conv_info = _haebara_gpcm(
            disc_old, steps_old, disc_new, steps_new, theta_grid, weights
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    fit_statistics = None
    if compute_diagnostics:
        fit_statistics = _compute_gpcm_fit(
            disc_old, steps_old, disc_new, steps_new, A, B, theta_grid, weights
        )

    return LinkingResult(
        constants=LinkingConstants(A=float(A), B=float(B), method=method),
        anchor_items=anchors_old,
        fit_statistics=fit_statistics,
        convergence_info=conv_info,
    )


def link_nrm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
) -> LinkingResult:
    """Link two Nominal Response Models.

    For NRM, the transformation involves both slope and intercept
    parameters for each category.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference NRM model.
    model_new : BaseItemModel
        New NRM model.
    anchors_old : list[int]
        Anchor item indices in old model.
    anchors_new : list[int]
        Anchor item indices in new model.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    weights : NDArray | None
        Theta point weights.

    Returns
    -------
    LinkingResult
        Linking constants and diagnostics.
    """
    if len(anchors_old) != len(anchors_new):
        raise ValueError("Anchor lists must have same length")

    slopes_old = _extract_nrm_slopes(model_old, anchors_old)
    slopes_new = _extract_nrm_slopes(model_new, anchors_new)
    intercepts_old = _extract_nrm_intercepts(model_old, anchors_old)
    intercepts_new = _extract_nrm_intercepts(model_new, anchors_new)

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    if weights is None:
        weights = stats.norm.pdf(theta_grid)
        weights = weights / np.sum(weights)

    A, B, conv_info = _stocking_lord_nrm(
        slopes_old, intercepts_old, slopes_new, intercepts_new, theta_grid, weights
    )

    return LinkingResult(
        constants=LinkingConstants(A=float(A), B=float(B), method="stocking_lord"),
        anchor_items=anchors_old,
        convergence_info=conv_info,
    )


def transform_polytomous_parameters(
    model: "BaseItemModel",
    A: float,
    B: float,
    model_type: Literal["grm", "gpcm", "nrm"] = "grm",
    in_place: bool = False,
) -> "BaseItemModel":
    """Transform polytomous model parameters.

    Parameters
    ----------
    model : BaseItemModel
        Model to transform.
    A : float
        Slope transformation.
    B : float
        Intercept transformation.
    model_type : str
        Type of polytomous model.
    in_place : bool
        Modify in place or return copy.

    Returns
    -------
    BaseItemModel
        Transformed model.
    """
    if not in_place:
        model = model.copy()

    disc = np.asarray(model.discrimination)
    new_disc = disc / A

    if model_type == "grm":
        thresholds = model.parameters.get(
            "thresholds", model.parameters.get("difficulty")
        )
        if thresholds is not None:
            new_thresholds = A * thresholds + B
            model.set_parameters(discrimination=new_disc, thresholds=new_thresholds)
        else:
            model.set_parameters(discrimination=new_disc)

    elif model_type == "gpcm":
        steps = model.parameters.get("steps", model.parameters.get("step_parameters"))
        location = model.parameters.get("location", model.parameters.get("difficulty"))

        if steps is not None:
            new_steps = steps / A
            if location is not None:
                new_location = A * location + B
                model.set_parameters(
                    discrimination=new_disc, steps=new_steps, location=new_location
                )
            else:
                model.set_parameters(discrimination=new_disc, steps=new_steps)
        else:
            model.set_parameters(discrimination=new_disc)

    elif model_type == "nrm":
        slopes = model.parameters.get("slopes")
        intercepts = model.parameters.get("intercepts")

        if slopes is not None:
            new_slopes = slopes / A
            model.set_parameters(slopes=new_slopes)

        if intercepts is not None:
            new_intercepts = intercepts + (slopes if slopes is not None else 0) * B / A
            model.set_parameters(intercepts=new_intercepts)

    return model


def _extract_thresholds(
    model: "BaseItemModel", items: list[int]
) -> list[NDArray[np.float64]]:
    """Extract threshold parameters from GRM model."""
    thresholds = model.parameters.get("thresholds", model.parameters.get("difficulty"))
    if thresholds is None:
        return [np.array([0.0]) for _ in items]

    result = []
    for idx in items:
        if thresholds.ndim == 1:
            result.append(np.array([thresholds[idx]]))
        else:
            result.append(thresholds[idx])
    return result


def _extract_steps(
    model: "BaseItemModel", items: list[int]
) -> list[NDArray[np.float64]]:
    """Extract step parameters from GPCM model."""
    steps = model.parameters.get("steps", model.parameters.get("step_parameters"))
    if steps is None:
        return [np.array([0.0]) for _ in items]

    result = []
    for idx in items:
        if steps.ndim == 1:
            result.append(np.array([steps[idx]]))
        else:
            result.append(steps[idx])
    return result


def _extract_nrm_slopes(
    model: "BaseItemModel", items: list[int]
) -> list[NDArray[np.float64]]:
    """Extract slope parameters from NRM model."""
    slopes = model.parameters.get("slopes")
    if slopes is None:
        disc = model.parameters.get("discrimination")
        if disc is not None:
            return [np.array([disc[idx]]) for idx in items]
        return [np.array([1.0]) for _ in items]

    result = []
    for idx in items:
        result.append(slopes[idx])
    return result


def _extract_nrm_intercepts(
    model: "BaseItemModel", items: list[int]
) -> list[NDArray[np.float64]]:
    """Extract intercept parameters from NRM model."""
    intercepts = model.parameters.get("intercepts")
    if intercepts is None:
        return [np.array([0.0]) for _ in items]

    result = []
    for idx in items:
        result.append(intercepts[idx])
    return result


def _mean_sigma_polytomous(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Mean/sigma method for polytomous models."""
    std_old = np.std(disc_old, ddof=1)
    std_new = np.std(disc_new, ddof=1)

    if std_new < 1e-10:
        A = 1.0
    else:
        A = std_old / std_new

    B = np.mean(diff_old) - A * np.mean(diff_new)

    return A, B, {"method": "mean_sigma"}


def _mean_mean_polytomous(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Mean/mean method for polytomous models."""
    mean_disc_old = np.mean(disc_old)
    mean_disc_new = np.mean(disc_new)

    if mean_disc_new < 1e-10:
        A = 1.0
    else:
        A = mean_disc_old / mean_disc_new

    B = np.mean(diff_old) - A * np.mean(diff_new)

    return A, B, {"method": "mean_mean"}


def _grm_category_probs(
    theta: NDArray[np.float64],
    disc: float,
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute GRM category probabilities."""
    n_theta = len(theta)
    n_cat = len(thresholds) + 1

    cumprobs = np.zeros((n_theta, n_cat + 1))
    cumprobs[:, 0] = 1.0
    cumprobs[:, -1] = 0.0

    for k, b_k in enumerate(thresholds):
        cumprobs[:, k + 1] = 1.0 / (1.0 + np.exp(-disc * (theta - b_k)))

    probs = np.zeros((n_theta, n_cat))
    for k in range(n_cat):
        probs[:, k] = cumprobs[:, k] - cumprobs[:, k + 1]

    return np.clip(probs, 1e-10, 1.0)


def _gpcm_category_probs(
    theta: NDArray[np.float64],
    disc: float,
    steps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute GPCM category probabilities."""
    n_theta = len(theta)
    n_cat = len(steps) + 1

    cumsum_d = np.zeros(n_cat)
    for k in range(1, n_cat):
        cumsum_d[k] = cumsum_d[k - 1] + steps[k - 1]

    numer = np.zeros((n_theta, n_cat))
    for k in range(n_cat):
        numer[:, k] = np.exp(disc * k * theta - disc * cumsum_d[k])

    denom = np.sum(numer, axis=1, keepdims=True)
    probs = numer / np.maximum(denom, 1e-10)

    return np.clip(probs, 1e-10, 1.0)


def _stocking_lord_grm(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord method for GRM."""
    n_items = len(disc_old)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params

        eap_old = np.zeros(len(theta_grid))
        eap_new = np.zeros(len(theta_grid))

        for j in range(n_items):
            n_cat = len(thresholds_old[j]) + 1
            probs_old = _grm_category_probs(theta_grid, disc_old[j], thresholds_old[j])

            disc_trans = disc_new[j] / A
            thresh_trans = A * thresholds_new[j] + B
            probs_new = _grm_category_probs(theta_grid, disc_trans, thresh_trans)

            for k in range(n_cat):
                eap_old += k * probs_old[:, k]
                eap_new += k * probs_new[:, k]

        diff_sq = (eap_old - eap_new) ** 2
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
        {"success": result.success, "fun": result.fun, "nit": result.nit},
    )


def _haebara_grm(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Haebara method for GRM."""
    n_items = len(disc_old)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params
        total = 0.0

        for j in range(n_items):
            probs_old = _grm_category_probs(theta_grid, disc_old[j], thresholds_old[j])

            disc_trans = disc_new[j] / A
            thresh_trans = A * thresholds_new[j] + B
            probs_new = _grm_category_probs(theta_grid, disc_trans, thresh_trans)

            diff_sq = np.sum((probs_old - probs_new) ** 2, axis=1)
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
        {"success": result.success, "fun": result.fun, "nit": result.nit},
    )


def _stocking_lord_gpcm(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord method for GPCM."""
    n_items = len(disc_old)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params

        eap_old = np.zeros(len(theta_grid))
        eap_new = np.zeros(len(theta_grid))

        for j in range(n_items):
            n_cat = len(steps_old[j]) + 1
            probs_old = _gpcm_category_probs(theta_grid, disc_old[j], steps_old[j])

            disc_trans = disc_new[j] / A
            steps_trans = steps_new[j] / A
            theta_trans = theta_grid - B / A
            probs_new = _gpcm_category_probs(theta_trans, disc_trans, steps_trans)

            for k in range(n_cat):
                eap_old += k * probs_old[:, k]
                eap_new += k * probs_new[:, k]

        diff_sq = (eap_old - eap_new) ** 2
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
        {"success": result.success, "fun": result.fun, "nit": result.nit},
    )


def _haebara_gpcm(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Haebara method for GPCM."""
    n_items = len(disc_old)

    def criterion(params: NDArray[np.float64]) -> float:
        A, B = params
        total = 0.0

        for j in range(n_items):
            probs_old = _gpcm_category_probs(theta_grid, disc_old[j], steps_old[j])

            disc_trans = disc_new[j] / A
            steps_trans = steps_new[j] / A
            theta_trans = theta_grid - B / A
            probs_new = _gpcm_category_probs(theta_trans, disc_trans, steps_trans)

            diff_sq = np.sum((probs_old - probs_new) ** 2, axis=1)
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
        {"success": result.success, "fun": result.fun, "nit": result.nit},
    )


def _stocking_lord_nrm(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord method for NRM."""
    n_items = len(slopes_old)

    def nrm_probs(theta, slopes, intercepts):
        n_theta = len(theta)
        n_cat = len(slopes)
        numer = np.zeros((n_theta, n_cat))
        for k in range(n_cat):
            numer[:, k] = np.exp(slopes[k] * theta + intercepts[k])
        denom = np.sum(numer, axis=1, keepdims=True)
        return numer / np.maximum(denom, 1e-10)

    def criterion(params):
        A, B = params
        eap_old = np.zeros(len(theta_grid))
        eap_new = np.zeros(len(theta_grid))

        for j in range(n_items):
            n_cat = len(slopes_old[j])
            probs_old = nrm_probs(theta_grid, slopes_old[j], intercepts_old[j])

            slopes_trans = slopes_new[j] / A
            intercepts_trans = intercepts_new[j] + slopes_new[j] * B / A
            probs_new = nrm_probs(theta_grid, slopes_trans, intercepts_trans)

            for k in range(n_cat):
                eap_old += k * probs_old[:, k]
                eap_new += k * probs_new[:, k]

        diff_sq = (eap_old - eap_new) ** 2
        return float(np.sum(weights * diff_sq))

    result = optimize.minimize(
        criterion,
        np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )

    A, B = result.x
    return (
        float(A),
        float(B),
        {"success": result.success, "fun": result.fun, "nit": result.nit},
    )


def _compute_grm_fit(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute fit statistics for GRM linking."""
    n_items = len(disc_old)

    disc_new_trans = disc_new / A
    diff_a = disc_old - disc_new_trans
    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    mad_a = float(np.mean(np.abs(diff_a)))

    all_diff_b: list[float] = []
    for j in range(n_items):
        thresh_trans = A * thresholds_new[j] + B
        all_diff_b.extend(thresholds_old[j] - thresh_trans)

    diff_b = np.array(all_diff_b)
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))
    mad_b = float(np.mean(np.abs(diff_b)))

    weighted_rmse = float(np.sqrt(rmse_a**2 + rmse_b**2))

    eap_old = np.zeros(len(theta_grid))
    eap_new = np.zeros(len(theta_grid))

    for j in range(n_items):
        n_cat = len(thresholds_old[j]) + 1
        probs_old = _grm_category_probs(theta_grid, disc_old[j], thresholds_old[j])

        disc_trans = disc_new[j] / A
        thresh_trans = A * thresholds_new[j] + B
        probs_new = _grm_category_probs(theta_grid, disc_trans, thresh_trans)

        for k in range(n_cat):
            eap_old += k * probs_old[:, k]
            eap_new += k * probs_new[:, k]

    tcc_diff = (eap_old - eap_new) ** 2
    tcc_rmse = float(np.sqrt(np.sum(weights * tcc_diff)))

    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=mad_a,
        mad_b=mad_b,
        weighted_rmse=weighted_rmse,
        tcc_rmse=tcc_rmse,
    )


def _compute_grm_diagnostics(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
) -> AnchorDiagnostics:
    """Compute anchor diagnostics for GRM linking."""
    n_items = len(disc_old)

    disc_new_trans = disc_new / A
    signed_diff_a = disc_old - disc_new_trans

    signed_diff_b = np.zeros(n_items)
    for j in range(n_items):
        thresh_trans = A * thresholds_new[j] + B
        signed_diff_b[j] = np.mean(thresholds_old[j] - thresh_trans)

    area_diff = np.zeros(n_items)
    for j in range(n_items):
        probs_old = _grm_category_probs(theta_grid, disc_old[j], thresholds_old[j])
        disc_trans = disc_new[j] / A
        thresh_trans = A * thresholds_new[j] + B
        probs_new = _grm_category_probs(theta_grid, disc_trans, thresh_trans)

        area_diff[j] = np.trapezoid(
            np.sum(np.abs(probs_old - probs_new), axis=1), theta_grid
        )

    combined = np.sqrt(signed_diff_a**2 + signed_diff_b**2)
    median_diff = np.median(combined)
    mad = np.median(np.abs(combined - median_diff)) * 1.4826

    if mad < 1e-10:
        robust_z = np.zeros(n_items)
    else:
        robust_z = (combined - median_diff) / mad

    flagged = np.abs(robust_z) > 2.5

    return AnchorDiagnostics(
        item_indices=anchor_indices,
        signed_diff_a=signed_diff_a,
        signed_diff_b=signed_diff_b,
        area_diff=area_diff,
        robust_z=robust_z,
        flagged=flagged,
    )


def _compute_gpcm_fit(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute fit statistics for GPCM linking."""
    n_items = len(disc_old)

    disc_new_trans = disc_new / A
    diff_a = disc_old - disc_new_trans
    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    mad_a = float(np.mean(np.abs(diff_a)))

    all_diff_b: list[float] = []
    for j in range(n_items):
        steps_trans = steps_new[j] / A
        all_diff_b.extend(steps_old[j] - steps_trans)

    diff_b = np.array(all_diff_b) if all_diff_b else np.array([0.0])
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))
    mad_b = float(np.mean(np.abs(diff_b)))

    weighted_rmse = float(np.sqrt(rmse_a**2 + rmse_b**2))

    eap_old = np.zeros(len(theta_grid))
    eap_new = np.zeros(len(theta_grid))

    for j in range(n_items):
        n_cat = len(steps_old[j]) + 1
        probs_old = _gpcm_category_probs(theta_grid, disc_old[j], steps_old[j])

        disc_trans = disc_new[j] / A
        steps_trans = steps_new[j] / A
        theta_trans = theta_grid - B / A
        probs_new = _gpcm_category_probs(theta_trans, disc_trans, steps_trans)

        for k in range(n_cat):
            eap_old += k * probs_old[:, k]
            eap_new += k * probs_new[:, k]

    tcc_diff = (eap_old - eap_new) ** 2
    tcc_rmse = float(np.sqrt(np.sum(weights * tcc_diff)))

    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=mad_a,
        mad_b=mad_b,
        weighted_rmse=weighted_rmse,
        tcc_rmse=tcc_rmse,
    )
