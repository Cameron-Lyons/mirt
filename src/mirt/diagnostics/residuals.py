"""Response pattern residuals for IRT model diagnostics.

This module provides functions to compute and analyze residuals from
IRT models, which are useful for detecting model misfit and identifying
aberrant response patterns.

Residual types:
- Raw residuals: O - E
- Standardized residuals: (O - E) / sqrt(E * (1 - E))
- Pearson residuals: (O - E) / sqrt(E)
- Deviance residuals: sign(O - E) * sqrt(2 * |log(p)|)

References:
    Hambleton, R. K., & Swaminathan, H. (1985). Item response theory:
        Principles and applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_RESIDUAL_TYPES = frozenset({"raw", "standardized", "pearson", "deviance"})


@dataclass
class _ResidualComputation:
    """Intermediate arrays shared by residual diagnostics."""

    residuals: dict[str, NDArray[np.float64]]
    expected_values: NDArray[np.float64] | None
    variances: NDArray[np.float64] | None


@dataclass
class ResidualAnalysisResult:
    """Result from response pattern residual analysis.

    Attributes
    ----------
    raw_residuals : NDArray
        Raw residuals (observed - expected)
    standardized_residuals : NDArray
        Standardized residuals
    pearson_residuals : NDArray
        Pearson residuals
    deviance_residuals : NDArray
        Deviance (likelihood) residuals
    expected_values : NDArray
        Expected values under the model
    theta_estimates : NDArray
        Ability estimates used
    pattern_residuals : dict
        Residual statistics aggregated by response pattern
    item_residuals : dict
        Residual statistics aggregated by item
    """

    raw_residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    pearson_residuals: NDArray[np.float64]
    deviance_residuals: NDArray[np.float64]
    expected_values: NDArray[np.float64]
    theta_estimates: NDArray[np.float64]
    pattern_residuals: dict
    item_residuals: dict

    def summary(self) -> str:
        """Generate summary of residual analysis."""
        lines = [
            "Response Pattern Residual Analysis",
            "=" * 60,
            "",
            "Overall Residual Statistics:",
            f"  Mean raw residual:          {np.nanmean(self.raw_residuals):.6f}",
            f"  SD raw residual:            {np.nanstd(self.raw_residuals):.4f}",
            f"  Mean standardized:          {np.nanmean(self.standardized_residuals):.6f}",
            f"  SD standardized:            {np.nanstd(self.standardized_residuals):.4f}",
            "",
            "Item-Level Residual Statistics:",
        ]

        for item_idx, stats in self.item_residuals.items():
            lines.append(
                f"  Item {item_idx + 1}: mean={stats['mean']:.4f}, "
                f"sd={stats['sd']:.4f}, max|z|={stats['max_abs_z']:.2f}"
            )

        lines.extend(
            [
                "",
                "Flagged Response Patterns (|mean z| > 2):",
            ]
        )

        flagged = [
            (k, v) for k, v in self.pattern_residuals.items() if abs(v["mean_z"]) > 2
        ]

        if flagged:
            for pattern, stats in sorted(flagged, key=lambda x: -abs(x[1]["mean_z"]))[
                :10
            ]:
                lines.append(
                    f"  {pattern}: mean_z={stats['mean_z']:.2f}, n={stats['n']}"
                )
        else:
            lines.append("  None")

        return "\n".join(lines)


def _resolve_theta(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return ability estimates in the shape expected by item models."""
    if theta is None:
        from mirt.scoring import fscores

        theta = fscores(model, responses, method="EAP").theta

    theta_array = np.asarray(theta)
    if theta_array.ndim == 1:
        return theta_array.reshape(-1, 1)

    theta_array = np.atleast_2d(theta_array)
    if theta_array.shape[0] == 1 and responses.shape[0] > 1:
        theta_array = theta_array.T
    return theta_array


def _compute_residual_arrays(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    residual_types: tuple[str, ...],
    *,
    store_expected: bool = False,
    store_variances: bool = False,
) -> _ResidualComputation:
    """Compute requested residual arrays in one pass over model probabilities."""
    unknown_types = set(residual_types) - _RESIDUAL_TYPES
    if unknown_types:
        unknown = next(kind for kind in residual_types if kind in unknown_types)
        raise ValueError(f"Unknown residual type: {unknown}")

    n_persons, n_items = responses.shape
    residuals = {kind: np.full((n_persons, n_items), np.nan) for kind in residual_types}
    expected_values = np.empty((n_persons, n_items)) if store_expected else None
    variances = np.full((n_persons, n_items), np.nan) if store_variances else None

    for j in range(n_items):
        probs = model.probability(theta, j)

        if probs.ndim == 2:
            categories = np.arange(probs.shape[1])
            expected = probs @ categories
            variance = probs @ (categories**2) - expected**2
        else:
            expected = probs
            variance = probs * (1 - probs)

        if expected_values is not None:
            expected_values[:, j] = expected

        valid = responses[:, j] >= 0
        observed = responses[valid, j]
        exp_valid = expected[valid]
        var_valid = variance[valid]
        raw = observed - exp_valid

        if variances is not None:
            variances[valid, j] = var_valid
        if "raw" in residuals:
            residuals["raw"][valid, j] = raw
        if "standardized" in residuals:
            residuals["standardized"][valid, j] = raw / np.sqrt(
                var_valid + PROB_EPSILON
            )
        if "pearson" in residuals:
            residuals["pearson"][valid, j] = raw / np.sqrt(exp_valid + PROB_EPSILON)
        if "deviance" in residuals:
            with np.errstate(divide="ignore", invalid="ignore"):
                if probs.ndim == 2:
                    p_obs = probs[valid, observed]
                else:
                    p_obs = np.where(observed == 1, probs[valid], 1 - probs[valid])
                p_obs = np.clip(p_obs, PROB_EPSILON, 1 - PROB_EPSILON)
                residuals["deviance"][valid, j] = np.sign(raw) * np.sqrt(
                    -2 * np.log(p_obs)
                )

    return _ResidualComputation(residuals, expected_values, variances)


def compute_residuals(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    residual_type: str = "standardized",
) -> NDArray[np.float64]:
    """Compute residuals for IRT model.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray of shape (n_persons, n_items)
        Response matrix
    theta : ndarray, optional
        Ability estimates. If None, EAP estimates are computed.
    residual_type : str
        Type of residual: "raw", "standardized", "pearson", or "deviance"

    Returns
    -------
    ndarray
        Residual matrix of same shape as responses
    """
    responses = np.asarray(responses)
    if residual_type not in _RESIDUAL_TYPES:
        raise ValueError(f"Unknown residual type: {residual_type}")

    theta_array = _resolve_theta(model, responses, theta)
    computation = _compute_residual_arrays(
        model,
        responses,
        theta_array,
        (residual_type,),
    )
    return computation.residuals[residual_type]


def _analyze_residuals(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None,
    *,
    store_variances: bool,
) -> tuple[ResidualAnalysisResult, NDArray[np.float64] | None]:
    """Build residual analysis and optionally retain variances for fit statistics."""
    theta_array = _resolve_theta(model, responses, theta)
    computation = _compute_residual_arrays(
        model,
        responses,
        theta_array,
        ("raw", "standardized", "pearson", "deviance"),
        store_expected=True,
        store_variances=store_variances,
    )
    raw = computation.residuals["raw"]
    standardized = computation.residuals["standardized"]
    pearson = computation.residuals["pearson"]
    deviance = computation.residuals["deviance"]
    expected = computation.expected_values
    assert expected is not None

    _, n_items = responses.shape
    item_residuals = {}
    for j in range(n_items):
        valid = ~np.isnan(standardized[:, j])
        z_j = standardized[valid, j]
        item_residuals[j] = {
            "mean": float(np.mean(z_j)),
            "sd": float(np.std(z_j)),
            "max_abs_z": float(np.max(np.abs(z_j))) if len(z_j) > 0 else 0,
        }

    pattern_residuals = {}
    for response_pattern, z_i in zip(responses, standardized, strict=True):
        pattern = tuple(response_pattern)
        valid = ~np.isnan(z_i)

        if pattern not in pattern_residuals:
            pattern_residuals[pattern] = {
                "sum_z": 0.0,
                "sum_z_sq": 0.0,
                "n": 0,
                "count": 0,
            }

        pattern_residuals[pattern]["sum_z"] += np.sum(z_i[valid])
        pattern_residuals[pattern]["sum_z_sq"] += np.sum(z_i[valid] ** 2)
        pattern_residuals[pattern]["n"] += np.sum(valid)
        pattern_residuals[pattern]["count"] += 1

    for stats in pattern_residuals.values():
        if stats["n"] > 0:
            stats["mean_z"] = stats["sum_z"] / stats["n"]
            stats["mean_z_sq"] = stats["sum_z_sq"] / stats["n"]
        else:
            stats["mean_z"] = 0
            stats["mean_z_sq"] = 0

    result = ResidualAnalysisResult(
        raw_residuals=raw,
        standardized_residuals=standardized,
        pearson_residuals=pearson,
        deviance_residuals=deviance,
        expected_values=expected,
        theta_estimates=(
            theta_array.ravel() if theta_array.shape[1] == 1 else theta_array
        ),
        pattern_residuals=pattern_residuals,
        item_residuals=item_residuals,
    )
    return result, computation.variances


def analyze_residuals(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> ResidualAnalysisResult:
    """Comprehensive residual analysis for IRT model.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    theta : ndarray, optional
        Ability estimates

    Returns
    -------
    ResidualAnalysisResult
        Complete residual analysis results
    """
    responses = np.asarray(responses)
    result, _ = _analyze_residuals(
        model,
        responses,
        theta,
        store_variances=False,
    )
    return result


def _fit_statistics(
    standardized_residuals: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Aggregate standardized residuals into item and person fit statistics."""
    z_sq = np.square(standardized_residuals)
    weighted_z_sq = z_sq * variances

    return {
        "item_outfit": np.nanmean(z_sq, axis=0),
        "item_infit": np.nansum(weighted_z_sq, axis=0)
        / (np.nansum(variances, axis=0) + PROB_EPSILON),
        "person_outfit": np.nanmean(z_sq, axis=1),
        "person_infit": np.nansum(weighted_z_sq, axis=1)
        / (np.nansum(variances, axis=1) + PROB_EPSILON),
    }


def compute_outfit_infit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute outfit and infit statistics for items and persons.

    Outfit: unweighted mean square residual
    Infit: information-weighted mean square residual

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    theta : ndarray, optional
        Ability estimates

    Returns
    -------
    dict
        Dictionary with 'item_outfit', 'item_infit', 'person_outfit', 'person_infit'
    """
    responses = np.asarray(responses)
    theta_array = _resolve_theta(model, responses, theta)
    computation = _compute_residual_arrays(
        model,
        responses,
        theta_array,
        ("standardized",),
        store_variances=True,
    )
    variances = computation.variances
    assert variances is not None
    return _fit_statistics(computation.residuals["standardized"], variances)


def identify_misfitting_patterns(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    z_threshold: float = 2.0,
    outfit_threshold: float = 1.5,
) -> dict[str, list]:
    """Identify misfitting persons and items.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    theta : ndarray, optional
        Ability estimates
    z_threshold : float
        Threshold for standardized residuals
    outfit_threshold : float
        Threshold for outfit statistics

    Returns
    -------
    dict
        Dictionary with 'misfitting_persons', 'misfitting_items', 'aberrant_responses'
    """
    responses = np.asarray(responses)
    analysis, variances = _analyze_residuals(
        model,
        responses,
        theta,
        store_variances=True,
    )
    assert variances is not None
    fit_stats = _fit_statistics(analysis.standardized_residuals, variances)

    misfitting_items = [
        {
            "item": int(j),
            "outfit": fit_stats["item_outfit"][j],
            "infit": fit_stats["item_infit"][j],
        }
        for j in np.flatnonzero(fit_stats["item_outfit"] > outfit_threshold)
    ]
    misfitting_persons = [
        {
            "person": int(i),
            "outfit": fit_stats["person_outfit"][i],
            "infit": fit_stats["person_infit"][i],
        }
        for i in np.flatnonzero(fit_stats["person_outfit"] > outfit_threshold)
    ]

    z = analysis.standardized_residuals
    aberrant = [
        {
            "person": int(i),
            "item": int(j),
            "response": responses[i, j],
            "expected": analysis.expected_values[i, j],
            "z": z[i, j],
        }
        for i, j in np.argwhere(np.isfinite(z) & (np.abs(z) > z_threshold))
    ]

    return {
        "misfitting_persons": misfitting_persons,
        "misfitting_items": misfitting_items,
        "aberrant_responses": aberrant,
    }
