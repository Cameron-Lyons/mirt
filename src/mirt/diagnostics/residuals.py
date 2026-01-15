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
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP")
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    residuals = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        probs = model.probability(theta, j)

        if probs.ndim == 2:
            n_cats = probs.shape[1]
            expected = np.sum(probs * np.arange(n_cats), axis=1)
            variance = np.sum(probs * (np.arange(n_cats) ** 2), axis=1) - expected**2
        else:
            expected = probs
            variance = probs * (1 - probs)

        valid = responses[:, j] >= 0
        observed = responses[valid, j]
        exp_valid = expected[valid]
        var_valid = variance[valid]

        raw = observed - exp_valid

        if residual_type == "raw":
            residuals[valid, j] = raw
        elif residual_type == "standardized":
            residuals[valid, j] = raw / np.sqrt(var_valid + PROB_EPSILON)
        elif residual_type == "pearson":
            residuals[valid, j] = raw / np.sqrt(exp_valid + PROB_EPSILON)
        elif residual_type == "deviance":
            with np.errstate(divide="ignore", invalid="ignore"):
                if probs.ndim == 2:
                    p_obs = probs[valid, observed]
                else:
                    p_obs = np.where(observed == 1, probs[valid], 1 - probs[valid])
                p_obs = np.clip(p_obs, PROB_EPSILON, 1 - PROB_EPSILON)
                deviance = np.sign(raw) * np.sqrt(-2 * np.log(p_obs))
            residuals[valid, j] = deviance
        else:
            raise ValueError(f"Unknown residual type: {residual_type}")

    return residuals


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
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP")
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    raw = compute_residuals(model, responses, theta, "raw")
    standardized = compute_residuals(model, responses, theta, "standardized")
    pearson = compute_residuals(model, responses, theta, "pearson")
    deviance = compute_residuals(model, responses, theta, "deviance")

    expected = np.zeros((n_persons, n_items))
    for j in range(n_items):
        probs = model.probability(theta, j)
        if probs.ndim == 2:
            expected[:, j] = np.sum(probs * np.arange(probs.shape[1]), axis=1)
        else:
            expected[:, j] = probs

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
    for i in range(n_persons):
        pattern = tuple(responses[i])
        z_i = standardized[i]
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

    for pattern, stats in pattern_residuals.items():
        if stats["n"] > 0:
            stats["mean_z"] = stats["sum_z"] / stats["n"]
            stats["mean_z_sq"] = stats["sum_z_sq"] / stats["n"]
        else:
            stats["mean_z"] = 0
            stats["mean_z_sq"] = 0

    return ResidualAnalysisResult(
        raw_residuals=raw,
        standardized_residuals=standardized,
        pearson_residuals=pearson,
        deviance_residuals=deviance,
        expected_values=expected,
        theta_estimates=theta.ravel() if theta.shape[1] == 1 else theta,
        pattern_residuals=pattern_residuals,
        item_residuals=item_residuals,
    )


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
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP")
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    z_sq = np.zeros((n_persons, n_items))
    variances = np.zeros((n_persons, n_items))

    for j in range(n_items):
        probs = model.probability(theta, j)

        if probs.ndim == 2:
            n_cats = probs.shape[1]
            expected = np.sum(probs * np.arange(n_cats), axis=1)
            var = np.sum(probs * (np.arange(n_cats) ** 2), axis=1) - expected**2
        else:
            expected = probs
            var = probs * (1 - probs)

        valid = responses[:, j] >= 0
        observed = responses[valid, j]

        raw = observed - expected[valid]
        z_sq[valid, j] = (raw**2) / (var[valid] + PROB_EPSILON)
        variances[valid, j] = var[valid]

    missing = responses < 0
    z_sq[missing] = np.nan
    variances[missing] = np.nan

    item_outfit = np.nanmean(z_sq, axis=0)
    item_infit = np.nansum(z_sq * variances, axis=0) / (
        np.nansum(variances, axis=0) + PROB_EPSILON
    )

    person_outfit = np.nanmean(z_sq, axis=1)
    person_infit = np.nansum(z_sq * variances, axis=1) / (
        np.nansum(variances, axis=1) + PROB_EPSILON
    )

    return {
        "item_outfit": item_outfit,
        "item_infit": item_infit,
        "person_outfit": person_outfit,
        "person_infit": person_infit,
    }


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
    analysis = analyze_residuals(model, responses, theta)
    fit_stats = compute_outfit_infit(model, responses, theta)

    misfitting_items = []
    for j in range(responses.shape[1]):
        if fit_stats["item_outfit"][j] > outfit_threshold:
            misfitting_items.append(
                {
                    "item": j,
                    "outfit": fit_stats["item_outfit"][j],
                    "infit": fit_stats["item_infit"][j],
                }
            )

    misfitting_persons = []
    for i in range(responses.shape[0]):
        if fit_stats["person_outfit"][i] > outfit_threshold:
            misfitting_persons.append(
                {
                    "person": i,
                    "outfit": fit_stats["person_outfit"][i],
                    "infit": fit_stats["person_infit"][i],
                }
            )

    aberrant = []
    z = analysis.standardized_residuals
    for i in range(responses.shape[0]):
        for j in range(responses.shape[1]):
            if not np.isnan(z[i, j]) and abs(z[i, j]) > z_threshold:
                aberrant.append(
                    {
                        "person": i,
                        "item": j,
                        "response": responses[i, j],
                        "expected": analysis.expected_values[i, j],
                        "z": z[i, j],
                    }
                )

    return {
        "misfitting_persons": misfitting_persons,
        "misfitting_items": misfitting_items,
        "aberrant_responses": aberrant,
    }
