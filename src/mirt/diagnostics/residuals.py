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
_FIT_STATISTICS_CHUNK_ELEMENTS = 1_000_000


@dataclass
class _ResidualComputation:
    """Intermediate arrays shared by residual diagnostics."""

    residuals: dict[str, NDArray[np.float64]]
    expected_values: NDArray[np.float64] | None
    variances: NDArray[np.float64] | None


@dataclass
class _FitAccumulator:
    """Sums and counts needed to finalize item and person fit statistics."""

    item_square_sum: NDArray[np.float64]
    item_weighted_sum: NDArray[np.float64]
    item_variance_sum: NDArray[np.float64]
    item_n: NDArray[np.intp]
    person_square_sum: NDArray[np.float64]
    person_weighted_sum: NDArray[np.float64]
    person_variance_sum: NDArray[np.float64]
    person_n: NDArray[np.intp]

    @classmethod
    def create(cls, n_persons: int, n_items: int) -> _FitAccumulator:
        """Create a zeroed accumulator for a response matrix shape."""
        return cls(
            item_square_sum=np.zeros(n_items),
            item_weighted_sum=np.zeros(n_items),
            item_variance_sum=np.zeros(n_items),
            item_n=np.zeros(n_items, dtype=np.intp),
            person_square_sum=np.zeros(n_persons),
            person_weighted_sum=np.zeros(n_persons),
            person_variance_sum=np.zeros(n_persons),
            person_n=np.zeros(n_persons, dtype=np.intp),
        )

    def finish(self) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
        """Finalize fit means and retain their observation counts."""
        return {
            "item_outfit": np.divide(
                self.item_square_sum,
                self.item_n,
                out=np.full(self.item_square_sum.shape, np.nan),
                where=self.item_n > 0,
            ),
            "item_infit": np.divide(
                self.item_weighted_sum,
                self.item_variance_sum + PROB_EPSILON,
                out=np.full(self.item_weighted_sum.shape, np.nan),
                where=self.item_n > 0,
            ),
            "person_outfit": np.divide(
                self.person_square_sum,
                self.person_n,
                out=np.full(self.person_square_sum.shape, np.nan),
                where=self.person_n > 0,
            ),
            "person_infit": np.divide(
                self.person_weighted_sum,
                self.person_variance_sum + PROB_EPSILON,
                out=np.full(self.person_weighted_sum.shape, np.nan),
                where=self.person_n > 0,
            ),
            "item_n": self.item_n,
            "person_n": self.person_n,
        }


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


def _item_expected_value_variance(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_index: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Evaluate one item and return probabilities, means, and variances."""
    probabilities = np.asarray(model.probability(theta, item_index), dtype=np.float64)
    n_persons = theta.shape[0]
    if probabilities.ndim == 2:
        if probabilities.shape[0] != n_persons or probabilities.shape[1] == 0:
            raise ValueError(
                "model probabilities must provide at least one category per person"
            )
        categories = np.arange(probabilities.shape[1], dtype=np.float64)
        expected = probabilities @ categories
        variance = probabilities @ np.square(categories) - np.square(expected)
        return probabilities, expected, variance

    if probabilities.ndim == 1 and probabilities.shape[0] == n_persons:
        expected = probabilities
        variance = probabilities * (1.0 - probabilities)
        return probabilities, expected, variance

    raise ValueError(
        "model probabilities must have shape (n_persons,) or (n_persons, n_categories)"
    )


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
        probs, expected, variance = _item_expected_value_variance(model, theta, j)

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
        if z_j.size:
            item_residuals[j] = {
                "mean": float(np.mean(z_j)),
                "sd": float(np.std(z_j)),
                "max_abs_z": float(np.max(np.abs(z_j))),
            }
        else:
            item_residuals[j] = {
                "mean": np.nan,
                "sd": np.nan,
                "max_abs_z": 0.0,
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
) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
    """Aggregate residuals in bounded chunks without full-size temporaries."""
    if standardized_residuals.shape != variances.shape:
        raise ValueError("standardized_residuals and variances must have equal shapes")

    n_persons, n_items = standardized_residuals.shape
    accumulator = _FitAccumulator.create(n_persons, n_items)

    chunk_size = min(
        n_items,
        max(1, _FIT_STATISTICS_CHUNK_ELEMENTS // max(1, n_persons)),
    )
    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        residual_block = standardized_residuals[:, start:stop]
        variance_block = variances[:, start:stop]
        valid = np.isfinite(residual_block)
        squared = np.zeros_like(residual_block)
        np.square(residual_block, out=squared, where=valid)

        accumulator.item_square_sum[start:stop] = np.sum(squared, axis=0)
        accumulator.item_n[start:stop] = np.sum(valid, axis=0, dtype=np.intp)
        accumulator.person_square_sum += np.sum(squared, axis=1)
        accumulator.person_n += np.sum(valid, axis=1, dtype=np.intp)
        accumulator.item_variance_sum[start:stop] = np.sum(
            variance_block,
            axis=0,
            where=valid,
            initial=0.0,
        )
        accumulator.person_variance_sum += np.sum(
            variance_block,
            axis=1,
            where=valid,
            initial=0.0,
        )

        np.multiply(squared, variance_block, out=squared, where=valid)
        accumulator.item_weighted_sum[start:stop] = np.sum(squared, axis=0)
        accumulator.person_weighted_sum += np.sum(squared, axis=1)

    return accumulator.finish()


def _stream_fit_statistics(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
    """Accumulate fit statistics directly from one item probability pass."""
    n_persons, n_items = responses.shape
    accumulator = _FitAccumulator.create(n_persons, n_items)

    for item_index in range(n_items):
        _, expected, variance = _item_expected_value_variance(
            model,
            theta,
            item_index,
        )
        valid = responses[:, item_index] >= 0
        observed = responses[valid, item_index]
        valid_variance = variance[valid]
        raw = observed - expected[valid]
        squared = np.square(raw) / (valid_variance + PROB_EPSILON)
        weighted = squared * valid_variance

        count = int(np.count_nonzero(valid))
        accumulator.item_n[item_index] = count
        accumulator.item_square_sum[item_index] = np.sum(squared)
        accumulator.item_weighted_sum[item_index] = np.sum(weighted)
        accumulator.item_variance_sum[item_index] = np.sum(valid_variance)
        accumulator.person_n[valid] += 1
        accumulator.person_square_sum[valid] += squared
        accumulator.person_weighted_sum[valid] += weighted
        accumulator.person_variance_sum[valid] += valid_variance

    return accumulator.finish()


def compute_outfit_infit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    *,
    include_counts: bool = False,
) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
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
    include_counts : bool, default=False
        Include valid observation counts as ``item_n`` and ``person_n``.

    Returns
    -------
    dict
        Dictionary with ``item_outfit``, ``item_infit``, ``person_outfit``, and
        ``person_infit``. When requested, ``item_n`` and ``person_n`` contain
        the corresponding valid observation counts.
    """
    if not isinstance(include_counts, (bool, np.bool_)):
        raise ValueError("include_counts must be boolean")
    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    theta_array = _resolve_theta(model, responses, theta)
    statistics = _stream_fit_statistics(model, responses, theta_array)
    if not include_counts:
        del statistics["item_n"]
        del statistics["person_n"]
    return statistics


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
