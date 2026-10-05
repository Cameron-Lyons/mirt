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
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON
from mirt.utils.data import _missing_coded_responses
from mirt.utils.numeric import (
    _fit_cell_terms,
    _FitStatsAccumulator,
    _fourth_central_moment,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_RESIDUAL_TYPES = frozenset({"raw", "standardized", "pearson", "deviance"})
_RESIDUAL_MAX_PROBABILITY_VALUES = 1_000_000
_MISFIT_TARGET_CHUNK_ELEMENTS = 262_144


@dataclass
class _ResidualComputation:
    """Intermediate arrays shared by residual diagnostics."""

    residuals: dict[str, NDArray[np.float64]]
    expected_values: NDArray[np.float64] | None
    variances: NDArray[np.float64] | None


class _ItemPersonFit:
    """Item and person mean-square totals built on the shared fit accumulator.

    Item and person statistics therefore follow exactly the same rules as
    :func:`~mirt.diagnostics.itemfit.compute_itemfit` and
    :func:`~mirt.diagnostics.personfit.compute_personfit`.
    """

    def __init__(
        self, n_persons: int, n_items: int, *, standardized: bool = False
    ) -> None:
        self.items = _FitStatsAccumulator(n_items, standardized=standardized)
        self.persons = _FitStatsAccumulator(n_persons, standardized=standardized)
        self.item_n = np.zeros(n_items, dtype=np.intp)
        self.person_n = np.zeros(n_persons, dtype=np.intp)

    def add(
        self,
        responses: NDArray[np.float64],
        expected: NDArray[np.float64],
        variance: NDArray[np.float64],
        *,
        items: slice = slice(None),
        persons: slice = slice(None),
        fourth_moment: NDArray[np.float64] | None = None,
    ) -> None:
        """Add a person-by-item block; negative codes are missing."""
        terms = _fit_cell_terms(responses, expected, variance, fourth_moment)
        self.item_n[items] += np.count_nonzero(terms.observed, axis=0)
        self.person_n[persons] += np.count_nonzero(terms.observed, axis=1)
        self.items.add_terms(terms, axis=0, target=items)
        self.persons.add_terms(terms, axis=1, target=persons)

    def finish(self) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
        """Finalize item and person statistics and their valid counts."""
        result: dict[str, NDArray[np.float64] | NDArray[np.intp]] = {}
        for prefix, accumulator in (("item", self.items), ("person", self.persons)):
            for name, values in accumulator.statistics().items():
                result[f"{prefix}_{name}"] = values
        result["item_n"] = self.item_n
        result["person_n"] = self.person_n
        return result


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
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None,
) -> tuple[BaseItemModel, NDArray[np.float64]]:
    """Return the item model and abilities in the shape it expects.

    Omitted abilities are EAP scores under the latent population of a
    ``FitResult``, as :func:`mirt.fscores` computes them.
    """
    from mirt.results._common import resolve_item_model

    model: BaseItemModel = resolve_item_model(model_or_result)
    if theta is None:
        from mirt.scoring import fscores

        theta = fscores(model_or_result, responses, method="EAP").theta

    theta_array = np.asarray(theta)
    if theta_array.ndim == 1:
        return model, theta_array.reshape(-1, 1)

    theta_array = np.atleast_2d(theta_array)
    if theta_array.shape[0] == 1 and responses.shape[0] > 1:
        theta_array = theta_array.T
    return model, theta_array


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


def _probability_values_per_row(model: BaseItemModel, n_items: int) -> int:
    """Return the batch probability width, including padded item categories."""
    is_polytomous = bool(getattr(model, "is_polytomous", False))
    n_categories = 1
    if is_polytomous:
        n_categories = getattr(model, "max_categories", None)
        if n_categories is None:
            category_counts = getattr(model, "n_categories", 1)
            if np.isscalar(category_counts):
                n_categories = int(category_counts)
            else:
                n_categories = max(category_counts)

    return n_items * int(n_categories)


def _all_item_expected_value_variance(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_items: int,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None
):
    """Evaluate all item moments when the bounded batch contract is available."""
    if getattr(model, "n_items", None) != n_items:
        return None

    n_persons = theta.shape[0]
    probability_values = n_persons * _probability_values_per_row(model, n_items)
    if probability_values > _RESIDUAL_MAX_PROBABILITY_VALUES:
        return None

    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if bool(getattr(model, "is_polytomous", False)):
        if probabilities.ndim == 2 and n_items == 1:
            probabilities = probabilities[:, None, :]
        if (
            probabilities.ndim != 3
            or probabilities.shape[:2] != (n_persons, n_items)
            or probabilities.shape[2] == 0
        ):
            return None
        categories = np.arange(probabilities.shape[2], dtype=np.float64)
        expected = probabilities @ categories
        variance = probabilities @ np.square(categories) - np.square(expected)
        return probabilities, expected, variance

    if probabilities.ndim == 1 and n_items == 1:
        probabilities = probabilities[:, None]
    if probabilities.shape != (n_persons, n_items):
        return None
    expected = probabilities
    variance = probabilities * (1.0 - probabilities)
    return probabilities, expected, variance


def _compute_batched_residual_arrays(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    residual_types: tuple[str, ...],
    *,
    store_expected: bool,
    store_variances: bool,
) -> _ResidualComputation | None:
    """Compute residual matrices from one bounded all-item probability call."""
    n_persons, n_items = responses.shape
    moments = _all_item_expected_value_variance(model, theta, n_items)
    if moments is None:
        return None

    probabilities, expected, variance = moments
    valid = responses >= 0
    raw = np.where(valid, responses - expected, np.nan)
    residuals: dict[str, NDArray[np.float64]] = {}

    if "raw" in residual_types:
        residuals["raw"] = raw.copy()
    if "standardized" in residual_types:
        residuals["standardized"] = raw / np.sqrt(variance + PROB_EPSILON)
    if "pearson" in residual_types:
        residuals["pearson"] = raw / np.sqrt(expected + PROB_EPSILON)
    if "deviance" in residual_types:
        if probabilities.ndim == 3:
            safe_responses = np.where(valid, responses, 0).astype(np.intp)
            observed_probability = np.take_along_axis(
                probabilities,
                safe_responses[:, :, None],
                axis=2,
            )[:, :, 0]
        else:
            observed_probability = np.where(
                responses == 1,
                probabilities,
                1.0 - probabilities,
            )
        observed_probability = np.clip(
            observed_probability,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        residuals["deviance"] = np.where(
            valid,
            np.sign(raw) * np.sqrt(-2.0 * np.log(observed_probability)),
            np.nan,
        )

    expected_values = expected if store_expected else None
    variances = np.where(valid, variance, np.nan) if store_variances else None
    return _ResidualComputation(residuals, expected_values, variances)


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

    batched = _compute_batched_residual_arrays(
        model,
        responses,
        theta,
        residual_types,
        store_expected=store_expected,
        store_variances=store_variances,
    )
    if batched is not None:
        return batched

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
                    p_obs = probs[valid, observed.astype(np.intp)]
                else:
                    p_obs = np.where(observed == 1, probs[valid], 1 - probs[valid])
                p_obs = np.clip(p_obs, PROB_EPSILON, 1 - PROB_EPSILON)
                residuals["deviance"][valid, j] = np.sign(raw) * np.sqrt(
                    -2 * np.log(p_obs)
                )

    return _ResidualComputation(residuals, expected_values, variances)


def compute_residuals(
    model: BaseItemModel | FitResult,
    responses: ArrayLike,
    theta: NDArray[np.float64] | None = None,
    residual_type: str = "standardized",
) -> NDArray[np.float64]:
    """Compute residuals for IRT model.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : array-like of shape (n_persons, n_items)
        Response matrix. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses.
    theta : ndarray, optional
        Ability estimates. If None, EAP estimates are computed.
    residual_type : str
        Type of residual: "raw", "standardized", "pearson", or "deviance"

    Returns
    -------
    ndarray
        Residual matrix of same shape as responses
    """
    responses = _missing_coded_responses(responses)
    if residual_type not in _RESIDUAL_TYPES:
        raise ValueError(f"Unknown residual type: {residual_type}")

    model, theta_array = _resolve_theta(model, responses, theta)
    computation = _compute_residual_arrays(
        model,
        responses,
        theta_array,
        (residual_type,),
    )
    return computation.residuals[residual_type]


def analyze_residuals(
    model: BaseItemModel | FitResult,
    responses: ArrayLike,
    theta: NDArray[np.float64] | None = None,
) -> ResidualAnalysisResult:
    """Comprehensive residual analysis for IRT model.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : array-like of shape (n_persons, n_items)
        Response matrix. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses.
    theta : ndarray, optional
        Ability estimates

    Returns
    -------
    ResidualAnalysisResult
        Complete residual analysis results
    """
    responses = _missing_coded_responses(responses)
    model, theta_array = _resolve_theta(model, responses, theta)
    computation = _compute_residual_arrays(
        model,
        responses,
        theta_array,
        ("raw", "standardized", "pearson", "deviance"),
        store_expected=True,
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

    return ResidualAnalysisResult(
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


def _stream_fit_statistics(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    *,
    standardized: bool = False,
) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
    """Accumulate fit statistics directly from one item probability pass."""
    n_persons, n_items = responses.shape
    totals = _ItemPersonFit(n_persons, n_items, standardized=standardized)

    for item_index in range(n_items):
        probabilities, expected, variance = _item_expected_value_variance(
            model,
            theta,
            item_index,
        )
        column = responses[:, item_index]
        totals.add(
            column[:, None],
            expected[:, None],
            variance[:, None],
            items=slice(item_index, item_index + 1),
            fourth_moment=(
                _fourth_central_moment(probabilities, expected)[:, None]
                if standardized
                else None
            ),
        )

    return totals.finish()


def compute_outfit_infit(
    model: BaseItemModel | FitResult,
    responses: ArrayLike,
    theta: NDArray[np.float64] | None = None,
    *,
    include_counts: bool = False,
    include_standardized: bool = False,
) -> dict[str, NDArray[np.float64] | NDArray[np.intp]]:
    """Compute outfit and infit statistics for items and persons.

    Outfit: unweighted mean square residual
    Infit: information-weighted mean square residual

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : array-like of shape (n_persons, n_items)
        Response matrix. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses.
    theta : ndarray, optional
        Ability estimates
    include_counts : bool, default=False
        Include valid observation counts as ``item_n`` and ``person_n``.
    include_standardized : bool, default=False
        Include Wilson-Hilferty standardized mean squares as
        ``item_z_outfit``, ``item_z_infit``, ``person_z_outfit`` and
        ``person_z_infit``. With the default EAP abilities the item z
        statistics are biased toward overfit and only descriptive (see
        :func:`~mirt.diagnostics.itemfit.compute_itemfit`).

    Returns
    -------
    dict
        Dictionary with ``item_outfit``, ``item_infit``, ``person_outfit``, and
        ``person_infit``. When requested, ``item_n`` and ``person_n`` contain
        the corresponding valid observation counts.

    Notes
    -----
    The statistics match :func:`~mirt.diagnostics.itemfit.compute_itemfit`
    and :func:`~mirt.diagnostics.personfit.compute_personfit`: infit is
    ``sum((x - E)^2) / sum(W)`` over observed responses, and outfit averages
    ``(x - E)^2 / W`` over observed responses whose modeled variance ``W``
    exceeds ``PROB_EPSILON``. Near-deterministic responses are excluded from
    outfit because their squared standardized residuals are unbounded.
    """
    if not isinstance(include_counts, (bool, np.bool_)):
        raise ValueError("include_counts must be boolean")
    if not isinstance(include_standardized, (bool, np.bool_)):
        raise ValueError("include_standardized must be boolean")
    responses = _missing_coded_responses(responses)
    if responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    model, theta_array = _resolve_theta(model, responses, theta)
    statistics = _stream_fit_statistics(
        model,
        responses,
        theta_array,
        standardized=bool(include_standardized),
    )
    if not include_counts:
        del statistics["item_n"]
        del statistics["person_n"]
    return statistics


def identify_misfitting_patterns(
    model: BaseItemModel | FitResult,
    responses: ArrayLike,
    theta: NDArray[np.float64] | None = None,
    z_threshold: float = 2.0,
    outfit_threshold: float = 1.5,
) -> dict[str, list]:
    """Identify misfitting persons and items.

    Compute standardized residuals in bounded probability blocks and retain only
    flagged entries and fit-statistic totals. Models without batch metadata
    retain the itemwise probability fallback. Item and person outfit and infit
    follow :func:`compute_outfit_infit`, so near-deterministic responses do
    not enter outfit.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : array-like of shape (n_persons, n_items)
        Response matrix. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses.
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
    responses = _missing_coded_responses(responses)
    model, theta_array = _resolve_theta(model, responses, theta)
    n_persons, n_items = responses.shape
    totals = _ItemPersonFit(n_persons, n_items)
    rows_per_chunk = max(1, n_persons)
    if getattr(model, "n_items", None) == n_items:
        rows_per_chunk = max(
            1,
            min(_MISFIT_TARGET_CHUNK_ELEMENTS, _RESIDUAL_MAX_PROBABILITY_VALUES)
            // max(1, _probability_values_per_row(model, n_items)),
        )

    aberrant: list[dict] = []
    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        block = responses[start:stop]
        computation = _compute_residual_arrays(
            model,
            block,
            theta_array[start:stop],
            ("standardized",),
            store_expected=True,
            store_variances=True,
        )
        z = computation.residuals["standardized"]
        expected = computation.expected_values
        variances = computation.variances
        assert expected is not None and variances is not None
        usable = np.isfinite(z)
        totals.add(
            np.where(usable, block, -1),
            np.where(usable, expected, 0.0),
            np.where(usable, variances, 0.0),
            persons=slice(start, stop),
        )
        aberrant.extend(
            {
                "person": int(start + i),
                "item": int(j),
                "response": block[i, j],
                "expected": expected[i, j],
                "z": z[i, j],
            }
            for i, j in np.argwhere(np.isfinite(z) & (np.abs(z) > z_threshold))
        )
    fit_stats = totals.finish()

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

    return {
        "misfitting_persons": misfitting_persons,
        "misfitting_items": misfitting_items,
        "aberrant_responses": aberrant,
    }
