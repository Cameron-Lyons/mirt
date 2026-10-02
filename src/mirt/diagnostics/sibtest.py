"""SIBTEST (Simultaneous Item Bias Test) procedure.

SIBTEST is a nonparametric DIF detection method that uses a matching
criterion based on valid subtest scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import split_groups
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)

SIBTESTMethod: TypeAlias = Literal["original", "crossing"]
SIBTESTResult: TypeAlias = dict[
    str, NDArray[np.float64] | NDArray[np.bool_] | float | int | str
]


@dataclass(frozen=True)
class _SIBTESTStatistics:
    beta: float = np.nan
    standard_error: float = np.nan
    chi2: float = np.nan
    df: int = 0
    crossing_point: float = np.nan
    n_strata: int = 0

    @property
    def z(self) -> float:
        if self.standard_error > PROB_EPSILON:
            return self.beta / self.standard_error
        return float("nan")

    @property
    def p_value(self) -> float:
        if self.df == 0:
            return float("nan")
        return float(stats.chi2.sf(self.chi2, self.df))


def _validate_min_cell_size(min_cell_size: int) -> int:
    if (
        isinstance(min_cell_size, (bool, np.bool_))
        or not isinstance(min_cell_size, (int, np.integer))
        or min_cell_size < 2
    ):
        raise ValueError("min_cell_size must be an integer of at least 2")
    return int(min_cell_size)


def _binary_item_variances(data: NDArray[np.int64]) -> NDArray[np.float64]:
    """Unbiased binary variances without allocating a centered response matrix."""
    proportions = np.mean(data, axis=0)
    return proportions * (1.0 - proportions) * (data.shape[0] / (data.shape[0] - 1))


def _validate_response_data(data: NDArray[np.int_]) -> NDArray[np.int64]:
    """Return a finite binary response matrix suitable for SIBTEST."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"data must be a 2D response matrix, got {values.ndim}D")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must contain at least one person and one item")
    if values.dtype.kind not in "biuf":
        raise ValueError("data must contain numeric binary responses")
    if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite responses")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("data must contain only binary responses coded 0 or 1")
    return values.astype(np.int64, copy=False)


def _validate_groups(groups: NDArray, n_persons: int) -> NDArray:
    """Validate group labels before attempting a two-group split."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.shape[0] != n_persons:
        raise ValueError(f"groups must have shape ({n_persons},)")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must contain only finite labels")
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    try:
        np.unique(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("groups must contain comparable labels") from exc
    return labels


def _validate_item_indices(
    items: list[int] | NDArray[np.int_],
    *,
    name: str,
    n_items: int,
) -> NDArray[np.int64]:
    """Validate a one-dimensional, unique item-index collection."""
    indices = np.asarray(items)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional collection of indices")
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one item")
    if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    normalized = indices.astype(np.int64, copy=False)
    if np.any((normalized < 0) | (normalized >= n_items)):
        raise ValueError(f"{name} contains an item index outside [0, {n_items})")
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return normalized


def _split_validated_groups(
    data: NDArray[np.int64], groups: NDArray, focal_group: Any | None = None
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Split the response matrix and require estimable group sizes."""
    ref_data, focal_data, ref_mask, focal_mask, _, _ = split_groups(
        data, groups, focal_group=focal_group
    )
    if ref_data.shape[0] < 2 or focal_data.shape[0] < 2:
        raise ValueError("each group must contain at least two persons")
    return ref_data, focal_data, ref_mask, focal_mask


def sibtest(
    data: NDArray[np.int_],
    groups: NDArray,
    suspect_items: list[int] | NDArray[np.int_],
    matching_items: list[int] | NDArray[np.int_] | None = None,
    method: SIBTESTMethod = "original",
    correction: bool = True,
    *,
    min_cell_size: int = 2,
    focal_group: Any | None = None,
) -> SIBTESTResult:
    """SIBTEST procedure for DIF detection.

    SIBTEST compares the performance of reference and focal groups on
    suspect items after matching on valid (anchor) items.

    Parameters
    ----------
    data : NDArray
        Response matrix (n_persons, n_items)
    groups : NDArray
        Group membership (n_persons,) with exactly 2 unique values
    suspect_items : list or NDArray
        Indices of items to test for DIF
    matching_items : list or NDArray, optional
        Indices of items to use for matching (anchor items).
        If None, uses all items except suspect items.
    method : str
        SIBTEST method:

        - 'original': Standard unidirectional SIBTEST (β_uni)
        - 'crossing': Crossing SIBTEST for non-uniform DIF (β_cross)
    correction : bool
        Apply the Shealy-Stout true-score regression correction using group
        KR-20 reliability. Correction requires at least two matching items,
        positive reliability, and observed adjacent matching-score strata.
    min_cell_size : int, default=2
        Minimum persons in each group at a retained matching score. Larger
        values (for example, 5) exclude sparse strata from asymptotic inference.
    focal_group : optional
        Group label treated as focal. By default, the second sorted label.

    Returns
    -------
    dict
        Dictionary with:

        - 'beta': SIBTEST β statistic
        - 'beta_se': Standard error of β
        - 'z': Z-statistic
        - 'p_value': Two-sided p-value
        - 'effect_size': Standardized effect size
        - 'chi2', 'df': Chi-square test statistic and degrees of freedom.
        - 'crossing_point': Estimated crossing location for the crossing method
        - 'n_strata': Number of retained matching-score strata

    Notes
    -----
    Responses must be complete and binary. Unestimable statistics are ``NaN``
    with zero degrees of freedom. Stratum weights use the pooled population,
    and standard errors use within-group suspect-score sampling variances.
    Crossing inference uses Chalmers (2018)'s one- or two-region chi-square
    test; its p-value is not a normal-tail transformation of ``z``.
    """
    if not isinstance(method, str) or method not in {"original", "crossing"}:
        raise ValueError(f"Unknown SIBTEST method: {method}")
    if not isinstance(correction, (bool, np.bool_)):
        raise ValueError("correction must be boolean")
    min_cell_size = _validate_min_cell_size(min_cell_size)

    response_data = _validate_response_data(data)
    group_labels = _validate_groups(groups, response_data.shape[0])
    n_items = response_data.shape[1]
    suspect_indices = _validate_item_indices(
        suspect_items, name="suspect_items", n_items=n_items
    )

    if matching_items is None:
        selected = np.ones(n_items, dtype=np.bool_)
        selected[suspect_indices] = False
        matching_indices = np.flatnonzero(selected)
        if matching_indices.size == 0:
            raise ValueError("No matching items available")
    else:
        matching_indices = _validate_item_indices(
            matching_items, name="matching_items", n_items=n_items
        )
        if np.intersect1d(suspect_indices, matching_indices).size:
            raise ValueError("suspect_items and matching_items must not overlap")

    ref_data, focal_data, ref_mask, focal_mask = _split_validated_groups(
        response_data, group_labels, focal_group
    )
    matching_scores = np.sum(response_data[:, matching_indices], axis=1)
    suspect_scores_ref = np.sum(ref_data[:, suspect_indices], axis=1)
    suspect_scores_focal = np.sum(focal_data[:, suspect_indices], axis=1)
    result = _compute_sibtest_statistics(
        suspect_scores_ref,
        suspect_scores_focal,
        matching_scores[ref_mask],
        matching_scores[focal_mask],
        method,
        bool(correction),
        min_cell_size=min_cell_size,
        n_matching_items=int(matching_indices.size),
        matching_variances=(
            float(_binary_item_variances(ref_data[:, matching_indices]).sum()),
            float(_binary_item_variances(focal_data[:, matching_indices]).sum()),
        )
        if correction
        else None,
    )
    effect_size = _effect_size(result.beta, suspect_scores_ref, suspect_scores_focal)

    return {
        "beta": result.beta,
        "beta_se": result.standard_error,
        "z": result.z,
        "p_value": result.p_value,
        "chi2": result.chi2,
        "df": result.df,
        "crossing_point": result.crossing_point,
        "n_strata": result.n_strata,
        "effect_size": float(effect_size),
        "method": method,
        "n_suspect_items": int(suspect_indices.size),
        "n_matching_items": int(matching_indices.size),
    }


@dataclass(frozen=True)
class _StratumStatistics:
    ref_counts: NDArray[np.int64]
    focal_counts: NDArray[np.int64]
    ref_means: NDArray[np.float64]
    focal_means: NDArray[np.float64]
    sampling_variances: NDArray[np.float64]


def _stratum_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
) -> _StratumStatistics:
    """Reduce group score moments in linear time without person/stratum scans."""
    n_levels = int(max(np.max(ref_scores), np.max(focal_scores))) + 1
    moments = []
    for scores, suspect in ((ref_scores, ref_suspect), (focal_scores, focal_suspect)):
        counts = np.bincount(scores, minlength=n_levels)
        values = np.asarray(suspect, dtype=np.float64)
        sums = np.bincount(scores, weights=values, minlength=n_levels)
        seconds = np.bincount(scores, weights=values**2, minlength=n_levels)
        means = sums / np.maximum(counts, 1)
        # Unbiased within-stratum score variance, then variance of its mean.
        centered = np.maximum(seconds - sums * means, 0.0)
        mean_variances = centered / (np.maximum(counts - 1, 1) * np.maximum(counts, 1))
        moments.append((counts, means, mean_variances))
    return _StratumStatistics(
        moments[0][0],
        moments[1][0],
        moments[0][1],
        moments[1][1],
        moments[0][2] + moments[1][2],
    )


def _matching_reliability(
    scores: NDArray[np.int64], n_items: int, item_variance_sum: float
) -> float:
    """KR-20 with consistently scaled sample item and total-score variances."""
    if n_items < 2:
        return float("nan")
    score_variance = float(np.var(scores, ddof=1))
    if score_variance <= PROB_EPSILON:
        return float("nan")
    reliability = n_items / (n_items - 1) * (1.0 - item_variance_sum / score_variance)
    return float(reliability) if reliability > PROB_EPSILON else float("nan")


def _regression_corrected_differences(
    strata: _StratumStatistics,
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    n_matching_items: int,
    matching_variances: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Extrapolate each conditional mean to the common estimated true score.

    Shealy and Stout (1993) regress true matched scores on observed scores
    using group-specific reliability, then estimate each suspect-score slope
    from the neighboring conditional means. Endpoints and absent neighbors
    cannot supply that central slope and are excluded.
    """
    n_levels = strata.ref_counts.size
    differences = np.full(n_levels, np.nan)
    usable = np.zeros(n_levels, dtype=bool)
    reliabilities = (
        _matching_reliability(ref_scores, n_matching_items, matching_variances[0]),
        _matching_reliability(focal_scores, n_matching_items, matching_variances[1]),
    )
    if n_levels < 3 or not np.all(np.isfinite(reliabilities)):
        return differences, usable

    ref_mean = float(np.mean(ref_scores))
    focal_mean = float(np.mean(focal_scores))
    levels = np.arange(1, n_levels - 1)
    ref_true = ref_mean + reliabilities[0] * (levels - ref_mean)
    focal_true = focal_mean + reliabilities[1] * (levels - focal_mean)
    common_true = (ref_true + focal_true) / 2.0
    ref_slope = (strata.ref_means[2:] - strata.ref_means[:-2]) / (2 * reliabilities[0])
    focal_slope = (strata.focal_means[2:] - strata.focal_means[:-2]) / (
        2 * reliabilities[1]
    )
    differences[levels] = (
        strata.ref_means[levels]
        + ref_slope * (common_true - ref_true)
        - strata.focal_means[levels]
        - focal_slope * (common_true - focal_true)
    )
    usable[levels] = (
        (strata.ref_counts[:-2] > 0)
        & (strata.ref_counts[2:] > 0)
        & (strata.focal_counts[:-2] > 0)
        & (strata.focal_counts[2:] > 0)
    )
    return differences, usable


def _crossing_location(
    levels: NDArray[np.int64],
    differences: NDArray[np.float64],
    regression_weights: NDArray[np.float64],
) -> float:
    """Estimate one crossing point by weighted conditional-difference regression."""
    normalized = regression_weights / np.sum(regression_weights)
    score_mean = float(np.dot(normalized, levels))
    difference_mean = float(np.dot(normalized, differences))
    centered_scores = levels - score_mean
    score_variance = float(np.dot(normalized, centered_scores**2))
    if score_variance <= PROB_EPSILON:
        return float("nan")
    slope = float(np.dot(normalized, centered_scores * differences) / score_variance)
    if abs(slope) <= PROB_EPSILON:
        return float("nan")
    return score_mean - difference_mean / slope


def _compute_sibtest_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    method: SIBTESTMethod,
    correction: bool,
    *,
    min_cell_size: int = 2,
    n_matching_items: int = 0,
    matching_variances: tuple[float, float] | None = None,
) -> _SIBTESTStatistics:
    """Compute pooled conditional effects and their sampling uncertainty.

    The variance is sum(p_k**2 * (s_Rk**2/n_Rk + s_Fk**2/n_Fk)).
    Crossing inference follows Chalmers (2018), equation 14, summing squared
    standardized effects on the independent sides of the estimated crossing.
    """
    strata = _stratum_statistics(ref_suspect, focal_suspect, ref_scores, focal_scores)
    eligible = (strata.ref_counts >= min_cell_size) & (
        strata.focal_counts >= min_cell_size
    )
    differences = strata.ref_means - strata.focal_means
    if correction:
        if matching_variances is None:
            return _SIBTESTStatistics()
        differences, correction_usable = _regression_corrected_differences(
            strata, ref_scores, focal_scores, n_matching_items, matching_variances
        )
        eligible &= correction_usable
    levels = np.flatnonzero(eligible)
    if levels.size == 0:
        return _SIBTESTStatistics()

    differences = differences[eligible]
    pooled_counts = strata.ref_counts[eligible] + strata.focal_counts[eligible]
    weights = pooled_counts / np.sum(pooled_counts)
    variance_components = weights**2 * strata.sampling_variances[eligible]
    standard_error = float(np.sqrt(np.sum(variance_components)))
    beta = float(np.dot(weights, differences))
    if method == "original":
        chi2 = (beta / standard_error) ** 2 if standard_error > PROB_EPSILON else np.nan
        return _SIBTESTStatistics(
            beta,
            standard_error,
            chi2,
            int(standard_error > PROB_EPSILON),
            n_strata=int(levels.size),
        )

    crossing = _crossing_location(
        levels,
        differences,
        np.maximum(strata.ref_counts[eligible], strata.focal_counts[eligible]),
    )
    low = (
        levels <= crossing
        if np.isfinite(crossing)
        else np.ones(levels.size, dtype=bool)
    )
    region_betas = np.array(
        [
            np.dot(weights[low], differences[low]),
            np.dot(weights[~low], differences[~low]),
        ]
    )
    region_variances = np.array(
        [
            np.sum(variance_components[low]),
            np.sum(variance_components[~low]),
        ]
    )
    estimable = region_variances > PROB_EPSILON**2
    df = int(np.count_nonzero(estimable))
    chi2 = (
        float(np.sum(region_betas[estimable] ** 2 / region_variances[estimable]))
        if df
        else np.nan
    )
    return _SIBTESTStatistics(
        abs(float(region_betas[0] - region_betas[1])),
        standard_error,
        chi2,
        df,
        crossing,
        int(levels.size),
    )


def _effect_size(
    beta: float,
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
) -> float:
    """Standardize beta by the equal-weight pooled suspect-score deviation."""
    pooled_variance = (
        float(np.var(ref_suspect, ddof=1)) + float(np.var(focal_suspect, ddof=1))
    ) / 2.0
    if pooled_variance <= PROB_EPSILON:
        return float("nan")
    return beta / np.sqrt(pooled_variance)


def _adjust_p_values(
    p_values: NDArray[np.float64], method: PValueAdjustment
) -> NDArray[np.float64]:
    """Compatibility wrapper for the shared adjustment utility."""
    return adjust_p_values(p_values, method)


def sibtest_items(
    data: NDArray[np.int_],
    groups: NDArray,
    anchor_items: list[int] | NDArray[np.int_] | None = None,
    method: SIBTESTMethod = "original",
    correction: bool = True,
    alpha: float = 0.05,
    p_adjust: PValueAdjustment = "bonferroni",
    *,
    min_cell_size: int = 2,
    focal_group: Any | None = None,
) -> SIBTESTResult:
    """Run SIBTEST for each item individually.

    Parameters
    ----------
    data : NDArray
        Response matrix
    groups : NDArray
        Group membership
    anchor_items : list or NDArray, optional
        Items to use for matching. The item under test is always excluded.
        If None, all other items are used.
    method : str
        SIBTEST method
    correction : bool
        Apply the Shealy-Stout true-score correction for either method.
    alpha : float
        Family-wise significance level. Default 0.05.
    p_adjust : {"bonferroni", "holm", "fdr_bh", "none"}
        Multiple-testing adjustment. Default "bonferroni".
    min_cell_size : int, default=2
        Minimum persons per group in a retained matching-score stratum.
    focal_group : optional
        Label treated as focal. See :func:`sibtest` for inference details.

    Returns
    -------
    dict
        Dictionary with arrays for each item:

        - 'beta': β statistics
        - 'beta_se': standard errors
        - 'z': Z-statistics
        - 'p_value': P-values
        - 'p_value_adjusted': multiplicity-adjusted P-values
        - 'effect_size': standardized effect sizes
        - 'chi2', 'df': chi-square statistics and degrees of freedom
        - 'crossing_point', 'n_strata': crossing locations and retained strata
        - 'flagged': Boolean flags for significant DIF
    """
    if not isinstance(method, str) or method not in {"original", "crossing"}:
        raise ValueError(f"Unknown SIBTEST method: {method}")
    if not isinstance(correction, (bool, np.bool_)):
        raise ValueError("correction must be boolean")
    min_cell_size = _validate_min_cell_size(min_cell_size)
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and between 0 and 1")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and between 0 and 1") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")

    response_data = _validate_response_data(data)
    group_labels = _validate_groups(groups, response_data.shape[0])
    n_items = response_data.shape[1]
    if n_items < 2:
        raise ValueError("sibtest_items requires at least two items")
    ref_data, focal_data, ref_mask, focal_mask = _split_validated_groups(
        response_data, group_labels, focal_group
    )

    if anchor_items is None:
        anchors = np.arange(n_items, dtype=np.int64)
        matching_totals = np.sum(response_data, axis=1)
    else:
        anchors = _validate_item_indices(
            anchor_items, name="anchor_items", n_items=n_items
        )
        matching_totals = np.sum(response_data[:, anchors], axis=1)
    anchor_membership = np.zeros(n_items, dtype=np.bool_)
    anchor_membership[anchors] = True
    if correction:
        ref_item_variances = _binary_item_variances(ref_data)
        focal_item_variances = _binary_item_variances(focal_data)
        ref_anchor_variance = float(np.sum(ref_item_variances[anchors]))
        focal_anchor_variance = float(np.sum(focal_item_variances[anchors]))

    betas = np.full(n_items, np.nan)
    standard_errors = np.full(n_items, np.nan)
    zs = np.full(n_items, np.nan)
    p_values = np.full(n_items, np.nan)
    effect_sizes = np.full(n_items, np.nan)
    chi2s = np.full(n_items, np.nan)
    degrees = np.zeros(n_items, dtype=np.float64)
    crossing_points = np.full(n_items, np.nan)
    stratum_counts = np.zeros(n_items, dtype=np.float64)

    for item_index in range(n_items):
        if anchors.size == 1 and anchor_membership[item_index]:
            continue
        item_responses = response_data[:, item_index]
        matching_scores = (
            matching_totals - item_responses
            if anchor_membership[item_index]
            else matching_totals
        )
        ref_suspect = ref_data[:, item_index]
        focal_suspect = focal_data[:, item_index]
        matching_variances = None
        if correction:
            matching_variances = (
                ref_anchor_variance - ref_item_variances[item_index]
                if anchor_membership[item_index]
                else ref_anchor_variance,
                focal_anchor_variance - focal_item_variances[item_index]
                if anchor_membership[item_index]
                else focal_anchor_variance,
            )
        item_result = _compute_sibtest_statistics(
            ref_suspect,
            focal_suspect,
            matching_scores[ref_mask],
            matching_scores[focal_mask],
            method,
            bool(correction),
            min_cell_size=min_cell_size,
            n_matching_items=int(anchors.size - anchor_membership[item_index]),
            matching_variances=matching_variances,
        )
        betas[item_index] = item_result.beta
        standard_errors[item_index] = item_result.standard_error
        effect_sizes[item_index] = _effect_size(
            item_result.beta, ref_suspect, focal_suspect
        )
        zs[item_index] = item_result.z
        p_values[item_index] = item_result.p_value
        chi2s[item_index] = item_result.chi2
        degrees[item_index] = item_result.df
        crossing_points[item_index] = item_result.crossing_point
        stratum_counts[item_index] = item_result.n_strata

    adjusted = adjust_p_values(p_values, p_adjust)
    flagged = adjusted < alpha
    n_finite_tests = int(np.count_nonzero(np.isfinite(p_values)))
    corrected_alpha = (
        alpha / n_finite_tests
        if p_adjust == "bonferroni" and n_finite_tests > 0
        else alpha
    )

    return {
        "beta": betas,
        "beta_se": standard_errors,
        "z": zs,
        "p_value": p_values,
        "p_value_adjusted": adjusted,
        "effect_size": effect_sizes,
        "chi2": chi2s,
        "df": degrees,
        "crossing_point": crossing_points,
        "n_strata": stratum_counts,
        "flagged": flagged,
        "alpha": float(alpha),
        "alpha_corrected": float(corrected_alpha),
        "adjustment": p_adjust,
    }
