"""SIBTEST (Simultaneous Item Bias Test) procedure.

SIBTEST is a nonparametric DIF detection method that uses a matching
criterion based on valid subtest scores.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import split_groups

SIBTESTMethod: TypeAlias = Literal["original", "crossing"]
PValueAdjustment: TypeAlias = Literal["bonferroni", "holm", "fdr_bh", "none"]
SIBTESTResult: TypeAlias = dict[
    str, NDArray[np.float64] | NDArray[np.bool_] | float | int | str
]


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
    data: NDArray[np.int64], groups: NDArray
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Split the response matrix and require estimable group sizes."""
    ref_data, focal_data, ref_mask, focal_mask, _, _ = split_groups(data, groups)
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
        Whether to apply Shealy-Stout regression correction

    Returns
    -------
    dict
        Dictionary with:
        - 'beta': SIBTEST β statistic
        - 'beta_se': Standard error of β
        - 'z': Z-statistic
        - 'p_value': Two-sided p-value
        - 'effect_size': Standardized effect size
    """
    if not isinstance(method, str) or method not in {"original", "crossing"}:
        raise ValueError(f"Unknown SIBTEST method: {method}")
    if not isinstance(correction, (bool, np.bool_)):
        raise ValueError("correction must be boolean")

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
        response_data, group_labels
    )
    matching_scores = np.sum(response_data[:, matching_indices], axis=1)
    suspect_scores_ref = np.sum(ref_data[:, suspect_indices], axis=1)
    suspect_scores_focal = np.sum(focal_data[:, suspect_indices], axis=1)
    beta, beta_se = _compute_sibtest_statistics(
        suspect_scores_ref,
        suspect_scores_focal,
        matching_scores[ref_mask],
        matching_scores[focal_mask],
        method,
        bool(correction),
    )

    if beta_se > PROB_EPSILON:
        z = beta / beta_se
        p_value = 2.0 * stats.norm.sf(abs(z))
    else:
        z = np.nan
        p_value = np.nan

    effect_size = _effect_size(beta, suspect_scores_ref, suspect_scores_focal)

    return {
        "beta": float(beta),
        "beta_se": float(beta_se),
        "z": float(z),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "method": method,
        "n_suspect_items": int(suspect_indices.size),
        "n_matching_items": int(matching_indices.size),
    }


def _stratum_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate group differences and harmonic weights by matching score."""
    n_levels = int(max(np.max(ref_scores), np.max(focal_scores))) + 1
    ref_counts = np.bincount(ref_scores, minlength=n_levels)
    focal_counts = np.bincount(focal_scores, minlength=n_levels)
    common = (ref_counts > 0) & (focal_counts > 0)
    levels = np.flatnonzero(common)
    if levels.size == 0:
        return levels, np.empty(0), np.empty(0)

    ref_sums = np.bincount(ref_scores, weights=ref_suspect, minlength=n_levels)[common]
    focal_sums = np.bincount(focal_scores, weights=focal_suspect, minlength=n_levels)[
        common
    ]
    ref_common = ref_counts[common].astype(np.float64, copy=False)
    focal_common = focal_counts[common].astype(np.float64, copy=False)
    differences = ref_sums / ref_common - focal_sums / focal_common
    weights = 2.0 * ref_common * focal_common / (ref_common + focal_common)
    return levels, differences, weights


def _weighted_mean(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    """Return a stable weighted mean for non-empty arrays."""
    return float(np.dot(weights, values) / np.sum(weights))


def _original_standard_error(
    differences: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_total: int,
) -> float:
    """Compute the existing stratum-weighted SIBTEST standard error."""
    if differences.size < 2:
        return float("nan")
    normalized = weights / np.sum(weights)
    centered = differences - np.dot(normalized, differences)
    return float(np.sqrt(np.dot(normalized, centered * centered) / n_total))


def _regression_correction(
    beta: float,
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
) -> float:
    """Apply the Shealy-Stout regression correction to beta."""
    all_scores = np.concatenate((ref_scores, focal_scores))
    score_variance = float(np.var(all_scores))
    if score_variance <= PROB_EPSILON:
        return beta

    all_suspect = np.concatenate((ref_suspect, focal_suspect))
    slope = float(np.cov(all_scores, all_suspect)[0, 1] / score_variance)
    mean_difference = float(np.mean(ref_scores) - np.mean(focal_scores))
    return beta - slope * mean_difference


def _compute_sibtest_statistics(
    ref_suspect: NDArray[np.int64],
    focal_suspect: NDArray[np.int64],
    ref_scores: NDArray[np.int64],
    focal_scores: NDArray[np.int64],
    method: SIBTESTMethod,
    correction: bool,
) -> tuple[float, float]:
    """Compute a SIBTEST statistic from precomputed subtest scores."""
    levels, differences, weights = _stratum_statistics(
        ref_suspect, focal_suspect, ref_scores, focal_scores
    )
    if differences.size == 0:
        return float("nan"), float("nan")

    n_total = ref_scores.size + focal_scores.size
    if method == "original":
        beta = _weighted_mean(differences, weights)
        if correction:
            beta = _regression_correction(
                beta,
                ref_scores,
                focal_scores,
                ref_suspect,
                focal_suspect,
            )
        return beta, _original_standard_error(differences, weights, n_total)

    median_score = float(np.median(np.concatenate((ref_scores, focal_scores))))
    low = levels <= median_score
    high = ~low
    if not np.any(low) or not np.any(high):
        return float("nan"), float("nan")
    beta_low = _weighted_mean(differences[low], weights[low])
    beta_high = _weighted_mean(differences[high], weights[high])
    beta = beta_high - beta_low
    standard_error = float(np.std(np.array([beta_low, beta_high])) / np.sqrt(n_total))
    return beta, standard_error


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
    """Adjust a family of p-values without an additional dependency."""
    adjusted = np.full_like(p_values, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted

    values = p_values[finite]
    family_size = finite.size
    if method == "none":
        corrected = values
    elif method == "bonferroni":
        corrected = values * family_size
    else:
        order = np.argsort(values)
        ordered = values[order]
        if method == "holm":
            ordered_adjusted = np.maximum.accumulate(
                ordered * (family_size - np.arange(ordered.size))
            )
        else:
            ranks = np.arange(1, ordered.size + 1)
            ordered_adjusted = np.minimum.accumulate(
                (ordered * family_size / ranks)[::-1]
            )[::-1]
        corrected = np.empty_like(values)
        corrected[order] = ordered_adjusted
    adjusted[finite] = np.clip(corrected, 0.0, 1.0)
    return adjusted


def sibtest_items(
    data: NDArray[np.int_],
    groups: NDArray,
    anchor_items: list[int] | NDArray[np.int_] | None = None,
    method: SIBTESTMethod = "original",
    correction: bool = True,
    alpha: float = 0.05,
    p_adjust: PValueAdjustment = "bonferroni",
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
        Whether to apply the Shealy-Stout correction for the original method.
    alpha : float
        Family-wise significance level. Default 0.05.
    p_adjust : {"bonferroni", "holm", "fdr_bh", "none"}
        Multiple-testing adjustment. Default "bonferroni".

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
        - 'flagged': Boolean flags for significant DIF
    """
    if not isinstance(method, str) or method not in {"original", "crossing"}:
        raise ValueError(f"Unknown SIBTEST method: {method}")
    if not isinstance(correction, (bool, np.bool_)):
        raise ValueError("correction must be boolean")
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and between 0 and 1")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and between 0 and 1") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    if not isinstance(p_adjust, str) or p_adjust not in {
        "bonferroni",
        "holm",
        "fdr_bh",
        "none",
    }:
        raise ValueError("p_adjust must be 'bonferroni', 'holm', 'fdr_bh', or 'none'")

    response_data = _validate_response_data(data)
    group_labels = _validate_groups(groups, response_data.shape[0])
    n_items = response_data.shape[1]
    if n_items < 2:
        raise ValueError("sibtest_items requires at least two items")
    ref_data, focal_data, ref_mask, focal_mask = _split_validated_groups(
        response_data, group_labels
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

    betas = np.full(n_items, np.nan)
    standard_errors = np.full(n_items, np.nan)
    zs = np.full(n_items, np.nan)
    p_values = np.full(n_items, np.nan)
    effect_sizes = np.full(n_items, np.nan)

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
        beta, standard_error = _compute_sibtest_statistics(
            ref_suspect,
            focal_suspect,
            matching_scores[ref_mask],
            matching_scores[focal_mask],
            method,
            bool(correction),
        )
        betas[item_index] = beta
        standard_errors[item_index] = standard_error
        effect_sizes[item_index] = _effect_size(beta, ref_suspect, focal_suspect)
        if standard_error > PROB_EPSILON:
            z = beta / standard_error
            zs[item_index] = z
            p_values[item_index] = 2.0 * stats.norm.sf(abs(z))

    adjusted = _adjust_p_values(p_values, p_adjust)
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
        "flagged": flagged,
        "alpha": float(alpha),
        "alpha_corrected": float(corrected_alpha),
        "adjustment": p_adjust,
    }
