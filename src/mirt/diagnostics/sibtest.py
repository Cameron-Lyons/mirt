"""SIBTEST (Simultaneous Item Bias Test) procedure.

SIBTEST is a nonparametric DIF detection method that uses a matching
criterion based on valid subtest scores.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def sibtest(
    data: NDArray[np.int_],
    groups: NDArray,
    suspect_items: list[int] | NDArray[np.int_],
    matching_items: list[int] | NDArray[np.int_] | None = None,
    method: Literal["original", "crossing"] = "original",
    correction: bool = True,
) -> dict[str, NDArray[np.float64] | float]:
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
    data = np.asarray(data)
    groups = np.asarray(groups)
    suspect_items = np.asarray(suspect_items)

    n_persons, n_items = data.shape

    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(unique_groups)}")

    ref_group, focal_group = unique_groups[0], unique_groups[1]

    if matching_items is None:
        all_items = set(range(n_items))
        matching_items = np.array(sorted(all_items - set(suspect_items)))
    else:
        matching_items = np.asarray(matching_items)

    if len(matching_items) == 0:
        raise ValueError("No matching items available")

    matching_scores = data[:, matching_items].sum(axis=1)

    ref_mask = groups == ref_group
    focal_mask = groups == focal_group

    ref_scores = matching_scores[ref_mask]
    focal_scores = matching_scores[focal_mask]

    ref_data = data[ref_mask]
    focal_data = data[focal_mask]

    if method == "original":
        beta, beta_se = _compute_sibtest_original(
            ref_data,
            focal_data,
            ref_scores,
            focal_scores,
            suspect_items,
            correction,
        )
    elif method == "crossing":
        beta, beta_se = _compute_sibtest_crossing(
            ref_data,
            focal_data,
            ref_scores,
            focal_scores,
            suspect_items,
        )
    else:
        raise ValueError(f"Unknown SIBTEST method: {method}")

    if beta_se > 1e-10:
        z = beta / beta_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z = np.nan
        p_value = np.nan

    suspect_scores_ref = ref_data[:, suspect_items].sum(axis=1)
    suspect_scores_focal = focal_data[:, suspect_items].sum(axis=1)
    pooled_sd = np.sqrt(
        (np.var(suspect_scores_ref, ddof=1) + np.var(suspect_scores_focal, ddof=1)) / 2
    )

    if pooled_sd > 1e-10:
        effect_size = beta / pooled_sd
    else:
        effect_size = np.nan

    return {
        "beta": float(beta),
        "beta_se": float(beta_se),
        "z": float(z),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "method": method,
        "n_suspect_items": len(suspect_items),
        "n_matching_items": len(matching_items),
    }


def _compute_sibtest_original(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
    correction: bool,
) -> tuple[float, float]:
    """Compute original SIBTEST β_uni statistic."""
    all_scores = np.concatenate([ref_scores, focal_scores])
    unique_scores = np.unique(all_scores)

    beta_k = []
    n_k = []

    for k in unique_scores:
        ref_at_k = ref_data[ref_scores == k]
        focal_at_k = focal_data[focal_scores == k]

        n_ref_k = len(ref_at_k)
        n_focal_k = len(focal_at_k)

        if n_ref_k > 0 and n_focal_k > 0:
            mean_ref_k = ref_at_k[:, suspect_items].sum(axis=1).mean()
            mean_focal_k = focal_at_k[:, suspect_items].sum(axis=1).mean()

            beta_k.append(mean_ref_k - mean_focal_k)

            n_k.append(2 * n_ref_k * n_focal_k / (n_ref_k + n_focal_k))

    if len(beta_k) == 0:
        return np.nan, np.nan

    beta_k = np.array(beta_k)
    n_k = np.array(n_k)

    beta = np.sum(n_k * beta_k) / np.sum(n_k)

    if correction:
        beta = _regression_correction(
            beta, ref_scores, focal_scores, suspect_items, ref_data, focal_data
        )

    se = _compute_sibtest_se(
        ref_data, focal_data, ref_scores, focal_scores, suspect_items, beta_k, n_k
    )

    return beta, se


def _compute_sibtest_crossing(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
) -> tuple[float, float]:
    """Compute crossing SIBTEST β_cross statistic for non-uniform DIF."""
    all_scores = np.concatenate([ref_scores, focal_scores])
    unique_scores = np.unique(all_scores)

    median_score = np.median(all_scores)

    beta_low_k = []
    beta_high_k = []
    n_low_k = []
    n_high_k = []

    for k in unique_scores:
        ref_at_k = ref_data[ref_scores == k]
        focal_at_k = focal_data[focal_scores == k]

        n_ref_k = len(ref_at_k)
        n_focal_k = len(focal_at_k)

        if n_ref_k > 0 and n_focal_k > 0:
            mean_ref_k = ref_at_k[:, suspect_items].sum(axis=1).mean()
            mean_focal_k = focal_at_k[:, suspect_items].sum(axis=1).mean()
            beta_k = mean_ref_k - mean_focal_k
            weight = 2 * n_ref_k * n_focal_k / (n_ref_k + n_focal_k)

            if k <= median_score:
                beta_low_k.append(beta_k)
                n_low_k.append(weight)
            else:
                beta_high_k.append(beta_k)
                n_high_k.append(weight)

    if len(beta_low_k) == 0 or len(beta_high_k) == 0:
        return np.nan, np.nan

    beta_low = np.sum(np.array(n_low_k) * np.array(beta_low_k)) / np.sum(n_low_k)
    beta_high = np.sum(np.array(n_high_k) * np.array(beta_high_k)) / np.sum(n_high_k)

    beta_cross = beta_high - beta_low

    n_total = len(ref_scores) + len(focal_scores)
    se = np.sqrt(1 / n_total) * np.std([beta_low, beta_high])

    return beta_cross, se


def _regression_correction(
    beta: float,
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
) -> float:
    """Apply Shealy-Stout regression correction to β."""
    len(suspect_items)

    all_scores = np.concatenate([ref_scores, focal_scores])
    all_suspect = np.concatenate(
        [
            ref_data[:, suspect_items].sum(axis=1),
            focal_data[:, suspect_items].sum(axis=1),
        ]
    )

    if np.var(all_scores) > 1e-10:
        slope = np.cov(all_scores, all_suspect)[0, 1] / np.var(all_scores)
    else:
        slope = 0

    mean_ref = np.mean(ref_scores)
    mean_focal = np.mean(focal_scores)

    correction = slope * (mean_ref - mean_focal)

    return beta - correction


def _compute_sibtest_se(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
    beta_k: NDArray[np.float64],
    n_k: NDArray[np.float64],
) -> float:
    """Compute standard error for SIBTEST β."""
    if len(beta_k) < 2:
        return np.nan

    weights = n_k / n_k.sum()
    weighted_mean = np.sum(weights * beta_k)
    weighted_var = np.sum(weights * (beta_k - weighted_mean) ** 2)

    n_total = len(ref_scores) + len(focal_scores)

    se = np.sqrt(weighted_var / n_total)

    return float(se)


def sibtest_items(
    data: NDArray[np.int_],
    groups: NDArray,
    anchor_items: list[int] | NDArray[np.int_] | None = None,
    method: Literal["original", "crossing"] = "original",
) -> dict[str, NDArray[np.float64]]:
    """Run SIBTEST for each item individually.

    Parameters
    ----------
    data : NDArray
        Response matrix
    groups : NDArray
        Group membership
    anchor_items : list or NDArray, optional
        Items to use for matching. If None, uses iterative purification.
    method : str
        SIBTEST method

    Returns
    -------
    dict
        Dictionary with arrays for each item:
        - 'beta': β statistics
        - 'z': Z-statistics
        - 'p_value': P-values
        - 'flagged': Boolean flags for significant DIF
    """
    data = np.asarray(data)
    n_items = data.shape[1]

    betas = np.zeros(n_items)
    zs = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for i in range(n_items):
        if anchor_items is None:
            matching = [j for j in range(n_items) if j != i]
        else:
            matching = [j for j in anchor_items if j != i]

        if len(matching) == 0:
            betas[i] = np.nan
            zs[i] = np.nan
            p_values[i] = np.nan
            continue

        result = sibtest(
            data, groups, suspect_items=[i], matching_items=matching, method=method
        )

        betas[i] = result["beta"]
        zs[i] = result["z"]
        p_values[i] = result["p_value"]

    alpha = 0.05 / n_items
    flagged = p_values < alpha

    return {
        "beta": betas,
        "z": zs,
        "p_value": p_values,
        "flagged": flagged,
        "alpha_corrected": alpha,
    }
