"""Classical Test Theory statistics.

Provides functions for computing traditional CTT statistics
from response data.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt._rust_backend import RUST_AVAILABLE
from mirt._rust_backend import compute_alpha_if_deleted as _rust_alpha_if_deleted


@dataclass
class TraditionalStats:
    """Container for classical test theory statistics.

    Attributes
    ----------
    difficulty : NDArray[np.float64]
        Item difficulty (proportion correct). Shape: (n_items,).
    discrimination : NDArray[np.float64]
        Item-total correlation. Shape: (n_items,).
    alpha : float
        Cronbach's alpha reliability coefficient.
    n_persons : int
        Number of examinees.
    n_items : int
        Number of items.
    mean_score : float
        Mean total score.
    sd_score : float
        Standard deviation of total scores.
    alpha_if_deleted : NDArray[np.float64]
        Alpha if each item is deleted. Shape: (n_items,).
    """

    difficulty: NDArray[np.float64]
    discrimination: NDArray[np.float64]
    alpha: float
    n_persons: int
    n_items: int
    mean_score: float
    sd_score: float
    alpha_if_deleted: NDArray[np.float64]


def traditional(
    responses: NDArray[np.float64],
    use_corrected_correlation: bool = True,
) -> TraditionalStats:
    """Compute classical test theory statistics.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
        Values should be 0/1 for dichotomous items.
    use_corrected_correlation : bool
        If True, use corrected item-total correlation
        (excludes the item from the total). Default True.

    Returns
    -------
    TraditionalStats
        Object containing CTT statistics.

    Examples
    --------
    >>> stats = traditional(responses)
    >>> print(f"Cronbach's alpha: {stats.alpha:.3f}")
    >>> print(f"Mean difficulty: {np.mean(stats.difficulty):.3f}")
    """
    responses = np.asarray(responses, dtype=np.float64)
    n_persons, n_items = responses.shape

    difficulty = np.nanmean(responses, axis=0)

    total_scores = np.nansum(responses, axis=1)

    discrimination = np.zeros(n_items)
    for j in range(n_items):
        item_responses = responses[:, j]
        valid = ~np.isnan(item_responses)

        if use_corrected_correlation:
            corrected_total = total_scores[valid] - item_responses[valid]
        else:
            corrected_total = total_scores[valid]

        if np.std(item_responses[valid]) > 0 and np.std(corrected_total) > 0:
            discrimination[j] = np.corrcoef(item_responses[valid], corrected_total)[
                0, 1
            ]
        else:
            discrimination[j] = 0.0

    item_variances = np.nanvar(responses, axis=0, ddof=1)
    total_variance = np.nanvar(total_scores, ddof=1)

    if total_variance > 0:
        alpha = (n_items / (n_items - 1)) * (
            1 - np.sum(item_variances) / total_variance
        )
    else:
        alpha = 0.0

    if RUST_AVAILABLE:
        alpha_if_deleted = _rust_alpha_if_deleted(responses)
    else:
        alpha_if_deleted = np.zeros(n_items)
        for j in range(n_items):
            remaining_items = np.delete(responses, j, axis=1)
            remaining_variances = np.nanvar(remaining_items, axis=0, ddof=1)
            remaining_total = np.nansum(remaining_items, axis=1)
            remaining_total_var = np.nanvar(remaining_total, ddof=1)

            k = n_items - 1
            if remaining_total_var > 0 and k > 1:
                alpha_if_deleted[j] = (k / (k - 1)) * (
                    1 - np.sum(remaining_variances) / remaining_total_var
                )
            else:
                alpha_if_deleted[j] = 0.0

    return TraditionalStats(
        difficulty=difficulty,
        discrimination=discrimination,
        alpha=float(alpha),
        n_persons=n_persons,
        n_items=n_items,
        mean_score=float(np.nanmean(total_scores)),
        sd_score=float(np.nanstd(total_scores, ddof=1)),
        alpha_if_deleted=alpha_if_deleted,
    )


def item_fit_chisq(
    responses: NDArray[np.float64],
    expected: NDArray[np.float64],
    n_groups: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute chi-square item fit statistics.

    Groups examinees by ability and computes chi-square comparing
    observed and expected proportions.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Observed response matrix. Shape: (n_persons, n_items).
    expected : NDArray[np.float64]
        Expected probabilities. Shape: (n_persons, n_items).
    n_groups : int
        Number of ability groups. Default 10.

    Returns
    -------
    chisq : NDArray[np.float64]
        Chi-square statistics for each item.
    p_values : NDArray[np.float64]
        P-values for each item.
    """
    n_persons, n_items = responses.shape

    total_scores = np.nansum(responses, axis=1)
    group_idx = np.floor(
        np.argsort(np.argsort(total_scores)) / n_persons * n_groups
    ).astype(int)
    group_idx = np.minimum(group_idx, n_groups - 1)

    chisq = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for j in range(n_items):
        chi2 = 0.0
        valid_groups = 0

        for g in range(n_groups):
            mask = group_idx == g
            n_g = np.sum(mask)

            if n_g < 5:
                continue

            obs = np.nanmean(responses[mask, j])
            exp = np.nanmean(expected[mask, j])

            if exp > 0.01 and exp < 0.99:
                chi2 += n_g * (obs - exp) ** 2 / (exp * (1 - exp))
                valid_groups += 1

        chisq[j] = chi2
        df = max(valid_groups - 1, 1)
        from scipy import stats

        p_values[j] = 1 - stats.chi2.cdf(chi2, df)

    return chisq, p_values


@dataclass
class ItemStats:
    """Container for generic item summary statistics.

    Attributes
    ----------
    n : NDArray[np.intp]
        Number of valid responses per item.
    mean : NDArray[np.float64]
        Mean response per item (p-value for binary).
    sd : NDArray[np.float64]
        Standard deviation per item.
    min : NDArray[np.float64]
        Minimum response per item.
    max : NDArray[np.float64]
        Maximum response per item.
    skewness : NDArray[np.float64]
        Skewness per item.
    kurtosis : NDArray[np.float64]
        Excess kurtosis per item.
    n_missing : NDArray[np.intp]
        Number of missing values per item.
    pct_missing : NDArray[np.float64]
        Percentage missing per item.
    frequencies : list[dict]
        Response frequency tables per item.
    """

    n: NDArray[np.intp]
    mean: NDArray[np.float64]
    sd: NDArray[np.float64]
    min: NDArray[np.float64]
    max: NDArray[np.float64]
    skewness: NDArray[np.float64]
    kurtosis: NDArray[np.float64]
    n_missing: NDArray[np.intp]
    pct_missing: NDArray[np.float64]
    frequencies: list[dict[int, int]]


def itemstats(
    responses: NDArray[np.float64],
    missing_code: int = -1,
    na_rm: bool = True,
) -> ItemStats:
    """Compute generic item summary statistics.

    Provides descriptive statistics for each item, useful for initial
    data inspection before IRT analysis.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    missing_code : int
        Code used to indicate missing values. Default -1.
    na_rm : bool
        If True, exclude missing values from calculations. Default True.

    Returns
    -------
    ItemStats
        Object containing:
        - n: Valid response count per item
        - mean: Mean (proportion correct for binary)
        - sd: Standard deviation
        - min, max: Range of responses
        - skewness, kurtosis: Distribution shape
        - n_missing, pct_missing: Missing data info
        - frequencies: Response distribution tables

    Examples
    --------
    >>> from mirt import load_dataset, itemstats
    >>> data = load_dataset('LSAT7')['data']
    >>> stats = itemstats(data)
    >>> print(f"Item means (p-values): {stats.mean}")
    >>> print(f"Missing rate: {stats.pct_missing.mean():.1%}")

    Notes
    -----
    For binary items, the mean is the proportion correct (p-value).
    For polytomous items, interpret as average category selected.
    """
    from scipy import stats as sp_stats

    responses = np.asarray(responses, dtype=np.float64)
    n_persons, n_items = responses.shape

    missing_mask = (responses == missing_code) | np.isnan(responses)

    n = np.zeros(n_items, dtype=np.intp)
    mean = np.zeros(n_items)
    sd = np.zeros(n_items)
    min_val = np.zeros(n_items)
    max_val = np.zeros(n_items)
    skewness = np.zeros(n_items)
    kurtosis = np.zeros(n_items)
    n_missing = np.zeros(n_items, dtype=np.intp)
    pct_missing = np.zeros(n_items)
    frequencies: list[dict[int, int]] = []

    for j in range(n_items):
        item_missing = missing_mask[:, j]
        n_missing[j] = int(np.sum(item_missing))
        pct_missing[j] = n_missing[j] / n_persons * 100

        if na_rm:
            item_data = responses[~item_missing, j]
        else:
            item_data = responses[:, j]

        n[j] = len(item_data)

        if n[j] > 0:
            mean[j] = np.nanmean(item_data)
            sd[j] = np.nanstd(item_data, ddof=1) if n[j] > 1 else 0.0
            min_val[j] = np.nanmin(item_data)
            max_val[j] = np.nanmax(item_data)

            if n[j] > 2 and sd[j] > 0:
                skewness[j] = float(sp_stats.skew(item_data, nan_policy="omit"))
                kurtosis[j] = float(sp_stats.kurtosis(item_data, nan_policy="omit"))

        valid_responses = responses[~item_missing, j].astype(int)
        unique, counts = np.unique(valid_responses, return_counts=True)
        freq_dict = {int(k): int(v) for k, v in zip(unique, counts)}
        frequencies.append(freq_dict)

    return ItemStats(
        n=n,
        mean=mean,
        sd=sd,
        min=min_val,
        max=max_val,
        skewness=skewness,
        kurtosis=kurtosis,
        n_missing=n_missing,
        pct_missing=pct_missing,
        frequencies=frequencies,
    )


def itemstats_to_dataframe(
    stats: ItemStats, item_names: list[str] | None = None
) -> Any:
    """Convert ItemStats to a DataFrame.

    Parameters
    ----------
    stats : ItemStats
        Item statistics object.
    item_names : list of str, optional
        Names for items. If None, uses Item_1, Item_2, etc.

    Returns
    -------
    DataFrame
        DataFrame with item statistics.
    """
    from mirt.utils.dataframe import create_dataframe

    n_items = len(stats.n)
    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    data = {
        "n": stats.n,
        "mean": stats.mean,
        "sd": stats.sd,
        "min": stats.min,
        "max": stats.max,
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,
        "n_missing": stats.n_missing,
        "pct_missing": stats.pct_missing,
    }

    return create_dataframe(data, index=item_names, index_name="item")
