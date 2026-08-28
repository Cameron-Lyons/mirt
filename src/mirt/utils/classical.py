"""Classical Test Theory statistics.

Provides functions for computing traditional CTT statistics
from response data.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._rust_backend import RUST_AVAILABLE
from mirt._rust_backend import compute_alpha_if_deleted as _rust_alpha_if_deleted

_ITEM_FIT_CHUNK_ELEMENTS = 1_000_000


def _clean_response_matrix(
    responses: ArrayLike,
    *,
    missing_code: float,
    binary: bool,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate responses and normalize missing values to NaN."""
    raw = np.asarray(responses)
    if raw.ndim != 2:
        raise ValueError(f"responses must be a 2D matrix, got {raw.ndim}D")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("responses must contain at least one person and one item")
    if raw.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")

    values = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(values)
    missing = np.isnan(values) | ((values < 0.0) & finite) | (values == missing_code)
    observed = values[~missing]
    if not np.all(np.isfinite(observed)):
        raise ValueError("observed responses must contain only finite values")
    if np.any(observed != np.floor(observed)):
        raise ValueError("observed responses must be integer-valued")
    if binary and np.any((observed != 0.0) & (observed != 1.0)):
        raise ValueError("responses must contain only 0, 1, or missing values")

    return np.where(missing, np.nan, values), missing


def _sample_variance(
    values: NDArray[np.float64],
    *,
    axis: int | None = None,
) -> NDArray[np.float64] | float:
    """Compute sample variance without warnings for sparse columns."""
    valid = np.isfinite(values)
    counts = np.sum(valid, axis=axis)
    sums = np.nansum(values, axis=axis)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    if axis is None:
        squared_deviations = np.where(valid, (values - means) ** 2, 0.0)
    else:
        squared_deviations = np.where(
            valid,
            (values - np.expand_dims(means, axis=axis)) ** 2,
            0.0,
        )
    squared_sum = np.sum(squared_deviations, axis=axis)
    return np.divide(
        squared_sum,
        counts - 1,
        out=np.zeros_like(squared_sum, dtype=np.float64),
        where=counts > 1,
    )


def _cronbach_alpha(responses: NDArray[np.float64]) -> float:
    n_items = responses.shape[1]
    if n_items < 2:
        return 0.0

    item_variances = np.asarray(_sample_variance(responses, axis=0))
    total_scores = np.nansum(responses, axis=1)
    total_scores[np.all(np.isnan(responses), axis=1)] = np.nan
    total_variance = float(_sample_variance(total_scores))
    if total_variance <= 0.0:
        return 0.0
    return float(
        (n_items / (n_items - 1))
        * (1.0 - float(np.sum(item_variances)) / total_variance)
    )


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
    responses: ArrayLike,
    use_corrected_correlation: bool = True,
    missing_code: float = -1,
) -> TraditionalStats:
    """Compute classical test theory statistics.

    Parameters
    ----------
    responses : array-like
        Response matrix. Shape: (n_persons, n_items).
        Values must be 0/1 for dichotomous items. Negative values, NaN,
        and ``missing_code`` are treated as missing.
    use_corrected_correlation : bool
        If True, use corrected item-total correlation
        (excludes the item from the total). Default True.
    missing_code : float, default=-1
        Additional value used to identify missing responses.

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
    if not isinstance(use_corrected_correlation, (bool, np.bool_)):
        raise ValueError("use_corrected_correlation must be boolean")
    responses, _ = _clean_response_matrix(
        responses,
        missing_code=missing_code,
        binary=True,
    )
    n_persons, n_items = responses.shape
    if n_persons < 2:
        raise ValueError("traditional requires at least two respondents")
    if n_items < 2:
        raise ValueError("traditional requires at least two items")
    scored = np.any(np.isfinite(responses), axis=1)
    if np.count_nonzero(scored) < 2:
        raise ValueError("traditional requires at least two respondents with data")

    valid_counts = np.sum(np.isfinite(responses), axis=0)
    difficulty = np.divide(
        np.nansum(responses, axis=0),
        valid_counts,
        out=np.full(n_items, np.nan),
        where=valid_counts > 0,
    )

    total_scores = np.nansum(responses, axis=1)
    total_scores[np.all(np.isnan(responses), axis=1)] = np.nan

    valid = np.isfinite(responses)
    item_values = np.where(valid, responses, 0.0)
    if use_corrected_correlation:
        correlation_totals = np.where(
            valid,
            total_scores[:, None] - item_values,
            0.0,
        )
    else:
        correlation_totals = np.where(valid, total_scores[:, None], 0.0)

    counts = np.sum(valid, axis=0)
    item_sums = np.sum(item_values, axis=0)
    total_sums = np.sum(correlation_totals, axis=0)
    covariance_numerator = np.sum(item_values * correlation_totals, axis=0)
    covariance_numerator -= np.divide(
        item_sums * total_sums,
        counts,
        out=np.zeros(n_items),
        where=counts > 0,
    )
    item_squared = np.sum(item_values**2, axis=0) - np.divide(
        item_sums**2,
        counts,
        out=np.zeros(n_items),
        where=counts > 0,
    )
    total_squared = np.sum(correlation_totals**2, axis=0) - np.divide(
        total_sums**2,
        counts,
        out=np.zeros(n_items),
        where=counts > 0,
    )
    denominator = np.sqrt(np.maximum(item_squared * total_squared, 0.0))
    discrimination = np.divide(
        covariance_numerator,
        denominator,
        out=np.zeros(n_items),
        where=denominator > 0.0,
    )

    scored_responses = responses[scored]
    alpha = _cronbach_alpha(scored_responses)

    if RUST_AVAILABLE:
        alpha_if_deleted = _rust_alpha_if_deleted(scored_responses)
    else:
        alpha_if_deleted = np.array(
            [
                _cronbach_alpha(np.delete(scored_responses, j, axis=1))
                for j in range(n_items)
            ]
        )

    valid_scores = total_scores[np.isfinite(total_scores)]
    score_variance = float(_sample_variance(valid_scores))

    return TraditionalStats(
        difficulty=difficulty,
        discrimination=discrimination,
        alpha=float(alpha),
        n_persons=n_persons,
        n_items=n_items,
        mean_score=float(np.mean(valid_scores)),
        sd_score=float(np.sqrt(score_variance)),
        alpha_if_deleted=alpha_if_deleted,
    )


def _aggregate_item_fit_groups(
    responses: NDArray[np.float64],
    expected: NDArray[np.float64],
    missing: NDArray[np.bool_],
    group_idx: NDArray[np.intp],
    n_groups: int,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Aggregate every item-group table in bounded response-matrix chunks."""
    n_persons, n_items = responses.shape
    chunk_size = min(
        n_items,
        max(1, _ITEM_FIT_CHUNK_ELEMENTS // n_persons),
    )
    group_counts = np.empty((n_items, n_groups), dtype=np.intp)
    observed_counts = np.empty((n_items, n_groups), dtype=np.float64)
    expected_counts = np.empty((n_items, n_groups), dtype=np.float64)
    expected_variances = np.empty((n_items, n_groups), dtype=np.float64)
    grouped = group_idx >= 0

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        width = stop - start
        response_chunk = responses[:, start:stop]
        expected_chunk = expected[:, start:stop]
        valid = grouped[:, None] & ~missing[:, start:stop]
        item_offsets = n_groups * np.arange(width, dtype=np.intp)
        combined_groups = group_idx[:, None] + item_offsets[None, :]
        codes = combined_groups[valid]
        output_size = width * n_groups

        group_counts[start:stop] = np.bincount(
            codes,
            minlength=output_size,
        ).reshape(width, n_groups)
        observed_counts[start:stop] = np.bincount(
            codes,
            weights=response_chunk[valid],
            minlength=output_size,
        ).reshape(width, n_groups)
        probabilities = expected_chunk[valid]
        expected_counts[start:stop] = np.bincount(
            codes,
            weights=probabilities,
            minlength=output_size,
        ).reshape(width, n_groups)
        expected_variances[start:stop] = np.bincount(
            codes,
            weights=probabilities * (1.0 - probabilities),
            minlength=output_size,
        ).reshape(width, n_groups)

    return group_counts, observed_counts, expected_counts, expected_variances


def item_fit_chisq(
    responses: ArrayLike,
    expected: ArrayLike,
    n_groups: int = 10,
    grouping: ArrayLike | None = None,
    min_group_size: int = 5,
    missing_code: float = -1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute chi-square item fit statistics.

    Groups examinees by ability and computes chi-square comparing
    observed and expected proportions.

    Parameters
    ----------
    responses : array-like
        Observed response matrix. Shape: (n_persons, n_items).
        Negative values, NaN, and ``missing_code`` are treated as missing.
    expected : array-like
        Expected probabilities. Shape: (n_persons, n_items).
    n_groups : int, default=10
        Number of ability groups. Default 10.
    grouping : array-like, optional
        Ability estimate used to order respondents. If omitted, observed
        total scores are used.
    min_group_size : int, default=5
        Minimum observed responses required for an item-group contribution.
    missing_code : float, default=-1
        Additional value used to identify missing responses.

    Returns
    -------
    chisq : NDArray[np.float64]
        Chi-square statistics for each item.
    p_values : NDArray[np.float64]
        P-values for each item.
    """
    responses, missing = _clean_response_matrix(
        responses,
        missing_code=missing_code,
        binary=True,
    )
    expected = np.asarray(expected, dtype=np.float64)
    if expected.shape != responses.shape:
        raise ValueError(
            f"expected has shape {expected.shape}, expected {responses.shape}"
        )

    n_persons, n_items = responses.shape
    if (
        isinstance(n_groups, bool)
        or not isinstance(n_groups, (int, np.integer))
        or n_groups < 2
    ):
        raise ValueError("n_groups must be an integer greater than or equal to 2")
    if (
        isinstance(min_group_size, bool)
        or not isinstance(min_group_size, (int, np.integer))
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer")

    observed_expected = expected[~missing]
    if not np.all(np.isfinite(observed_expected)):
        raise ValueError("expected probabilities must be finite for observed responses")
    if np.any((observed_expected < 0.0) | (observed_expected > 1.0)):
        raise ValueError("expected probabilities must be between 0 and 1")

    grouped = np.any(~missing, axis=1)
    if not np.any(grouped):
        raise ValueError("responses must contain at least one observed response")

    if grouping is None:
        grouping_values = np.nansum(responses, axis=1)
    else:
        grouping_values = np.asarray(grouping, dtype=np.float64)
        if grouping_values.ndim != 1 or grouping_values.shape[0] != n_persons:
            raise ValueError("grouping must contain one value per respondent")
        if not np.all(np.isfinite(grouping_values[grouped])):
            raise ValueError("grouping must contain only finite values")

    grouped_indices = np.flatnonzero(grouped)
    order = grouped_indices[np.argsort(grouping_values[grouped_indices], kind="stable")]
    group_idx = np.full(n_persons, -1, dtype=np.intp)
    group_idx[order] = np.minimum(
        np.arange(order.size) * n_groups // order.size,
        int(n_groups) - 1,
    )

    group_counts, observed_counts, expected_counts, expected_variances = (
        _aggregate_item_fit_groups(
            responses,
            expected,
            missing,
            group_idx,
            int(n_groups),
        )
    )
    contributing = (group_counts >= min_group_size) & (
        expected_variances > np.finfo(np.float64).eps
    )
    residuals = observed_counts - expected_counts
    chisq = np.sum(
        np.divide(
            residuals**2,
            expected_variances,
            out=np.zeros_like(expected_variances),
            where=contributing,
        ),
        axis=1,
    )
    degrees_of_freedom = np.maximum(
        np.count_nonzero(contributing, axis=1) - 1,
        1,
    )

    from scipy import stats

    return chisq, np.asarray(stats.chi2.sf(chisq, degrees_of_freedom))


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
        Proportion missing per item, in the interval [0, 1].
    frequencies : list[dict[int, int]]
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
    responses: ArrayLike,
    missing_code: float = -1,
    na_rm: bool = True,
) -> ItemStats:
    """Compute generic item summary statistics.

    Provides descriptive statistics for each item, useful for initial
    data inspection before IRT analysis.

    Parameters
    ----------
    responses : array-like
        Response matrix. Shape: (n_persons, n_items).
        Observed responses must be non-negative integers.
    missing_code : float, default=-1
        Additional value used to identify missing responses. All negative
        values and NaN are also treated as missing.
    na_rm : bool, default=True
        If True, exclude missing values from calculations. If False,
        descriptive statistics are NaN for items containing missing data.

    Returns
    -------
    ItemStats
        Object containing:
        - n: Valid response count per item
        - mean: Mean (proportion correct for binary)
        - sd: Standard deviation
        - min, max: Range of responses
        - skewness, kurtosis: Distribution shape
        - n_missing, pct_missing: Missing count and proportion
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
    if not isinstance(na_rm, (bool, np.bool_)):
        raise ValueError("na_rm must be boolean")
    responses, missing_mask = _clean_response_matrix(
        responses,
        missing_code=missing_code,
        binary=False,
    )
    n_persons, n_items = responses.shape

    valid = ~missing_mask
    n = np.sum(valid, axis=0, dtype=np.intp)
    n_missing = np.sum(missing_mask, axis=0, dtype=np.intp)
    pct_missing = n_missing / n_persons

    mean = np.divide(
        np.nansum(responses, axis=0),
        n,
        out=np.full(n_items, np.nan),
        where=n > 0,
    )
    variances = np.asarray(_sample_variance(responses, axis=0))
    sd = np.sqrt(variances)
    sd[n == 0] = np.nan

    min_val = np.min(np.where(valid, responses, np.inf), axis=0)
    max_val = np.max(np.where(valid, responses, -np.inf), axis=0)
    min_val[n == 0] = np.nan
    max_val[n == 0] = np.nan

    centered = np.where(valid, responses - mean, 0.0)
    second_moment = np.divide(
        np.sum(centered**2, axis=0),
        n,
        out=np.zeros(n_items),
        where=n > 0,
    )
    third_moment = np.divide(
        np.sum(centered**3, axis=0),
        n,
        out=np.zeros(n_items),
        where=n > 0,
    )
    fourth_moment = np.divide(
        np.sum(centered**4, axis=0),
        n,
        out=np.zeros(n_items),
        where=n > 0,
    )
    shaped = (n > 2) & (second_moment > 0.0)
    skewness = np.zeros(n_items)
    kurtosis = np.zeros(n_items)
    skewness[shaped] = third_moment[shaped] / second_moment[shaped] ** 1.5
    kurtosis[shaped] = fourth_moment[shaped] / second_moment[shaped] ** 2 - 3.0
    skewness[n == 0] = np.nan
    kurtosis[n == 0] = np.nan

    if not na_rm:
        affected = n_missing > 0
        for statistic in (
            mean,
            sd,
            min_val,
            max_val,
            skewness,
            kurtosis,
        ):
            statistic[affected] = np.nan

    frequencies: list[dict[int, int]] = []

    for j in range(n_items):
        valid_responses = responses[valid[:, j], j].astype(int)
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
    elif len(item_names) != n_items:
        raise ValueError(f"item_names has length {len(item_names)}, expected {n_items}")

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
