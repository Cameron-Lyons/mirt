"""Classical Test Theory statistics.

Provides functions for computing traditional CTT statistics
from response data.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._classical import _alpha_if_deleted_numpy, _sample_variance
from mirt._rust_backend import RUST_AVAILABLE
from mirt._rust_backend import compute_alpha_if_deleted as _rust_alpha_if_deleted

_ITEM_FIT_CHUNK_ELEMENTS = 1_000_000
_RESPONSE_VALIDATION_CHUNK_ELEMENTS = 1_000_000
_ITEM_STATS_MOMENT_CHUNK_ELEMENTS = 1_000_000
_ITEM_STATS_FREQUENCY_CHUNK_ELEMENTS = 8_000_000
_ITEM_STATS_MAX_VECTORIZED_CATEGORIES = 16
_ITEM_STATS_MAX_FREQUENCY_ENTRIES = 5_000_000


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
    del finite

    flat_values = values.ravel()
    flat_missing = missing.ravel()
    for start in range(0, flat_values.size, _RESPONSE_VALIDATION_CHUNK_ELEMENTS):
        stop = min(start + _RESPONSE_VALIDATION_CHUNK_ELEMENTS, flat_values.size)
        observed = flat_values[start:stop][~flat_missing[start:stop]]
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed responses must contain only finite values")
        if np.any(observed != np.floor(observed)):
            raise ValueError("observed responses must be integer-valued")
        if binary and np.any((observed != 0.0) & (observed != 1.0)):
            raise ValueError("responses must contain only 0, 1, or missing values")

    return np.where(missing, np.nan, values), missing


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
        alpha_if_deleted = _alpha_if_deleted_numpy(scored_responses)

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

    Derived properties also provide the observed category count, modal
    response, Shannon entropy, and effective category count for each item.
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

    @property
    def n_categories(self) -> NDArray[np.intp]:
        """Number of distinct observed response categories per item."""
        return np.fromiter(
            (len(frequency) for frequency in self.frequencies),
            dtype=np.intp,
            count=len(self.frequencies),
        )

    @property
    def mode(self) -> NDArray[np.float64]:
        """Modal response per item, using the smaller category to break ties."""
        modes = np.full(len(self.frequencies), np.nan)
        for index, frequency in enumerate(self.frequencies):
            if frequency:
                modes[index] = min(
                    frequency,
                    key=lambda category: (-frequency[category], category),
                )
        return modes

    @property
    def entropy(self) -> NDArray[np.float64]:
        """Shannon response entropy in natural-log units for each item."""
        entropy = np.full(len(self.frequencies), np.nan)
        for index, frequency in enumerate(self.frequencies):
            if not frequency:
                continue
            counts = np.fromiter(frequency.values(), dtype=np.float64)
            probabilities = counts / np.sum(counts)
            entropy[index] = -float(np.sum(probabilities * np.log(probabilities)))
        return entropy

    @property
    def effective_categories(self) -> NDArray[np.float64]:
        """Entropy-equivalent number of equally likely response categories."""
        return np.exp(self.entropy)


def _item_shape_moments(
    responses: NDArray[np.float64],
    valid: NDArray[np.bool_],
    means: NDArray[np.float64],
    counts: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute central moments in bounded item chunks."""
    n_persons, n_items = responses.shape
    chunk_size = min(
        n_items,
        max(1, _ITEM_STATS_MOMENT_CHUNK_ELEMENTS // n_persons),
    )
    second = np.zeros(n_items)
    third = np.zeros(n_items)
    fourth = np.zeros(n_items)

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        block_valid = valid[:, start:stop]
        centered = np.zeros((n_persons, stop - start), dtype=np.float64)
        np.subtract(
            responses[:, start:stop],
            means[None, start:stop],
            out=centered,
            where=block_valid,
        )
        squared = centered * centered
        second[start:stop] = np.sum(squared, axis=0)
        third[start:stop] = np.sum(squared * centered, axis=0)
        fourth[start:stop] = np.sum(squared * squared, axis=0)

    denominator = counts.astype(np.float64)
    for moment in (second, third, fourth):
        np.divide(moment, denominator, out=moment, where=counts > 0)
    return second, third, fourth


def _item_frequency_tables(
    responses: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> list[dict[int, int]]:
    """Count compact category codes across item chunks, with a sparse fallback."""
    n_persons, n_items = responses.shape
    maximum = int(np.max(responses, where=valid, initial=-1.0))
    n_categories = maximum + 1
    vectorized = (
        0 < n_categories <= _ITEM_STATS_MAX_VECTORIZED_CATEGORIES
        and n_categories * n_items <= _ITEM_STATS_MAX_FREQUENCY_ENTRIES
    )
    if not vectorized:
        frequencies: list[dict[int, int]] = []
        for item_index in range(n_items):
            observed = responses[valid[:, item_index], item_index].astype(np.int_)
            categories, counts = np.unique(observed, return_counts=True)
            frequencies.append(
                {
                    int(category): int(count)
                    for category, count in zip(categories, counts, strict=True)
                }
            )
        return frequencies

    chunk_size = min(
        n_items,
        max(1, _ITEM_STATS_FREQUENCY_CHUNK_ELEMENTS // n_persons),
    )
    frequency_counts = np.empty((n_categories, n_items), dtype=np.intp)
    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        block = responses[:, start:stop]
        for category in range(n_categories):
            frequency_counts[category, start:stop] = np.count_nonzero(
                block == category,
                axis=0,
            )

    return [
        {
            int(category): int(frequency_counts[category, item_index])
            for category in np.flatnonzero(frequency_counts[:, item_index])
        }
        for item_index in range(n_items)
    ]


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
        - mode, n_categories: Modal response and observed category count
        - entropy, effective_categories: Response diversity measures

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
    del missing_mask

    mean = np.divide(
        np.sum(responses, axis=0, where=valid, initial=0.0),
        n,
        out=np.full(n_items, np.nan),
        where=n > 0,
    )
    min_val = np.min(responses, axis=0, where=valid, initial=np.inf)
    max_val = np.max(responses, axis=0, where=valid, initial=-np.inf)
    min_val[n == 0] = np.nan
    max_val[n == 0] = np.nan

    second_moment, third_moment, fourth_moment = _item_shape_moments(
        responses,
        valid,
        mean,
        n,
    )
    variances = np.divide(
        second_moment * n,
        n - 1,
        out=np.zeros(n_items),
        where=n > 1,
    )
    sd = np.sqrt(variances)
    sd[n == 0] = np.nan

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

    frequencies = _item_frequency_tables(responses, valid)

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

    entropy = stats.entropy
    data = {
        "n": stats.n,
        "mean": stats.mean,
        "sd": stats.sd,
        "min": stats.min,
        "max": stats.max,
        "mode": stats.mode,
        "n_categories": stats.n_categories,
        "entropy": entropy,
        "effective_categories": np.exp(entropy),
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,
        "n_missing": stats.n_missing,
        "pct_missing": stats.pct_missing,
    }

    return create_dataframe(data, index=item_names, index_name="item")
