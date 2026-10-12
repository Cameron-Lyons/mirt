"""Shared array kernels for classical test statistics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_ALPHA_DELETED_CHUNK_ELEMENTS = 2_000_000


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
    squared_deviations = np.array(values, dtype=np.float64, copy=True)
    squared_deviations -= means if axis is None else np.expand_dims(means, axis=axis)
    np.square(squared_deviations, out=squared_deviations)
    np.copyto(squared_deviations, 0.0, where=~valid)
    squared_sum = np.sum(squared_deviations, axis=axis)
    return np.divide(
        squared_sum,
        counts - 1,
        out=np.zeros_like(squared_sum, dtype=np.float64),
        where=counts > 1,
    )


def _alpha_if_deleted_numpy(
    responses: NDArray[np.float64],
    *,
    total_scores: NDArray[np.float64] | None = None,
    observed_counts: NDArray[np.intp] | None = None,
    item_variances: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute deletion coefficients, optionally reusing prepared statistics.

    Missing (NaN) responses add nothing to the totals, so the coefficients
    are the usual ones only for complete data; see
    :func:`_complete_case_alpha_if_deleted` for incomplete responses.
    """
    n_persons, n_items = responses.shape
    remaining_items = n_items - 1
    alpha = np.zeros(n_items, dtype=np.float64)
    if remaining_items < 2:
        return alpha

    if observed_counts is None:
        observed_counts = np.sum(np.isfinite(responses), axis=1, dtype=np.intp)
    if total_scores is None:
        total_scores = np.nansum(responses, axis=1)
    if item_variances is None:
        item_variances = np.asarray(_sample_variance(responses, axis=0))
    remaining_variance_sums = np.sum(item_variances) - item_variances
    chunk_size = min(
        n_items,
        max(1, _ALPHA_DELETED_CHUNK_ELEMENTS // max(1, n_persons)),
    )

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        valid_chunk = np.isfinite(responses[:, start:stop])
        deleted_scores = total_scores[:, None] - np.where(
            valid_chunk,
            responses[:, start:stop],
            0.0,
        )
        deleted_scores[observed_counts[:, None] == valid_chunk] = np.nan
        total_variances = np.asarray(_sample_variance(deleted_scores, axis=0))
        usable = total_variances > 0.0
        alpha[start:stop] = np.divide(
            remaining_variance_sums[start:stop],
            total_variances,
            out=np.zeros(stop - start, dtype=np.float64),
            where=usable,
        )
        alpha[start:stop] = np.where(
            usable,
            (remaining_items / (remaining_items - 1)) * (1.0 - alpha[start:stop]),
            0.0,
        )

    alpha[np.abs(alpha) <= 4.0 * np.finfo(np.float64).eps] = 0.0
    return alpha


def _complete_case_alpha(
    rows: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Return alpha, standardized alpha and total-score mean and SD of rows.

    ``rows`` holds complete responses only. Coefficients are NaN with fewer
    than two rows, alpha is zero when the totals do not vary, and the
    standardized coefficient is NaN when some item does not vary (as R's
    ``CA(scale(x))``).
    """
    n_rows, n_items = rows.shape
    if n_rows == 0:
        return np.nan, np.nan, np.nan, np.nan
    totals = np.sum(rows, axis=1)
    mean_total = float(np.mean(totals))
    if n_rows < 2:
        return np.nan, np.nan, mean_total, np.nan
    item_variances = np.asarray(_sample_variance(rows, axis=0))
    total_variance = float(_sample_variance(totals))
    alpha = (
        n_items / (n_items - 1) * (1.0 - float(np.sum(item_variances)) / total_variance)
        if total_variance > 0.0
        else 0.0
    )
    std_alpha = _standardized_alpha(rows, item_variances)
    return float(alpha), std_alpha, mean_total, float(np.sqrt(total_variance))


def _standardized_alpha(
    rows: NDArray[np.float64],
    item_variances: NDArray[np.float64],
) -> float:
    """Alpha of standardized complete responses, R's ``CA(scale(x))``.

    The sum of standardized items is a weighted total, so its variance needs
    no standardized copy of the responses. Constant items make it NaN.
    """
    n_items = rows.shape[1]
    if rows.shape[0] < 2 or not np.all(item_variances > 0.0):
        return np.nan
    standardized_variance = float(
        _sample_variance(rows @ (1.0 / np.sqrt(item_variances)))
    )
    if standardized_variance <= 0.0:
        return 0.0
    return float(n_items / (n_items - 1) * (1.0 - n_items / standardized_variance))


def _complete_case_alpha_if_deleted(
    responses: NDArray[np.float64],
    missing: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Alpha without each item over respondents who answered all others.

    This is R's ``itemstats`` ``alpha_if_rm``: deleting item ``j`` keeps the
    complete respondents and those whose only missing response is ``j``.
    Shared sums over complete rows make the cost linear in the responses.
    Items no one answered, and deletions with fewer than two respondents,
    are NaN; tests left with fewer than two items are zero.
    """
    n_persons, n_items = responses.shape
    remaining_items = n_items - 1
    if remaining_items < 2:
        return np.zeros(n_items, dtype=np.float64)

    observed = ~missing
    counts = np.sum(observed, axis=0)
    offsets = np.divide(
        np.sum(responses, axis=0, where=observed, initial=0.0),
        counts,
        out=np.zeros(n_items),
        where=counts > 0,
    )
    # Centering by item means keeps the sums of squares well conditioned.
    centered = np.where(observed, responses - offsets, 0.0)
    missing_counts = np.sum(missing, axis=1)

    base = centered[missing_counts == 0]
    n_base = base.shape[0]
    base_sums = np.sum(base, axis=0)
    base_squares = np.einsum("ij,ij->j", base, base)
    base_totals = np.sum(base, axis=1)
    total_sum = float(np.sum(base_totals))
    total_square = float(base_totals @ base_totals)
    total_cross = base_totals @ base
    del base

    single_rows = np.flatnonzero(missing_counts == 1)
    single_items = np.argmax(missing[single_rows], axis=1)
    order = np.argsort(single_items, kind="stable")
    single_rows = single_rows[order]
    boundaries = np.searchsorted(single_items[order], np.arange(n_items + 1))

    alpha = np.full(n_items, np.nan, dtype=np.float64)
    factor = remaining_items / (remaining_items - 1)
    for item in range(n_items):
        if counts[item] == 0:
            continue
        extra = centered[single_rows[boundaries[item] : boundaries[item + 1]]]
        n_rows = n_base + extra.shape[0]
        if n_rows < 2:
            continue
        sums = base_sums + np.sum(extra, axis=0)
        squares = base_squares + np.einsum("ij,ij->j", extra, extra)
        variances = np.maximum(squares - sums * sums / n_rows, 0.0) / (n_rows - 1)
        item_variance_sum = float(np.sum(variances) - variances[item])

        # Rest scores: complete rows drop item j; rows missing only j are
        # already rest scores because their j entry is zero.
        extra_rest = np.sum(extra, axis=1)
        rest_sum = total_sum - base_sums[item] + float(np.sum(extra_rest))
        rest_square = (
            total_square
            - 2.0 * total_cross[item]
            + base_squares[item]
            + float(extra_rest @ extra_rest)
        )
        rest_variance = max(rest_square - rest_sum * rest_sum / n_rows, 0.0) / (
            n_rows - 1
        )
        alpha[item] = (
            factor * (1.0 - item_variance_sum / rest_variance)
            if rest_variance > 0.0
            else 0.0
        )
    return alpha


def _pairwise_covariance_sums(
    responses: NDArray[np.float64],
    missing: NDArray[np.bool_],
    *,
    chunk_elements: int = _ALPHA_DELETED_CHUNK_ELEMENTS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reduce pairwise-complete covariances and correlations by item.

    Returns the item variances (the diagonal), the row sums of the pairwise
    covariance matrix and the row sums of the pairwise correlation matrix.
    Each pair uses the respondents who answered both items, as R's
    ``cov(x, use = "pairwise.complete.obs")``. Pairs with fewer than two
    shared responses make their rows NaN. Item blocks bound the memory of
    the ``n_items``-wide pair statistics.
    """
    n_persons, n_items = responses.shape
    observed = ~missing
    counts = np.sum(observed, axis=0)
    offsets = np.divide(
        np.sum(responses, axis=0, where=observed, initial=0.0),
        counts,
        out=np.zeros(n_items),
        where=counts > 0,
    )
    centered = np.where(observed, responses - offsets, 0.0)
    squares = centered * centered
    mask = observed.astype(np.float64)

    diagonal = np.empty(n_items, dtype=np.float64)
    covariance_sums = np.empty(n_items, dtype=np.float64)
    correlation_sums = np.empty(n_items, dtype=np.float64)
    block = max(1, chunk_elements // max(1, n_items))
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        columns = slice(start, stop)
        pair_counts = mask[:, columns].T @ mask
        row_sums = centered[:, columns].T @ mask
        column_sums = mask[:, columns].T @ centered
        row_squares = squares[:, columns].T @ mask
        column_squares = mask[:, columns].T @ squares
        products = centered[:, columns].T @ centered

        defined = pair_counts >= 2.0
        safe_counts = np.where(defined, pair_counts, 2.0)
        covariance = np.where(
            defined,
            (products - row_sums * column_sums / safe_counts) / (safe_counts - 1.0),
            np.nan,
        )
        row_variance = (row_squares - row_sums * row_sums / safe_counts) / (
            safe_counts - 1.0
        )
        column_variance = (column_squares - column_sums * column_sums / safe_counts) / (
            safe_counts - 1.0
        )
        scale = np.sqrt(
            np.maximum(row_variance, 0.0) * np.maximum(column_variance, 0.0)
        )
        correlation = np.full_like(covariance, np.nan)
        np.divide(covariance, scale, out=correlation, where=defined & (scale > 0.0))
        np.clip(correlation, -1.0, 1.0, out=correlation)

        positions = np.arange(stop - start)
        diagonal[columns] = covariance[positions, positions + start]
        covariance_sums[columns] = np.sum(covariance, axis=1)
        correlation_sums[columns] = np.sum(correlation, axis=1)
    return diagonal, covariance_sums, correlation_sums
