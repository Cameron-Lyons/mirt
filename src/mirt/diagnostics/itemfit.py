from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)
from mirt.utils.numeric import _FitStatsAccumulator, compute_expected_variance

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_SX2_TARGET_CHUNK_ELEMENTS = 1_000_000
_ITEMFIT_TARGET_CHUNK_ELEMENTS = 262_144


def compute_itemfit(
    model: BaseItemModel,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
    theta: NDArray[np.float64] | None = None,
    n_groups: int = 10,
    p_adjust: PValueAdjustment = "none",
) -> dict[str, NDArray[np.float64]]:
    """Compute requested item-fit statistics.

    S-X2 results include the statistic, degrees of freedom, and p-value under
    the keys ``"S_X2"``, ``"df"``, and ``"p_value"``. When ``p_adjust`` is
    not ``"none"``, ``"p_value_adjusted"`` contains multiplicity-adjusted
    p-values across items.

    Probabilities, score moments, and group reductions use bounded row blocks.
    Score groups and multiplicity corrections use the full population.
    """
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    if statistics is None:
        statistics = ["infit", "outfit"]

    if responses is None:
        raise ValueError("responses required for item fit statistics")

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape
    if n_persons == 0:
        raise ValueError("responses must contain at least one person")

    compute_mean_squares = "infit" in statistics or "outfit" in statistics
    compute_sx2 = "S_X2" in statistics
    if not compute_mean_squares and not compute_sx2:
        return {}
    if compute_sx2:
        n_groups = _validate_n_groups(n_groups)

    if theta is None:
        from mirt.scoring import fscores

        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if theta.ndim != 2 or theta.shape[0] != n_persons:
        raise ValueError("theta must be a matrix with one row per person")

    category_width = max(model.n_categories) if model.is_polytomous else 1
    rows_per_chunk = max(
        1, _ITEMFIT_TARGET_CHUNK_ELEMENTS // max(n_items * category_width, 1)
    )
    accumulator = _FitStatsAccumulator(n_items) if compute_mean_squares else None
    group_indices = None
    if compute_sx2:
        group_indices = _score_group_indices(responses, n_groups)
        score_scales = (
            np.asarray(model.n_categories, dtype=np.float64) - 1.0
            if model.is_polytomous
            else np.ones(n_items)
        )
        group_sums = tuple(np.zeros((n_groups, n_items)) for _ in range(3))

    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        block_responses = responses[start:stop]
        expected, variance = compute_expected_variance(
            model, theta[start:stop], n_items
        )
        if accumulator is not None:
            accumulator.add(block_responses, expected, variance)
        if group_indices is not None:
            block_sums = _grouped_item_sums(
                block_responses,
                expected,
                block_responses >= 0,
                group_indices[start:stop],
                score_scales,
                n_groups,
            )
            for total, block_sum in zip(group_sums, block_sums, strict=True):
                total += block_sum

    result: dict[str, NDArray[np.float64]] = {}
    if accumulator is not None:
        infit, outfit = accumulator.finish()

        if "outfit" in statistics:
            result["outfit"] = outfit

        if "infit" in statistics:
            result["infit"] = infit

    if group_indices is not None:
        s_x2_result = _s_x2_from_group_sums(*group_sums)
        _include_adjusted_p_values(s_x2_result, p_adjust)
        result.update(s_x2_result)

    return result


def _include_adjusted_p_values(
    result: dict[str, NDArray[np.float64]],
    p_adjust: PValueAdjustment,
) -> None:
    """Add adjusted S-X2 p-values only when explicitly requested."""
    if p_adjust != "none":
        result["p_value_adjusted"] = adjust_p_values(result["p_value"], p_adjust)


def _validate_n_groups(n_groups: int) -> int:
    """Validate and normalize the number of score groups."""
    if isinstance(n_groups, (bool, np.bool_)) or not isinstance(
        n_groups, (int, np.integer)
    ):
        raise ValueError("n_groups must be an integer")
    if int(n_groups) < 2:
        raise ValueError("n_groups must be at least 2")
    return int(n_groups)


def _score_group_indices(
    responses: NDArray[np.int_],
    n_groups: int,
) -> NDArray[np.intp]:
    """Build population-wide score groups without a full-size missing mask."""
    n_persons, n_items = responses.shape
    sum_scores = np.empty(n_persons, dtype=np.sum(responses[:0], axis=1).dtype)
    rows_per_chunk = max(1, _SX2_TARGET_CHUNK_ELEMENTS // max(n_items, 1))
    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        block = responses[start:stop]
        sum_scores[start:stop] = np.sum(np.where(block >= 0, block, 0), axis=1)
    score_cuts = np.percentile(
        sum_scores,
        np.linspace(0.0, 100.0, n_groups + 1),
    )
    return np.searchsorted(
        score_cuts[1:-1],
        sum_scores,
        side="right",
    )


def _s_x2_from_group_sums(
    counts: NDArray[np.float64],
    observed_sums: NDArray[np.float64],
    expected_sums: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Apply count floors and compute S-X2 after all blocks have accumulated."""
    from scipy import special

    eligible = counts >= 5
    safe_counts = np.where(eligible, counts, 1.0)
    observed_means = observed_sums / safe_counts
    expected_means = np.clip(
        expected_sums / safe_counts,
        PROB_CLIP_MIN,
        PROB_CLIP_MAX,
    )
    components = (
        counts
        * (observed_means - expected_means) ** 2
        / (expected_means * (1.0 - expected_means))
    )
    s_x2 = np.sum(np.where(eligible, components, 0.0), axis=0)
    degrees = np.count_nonzero(eligible, axis=0)

    df = np.maximum(degrees - 1, 1).astype(np.float64)
    return {
        "S_X2": s_x2,
        "df": df,
        "p_value": special.chdtrc(df, s_x2),
    }


def _grouped_item_sums(
    responses: NDArray[np.int_],
    expected: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    group_indices: NDArray[np.int_],
    score_scales: NDArray[np.float64],
    n_groups: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Reduce all S-X2 score-group and item combinations in bounded chunks."""
    n_persons, n_items = responses.shape
    n_bins = n_groups * n_items
    counts = np.zeros(n_bins, dtype=np.float64)
    observed_sums = np.zeros(n_bins, dtype=np.float64)
    expected_sums = np.zeros(n_bins, dtype=np.float64)
    item_offsets = np.arange(n_items, dtype=np.intp)
    rows_per_chunk = max(
        1,
        min(
            n_persons,
            _SX2_TARGET_CHUNK_ELEMENTS // max(n_items, 1),
        ),
    )

    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        group_item_codes = (
            group_indices[start:stop, None] * n_items + item_offsets
        ).ravel()
        block_valid = valid_mask[start:stop]
        counts += np.bincount(
            group_item_codes,
            weights=block_valid.ravel(),
            minlength=n_bins,
        )
        observed_sums += np.bincount(
            group_item_codes,
            weights=np.where(
                block_valid,
                responses[start:stop] / score_scales,
                0.0,
            ).ravel(),
            minlength=n_bins,
        )
        expected_sums += np.bincount(
            group_item_codes,
            weights=np.where(
                block_valid,
                expected[start:stop] / score_scales,
                0.0,
            ).ravel(),
            minlength=n_bins,
        )

    shape = (n_groups, n_items)
    return (
        counts.reshape(shape),
        observed_sums.reshape(shape),
        expected_sums.reshape(shape),
    )


def compute_s_x2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_groups: int = 10,
    p_adjust: PValueAdjustment = "none",
) -> dict[str, NDArray[np.float64]]:
    """Compute grouped Orlando-Thissen S-X2 item-fit statistics.

    Set ``p_adjust`` to ``"bonferroni"``, ``"holm"``, or ``"fdr_bh"`` to
    include multiplicity-adjusted p-values across items under the
    ``"p_value_adjusted"`` key. The default preserves the original result
    shape.
    """
    return compute_itemfit(
        model,
        responses,
        statistics=["S_X2"],
        theta=theta,
        n_groups=n_groups,
        p_adjust=p_adjust,
    )
