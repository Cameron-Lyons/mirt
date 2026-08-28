from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN
from mirt.utils.numeric import compute_expected_variance, compute_fit_stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def compute_itemfit(
    model: BaseItemModel,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
    theta: NDArray[np.float64] | None = None,
    n_groups: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Compute requested item-fit statistics.

    S-X2 results include the statistic, degrees of freedom, and p-value under
    the keys ``"S_X2"``, ``"df"``, and ``"p_value"``.
    """
    if statistics is None:
        statistics = ["infit", "outfit"]

    if responses is None:
        raise ValueError("responses required for item fit statistics")

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    expected, variance = compute_expected_variance(model, theta, n_items)
    result: dict[str, NDArray[np.float64]] = {}

    if "outfit" in statistics or "infit" in statistics:
        infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)

        if "outfit" in statistics:
            result["outfit"] = outfit

        if "infit" in statistics:
            result["infit"] = infit

    if "S_X2" in statistics:
        result.update(
            _compute_s_x2_from_expected(
                model,
                responses,
                expected,
                n_groups=n_groups,
            )
        )

    return result


def _validate_n_groups(n_groups: int) -> int:
    """Validate and normalize the number of score groups."""
    if isinstance(n_groups, (bool, np.bool_)) or not isinstance(
        n_groups, (int, np.integer)
    ):
        raise ValueError("n_groups must be an integer")
    if int(n_groups) < 2:
        raise ValueError("n_groups must be at least 2")
    return int(n_groups)


def _compute_s_x2_from_expected(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    expected: NDArray[np.float64],
    *,
    n_groups: int,
) -> dict[str, NDArray[np.float64]]:
    """Aggregate observed and expected scores into S-X2 statistics."""
    from scipy import special

    n_groups = _validate_n_groups(n_groups)
    n_persons, n_items = responses.shape
    if n_persons == 0:
        raise ValueError("responses must contain at least one person")

    valid_mask = responses >= 0
    sum_scores = np.sum(np.where(valid_mask, responses, 0), axis=1)
    score_cuts = np.percentile(
        sum_scores,
        np.linspace(0.0, 100.0, n_groups + 1),
    )
    group_indices = np.searchsorted(
        score_cuts[1:-1],
        sum_scores,
        side="right",
    )

    if model.is_polytomous:
        score_scales = np.asarray(model.n_categories, dtype=np.float64) - 1.0
    else:
        score_scales = np.ones(n_items, dtype=np.float64)

    observed_scaled = responses / score_scales
    expected_scaled = expected / score_scales
    s_x2 = np.zeros(n_items, dtype=np.float64)
    degrees = np.zeros(n_items, dtype=np.int64)

    for item_idx in range(n_items):
        item_valid = valid_mask[:, item_idx]
        item_groups = group_indices[item_valid]
        counts = np.bincount(item_groups, minlength=n_groups)
        eligible = counts >= 5
        if not np.any(eligible):
            continue

        observed_sums = np.bincount(
            item_groups,
            weights=observed_scaled[item_valid, item_idx],
            minlength=n_groups,
        )
        expected_sums = np.bincount(
            item_groups,
            weights=expected_scaled[item_valid, item_idx],
            minlength=n_groups,
        )
        observed_means = observed_sums[eligible] / counts[eligible]
        expected_means = np.clip(
            expected_sums[eligible] / counts[eligible],
            PROB_CLIP_MIN,
            PROB_CLIP_MAX,
        )
        s_x2[item_idx] = np.sum(
            counts[eligible]
            * (observed_means - expected_means) ** 2
            / (expected_means * (1.0 - expected_means))
        )
        degrees[item_idx] = np.count_nonzero(eligible)

    df = np.maximum(degrees - 1, 1).astype(np.float64)
    return {
        "S_X2": s_x2,
        "df": df,
        "p_value": special.chdtrc(df, s_x2),
    }


def compute_s_x2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_groups: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Compute grouped Orlando-Thissen S-X2 item-fit statistics."""
    n_groups = _validate_n_groups(n_groups)
    responses = np.asarray(responses)
    _, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta_array = np.asarray(theta)
    if theta_array.ndim == 1:
        theta_array = theta_array.reshape(-1, 1)

    expected, _ = compute_expected_variance(model, theta_array, n_items)
    return _compute_s_x2_from_expected(
        model,
        responses,
        expected,
        n_groups=n_groups,
    )
