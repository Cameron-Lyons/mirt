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
) -> dict[str, NDArray[np.float64]]:
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

    result: dict[str, NDArray[np.float64]] = {}

    expected, variance = compute_expected_variance(model, theta, n_items)
    infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)

    if "outfit" in statistics:
        result["outfit"] = outfit

    if "infit" in statistics:
        result["infit"] = infit

    return result


def compute_s_x2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_groups: int = 10,
) -> dict[str, NDArray[np.float64]]:
    from scipy import stats

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta = np.asarray(theta).ravel()

    valid_mask = responses >= 0
    sum_scores = np.sum(np.where(valid_mask, responses, 0), axis=1)

    percentiles = np.linspace(0, 100, n_groups + 1)
    score_cuts = np.percentile(sum_scores, percentiles)

    s_x2 = np.zeros(n_items)
    df = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for i in range(n_items):
        chi2 = 0.0
        degrees = 0

        for g in range(n_groups):
            if g < n_groups - 1:
                in_group = (sum_scores >= score_cuts[g]) & (
                    sum_scores < score_cuts[g + 1]
                )
            else:
                in_group = sum_scores >= score_cuts[g]

            valid_in_group = in_group & valid_mask[:, i]
            n_g = valid_in_group.sum()

            if n_g < 5:
                continue

            observed = responses[valid_in_group, i].mean()

            group_theta = theta[valid_in_group]
            probs = model.probability(group_theta.reshape(-1, 1), i)
            if probs.ndim == 1:
                expected = probs.mean()
            else:
                n_cat = probs.shape[1]
                exp_score = np.sum(probs * np.arange(n_cat), axis=1).mean()
                expected = exp_score / (n_cat - 1)

            expected = np.clip(expected, PROB_CLIP_MIN, PROB_CLIP_MAX)

            if expected > 0 and expected < 1:
                chi2 += n_g * (observed - expected) ** 2 / (expected * (1 - expected))
                degrees += 1

        s_x2[i] = chi2
        df[i] = max(degrees - 1, 1)
        p_values[i] = 1 - stats.chi2.cdf(chi2, df[i])

    return {
        "S_X2": s_x2,
        "df": df,
        "p_value": p_values,
    }
