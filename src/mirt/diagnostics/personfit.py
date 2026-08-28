from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.utils.numeric import compute_fit_stats, compute_probability_moments

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_ZH_TARGET_CHUNK_ELEMENTS = 2_000_000


def compute_personfit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    statistics: list[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    responses = np.asarray(responses)
    theta = np.asarray(theta)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    n_persons, n_items = responses.shape

    result: dict[str, NDArray[np.float64]] = {}
    compute_mean_squares = "outfit" in statistics or "infit" in statistics
    compute_zh = "Zh" in statistics or "lz" in statistics
    if not compute_mean_squares and not compute_zh:
        return result

    probabilities, expected, variance = compute_probability_moments(
        model,
        theta,
        n_items,
    )

    if compute_mean_squares:
        infit, outfit = compute_fit_stats(responses, expected, variance, axis=1)

        if "outfit" in statistics:
            result["outfit"] = outfit

        if "infit" in statistics:
            result["infit"] = infit

    if compute_zh:
        valid_mask = responses >= 0
        zh = _compute_zh_vectorized(model, responses, probabilities, valid_mask)

        if "Zh" in statistics:
            result["Zh"] = zh
        if "lz" in statistics:
            result["lz"] = zh

    return result


def _compute_zh_vectorized(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    probabilities: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_persons, n_items = responses.shape

    ll = np.zeros(n_persons)
    expected_ll = np.zeros(n_persons)
    var_ll = np.zeros(n_persons)

    category_width = probabilities.shape[2] if probabilities.ndim == 3 else 1
    chunk_size = max(
        1,
        min(
            n_items,
            _ZH_TARGET_CHUNK_ELEMENTS // max(n_persons * category_width, 1),
        ),
    )

    if probabilities.ndim == 2:
        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            probs = np.clip(
                probabilities[:, start:stop],
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            block_valid = valid_mask[:, start:stop]
            block_responses = responses[:, start:stop]
            log_p = np.log(probs)
            log_q = np.log(1 - probs)

            person_ll = np.where(block_responses == 1, log_p, log_q)
            ll += np.sum(np.where(block_valid, person_ll, 0.0), axis=1)

            item_expected_ll = probs * log_p + (1 - probs) * log_q
            expected_ll += np.sum(
                np.where(block_valid, item_expected_ll, 0.0),
                axis=1,
            )

            item_var_ll = probs * (1 - probs) * (log_p - log_q) ** 2
            var_ll += np.sum(
                np.where(block_valid, item_var_ll, 0.0),
                axis=1,
            )
    else:
        category_counts = np.asarray(model.n_categories, dtype=np.int64)
        category_indices = np.arange(probabilities.shape[2])

        for start in range(0, n_items, chunk_size):
            stop = min(start + chunk_size, n_items)
            block_valid = valid_mask[:, start:stop]
            block_responses = responses[:, start:stop]
            category_mask = category_indices < category_counts[start:stop, None]
            clipped_probs = np.clip(
                probabilities[:, start:stop, :],
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            probs = np.where(category_mask[None, :, :], clipped_probs, 0.0)
            log_probs = np.where(
                category_mask[None, :, :],
                np.log(clipped_probs),
                0.0,
            )

            safe_responses = np.where(block_valid, block_responses, 0)
            safe_responses = np.clip(
                safe_responses,
                0,
                category_counts[None, start:stop] - 1,
            )
            person_ll = np.take_along_axis(
                log_probs,
                safe_responses[..., None],
                axis=2,
            )[..., 0]
            ll += np.sum(np.where(block_valid, person_ll, 0.0), axis=1)

            item_expected_ll = np.sum(probs * log_probs, axis=2)
            expected_ll += np.sum(
                np.where(block_valid, item_expected_ll, 0.0),
                axis=1,
            )

            item_var_ll = np.sum(probs * log_probs**2, axis=2) - item_expected_ll**2
            var_ll += np.sum(
                np.where(block_valid, item_var_ll, 0.0),
                axis=1,
            )

    valid_count = valid_mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zh = np.where(
            (valid_count >= 2) & (var_ll > PROB_EPSILON),
            (ll - expected_ll) / np.sqrt(var_ll),
            np.nan,
        )

    return zh


def flag_aberrant_persons(
    fit_stats: dict[str, NDArray[np.float64]],
    criteria: dict[str, tuple[float, float]] | None = None,
) -> NDArray[np.bool_]:
    if criteria is None:
        criteria = {
            "infit": (0.5, 1.5),
            "outfit": (0.5, 1.5),
            "Zh": (-2.0, 2.0),
            "lz": (-2.0, 2.0),
        }

    if not fit_stats:
        raise ValueError("fit_stats must contain at least one statistic")

    first_stat = next(iter(fit_stats.values()))
    n_persons = len(first_stat)

    flags = np.zeros(n_persons, dtype=bool)

    for stat_name, (lower, upper) in criteria.items():
        if stat_name in fit_stats:
            values = fit_stats[stat_name]
            flags |= (values < lower) | (values > upper)

    return flags
