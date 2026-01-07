from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.utils.numeric import compute_expected_variance, compute_fit_stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


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

    expected, variance = compute_expected_variance(model, theta, n_items)
    infit, outfit = compute_fit_stats(responses, expected, variance, axis=1)

    result: dict[str, NDArray[np.float64]] = {}

    if "outfit" in statistics:
        result["outfit"] = outfit

    if "infit" in statistics:
        result["infit"] = infit

    if "Zh" in statistics or "lz" in statistics:
        valid_mask = responses >= 0
        zh = _compute_zh_vectorized(model, responses, theta, valid_mask)

        if "Zh" in statistics:
            result["Zh"] = zh
        if "lz" in statistics:
            result["lz"] = zh

    return result


def _compute_zh_vectorized(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_persons, n_items = responses.shape

    ll = np.zeros(n_persons)
    expected_ll = np.zeros(n_persons)
    var_ll = np.zeros(n_persons)

    for i in range(n_items):
        probs = model.probability(theta, i)

        if probs.ndim == 1:
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            item_valid = valid_mask[:, i]
            resp = responses[:, i]

            log_p = np.log(probs)
            log_q = np.log(1 - probs)

            person_ll = np.where(resp == 1, log_p, log_q)
            ll += np.where(item_valid, person_ll, 0.0)

            item_expected_ll = probs * log_p + (1 - probs) * log_q
            expected_ll += np.where(item_valid, item_expected_ll, 0.0)

            item_var_ll = probs * (1 - probs) * (log_p - log_q) ** 2
            var_ll += np.where(item_valid, item_var_ll, 0.0)

        else:
            n_cat = probs.shape[1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            log_probs = np.log(probs)
            item_valid = valid_mask[:, i]
            resp = responses[:, i]

            resp_safe = np.where(item_valid, resp, 0)
            resp_safe = np.clip(resp_safe, 0, n_cat - 1)
            person_ll = log_probs[np.arange(n_persons), resp_safe]
            ll += np.where(item_valid, person_ll, 0.0)

            item_expected_ll = np.sum(probs * log_probs, axis=1)
            expected_ll += np.where(item_valid, item_expected_ll, 0.0)

            item_var_ll = np.sum(probs * log_probs**2, axis=1) - item_expected_ll**2
            var_ll += np.where(item_valid, item_var_ll, 0.0)

    valid_count = valid_mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zh = np.where(
            (valid_count >= 2) & (var_ll > 1e-10),
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
        }

    first_stat = next(iter(fit_stats.values()))
    n_persons = len(first_stat)

    flags = np.zeros(n_persons, dtype=bool)

    for stat_name, (lower, upper) in criteria.items():
        if stat_name in fit_stats:
            values = fit_stats[stat_name]
            flags |= (values < lower) | (values > upper)

    return flags
