"""Person fit statistics for IRT models."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def compute_personfit(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    statistics: Optional[list[str]] = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute person fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors)
        Person ability values.
    statistics : list of str, optional
        Statistics to compute. Options: 'infit', 'outfit', 'Zh', 'lz'.
        Default: ['infit', 'outfit', 'Zh'].

    Returns
    -------
    dict
        Dictionary mapping statistic names to arrays of values.
    """
    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    responses = np.asarray(responses)
    theta = np.asarray(theta)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    n_persons, n_items = responses.shape

    # Compute expected values and variances
    expected = np.zeros((n_persons, n_items))
    variance = np.zeros((n_persons, n_items))

    for i in range(n_items):
        probs = model.probability(theta, i)
        if probs.ndim == 1:
            expected[:, i] = probs
            variance[:, i] = probs * (1 - probs)
        else:
            n_cat = probs.shape[1]
            categories = np.arange(n_cat)
            expected[:, i] = np.sum(probs * categories, axis=1)
            expected_sq = np.sum(probs * (categories ** 2), axis=1)
            variance[:, i] = expected_sq - expected[:, i] ** 2

    # Compute residuals
    valid_mask = responses >= 0
    residuals = np.where(valid_mask, responses - expected, 0.0)
    std_residuals = np.where(
        valid_mask & (variance > 1e-10),
        residuals / np.sqrt(variance + 1e-10),
        0.0,
    )

    result: dict[str, NDArray[np.float64]] = {}

    if "outfit" in statistics:
        # Outfit: unweighted mean square
        outfit = np.zeros(n_persons)
        for p in range(n_persons):
            valid = valid_mask[p, :]
            if valid.sum() > 0:
                outfit[p] = np.mean(std_residuals[p, valid] ** 2)
            else:
                outfit[p] = np.nan
        result["outfit"] = outfit

    if "infit" in statistics:
        # Infit: weighted mean square
        infit = np.zeros(n_persons)
        for p in range(n_persons):
            valid = valid_mask[p, :]
            if valid.sum() > 0:
                numerator = np.sum(residuals[p, valid] ** 2)
                denominator = np.sum(variance[p, valid])
                if denominator > 1e-10:
                    infit[p] = numerator / denominator
                else:
                    infit[p] = np.nan
            else:
                infit[p] = np.nan
        result["infit"] = infit

    if "Zh" in statistics or "lz" in statistics:
        # Zh (Drasgow et al.) / lz (person fit)
        # Standardized log-likelihood
        zh = np.zeros(n_persons)

        for p in range(n_persons):
            valid = valid_mask[p, :]
            if valid.sum() < 2:
                zh[p] = np.nan
                continue

            # Log-likelihood for this person
            ll = 0.0
            expected_ll = 0.0
            var_ll = 0.0

            for i in range(n_items):
                if not valid[i]:
                    continue

                resp = responses[p, i]
                probs = model.probability(theta[p:p+1], i)

                if probs.ndim == 1:
                    # Dichotomous
                    prob = probs[0]
                    prob = np.clip(prob, 1e-10, 1 - 1e-10)

                    if resp == 1:
                        ll += np.log(prob)
                    else:
                        ll += np.log(1 - prob)

                    # Expected log-likelihood
                    expected_ll += prob * np.log(prob) + (1 - prob) * np.log(1 - prob)

                    # Variance of log-likelihood
                    log_p = np.log(prob)
                    log_q = np.log(1 - prob)
                    var_ll += prob * (1 - prob) * (log_p - log_q) ** 2

                else:
                    # Polytomous
                    prob = probs[0, resp]
                    prob = max(prob, 1e-10)
                    ll += np.log(prob)

                    # Expected and variance for polytomous
                    probs_item = probs[0]
                    probs_item = np.clip(probs_item, 1e-10, 1 - 1e-10)
                    log_probs = np.log(probs_item)

                    expected_ll += np.sum(probs_item * log_probs)
                    var_ll += np.sum(probs_item * log_probs ** 2) - \
                              np.sum(probs_item * log_probs) ** 2

            # Standardize
            if var_ll > 1e-10:
                zh[p] = (ll - expected_ll) / np.sqrt(var_ll)
            else:
                zh[p] = np.nan

        if "Zh" in statistics:
            result["Zh"] = zh
        if "lz" in statistics:
            result["lz"] = zh  # Same computation

    return result


def flag_aberrant_persons(
    fit_stats: dict[str, NDArray[np.float64]],
    criteria: Optional[dict[str, tuple[float, float]]] = None,
) -> NDArray[np.bool_]:
    """Identify persons with aberrant response patterns.

    Parameters
    ----------
    fit_stats : dict
        Person fit statistics from compute_personfit.
    criteria : dict, optional
        Flagging criteria as {statistic: (lower, upper)}.
        Default: infit/outfit in (0.5, 1.5), Zh in (-2, 2).

    Returns
    -------
    ndarray of bool
        True for persons flagged as aberrant.
    """
    if criteria is None:
        criteria = {
            "infit": (0.5, 1.5),
            "outfit": (0.5, 1.5),
            "Zh": (-2.0, 2.0),
        }

    # Get number of persons from first statistic
    first_stat = next(iter(fit_stats.values()))
    n_persons = len(first_stat)

    flags = np.zeros(n_persons, dtype=bool)

    for stat_name, (lower, upper) in criteria.items():
        if stat_name in fit_stats:
            values = fit_stats[stat_name]
            flags |= (values < lower) | (values > upper)

    return flags
