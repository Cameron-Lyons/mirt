"""Item fit statistics for IRT models."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def compute_itemfit(
    model: "BaseItemModel",
    responses: Optional[NDArray[np.int_]] = None,
    statistics: Optional[list[str]] = None,
    theta: Optional[NDArray[np.float64]] = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute item fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    responses : ndarray of shape (n_persons, n_items), optional
        Response matrix. Required for residual-based statistics.
    statistics : list of str, optional
        Statistics to compute. Options: 'infit', 'outfit'.
        Default: ['infit', 'outfit'].
    theta : ndarray, optional
        Person ability values. If None and responses provided,
        estimated via EAP.

    Returns
    -------
    dict
        Dictionary mapping statistic names to arrays of values.
    """
    if statistics is None:
        statistics = ["infit", "outfit"]

    if responses is None:
        raise ValueError("responses required for item fit statistics")

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    # Estimate theta if not provided
    if theta is None:
        from mirt.scoring import fscores
        from mirt.results.fit_result import FitResult

        # Create minimal FitResult for scoring
        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    result: dict[str, NDArray[np.float64]] = {}

    # Compute expected values and variances for each person-item
    expected = np.zeros((n_persons, n_items))
    variance = np.zeros((n_persons, n_items))

    for i in range(n_items):
        probs = model.probability(theta, i)
        if probs.ndim == 1:
            # Dichotomous
            expected[:, i] = probs
            variance[:, i] = probs * (1 - probs)
        else:
            # Polytomous: expected = sum(k * P(k))
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

    # Compute statistics for each item
    if "outfit" in statistics:
        # Outfit (unweighted mean square): mean of squared std residuals
        outfit = np.zeros(n_items)
        for i in range(n_items):
            valid = valid_mask[:, i]
            if valid.sum() > 0:
                outfit[i] = np.mean(std_residuals[valid, i] ** 2)
            else:
                outfit[i] = np.nan
        result["outfit"] = outfit

    if "infit" in statistics:
        # Infit (weighted mean square): variance-weighted
        infit = np.zeros(n_items)
        for i in range(n_items):
            valid = valid_mask[:, i]
            if valid.sum() > 0:
                numerator = np.sum((residuals[valid, i] ** 2))
                denominator = np.sum(variance[valid, i])
                if denominator > 1e-10:
                    infit[i] = numerator / denominator
                else:
                    infit[i] = np.nan
            else:
                infit[i] = np.nan
        result["infit"] = infit

    return result


def compute_s_x2(
    model: "BaseItemModel",
    responses: NDArray[np.int_],
    theta: Optional[NDArray[np.float64]] = None,
    n_groups: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Compute S-X² item fit statistic (Orlando & Thissen, 2000).

    Groups examinees by score level and compares observed vs expected
    proportions correct.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    responses : ndarray
        Response matrix.
    theta : ndarray, optional
        Person abilities.
    n_groups : int, default=10
        Number of score groups.

    Returns
    -------
    dict
        S_X2 values and p-values for each item.
    """
    from scipy import stats

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores
        score_result = fscores(model, responses, method="EAP")
        theta = score_result.theta

    theta = np.asarray(theta).ravel()

    # Compute sum scores for grouping
    valid_mask = responses >= 0
    sum_scores = np.sum(np.where(valid_mask, responses, 0), axis=1)

    # Create score groups
    percentiles = np.linspace(0, 100, n_groups + 1)
    score_cuts = np.percentile(sum_scores, percentiles)

    s_x2 = np.zeros(n_items)
    df = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for i in range(n_items):
        chi2 = 0.0
        degrees = 0

        for g in range(n_groups):
            # Find persons in this score group
            if g < n_groups - 1:
                in_group = (sum_scores >= score_cuts[g]) & (sum_scores < score_cuts[g + 1])
            else:
                in_group = sum_scores >= score_cuts[g]

            valid_in_group = in_group & valid_mask[:, i]
            n_g = valid_in_group.sum()

            if n_g < 5:
                continue

            # Observed proportion correct
            observed = responses[valid_in_group, i].mean()

            # Expected proportion (mean probability at group theta)
            group_theta = theta[valid_in_group]
            probs = model.probability(group_theta.reshape(-1, 1), i)
            if probs.ndim == 1:
                expected = probs.mean()
            else:
                # Polytomous: use expected score / (max score)
                n_cat = probs.shape[1]
                exp_score = np.sum(probs * np.arange(n_cat), axis=1).mean()
                expected = exp_score / (n_cat - 1)

            # Avoid extreme probabilities
            expected = np.clip(expected, 0.01, 0.99)

            # Chi-square contribution
            if expected > 0 and expected < 1:
                chi2 += n_g * (observed - expected) ** 2 / (expected * (1 - expected))
                degrees += 1

        s_x2[i] = chi2
        df[i] = max(degrees - 1, 1)  # Subtract 1 for item parameters
        p_values[i] = 1 - stats.chi2.cdf(chi2, df[i])

    return {
        "S_X2": s_x2,
        "df": df,
        "p_value": p_values,
    }
