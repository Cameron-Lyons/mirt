"""Empirical analysis functions for IRT models.

Provides functions for computing DIF effect sizes and generating
data for observed vs expected plots.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class DIFEffectSize:
    """Container for DIF effect size statistics.

    Attributes
    ----------
    item_idx : int
        Item index.
    signed_es : float
        Signed effect size (positive = favors focal group).
    unsigned_es : float
        Unsigned (absolute) effect size.
    sids : float
        Signed Item Difference in the Sample.
    uids : float
        Unsigned Item Difference in the Sample.
    classification : str
        ETS classification ("A", "B", or "C").
    """

    item_idx: int
    signed_es: float
    unsigned_es: float
    sids: float
    uids: float
    classification: str


@dataclass
class EmpiricalPlotData:
    """Container for empirical plot data.

    Attributes
    ----------
    item_idx : int
        Item index.
    theta_bins : NDArray[np.float64]
        Theta bin midpoints.
    observed_prop : NDArray[np.float64]
        Observed proportions correct in each bin.
    expected_prop : NDArray[np.float64]
        Model-predicted proportions.
    n_per_bin : NDArray[np.intp]
        Number of observations in each bin.
    residuals : NDArray[np.float64]
        Observed - expected differences.
    """

    item_idx: int
    theta_bins: NDArray[np.float64]
    observed_prop: NDArray[np.float64]
    expected_prop: NDArray[np.float64]
    n_per_bin: NDArray[np.intp]
    residuals: NDArray[np.float64]


def empirical_ES(
    model_ref: "BaseItemModel",
    model_focal: "BaseItemModel",
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
    focal_weight: float = 0.5,
) -> DIFEffectSize:
    """Compute empirical effect size for DIF.

    Computes effect sizes comparing item response functions between
    reference and focal groups.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.
    focal_weight : float
        Weight for focal group in combined distribution.
        Default 0.5 (equal weighting).

    Returns
    -------
    DIFEffectSize
        Container with effect size statistics.

    Examples
    --------
    >>> model_ref = fit_mirt(responses_ref, model="2PL").model
    >>> model_focal = fit_mirt(responses_focal, model="2PL").model
    >>> es = empirical_ES(model_ref, model_focal, item_idx=0)
    >>> print(f"Signed ES: {es.signed_es:.3f}")
    >>> print(f"ETS Classification: {es.classification}")
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta.reshape(-1, 1)

    from scipy import stats

    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    prob_ref = model_ref.probability(theta_2d, item_idx=item_idx)
    prob_focal = model_focal.probability(theta_2d, item_idx=item_idx)

    if prob_ref.ndim > 1:
        prob_ref = prob_ref[:, 0] if prob_ref.shape[1] == 1 else prob_ref.ravel()
    if prob_focal.ndim > 1:
        prob_focal = (
            prob_focal[:, 0] if prob_focal.shape[1] == 1 else prob_focal.ravel()
        )

    diff = prob_focal - prob_ref

    sids = np.sum(weights * diff)
    uids = np.sum(weights * np.abs(diff))

    signed_es = sids
    unsigned_es = uids

    if unsigned_es < 0.05:
        classification = "A"
    elif unsigned_es < 0.10:
        classification = "B"
    else:
        classification = "C"

    return DIFEffectSize(
        item_idx=item_idx,
        signed_es=float(signed_es),
        unsigned_es=float(unsigned_es),
        sids=float(sids),
        uids=float(uids),
        classification=classification,
    )


def empirical_plot(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    item_idx: int,
    n_bins: int = 10,
) -> EmpiricalPlotData:
    """Compute data for observed vs expected empirical plot.

    Groups examinees by theta estimate and computes observed vs
    model-predicted proportions for model-data fit assessment.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, 1).
    item_idx : int
        Index of item to plot.
    n_bins : int
        Number of theta bins. Default 10.

    Returns
    -------
    EmpiricalPlotData
        Container with plot data.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> plot_data = empirical_plot(result.model, responses, result.theta, item_idx=0)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(plot_data.theta_bins, plot_data.observed_prop)
    >>> plt.plot(plot_data.theta_bins, plot_data.expected_prop)
    """
    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_1d(theta).ravel()

    item_responses = responses[:, item_idx]
    valid_mask = ~np.isnan(item_responses)
    item_responses = item_responses[valid_mask]
    theta_valid = theta[valid_mask]

    n_valid = len(theta_valid)
    if n_valid == 0:
        return EmpiricalPlotData(
            item_idx=item_idx,
            theta_bins=np.array([]),
            observed_prop=np.array([]),
            expected_prop=np.array([]),
            n_per_bin=np.array([], dtype=np.intp),
            residuals=np.array([]),
        )

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(theta_valid, percentiles)
    bin_edges[-1] += 1e-10

    bin_indices = np.digitize(theta_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    theta_bins = np.zeros(n_bins)
    observed_prop = np.zeros(n_bins)
    expected_prop = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins, dtype=np.intp)

    for b in range(n_bins):
        mask = bin_indices == b
        n_in_bin = np.sum(mask)
        n_per_bin[b] = n_in_bin

        if n_in_bin > 0:
            theta_bins[b] = np.mean(theta_valid[mask])
            observed_prop[b] = np.mean(item_responses[mask])

            theta_bin_2d = theta_valid[mask].reshape(-1, 1)
            probs = model.probability(theta_bin_2d, item_idx=item_idx)
            if probs.ndim > 1:
                probs = probs[:, 0] if probs.shape[1] == 1 else probs.ravel()
            expected_prop[b] = np.mean(probs)
        else:
            theta_bins[b] = (bin_edges[b] + bin_edges[b + 1]) / 2

    residuals = observed_prop - expected_prop

    return EmpiricalPlotData(
        item_idx=item_idx,
        theta_bins=theta_bins,
        observed_prop=observed_prop,
        expected_prop=expected_prop,
        n_per_bin=n_per_bin,
        residuals=residuals,
    )


def empirical_rmsea(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_bins: int = 10,
) -> NDArray[np.float64]:
    """Compute RMSEA-like fit statistic per item.

    Measures root mean square error of approximation between
    observed and expected proportions across ability bins.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.
    n_bins : int
        Number of theta bins. Default 10.

    Returns
    -------
    NDArray[np.float64]
        RMSEA values for each item.
    """
    n_items = responses.shape[1]
    rmsea = np.zeros(n_items)

    for j in range(n_items):
        plot_data = empirical_plot(model, responses, theta, j, n_bins)

        valid_bins = plot_data.n_per_bin > 0
        if np.sum(valid_bins) > 0:
            residuals_sq = plot_data.residuals[valid_bins] ** 2
            rmsea[j] = np.sqrt(np.mean(residuals_sq))
        else:
            rmsea[j] = np.nan

    return rmsea


def mantel_haenszel(
    responses: NDArray[np.float64],
    group: NDArray[np.intp],
    theta: NDArray[np.float64],
    item_idx: int,
    n_strata: int = 5,
) -> tuple[float, float, float]:
    """Compute Mantel-Haenszel DIF statistic.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix.
    group : NDArray[np.intp]
        Group membership (0 = reference, 1 = focal).
    theta : NDArray[np.float64]
        Matching variable (e.g., total score or theta estimate).
    item_idx : int
        Index of item to test.
    n_strata : int
        Number of matching strata. Default 5.

    Returns
    -------
    mh_chisq : float
        Mantel-Haenszel chi-square statistic.
    p_value : float
        P-value.
    mh_odds : float
        Mantel-Haenszel odds ratio.
    """
    from scipy import stats

    responses = np.asarray(responses, dtype=np.float64)
    group = np.asarray(group, dtype=np.intp)
    theta = np.atleast_1d(theta).ravel()

    item_resp = responses[:, item_idx]
    valid = ~np.isnan(item_resp)
    item_resp = item_resp[valid]
    group_valid = group[valid]
    theta_valid = theta[valid]

    percentiles = np.linspace(0, 100, n_strata + 1)
    bins = np.percentile(theta_valid, percentiles)
    bins[-1] += 1e-10
    stratum = np.digitize(theta_valid, bins) - 1
    stratum = np.clip(stratum, 0, n_strata - 1)

    numerator = 0.0
    denominator = 0.0
    variance = 0.0

    for s in range(n_strata):
        mask = stratum == s

        ref_mask = mask & (group_valid == 0)
        focal_mask = mask & (group_valid == 1)

        n_ref = np.sum(ref_mask)
        n_focal = np.sum(focal_mask)
        n_total = n_ref + n_focal

        if n_ref < 1 or n_focal < 1:
            continue

        a = np.sum(item_resp[ref_mask])
        b = n_ref - a
        c = np.sum(item_resp[focal_mask])
        d = n_focal - c

        n1 = a + b
        n0 = c + d
        m1 = a + c
        m0 = b + d

        if n_total > 0:
            e_a = n1 * m1 / n_total
            numerator += a - e_a

            if n_total > 1:
                v_a = n1 * n0 * m1 * m0 / (n_total**2 * (n_total - 1))
                variance += v_a

            denominator += (a * d) / n_total

    if variance > 0:
        mh_chisq = (abs(numerator) - 0.5) ** 2 / variance
        p_value = 1 - stats.chi2.cdf(mh_chisq, 1)
    else:
        mh_chisq = 0.0
        p_value = 1.0

    if denominator > 0:
        bd_sum = sum(
            (n_ref - np.sum(item_resp[ref_mask]))
            * np.sum(item_resp[focal_mask])
            / n_total
            for s in range(n_strata)
            if np.sum((stratum == s) & (group_valid == 0)) > 0
            and np.sum((stratum == s) & (group_valid == 1)) > 0
            for ref_mask in [(stratum == s) & (group_valid == 0)]
            for focal_mask in [(stratum == s) & (group_valid == 1)]
            for n_ref in [np.sum(ref_mask)]
            for n_focal in [np.sum(focal_mask)]
            for n_total in [n_ref + n_focal]
        )
        if bd_sum > 0:
            mh_odds = denominator / bd_sum
        else:
            mh_odds = 1.0
    else:
        mh_odds = 1.0

    return float(mh_chisq), float(p_value), float(mh_odds)


def RMSD_DIF(
    model_ref: "BaseItemModel",
    model_focal: "BaseItemModel",
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
) -> float:
    """Compute RMSD-based DIF statistic.

    The Root Mean Square Difference compares item response functions
    between reference and focal groups.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.

    Returns
    -------
    float
        RMSD value. Larger values indicate more DIF.

    Examples
    --------
    >>> model_ref = fit_mirt(responses_ref, model="2PL").model
    >>> model_focal = fit_mirt(responses_focal, model="2PL").model
    >>> rmsd = RMSD_DIF(model_ref, model_focal, item_idx=0)
    >>> print(f"RMSD DIF: {rmsd:.4f}")

    Notes
    -----
    Guidelines for interpretation (Meade, 2010):
    - RMSD < 0.05: Negligible DIF
    - 0.05 <= RMSD < 0.10: Slight DIF
    - RMSD >= 0.10: Notable DIF
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta.reshape(-1, 1)

    prob_ref = model_ref.probability(theta_2d, item_idx=item_idx)
    prob_focal = model_focal.probability(theta_2d, item_idx=item_idx)

    if prob_ref.ndim > 1:
        prob_ref = prob_ref[:, 0] if prob_ref.shape[1] == 1 else prob_ref.ravel()
    if prob_focal.ndim > 1:
        prob_focal = (
            prob_focal[:, 0] if prob_focal.shape[1] == 1 else prob_focal.ravel()
        )

    squared_diff = (prob_ref - prob_focal) ** 2
    rmsd = np.sqrt(np.mean(squared_diff))

    return float(rmsd)


def weighted_RMSD_DIF(
    model_ref: "BaseItemModel",
    model_focal: "BaseItemModel",
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
) -> float:
    """Compute weighted RMSD-based DIF statistic.

    Weights the squared differences by the standard normal density,
    giving more weight to typical ability levels.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.

    Returns
    -------
    float
        Weighted RMSD value.
    """
    from scipy import stats

    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta.reshape(-1, 1)

    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    prob_ref = model_ref.probability(theta_2d, item_idx=item_idx)
    prob_focal = model_focal.probability(theta_2d, item_idx=item_idx)

    if prob_ref.ndim > 1:
        prob_ref = prob_ref[:, 0] if prob_ref.shape[1] == 1 else prob_ref.ravel()
    if prob_focal.ndim > 1:
        prob_focal = (
            prob_focal[:, 0] if prob_focal.shape[1] == 1 else prob_focal.ravel()
        )

    squared_diff = (prob_ref - prob_focal) ** 2
    weighted_rmsd = np.sqrt(np.sum(weights * squared_diff))

    return float(weighted_rmsd)
