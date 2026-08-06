"""Empirical analysis functions for IRT models.

Provides functions for computing DIF effect sizes and generating
data for observed vs expected plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

SILVERMAN_CONSTANT = 1.06
SILVERMAN_EXPONENT = -1 / 5
KERNEL_BLOCK_ELEMENTS = 2_000_000


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
        Observed mean item scores in each bin.
    expected_prop : NDArray[np.float64]
        Model-predicted mean item scores in each bin.
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


def _validate_positive_integer(value: int, name: str, minimum: int = 1) -> int:
    """Return a validated integer control parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_empirical_inputs(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the shared unidimensional empirical-diagnostic inputs."""
    if model.n_factors != 1:
        raise ValueError("empirical diagnostics require a unidimensional model")

    response_values = np.asarray(responses, dtype=np.float64)
    if response_values.ndim != 2:
        raise ValueError("responses must be a 2D matrix")
    if response_values.shape[0] == 0:
        raise ValueError("responses must contain at least one person")
    if response_values.shape[1] != model.n_items:
        raise ValueError(
            f"responses must contain {model.n_items} items, "
            f"got {response_values.shape[1]}"
        )
    if np.any(np.isinf(response_values)):
        raise ValueError("responses must not contain infinite values")

    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim == 1:
        theta_values = theta_values.reshape(-1, 1)
    if theta_values.ndim != 2 or theta_values.shape[1] != 1:
        raise ValueError("theta must have shape (n_persons,) or (n_persons, 1)")
    if theta_values.shape[0] != response_values.shape[0]:
        raise ValueError("theta and responses must contain the same number of persons")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    return response_values, theta_values


def _validate_item_index(model: BaseItemModel, item_idx: int) -> int:
    """Return a validated zero-based item index."""
    if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
        item_idx, (int, np.integer)
    ):
        raise ValueError("item_idx must be an integer")
    result = int(item_idx)
    if result < 0 or result >= model.n_items:
        raise ValueError(f"item_idx must be between 0 and {model.n_items - 1}")
    return result


def _validate_probabilities(probabilities: NDArray[np.float64]) -> None:
    """Reject malformed model probability output early."""
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    tolerance = 1e-8
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
        raise ValueError("model probabilities must be between 0 and 1")


def _expected_item_score(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> tuple[NDArray[np.float64], int]:
    """Return the item expected score and its maximum category score."""
    probabilities = np.asarray(
        model.probability(theta, item_idx=item_idx), dtype=np.float64
    )
    n_persons = theta.shape[0]

    if probabilities.ndim == 1 and probabilities.shape[0] == n_persons:
        _validate_probabilities(probabilities)
        return probabilities, 1

    if probabilities.ndim != 2 or probabilities.shape[0] != n_persons:
        raise ValueError(
            "item probability output must have shape (n_persons,) or "
            "(n_persons, n_categories)"
        )
    if probabilities.shape[1] == 0:
        raise ValueError("item probability output must contain a category")

    _validate_probabilities(probabilities)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0], 1
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("polytomous category probabilities must sum to 1")

    categories = np.arange(probabilities.shape[1], dtype=np.float64)
    return probabilities @ categories, probabilities.shape[1] - 1


def _expected_all_item_scores(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Evaluate expected scores for every item with one public model call."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    n_persons = theta.shape[0]

    if probabilities.ndim == 1 and model.n_items == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.ndim == 2 and probabilities.shape == (
        n_persons,
        model.n_items,
    ):
        _validate_probabilities(probabilities)
        return probabilities, np.ones(model.n_items, dtype=np.intp)

    if (
        probabilities.ndim != 3
        or probabilities.shape[0] != n_persons
        or probabilities.shape[1] != model.n_items
        or probabilities.shape[2] < 2
    ):
        raise ValueError(
            "model probability output must have shape (n_persons, n_items) "
            "or (n_persons, n_items, n_categories)"
        )

    _validate_probabilities(probabilities)
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("polytomous category probabilities must sum to 1")

    categories = np.arange(probabilities.shape[2], dtype=np.float64)
    expected_scores = probabilities @ categories

    category_counts = getattr(model, "n_categories", None)
    if category_counts is not None:
        counts = np.asarray(category_counts, dtype=np.intp)
        if counts.shape != (model.n_items,) or np.any(counts < 2):
            raise ValueError("model category counts are malformed")
        max_scores = counts - 1
    else:
        max_scores = np.full(model.n_items, probabilities.shape[2] - 1, dtype=np.intp)
    return expected_scores, max_scores


def _validate_observed_scores(
    values: NDArray[np.float64],
    max_score: int,
    item_idx: int,
) -> None:
    """Validate observed, non-missing category scores for one item."""
    observed = values[np.isfinite(values) & (values >= 0)]
    if observed.size == 0:
        return
    if np.any(observed != np.floor(observed)):
        raise ValueError(
            f"responses for item {item_idx} must be integer category scores"
        )
    if np.any(observed > max_score):
        raise ValueError(
            f"responses for item {item_idx} must be between 0 and {max_score}"
        )


def _build_theta_bins(
    theta: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Assign persons to shared theta quantile bins."""
    percentiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.percentile(theta, percentiles)
    bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)

    bin_indices = np.clip(np.digitize(theta, bin_edges) - 1, 0, n_bins - 1).astype(
        np.intp
    )
    n_per_bin = np.bincount(bin_indices, minlength=n_bins).astype(np.intp)
    nonempty = n_per_bin > 0

    theta_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    theta_sums = np.bincount(bin_indices, weights=theta, minlength=n_bins)
    theta_bins[nonempty] = theta_sums[nonempty] / n_per_bin[nonempty]

    return bin_indices, theta_bins


def _aggregate_item_bins(
    bin_indices: NDArray[np.intp],
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Aggregate observed and expected item scores into assigned bins."""
    n_per_bin = np.bincount(bin_indices, minlength=n_bins).astype(np.intp)
    nonempty = n_per_bin > 0
    observed_scores = np.zeros(n_bins, dtype=np.float64)
    expected_scores = np.zeros(n_bins, dtype=np.float64)

    observed_sums = np.bincount(bin_indices, weights=observed, minlength=n_bins)
    expected_sums = np.bincount(bin_indices, weights=expected, minlength=n_bins)
    observed_scores[nonempty] = observed_sums[nonempty] / n_per_bin[nonempty]
    expected_scores[nonempty] = expected_sums[nonempty] / n_per_bin[nonempty]

    return observed_scores, expected_scores, n_per_bin


def _validate_integration_grid(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float],
    n_points: int,
) -> tuple[int, NDArray[np.float64]]:
    """Validate DIF model inputs and construct a unidimensional grid."""
    if model_ref.n_factors != 1 or model_focal.n_factors != 1:
        raise ValueError("empirical DIF diagnostics require unidimensional models")
    ref_idx = _validate_item_index(model_ref, item_idx)
    _validate_item_index(model_focal, item_idx)
    point_count = _validate_positive_integer(n_points, "n_points", minimum=2)

    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range lower bound must be less than its upper bound")

    theta = np.linspace(limits[0], limits[1], point_count).reshape(-1, 1)
    return ref_idx, theta


def empirical_ES(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
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
        Relative focal-group weight, retained for API compatibility. Because
        this function integrates over one shared standard-normal density, the
        result is invariant to this mixture weight. Must be between 0 and 1.

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
    item_idx, theta_2d = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )
    if not np.isscalar(focal_weight) or not np.isfinite(focal_weight):
        raise ValueError("focal_weight must be a finite value between 0 and 1")
    if focal_weight < 0 or focal_weight > 1:
        raise ValueError("focal_weight must be between 0 and 1")

    from scipy import stats

    theta = theta_2d[:, 0]
    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    score_ref, ref_max_score = _expected_item_score(model_ref, theta_2d, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta_2d, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    diff = score_focal - score_ref

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
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    item_idx: int,
    n_bins: int = 10,
) -> EmpiricalPlotData:
    """Compute data for observed vs expected empirical plot.

    Groups examinees by theta estimate and computes observed vs
    model-predicted mean item scores for model-data fit assessment. For
    dichotomous items these scores are proportions correct; for polytomous
    items they are expected category scores. Missing item responses are
    excluded without changing the theta-bin boundaries.

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
    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    item_idx = _validate_item_index(model, item_idx)
    n_bins = _validate_positive_integer(n_bins, "n_bins")

    item_responses = responses[:, item_idx]
    valid_mask = np.isfinite(item_responses) & (item_responses >= 0)
    if not np.any(valid_mask):
        return EmpiricalPlotData(
            item_idx=item_idx,
            theta_bins=np.array([]),
            observed_prop=np.array([]),
            expected_prop=np.array([]),
            n_per_bin=np.array([], dtype=np.intp),
            residuals=np.array([]),
        )

    bin_indices, theta_bins = _build_theta_bins(theta_2d[:, 0], n_bins)
    item_responses = item_responses[valid_mask]
    expected_scores, max_score = _expected_item_score(
        model, theta_2d[valid_mask], item_idx
    )
    _validate_observed_scores(item_responses, max_score, item_idx)
    observed_prop, expected_prop, n_per_bin = _aggregate_item_bins(
        bin_indices[valid_mask], item_responses, expected_scores, n_bins
    )

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
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_bins: int = 10,
) -> NDArray[np.float64]:
    """Compute RMSEA-like fit statistic per item.

    Measures root mean square error of approximation between observed and
    expected item scores across ability bins. Model expectations for all
    items are evaluated together once, and every item uses the same theta-bin
    boundaries so item-level missingness cannot shift the conditioning groups.

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
    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    n_bins = _validate_positive_integer(n_bins, "n_bins")
    expected_scores, max_scores = _expected_all_item_scores(model, theta_2d)
    for item_idx, max_score in enumerate(max_scores):
        _validate_observed_scores(responses[:, item_idx], int(max_score), item_idx)

    bin_indices, _ = _build_theta_bins(theta_2d[:, 0], n_bins)
    membership = (
        bin_indices[:, None] == np.arange(n_bins, dtype=np.intp)[None, :]
    ).T.astype(np.float64)
    valid = np.isfinite(responses) & (responses >= 0)
    valid_float = valid.astype(np.float64)
    counts = membership @ valid_float
    observed_sums = membership @ np.where(valid, responses, 0.0)
    expected_sums = membership @ np.where(valid, expected_scores, 0.0)

    observed_bins = np.divide(
        observed_sums,
        counts,
        out=np.zeros_like(observed_sums),
        where=counts > 0,
    )
    expected_bins = np.divide(
        expected_sums,
        counts,
        out=np.zeros_like(expected_sums),
        where=counts > 0,
    )
    estimable = counts > 0
    squared_residuals = np.where(estimable, (observed_bins - expected_bins) ** 2, 0.0)
    estimable_bins = estimable.sum(axis=0)
    mean_squared = np.divide(
        squared_residuals.sum(axis=0),
        estimable_bins,
        out=np.full(model.n_items, np.nan, dtype=np.float64),
        where=estimable_bins > 0,
    )
    return np.sqrt(mean_squared)


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
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
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
    item_idx, theta = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )
    score_ref, ref_max_score = _expected_item_score(model_ref, theta, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    squared_diff = (score_ref - score_focal) ** 2
    rmsd = np.sqrt(np.mean(squared_diff))

    return float(rmsd)


def weighted_RMSD_DIF(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
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

    item_idx, theta_2d = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )

    theta = theta_2d[:, 0]
    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    score_ref, ref_max_score = _expected_item_score(model_ref, theta_2d, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta_2d, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    squared_diff = (score_ref - score_focal) ** 2
    weighted_rmsd = np.sqrt(np.sum(weights * squared_diff))

    return float(weighted_rmsd)


@dataclass
class ItemGAMResult:
    """Container for itemGAM results.

    Attributes
    ----------
    item_idx : int
        Item index.
    theta_grid : NDArray[np.float64]
        Grid of theta values for smooth curve.
    smoothed_probs : NDArray[np.float64]
        Smoothed empirical mean scores.
    model_probs : NDArray[np.float64]
        Model-predicted expected scores.
    se_bands : NDArray[np.float64]
        Standard error bands (lower, upper) for smoothed curve.
    raw_theta : NDArray[np.float64]
        Raw theta values from data.
    raw_probs : NDArray[np.float64]
        Raw observed item scores at each theta.
    """

    item_idx: int
    theta_grid: NDArray[np.float64]
    smoothed_probs: NDArray[np.float64]
    model_probs: NDArray[np.float64]
    se_bands: NDArray[np.float64]
    raw_theta: NDArray[np.float64]
    raw_probs: NDArray[np.float64]


def itemGAM(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    item_idx: int | list[int] | None = None,
    n_grid: int = 100,
    bandwidth: float | None = None,
    se: bool = True,
    alpha: float = 0.05,
    theta_margin: float = 0.1,
) -> ItemGAMResult | list[ItemGAMResult]:
    """Compute parametric smoothed regression lines for item response functions.

    Fits a kernel-smoothed regression to compare observed item performance
    with model predictions. Dichotomous results are proportions correct;
    polytomous results are collapsed to expected category scores.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, 1).
    item_idx : int, list of int, or None
        Item index or indices to analyze. If None, all items.
    n_grid : int
        Number of points in theta grid. Default 100.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses Silverman's rule of thumb.
    se : bool
        Whether to compute standard error bands. Default True.
    alpha : float
        Significance level for confidence bands. Default 0.05 (95% CI).
    theta_margin : float
        Fraction of theta range to extend grid beyond observed values.
        Default 0.1 (10% on each side).

    Returns
    -------
    ItemGAMResult or list of ItemGAMResult
        Smoothed regression results for each item.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> scores = fscores(result, responses)
    >>> gam = itemGAM(result.model, responses, scores.theta, item_idx=0)
    >>> # Plot smoothed vs model curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(gam.theta_grid, gam.smoothed_probs, label='Observed (smoothed)')
    >>> plt.plot(gam.theta_grid, gam.model_probs, label='Model')
    >>> plt.fill_between(gam.theta_grid, gam.se_bands[0], gam.se_bands[1], alpha=0.2)
    >>> plt.legend()
    """
    from scipy import stats

    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    theta_values = theta_2d[:, 0]
    n_grid = _validate_positive_integer(n_grid, "n_grid", minimum=2)

    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite value between 0 and 1")
    if (
        not np.isscalar(theta_margin)
        or not np.isfinite(theta_margin)
        or theta_margin < 0
    ):
        raise ValueError("theta_margin must be a finite non-negative value")

    z_crit = stats.norm.ppf(1 - alpha / 2)

    if item_idx is None:
        item_indices = list(range(model.n_items))
        single_item = False
    elif isinstance(item_idx, (int, np.integer)) and not isinstance(
        item_idx, (bool, np.bool_)
    ):
        item_indices = [_validate_item_index(model, int(item_idx))]
        single_item = True
    else:
        try:
            item_indices = [_validate_item_index(model, idx) for idx in list(item_idx)]
        except TypeError as exc:
            raise ValueError(
                "item_idx must be an integer, a list of integers, or None"
            ) from exc
        single_item = False

    if bandwidth is None:
        resolved_bandwidth = (
            SILVERMAN_CONSTANT
            * np.std(theta_values)
            * len(theta_values) ** SILVERMAN_EXPONENT
        )
        if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
            resolved_bandwidth = max(float(np.ptp(theta_values)), 1.0)
    else:
        if not np.isscalar(bandwidth) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be a finite positive value")
        resolved_bandwidth = float(bandwidth)

    theta_min, theta_max = np.min(theta_values), np.max(theta_values)
    margin = theta_margin * (theta_max - theta_min)
    theta_grid = np.linspace(theta_min - margin, theta_max + margin, n_grid)

    results = []
    for idx in item_indices:
        item_responses = responses[:, idx]
        valid_mask = np.isfinite(item_responses) & (item_responses >= 0)
        item_resp_valid = item_responses[valid_mask]
        theta_valid = theta_values[valid_mask]

        model_probs, max_score = _expected_item_score(
            model, theta_grid.reshape(-1, 1), idx
        )
        _validate_observed_scores(item_resp_valid, max_score, idx)

        if item_resp_valid.size == 0:
            smoothed_probs = np.full(n_grid, np.nan, dtype=np.float64)
            if se:
                se_lower = np.full(n_grid, np.nan, dtype=np.float64)
                se_upper = np.full(n_grid, np.nan, dtype=np.float64)
            else:
                se_lower = np.zeros(n_grid, dtype=np.float64)
                se_upper = np.zeros(n_grid, dtype=np.float64)
        else:
            smoothed_probs = np.empty(n_grid, dtype=np.float64)
            se_lower = np.zeros(n_grid, dtype=np.float64)
            se_upper = np.zeros(n_grid, dtype=np.float64)
            block_size = max(
                1,
                min(n_grid, KERNEL_BLOCK_ELEMENTS // item_resp_valid.size),
            )
            for start in range(0, n_grid, block_size):
                stop = min(start + block_size, n_grid)
                scaled_distance = (
                    theta_grid[start:stop, None] - theta_valid[None, :]
                ) / resolved_bandwidth
                log_weights = -0.5 * scaled_distance**2
                log_weights -= np.max(log_weights, axis=1, keepdims=True)
                weights = np.exp(log_weights)
                weights /= np.sum(weights, axis=1, keepdims=True)

                block_smoothed = weights @ item_resp_valid
                smoothed_probs[start:stop] = block_smoothed
                if se:
                    local_second_moment = weights @ (item_resp_valid**2)
                    local_variance = np.maximum(
                        local_second_moment - block_smoothed**2, 0.0
                    )
                    effective_n = 1.0 / np.sum(weights**2, axis=1)
                    se_values = np.sqrt(local_variance / effective_n)
                    se_lower[start:stop] = np.clip(
                        block_smoothed - z_crit * se_values, 0, max_score
                    )
                    se_upper[start:stop] = np.clip(
                        block_smoothed + z_crit * se_values, 0, max_score
                    )

        results.append(
            ItemGAMResult(
                item_idx=idx,
                theta_grid=theta_grid,
                smoothed_probs=smoothed_probs,
                model_probs=model_probs,
                se_bands=np.array([se_lower, se_upper]),
                raw_theta=theta_valid,
                raw_probs=item_resp_valid,
            )
        )

    if single_item:
        return results[0]
    return results
