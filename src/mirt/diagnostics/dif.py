"""Differential Item Functioning (DIF) analysis."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import extract_item_se, fit_group_models, split_groups

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


def compute_dif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    method: Literal["likelihood_ratio", "wald", "lord", "raju"] = "likelihood_ratio",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    focal_group: str | int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute Differential Item Functioning statistics.

    DIF analysis tests whether items function differently across groups
    after controlling for ability level.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Must have exactly 2 groups.
        model: IRT model type.
        method: DIF detection method:
            - 'likelihood_ratio': Likelihood ratio test (recommended)
            - 'wald': Wald test on parameter differences
            - 'lord': Lord's chi-square test
            - 'raju': Raju's area measures
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).

    Returns:
        Dictionary with DIF statistics:
            - 'statistic': Test statistic for each item
            - 'p_value': P-value for each item
            - 'effect_size': Effect size measure
            - 'classification': ETS classification (A/B/C)
    """
    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]

    ref_data, focal_data, _, _, _, _ = split_groups(data, groups, focal_group)

    ref_result, focal_result = fit_group_models(
        ref_data,
        focal_data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
    )

    if method == "likelihood_ratio":
        return _dif_likelihood_ratio(ref_result, focal_result, n_items)
    elif method == "wald":
        return _dif_wald(ref_result, focal_result, n_items)
    elif method == "lord":
        return _dif_wald(ref_result, focal_result, n_items)  # Lord's test uses Wald
    elif method == "raju":
        return _dif_raju(ref_result, focal_result, n_items)
    else:
        raise ValueError(f"Unknown DIF method: {method}")


def _dif_likelihood_ratio(
    ref_result: FitResult,
    focal_result: FitResult,
    n_items: int,
) -> dict[str, NDArray[np.float64]]:
    """Likelihood ratio test for DIF."""
    statistics = np.zeros(n_items)
    p_values = np.zeros(n_items)
    effect_sizes = np.zeros(n_items)

    for item_idx in range(n_items):
        ref_params = ref_result.model.get_item_parameters(item_idx)
        focal_params = focal_result.model.get_item_parameters(item_idx)

        diff_sum_sq = 0.0
        n_params = 0

        for param_name in ref_params:
            ref_val = np.atleast_1d(ref_params[param_name])
            focal_val = np.atleast_1d(focal_params[param_name])

            ref_se_full = ref_result.standard_errors.get(
                param_name, np.ones_like(ref_val)
            )
            focal_se_full = focal_result.standard_errors.get(
                param_name, np.ones_like(focal_val)
            )

            ref_se = extract_item_se(ref_se_full, item_idx)
            focal_se = extract_item_se(focal_se_full, item_idx)

            pooled_var = ref_se**2 + focal_se**2
            pooled_var = np.where(pooled_var > 0, pooled_var, 1.0)

            diff = ref_val - focal_val
            diff_sum_sq += np.sum(diff**2 / pooled_var)
            n_params += len(ref_val)

        statistics[item_idx] = diff_sum_sq
        p_values[item_idx] = 1 - stats.chi2.cdf(diff_sum_sq, df=max(1, n_params))

        if "difficulty" in ref_params and "difficulty" in focal_params:
            ref_b = float(np.atleast_1d(ref_params["difficulty"])[0])
            focal_b = float(np.atleast_1d(focal_params["difficulty"])[0])
            effect_sizes[item_idx] = abs(ref_b - focal_b)
        elif "thresholds" in ref_params and "thresholds" in focal_params:
            ref_b = np.mean(ref_params["thresholds"])
            focal_b = np.mean(focal_params["thresholds"])
            effect_sizes[item_idx] = abs(ref_b - focal_b)
        elif "intercepts" in ref_params and "intercepts" in focal_params:
            ref_b = float(np.atleast_1d(ref_params["intercepts"])[0])
            focal_b = float(np.atleast_1d(focal_params["intercepts"])[0])
            effect_sizes[item_idx] = abs(ref_b - focal_b)

    classification = _ets_classify(effect_sizes, p_values)

    return {
        "statistic": statistics,
        "p_value": p_values,
        "effect_size": effect_sizes,
        "classification": classification,
    }


def _dif_wald(
    ref_result: FitResult,
    focal_result: FitResult,
    n_items: int,
) -> dict[str, NDArray[np.float64]]:
    """Wald test for DIF."""
    statistics = np.zeros(n_items)
    p_values = np.zeros(n_items)
    effect_sizes = np.zeros(n_items)

    for item_idx in range(n_items):
        ref_params = ref_result.model.get_item_parameters(item_idx)
        focal_params = focal_result.model.get_item_parameters(item_idx)

        wald_sum = 0.0
        df = 0

        for param_name in ref_params:
            ref_val = np.atleast_1d(ref_params[param_name])
            focal_val = np.atleast_1d(focal_params[param_name])

            ref_se_full = ref_result.standard_errors.get(param_name)
            focal_se_full = focal_result.standard_errors.get(param_name)

            if ref_se_full is None or focal_se_full is None:
                continue

            ref_se = extract_item_se(ref_se_full, item_idx)
            focal_se = extract_item_se(focal_se_full, item_idx)

            pooled_var = ref_se**2 + focal_se**2
            valid = pooled_var > PROB_EPSILON

            if np.any(valid):
                diff = ref_val - focal_val
                wald_sum += np.sum((diff[valid] ** 2) / pooled_var[valid])
                df += np.sum(valid)

        statistics[item_idx] = wald_sum
        p_values[item_idx] = 1 - stats.chi2.cdf(wald_sum, df=max(1, df))

        if "difficulty" in ref_params and "difficulty" in focal_params:
            ref_b = float(np.atleast_1d(ref_params["difficulty"])[0])
            focal_b = float(np.atleast_1d(focal_params["difficulty"])[0])
            effect_sizes[item_idx] = abs(ref_b - focal_b)

    classification = _ets_classify(effect_sizes, p_values)

    return {
        "statistic": statistics,
        "p_value": p_values,
        "effect_size": effect_sizes,
        "classification": classification,
    }


def _dif_raju(
    ref_result: FitResult,
    focal_result: FitResult,
    n_items: int,
) -> dict[str, NDArray[np.float64]]:
    """Raju's area measures for DIF."""
    theta_range = np.linspace(-4, 4, 100)
    theta_2d = theta_range.reshape(-1, 1)

    statistics = np.zeros(n_items)
    effect_sizes = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for item_idx in range(n_items):
        ref_prob = ref_result.model.probability(theta_2d, item_idx)
        focal_prob = focal_result.model.probability(theta_2d, item_idx)

        if ref_prob.ndim > 1:
            n_cat = ref_prob.shape[1]
            categories = np.arange(n_cat)
            ref_expected = np.sum(ref_prob * categories, axis=1)
            focal_expected = np.sum(focal_prob * categories, axis=1)
            ref_prob = ref_expected / (n_cat - 1)
            focal_prob = focal_expected / (n_cat - 1)

        diff = ref_prob - focal_prob

        unsigned_area = trapezoid(np.abs(diff), theta_range)
        statistics[item_idx] = unsigned_area

        signed_area = trapezoid(diff, theta_range)
        effect_sizes[item_idx] = signed_area

        se_area = 0.1 * (1 + 0.5 * unsigned_area)
        z = unsigned_area / se_area
        p_values[item_idx] = 2 * (1 - stats.norm.cdf(abs(z)))

    classification = _ets_classify(np.abs(effect_sizes), p_values)

    return {
        "statistic": statistics,
        "p_value": p_values,
        "effect_size": effect_sizes,
        "classification": classification,
    }


def _ets_classify(
    effect_sizes: NDArray[np.float64],
    p_values: NDArray[np.float64],
) -> NDArray:
    """Classify DIF using ETS guidelines (A/B/C)."""
    n_items = len(effect_sizes)
    classification = np.empty(n_items, dtype="U1")

    for i in range(n_items):
        es = effect_sizes[i]
        p = p_values[i]

        if p > 0.05 or es < 0.426:
            classification[i] = "A"
        elif es < 0.638:
            classification[i] = "B"
        else:
            classification[i] = "C"

    return classification


def flag_dif_items(
    dif_results: dict[str, NDArray[np.float64]],
    alpha: float = 0.05,
    min_effect_size: float = 0.426,
    classification: str | None = None,
) -> NDArray[np.bool_]:
    """Flag items showing significant DIF.

    Args:
        dif_results: Output from compute_dif().
        alpha: Significance level for p-value.
        min_effect_size: Minimum effect size to flag.
        classification: If specified, flag items with this ETS class or worse.
            'B' flags B and C items, 'C' flags only C items.

    Returns:
        Boolean array indicating flagged items.
    """
    p_values = dif_results["p_value"]
    effect_sizes = dif_results["effect_size"]
    classes = dif_results["classification"]

    flags = (p_values <= alpha) & (np.abs(effect_sizes) >= min_effect_size)

    if classification is not None:
        if classification == "B":
            flags &= (classes == "B") | (classes == "C")
        elif classification == "C":
            flags &= classes == "C"

    return flags


def compute_grdif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
    alpha: float = 0.05,
    purify: bool = False,
    purify_by: Literal["grdif_rs", "grdif_r", "grdif_s"] = "grdif_rs",
    max_purify_iter: int = 10,
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    scaling_method: Literal["mean", "mad", "iqr"] = "mean",
) -> dict[str, Any]:
    """Compute Generalized Residual DIF (GRDIF) statistics for multiple groups.

    GRDIF is a generalized version of the RDIF detection framework designed
    to assess DIF across multiple groups simultaneously. It computes three
    chi-square distributed test statistics based on IRT residuals.

    This method has several advantages over traditional DIF approaches:
    - Works with any number of groups (G >= 2)
    - No separate calibration per group required
    - No matching variable or theta bins needed
    - Computationally efficient
    - Well-controlled Type I error rates

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Can have 2+ groups.
        model: IRT model type for aggregate calibration.
        scoring_method: Method for computing ability estimates.
        alpha: Significance level for flagging DIF items.
        purify: Whether to use iterative purification to identify anchor items.
        purify_by: Which statistic to use for purification decisions.
        max_purify_iter: Maximum purification iterations.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        scaling_method: Method for variance estimation:
            - 'mean': Standard sample variance (default)
            - 'mad': Median absolute deviation (robust to outliers)
            - 'iqr': Interquartile range (robust to outliers)

    Returns:
        Dictionary with GRDIF results:
            - 'grdif_r': GRDIF_R statistics (uniform DIF)
            - 'grdif_s': GRDIF_S statistics (nonuniform DIF)
            - 'grdif_rs': GRDIF_RS statistics (mixed DIF)
            - 'p_value_r': P-values for GRDIF_R
            - 'p_value_s': P-values for GRDIF_S
            - 'p_value_rs': P-values for GRDIF_RS
            - 'flagged_r': Items flagged by GRDIF_R
            - 'flagged_s': Items flagged by GRDIF_S
            - 'flagged_rs': Items flagged by GRDIF_RS
            - 'n_groups': Number of groups
            - 'group_labels': Unique group labels
            - 'group_sizes': Sample size per group
            - 'purification_history': Iteration details if purify=True

    References:
        Lim, H., et al. (2024). Detecting Differential Item Functioning among
        Multiple Groups Using IRT Residual DIF Framework. Journal of
        Educational Measurement.
    """
    from mirt import fit_mirt
    from mirt.scoring import fscores

    data = np.asarray(data)
    groups = np.asarray(groups)
    n_persons, n_items = data.shape

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError(f"GRDIF requires at least 2 groups, found {n_groups}")

    group_masks = {g: groups == g for g in unique_groups}
    group_sizes = {g: np.sum(mask) for g, mask in group_masks.items()}

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    score_result = fscores(
        fit_result.model,
        data,
        method=scoring_method,
        n_quadpts=n_quadpts,
    )
    theta = score_result.theta
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    anchor_items = np.ones(n_items, dtype=bool)
    purification_history = []

    if purify:
        for iteration in range(max_purify_iter):
            grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
                data,
                theta,
                fit_result.model,
                group_masks,
                unique_groups,
                anchor_items,
                scaling_method,
            )

            if purify_by == "grdif_rs":
                flagged = p_rs < alpha
            elif purify_by == "grdif_r":
                flagged = p_r < alpha
            else:
                flagged = p_s < alpha

            new_anchors = ~flagged

            if np.sum(new_anchors) < 2:
                break

            purification_history.append(
                {
                    "iteration": iteration + 1,
                    "n_flagged": int(np.sum(flagged)),
                    "flagged_items": np.where(flagged)[0].tolist(),
                }
            )

            if np.array_equal(anchor_items, new_anchors):
                break

            anchor_items = new_anchors

    grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
        data,
        theta,
        fit_result.model,
        group_masks,
        unique_groups,
        anchor_items,
        scaling_method,
    )

    return {
        "grdif_r": grdif_r,
        "grdif_s": grdif_s,
        "grdif_rs": grdif_rs,
        "p_value_r": p_r,
        "p_value_s": p_s,
        "p_value_rs": p_rs,
        "flagged_r": p_r < alpha,
        "flagged_s": p_s < alpha,
        "flagged_rs": p_rs < alpha,
        "n_groups": n_groups,
        "group_labels": unique_groups.tolist(),
        "group_sizes": group_sizes,
        "anchor_items": anchor_items,
        "purification_history": purification_history if purify else None,
    }


def _compute_grdif_statistics(
    data: NDArray[np.int_],
    theta: NDArray[np.float64],
    model: Any,
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray,
    anchor_items: NDArray[np.bool_],
    scaling_method: Literal["mean", "mad", "iqr"] = "mean",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute GRDIF_R, GRDIF_S, GRDIF_RS statistics.

    The statistics are based on the asymptotic multivariate normality of
    the mean raw residuals (MRR) and mean squared residuals (MSR).

    GRDIF_R detects uniform DIF (differences in difficulty)
    GRDIF_S detects nonuniform DIF (differences in discrimination)
    GRDIF_RS detects mixed DIF (both types)
    """
    n_items = data.shape[1]
    n_groups = len(unique_groups)
    df_r = n_groups - 1
    df_s = n_groups - 1
    df_rs = 2 * (n_groups - 1)

    grdif_r = np.zeros(n_items)
    grdif_s = np.zeros(n_items)
    grdif_rs = np.zeros(n_items)

    for item_idx in range(n_items):
        mrr_g = []
        msr_g = []
        var_mrr_g = []
        var_msr_g = []
        n_g = []

        for g in unique_groups:
            mask = group_masks[g]
            responses_g = data[mask, item_idx]
            theta_g = theta[mask]

            valid = responses_g >= 0
            if np.sum(valid) < 2:
                mrr_g.append(0.0)
                msr_g.append(0.0)
                var_mrr_g.append(1.0)
                var_msr_g.append(1.0)
                n_g.append(1)
                continue

            responses_valid = responses_g[valid]
            theta_valid = theta_g[valid]
            n_valid = len(responses_valid)
            n_g.append(n_valid)

            expected = model.probability(theta_valid, item_idx=item_idx)
            if expected.ndim > 1:
                n_cat = expected.shape[1]
                categories = np.arange(n_cat)
                expected = np.sum(expected * categories, axis=1)
            else:
                expected = expected.ravel()

            residuals = responses_valid - expected

            mrr = np.mean(residuals)
            msr = np.mean(residuals**2)

            var_mrr = (
                _compute_robust_scale(residuals, scaling_method) / n_valid
                if n_valid > 1
                else 1.0
            )
            var_msr = (
                _compute_robust_scale(residuals**2, scaling_method) / n_valid
                if n_valid > 1
                else 1.0
            )

            var_mrr = max(var_mrr, PROB_EPSILON)
            var_msr = max(var_msr, PROB_EPSILON)

            mrr_g.append(mrr)
            msr_g.append(msr)
            var_mrr_g.append(var_mrr)
            var_msr_g.append(var_msr)

        mrr_g = np.array(mrr_g)
        msr_g = np.array(msr_g)
        var_mrr_g = np.array(var_mrr_g)
        var_msr_g = np.array(var_msr_g)

        total_n = sum(n_g)
        weights = np.array(n_g) / total_n

        mrr_pooled = np.sum(weights * mrr_g)
        msr_pooled = np.sum(weights * msr_g)

        mrr_centered = mrr_g - mrr_pooled
        msr_centered = msr_g - msr_pooled

        chi2_r = np.sum(mrr_centered**2 / var_mrr_g)
        chi2_s = np.sum(msr_centered**2 / var_msr_g)

        grdif_r[item_idx] = chi2_r
        grdif_s[item_idx] = chi2_s
        grdif_rs[item_idx] = chi2_r + chi2_s

    p_r = 1 - stats.chi2.cdf(grdif_r, df=df_r)
    p_s = 1 - stats.chi2.cdf(grdif_s, df=df_s)
    p_rs = 1 - stats.chi2.cdf(grdif_rs, df=df_rs)

    return grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs


def _compute_robust_scale(
    data: NDArray[np.float64],
    method: Literal["mean", "mad", "iqr"] = "mean",
) -> float:
    """Compute scale estimate (variance-like) using specified method.

    Args:
        data: Array of values to compute scale for.
        method: Scaling method:
            - 'mean': Standard sample variance
            - 'mad': Median absolute deviation squared (robust)
            - 'iqr': Interquartile range squared (robust)

    Returns:
        Scale estimate (variance-like quantity).
    """
    if method == "mean":
        return float(np.var(data, ddof=1)) if len(data) > 1 else 1.0
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median)) * 1.4826
        return max(float(mad**2), PROB_EPSILON)
    elif method == "iqr":
        q75, q25 = np.percentile(data, [75, 25])
        iqr_scale = (q75 - q25) / 1.349
        return max(float(iqr_scale**2), PROB_EPSILON)
    raise ValueError(f"Unknown scaling method: {method}")


def compute_pairwise_rdif(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    scoring_method: Literal["EAP", "MAP", "ML", "WLE"] = "EAP",
    alpha: float = 0.05,
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Compute pairwise RDIF statistics for post-hoc analysis.

    After finding significant GRDIF, this function performs pairwise
    comparisons between all group pairs to identify which specific
    groups differ on each item.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array.
        model: IRT model type.
        scoring_method: Method for computing ability estimates.
        alpha: Significance level.
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.

    Returns:
        Dictionary with pairwise results:
            - 'pairs': List of group pairs compared
            - 'rdif_r': RDIF_R statistics per pair per item
            - 'rdif_s': RDIF_S statistics per pair per item
            - 'rdif_rs': RDIF_RS statistics per pair per item
            - 'p_values_r': P-values for RDIF_R
            - 'p_values_s': P-values for RDIF_S
            - 'p_values_rs': P-values for RDIF_RS
    """
    from mirt import fit_mirt
    from mirt.scoring import fscores

    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("Need at least 2 groups for pairwise comparison")

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    score_result = fscores(
        fit_result.model,
        data,
        method=scoring_method,
        n_quadpts=n_quadpts,
    )
    theta = score_result.theta
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    pairs = list(combinations(unique_groups, 2))
    n_pairs = len(pairs)

    rdif_r = np.zeros((n_pairs, n_items))
    rdif_s = np.zeros((n_pairs, n_items))
    rdif_rs = np.zeros((n_pairs, n_items))

    for pair_idx, (g1, g2) in enumerate(pairs):
        mask1 = groups == g1
        mask2 = groups == g2

        pair_masks = {g1: mask1, g2: mask2}
        pair_groups = np.array([g1, g2])
        anchor_items = np.ones(n_items, dtype=bool)

        r, s, rs, _, _, _ = _compute_grdif_statistics(
            data, theta, fit_result.model, pair_masks, pair_groups, anchor_items
        )

        rdif_r[pair_idx] = r
        rdif_s[pair_idx] = s
        rdif_rs[pair_idx] = rs

    p_r = 1 - stats.chi2.cdf(rdif_r, df=1)
    p_s = 1 - stats.chi2.cdf(rdif_s, df=1)
    p_rs = 1 - stats.chi2.cdf(rdif_rs, df=2)

    return {
        "pairs": pairs,
        "rdif_r": rdif_r,
        "rdif_s": rdif_s,
        "rdif_rs": rdif_rs,
        "p_values_r": p_r,
        "p_values_s": p_s,
        "p_values_rs": p_rs,
        "flagged_r": p_r < alpha,
        "flagged_s": p_s < alpha,
        "flagged_rs": p_rs < alpha,
    }


def grdif_effect_size(
    data: NDArray[np.int_],
    groups: NDArray,
    grdif_results: dict[str, Any],
    effect_type: Literal["delta_mrr", "delta_msr", "max_diff"] = "delta_mrr",
) -> NDArray[np.float64]:
    """Compute effect sizes for GRDIF flagged items.

    Args:
        data: Response matrix.
        groups: Group membership array.
        grdif_results: Output from compute_grdif().
        effect_type: Type of effect size:
            - 'delta_mrr': Maximum difference in mean raw residuals
            - 'delta_msr': Maximum difference in mean squared residuals
            - 'max_diff': Maximum of both

    Returns:
        Effect size array for each item.
    """
    from mirt import fit_mirt
    from mirt.scoring import fscores

    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]

    unique_groups = np.array(grdif_results["group_labels"])

    fit_result = fit_mirt(data, model="2PL", verbose=False)
    score_result = fscores(fit_result.model, data, method="EAP")
    theta = score_result.theta
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    effect_sizes = np.zeros(n_items)

    for item_idx in range(n_items):
        mrr_values = []
        msr_values = []

        for g in unique_groups:
            mask = groups == g
            responses_g = data[mask, item_idx]
            theta_g = theta[mask]

            valid = responses_g >= 0
            if np.sum(valid) < 2:
                continue

            responses_valid = responses_g[valid]
            theta_valid = theta_g[valid]

            expected = fit_result.model.probability(theta_valid, item_idx=item_idx)
            if expected.ndim > 1:
                n_cat = expected.shape[1]
                categories = np.arange(n_cat)
                expected = np.sum(expected * categories, axis=1)
            else:
                expected = expected.ravel()

            residuals = responses_valid - expected
            mrr_values.append(np.mean(residuals))
            msr_values.append(np.mean(residuals**2))

        if len(mrr_values) >= 2:
            delta_mrr = np.max(mrr_values) - np.min(mrr_values)
            delta_msr = np.max(msr_values) - np.min(msr_values)

            if effect_type == "delta_mrr":
                effect_sizes[item_idx] = delta_mrr
            elif effect_type == "delta_msr":
                effect_sizes[item_idx] = delta_msr
            else:
                effect_sizes[item_idx] = max(delta_mrr, delta_msr)

    return effect_sizes
