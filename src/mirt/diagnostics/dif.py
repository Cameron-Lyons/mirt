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
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


_GRDIF_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_GRDIF_SCORING_METHODS = frozenset({"EAP", "MAP", "ML", "WLE"})
_GRDIF_PURIFICATION_METHODS = frozenset({"grdif_rs", "grdif_r", "grdif_s"})
_GRDIF_SCALING_METHODS = frozenset({"mean", "mad", "iqr"})


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
    p_adjust: PValueAdjustment = "none",
) -> dict[str, Any]:
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
        p_adjust: Multiple-testing adjustment across items. Supported values are
            'none', 'bonferroni', 'holm', and 'fdr_bh'. Default 'none'.

    Returns:
        Dictionary with DIF statistics:
            - 'statistic': Test statistic for each item
            - 'p_value': P-value for each item
            - 'p_value_adjusted': Multiplicity-adjusted P-value for each item
            - 'effect_size': Effect size measure
            - 'classification': ETS classification using adjusted P-values
            - 'adjustment': Adjustment method for each row
    """
    if method not in {"likelihood_ratio", "wald", "lord", "raju"}:
        raise ValueError(f"Unknown DIF method: {method}")
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")

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

    result: dict[str, Any]
    if method == "likelihood_ratio":
        result = _dif_likelihood_ratio(ref_result, focal_result, n_items)
    elif method == "wald":
        result = _dif_wald(ref_result, focal_result, n_items)
    elif method == "lord":
        result = _dif_wald(ref_result, focal_result, n_items)
    else:
        result = _dif_raju(ref_result, focal_result, n_items)

    adjusted = adjust_p_values(result["p_value"], p_adjust)
    result["p_value_adjusted"] = adjusted
    result["classification"] = _ets_classify(result["effect_size"], adjusted)
    result["adjustment"] = np.full(n_items, p_adjust)
    return result


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
        es = abs(effect_sizes[i])
        p = p_values[i]

        if p > 0.05 or es < 0.426:
            classification[i] = "A"
        elif es < 0.638:
            classification[i] = "B"
        else:
            classification[i] = "C"

    return classification


def flag_dif_items(
    dif_results: dict[str, Any],
    alpha: float = 0.05,
    min_effect_size: float = 0.426,
    classification: str | None = None,
    p_adjust: PValueAdjustment = "none",
) -> NDArray[np.bool_]:
    """Flag items showing significant DIF.

    Args:
        dif_results: Output from compute_dif().
        alpha: Significance level for p-value.
        min_effect_size: Minimum effect size to flag.
        classification: If specified, flag items with this ETS class or worse.
            'B' flags B and C items, 'C' flags only C items.
        p_adjust: Multiple-testing adjustment applied before flagging. Supported
            values are 'none', 'bonferroni', 'holm', and 'fdr_bh'.

    Returns:
        Boolean array indicating flagged items.
    """
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite and in (0, 1)")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and in (0, 1)") from exc
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if isinstance(min_effect_size, (bool, np.bool_)):
        raise ValueError("min_effect_size must be finite and nonnegative")
    try:
        min_effect_size = float(min_effect_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_effect_size must be finite and nonnegative") from exc
    if not np.isfinite(min_effect_size) or min_effect_size < 0.0:
        raise ValueError("min_effect_size must be finite and nonnegative")
    if classification not in {None, "B", "C"}:
        raise ValueError("classification must be 'B', 'C', or None")

    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    p_values = adjust_p_values(dif_results["p_value"], p_adjust)
    effect_sizes = np.asarray(dif_results["effect_size"], dtype=np.float64)
    classes = _ets_classify(effect_sizes, p_values)

    flags = (p_values <= alpha) & (np.abs(effect_sizes) >= min_effect_size)

    if classification is not None:
        if classification == "B":
            flags &= (classes == "B") | (classes == "C")
        else:
            flags &= classes == "C"

    return flags


def _validate_grdif_inputs(
    *,
    data: object,
    groups: object,
    model: str,
    scoring_method: str,
    alpha: float,
    purify: object,
    purify_by: str,
    max_purify_iter: int,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    scaling_method: str,
) -> tuple[NDArray[np.int_], NDArray[Any], NDArray[Any]]:
    if model not in _GRDIF_MODELS:
        valid = ", ".join(sorted(_GRDIF_MODELS))
        raise ValueError(f"model must be one of: {valid}")
    if scoring_method not in _GRDIF_SCORING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCORING_METHODS))
        raise ValueError(f"scoring_method must be one of: {valid}")
    if purify_by not in _GRDIF_PURIFICATION_METHODS:
        valid = ", ".join(sorted(_GRDIF_PURIFICATION_METHODS))
        raise ValueError(f"purify_by must be one of: {valid}")
    if scaling_method not in _GRDIF_SCALING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCALING_METHODS))
        raise ValueError(f"scaling_method must be one of: {valid}")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not isinstance(purify, (bool, np.bool_)):
        raise ValueError("purify must be a boolean")

    integer_controls = (
        ("max_purify_iter", max_purify_iter, 1),
        ("n_quadpts", n_quadpts, 2),
        ("max_iter", max_iter, 1),
    )
    for name, value, minimum in integer_controls:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
        if value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    values = np.asarray(data)
    labels = np.asarray(groups)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty two-dimensional response matrix")
    if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
        raise ValueError("data must contain finite numeric response codes")
    if np.any(values < -1) or np.any(values != np.floor(values)):
        raise ValueError("responses must be integer coded with -1 reserved for missing")
    if labels.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if labels.shape[0] != values.shape[0]:
        raise ValueError("groups length must match the number of response-matrix rows")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must not contain missing or non-finite labels")
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    try:
        unique_groups = np.unique(labels)
    except TypeError as exc:
        raise ValueError("group labels must be mutually comparable") from exc
    if unique_groups.size < 2:
        raise ValueError(
            f"GRDIF requires at least 2 groups, found {unique_groups.size}"
        )
    return values.astype(np.int64, copy=False), labels, unique_groups


def _score_grdif_responses(
    model: Any,
    data: NDArray[np.int_],
    anchor_items: NDArray[np.bool_],
    *,
    scoring_method: str,
    n_quadpts: int,
) -> NDArray[np.float64]:
    """Estimate abilities from the current purified item set."""
    from mirt.scoring import fscores

    anchors = np.asarray(anchor_items, dtype=np.bool_)
    if anchors.shape != (data.shape[1],):
        raise ValueError("anchor_items must match the number of response columns")
    if not np.any(anchors):
        raise ValueError("at least one anchor item is required for scoring")

    if np.all(anchors):
        scoring_data = data
    else:
        scoring_data = data.copy()
        scoring_data[:, ~anchors] = -1
    score_result = fscores(
        model,
        scoring_data,
        method=scoring_method,
        n_quadpts=n_quadpts,
    )
    theta = np.asarray(score_result.theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    if theta.ndim != 2 or theta.shape[0] != data.shape[0]:
        raise ValueError("ability estimates must have one row per response record")
    if not np.all(np.isfinite(theta)):
        raise ValueError("ability estimates must contain only finite values")
    return theta


def _expected_response_matrix(
    model: Any,
    theta: NDArray[np.float64],
    n_items: int,
) -> NDArray[np.float64]:
    """Evaluate every expected item score in one model call."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    if np.any(probabilities < -PROB_EPSILON) or np.any(
        probabilities > 1.0 + PROB_EPSILON
    ):
        raise ValueError("model probabilities must lie in [0, 1]")

    expected_shape = (theta.shape[0], n_items)
    if probabilities.ndim == 2:
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"dichotomous probabilities must have shape {expected_shape}"
            )
        expected = probabilities
    elif probabilities.ndim == 3:
        if probabilities.shape[:2] != expected_shape:
            raise ValueError(
                "polytomous probabilities must have shape "
                f"({theta.shape[0]}, {n_items}, n_categories)"
            )
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        expected = probabilities @ category_scores
    else:
        raise ValueError("model probabilities must be two- or three-dimensional")
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected item scores must contain only finite values")
    return expected


def _column_scale(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    counts: NDArray[np.int_],
    means: NDArray[np.float64],
    method: Literal["mean", "mad", "iqr"],
) -> NDArray[np.float64]:
    """Compute one variance-like scale per item column."""
    n_items = values.shape[1]
    scales = np.ones(n_items, dtype=np.float64)
    sufficient = counts >= 2
    if method == "mean":
        deviations = np.where(valid, values - means[None, :], 0.0)
        scales[sufficient] = np.sum(deviations**2, axis=0)[sufficient] / (
            counts[sufficient] - 1
        )
        return scales

    for item_index in np.flatnonzero(sufficient):
        scales[item_index] = _compute_robust_scale(
            values[valid[:, item_index], item_index], method
        )
    return scales


def _group_residual_moments(
    residuals: NDArray[np.float64],
    valid: NDArray[np.bool_],
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray[Any],
    scaling_method: Literal["mean", "mad", "iqr"],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
]:
    """Compute group-by-item residual moments with itemwise missingness."""
    n_groups = len(unique_groups)
    n_items = residuals.shape[1]
    mrr = np.zeros((n_groups, n_items), dtype=np.float64)
    msr = np.zeros((n_groups, n_items), dtype=np.float64)
    var_mrr = np.ones((n_groups, n_items), dtype=np.float64)
    var_msr = np.ones((n_groups, n_items), dtype=np.float64)
    effective_counts = np.ones((n_groups, n_items), dtype=np.int64)

    for group_index, group in enumerate(unique_groups):
        mask = group_masks[group]
        group_valid = valid[mask]
        group_residuals = residuals[mask]
        counts = np.count_nonzero(group_valid, axis=0)
        sufficient = counts >= 2
        effective_counts[group_index, sufficient] = counts[sufficient]

        sums = np.sum(np.where(group_valid, group_residuals, 0.0), axis=0)
        squared = group_residuals**2
        squared_sums = np.sum(np.where(group_valid, squared, 0.0), axis=0)
        mrr[group_index, sufficient] = sums[sufficient] / counts[sufficient]
        msr[group_index, sufficient] = squared_sums[sufficient] / counts[sufficient]

        raw_scale = _column_scale(
            group_residuals,
            group_valid,
            counts,
            mrr[group_index],
            scaling_method,
        )
        squared_scale = _column_scale(
            squared,
            group_valid,
            counts,
            msr[group_index],
            scaling_method,
        )
        var_mrr[group_index, sufficient] = np.maximum(
            raw_scale[sufficient] / counts[sufficient], PROB_EPSILON
        )
        var_msr[group_index, sufficient] = np.maximum(
            squared_scale[sufficient] / counts[sufficient], PROB_EPSILON
        )

    return mrr, msr, var_mrr, var_msr, effective_counts


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
        purify: Whether to iteratively remove flagged items from ability scoring
            and re-estimate abilities from the remaining anchors.
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
            - 'anchor_items': Boolean mask of the final purified item set
            - 'purification_history': Iteration details if purify=True
            - 'purification_complete': Whether the anchor set converged
            - 'purification_stop_reason': Convergence or stopping condition
            - 'theta': Final ability estimates used by the reported statistics

    References:
        Lim, H., et al. (2024). Detecting Differential Item Functioning among
        Multiple Groups Using IRT Residual DIF Framework. Journal of
        Educational Measurement.
    """
    from mirt import fit_mirt

    data, groups, unique_groups = _validate_grdif_inputs(
        data=data,
        groups=groups,
        model=model,
        scoring_method=scoring_method,
        alpha=alpha,
        purify=purify,
        purify_by=purify_by,
        max_purify_iter=max_purify_iter,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        scaling_method=scaling_method,
    )
    _, n_items = data.shape
    n_groups = len(unique_groups)

    group_masks = {g: groups == g for g in unique_groups}
    group_sizes = {g: int(np.count_nonzero(mask)) for g, mask in group_masks.items()}

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    anchor_items = np.ones(n_items, dtype=np.bool_)
    theta = _score_grdif_responses(
        fit_result.model,
        data,
        anchor_items,
        scoring_method=scoring_method,
        n_quadpts=n_quadpts,
    )

    purification_history: list[dict[str, Any]] = []
    purification_complete: bool | None = None
    purification_stop_reason: str | None = None

    if purify:
        for iteration in range(max_purify_iter):
            grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
                data,
                theta,
                fit_result.model,
                group_masks,
                unique_groups,
                scaling_method,
            )

            p_values = {
                "grdif_rs": p_rs,
                "grdif_r": p_r,
                "grdif_s": p_s,
            }[purify_by]
            flagged = p_values < alpha

            new_anchors = ~flagged
            n_anchors = int(np.count_nonzero(new_anchors))
            purification_history.append(
                {
                    "iteration": iteration + 1,
                    "n_flagged": int(np.count_nonzero(flagged)),
                    "flagged_items": np.flatnonzero(flagged).tolist(),
                    "n_anchors": n_anchors,
                }
            )

            if n_anchors < 2:
                purification_complete = False
                purification_stop_reason = "insufficient_anchors"
                break

            if np.array_equal(anchor_items, new_anchors):
                purification_complete = True
                purification_stop_reason = "converged"
                break

            anchor_items = new_anchors
            theta = _score_grdif_responses(
                fit_result.model,
                data,
                anchor_items,
                scoring_method=scoring_method,
                n_quadpts=n_quadpts,
            )
        else:
            purification_complete = False
            purification_stop_reason = "max_iterations"

    grdif_r, grdif_s, grdif_rs, p_r, p_s, p_rs = _compute_grdif_statistics(
        data,
        theta,
        fit_result.model,
        group_masks,
        unique_groups,
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
        "purification_complete": purification_complete,
        "purification_stop_reason": purification_stop_reason,
        "theta": theta.copy(),
    }


def _compute_grdif_statistics(
    data: NDArray[np.int_],
    theta: NDArray[np.float64],
    model: Any,
    group_masks: dict[Any, NDArray[np.bool_]],
    unique_groups: NDArray,
    scaling_method: Literal["mean", "mad", "iqr"] = "mean",
    *,
    expected_responses: NDArray[np.float64] | None = None,
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

    Expected responses are evaluated in one batched model call. Residual
    moments are then reduced by group while retaining itemwise missingness.
    """
    n_groups = len(unique_groups)
    df_r = n_groups - 1
    df_s = n_groups - 1
    df_rs = 2 * (n_groups - 1)

    if scaling_method not in _GRDIF_SCALING_METHODS:
        valid = ", ".join(sorted(_GRDIF_SCALING_METHODS))
        raise ValueError(f"scaling_method must be one of: {valid}")
    values = np.asarray(data)
    theta_values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("data must be a two-dimensional response matrix")
    n_items = values.shape[1]
    if theta_values.ndim != 2 or theta_values.shape[0] != values.shape[0]:
        raise ValueError("theta must be two-dimensional with one row per person")

    if expected_responses is None:
        expected = _expected_response_matrix(model, theta_values, n_items)
    else:
        expected = np.asarray(expected_responses, dtype=np.float64)
        if expected.shape != values.shape or not np.all(np.isfinite(expected)):
            raise ValueError(
                "expected_responses must be finite and match the response matrix"
            )

    valid_responses = values >= 0
    residuals = np.where(valid_responses, values - expected, 0.0)
    mrr, msr, var_mrr, var_msr, effective_counts = _group_residual_moments(
        residuals,
        valid_responses,
        group_masks,
        unique_groups,
        scaling_method,
    )

    weights = effective_counts / np.sum(effective_counts, axis=0, keepdims=True)
    pooled_mrr = np.sum(weights * mrr, axis=0)
    pooled_msr = np.sum(weights * msr, axis=0)
    centered_mrr = mrr - pooled_mrr
    centered_msr = msr - pooled_msr

    grdif_r = np.sum(centered_mrr**2 / var_mrr, axis=0)
    grdif_s = np.sum(centered_msr**2 / var_msr, axis=0)
    grdif_rs = grdif_r + grdif_s

    p_r = stats.chi2.sf(grdif_r, df=df_r)
    p_s = stats.chi2.sf(grdif_s, df=df_s)
    p_rs = stats.chi2.sf(grdif_rs, df=df_rs)

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

    data, groups, unique_groups = _validate_grdif_inputs(
        data=data,
        groups=groups,
        model=model,
        scoring_method=scoring_method,
        alpha=alpha,
        purify=False,
        purify_by="grdif_rs",
        max_purify_iter=1,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        scaling_method="mean",
    )
    n_items = data.shape[1]

    fit_result = fit_mirt(
        data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    theta = _score_grdif_responses(
        fit_result.model,
        data,
        np.ones(n_items, dtype=np.bool_),
        scoring_method=scoring_method,
        n_quadpts=n_quadpts,
    )
    expected_responses = _expected_response_matrix(fit_result.model, theta, n_items)

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
        r, s, rs, _, _, _ = _compute_grdif_statistics(
            data,
            theta,
            fit_result.model,
            pair_masks,
            pair_groups,
            expected_responses=expected_responses,
        )

        rdif_r[pair_idx] = r
        rdif_s[pair_idx] = s
        rdif_rs[pair_idx] = rs

    p_r = stats.chi2.sf(rdif_r, df=1)
    p_s = stats.chi2.sf(rdif_s, df=1)
    p_rs = stats.chi2.sf(rdif_rs, df=2)

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
