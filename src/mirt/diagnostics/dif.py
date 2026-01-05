"""Differential Item Functioning (DIF) analysis."""

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import trapezoid

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
    from mirt import fit_mirt

    data = np.asarray(data)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(
            f"DIF analysis requires exactly 2 groups, got {len(unique_groups)}"
        )

    reference_group = unique_groups[0]
    if focal_group is None:
        focal_group = unique_groups[1]
    elif focal_group not in unique_groups:
        raise ValueError(f"focal_group {focal_group} not found in groups")

    if focal_group == reference_group:
        reference_group = unique_groups[1] if unique_groups[0] == focal_group else unique_groups[0]

    ref_mask = groups == reference_group
    focal_mask = groups == focal_group

    ref_data = data[ref_mask]
    focal_data = data[focal_mask]

    n_items = data.shape[1]

    ref_result = fit_mirt(
        ref_data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    focal_result = fit_mirt(
        focal_data,
        model=model,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    if method == "likelihood_ratio":
        return _dif_likelihood_ratio(ref_result, focal_result, n_items)
    elif method == "wald":
        return _dif_wald(ref_result, focal_result, n_items)
    elif method == "lord":
        return _dif_lord(ref_result, focal_result, n_items)
    elif method == "raju":
        return _dif_raju(ref_result, focal_result, n_items)
    else:
        raise ValueError(f"Unknown DIF method: {method}")


def _dif_likelihood_ratio(
    ref_result: "FitResult",
    focal_result: "FitResult",
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

            ref_se = ref_result.standard_errors.get(param_name, np.ones_like(ref_val))
            focal_se = focal_result.standard_errors.get(param_name, np.ones_like(focal_val))

            if ref_se.ndim > 1:
                ref_se = ref_se[item_idx]
            elif len(ref_se) > 1:
                ref_se = np.atleast_1d(ref_se[item_idx])

            if focal_se.ndim > 1:
                focal_se = focal_se[item_idx]
            elif len(focal_se) > 1:
                focal_se = np.atleast_1d(focal_se[item_idx])

            pooled_var = (ref_se**2 + focal_se**2)
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
    ref_result: "FitResult",
    focal_result: "FitResult",
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

            ref_se = ref_result.standard_errors.get(param_name)
            focal_se = focal_result.standard_errors.get(param_name)

            if ref_se is None or focal_se is None:
                continue

            if ref_se.ndim > 1:
                ref_se = ref_se[item_idx]
            elif len(ref_se) > 1:
                ref_se = np.atleast_1d(ref_se[item_idx])

            if focal_se.ndim > 1:
                focal_se = focal_se[item_idx]
            elif len(focal_se) > 1:
                focal_se = np.atleast_1d(focal_se[item_idx])

            pooled_var = ref_se**2 + focal_se**2
            valid = pooled_var > 1e-10

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


def _dif_lord(
    ref_result: "FitResult",
    focal_result: "FitResult",
    n_items: int,
) -> dict[str, NDArray[np.float64]]:
    """Lord's chi-square test for DIF."""
    return _dif_wald(ref_result, focal_result, n_items)


def _dif_raju(
    ref_result: "FitResult",
    focal_result: "FitResult",
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
