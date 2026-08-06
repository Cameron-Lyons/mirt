"""Model fit statistics for IRT models.

This module provides limited-information goodness-of-fit statistics:
- M2 statistic (Maydeu-Olivares & Joe, 2005)
- RMSEA (Root Mean Square Error of Approximation)
- CFI (Comparative Fit Index)
- TLI (Tucker-Lewis Index)
- SRMSR (Standardized Root Mean Square Residual)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass(frozen=True)
class _FitMoments:
    """Observed and model-implied first- and second-order score moments."""

    observed_uni: NDArray[np.float64]
    observed_bi: NDArray[np.float64]
    expected_uni: NDArray[np.float64]
    expected_bi: NDArray[np.float64]
    observed_corr: NDArray[np.float64]
    expected_corr: NDArray[np.float64]
    uni_counts: NDArray[np.float64]
    pair_counts: NDArray[np.float64]


def compute_m2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> dict[str, float]:
    """Compute M2 limited-information fit statistic.

    The statistic tests whether the model reproduces first- and second-order
    score moments. For polytomous items these are the collapsed score moments
    used by the M2* formulation.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix (n_persons, n_items)
    theta : NDArray, optional
        Person ability estimates used as the empirical latent distribution.
        If omitted, expected moments are integrated by quadrature.
    n_quadpts : int
        Number of quadrature points for integration

    Returns
    -------
    dict
        Dictionary with:
        - 'M2': M2 statistic value
        - 'df': Degrees of freedom
        - 'p_value': P-value
        - 'M2_df_ratio': M2/df ratio
    """
    response_values, valid_mask = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        valid_mask,
        theta,
        n_quadpts,
    )
    return _m2_from_moments(model, moments)


def compute_fit_indices(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> dict[str, float]:
    """Compute model fit indices (RMSEA, CFI, TLI, SRMSR).

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Person ability estimates used as the empirical latent distribution.
        If omitted, expected moments are integrated by quadrature.
    n_quadpts : int
        Number of quadrature points

    Returns
    -------
    dict
        Dictionary with:
        - 'RMSEA': Root Mean Square Error of Approximation
        - 'RMSEA_CI_lower': Lower bound of 90% CI for RMSEA
        - 'RMSEA_CI_upper': Upper bound of 90% CI for RMSEA
        - 'CFI': Comparative Fit Index
        - 'TLI': Tucker-Lewis Index (NNFI)
        - 'SRMSR': Standardized Root Mean Square Residual
    """
    response_values, valid_mask = _validate_diagnostic_inputs(model, responses)
    n_persons = response_values.shape[0]
    moments = _prepare_fit_moments(
        model,
        response_values,
        valid_mask,
        theta,
        n_quadpts,
    )
    m2_result = _m2_from_moments(model, moments)
    M2 = m2_result["M2"]
    df = m2_result["df"]

    M2_0, df_0 = _compute_baseline_m2(response_values)

    rmsea = _compute_rmsea(M2, df, n_persons)
    rmsea_ci = _compute_rmsea_ci(M2, df, n_persons)

    cfi = _compute_cfi(M2, df, M2_0, df_0)

    tli = _compute_tli(M2, df, M2_0, df_0)

    srmsr = _srmsr_from_moments(moments)

    return {
        "RMSEA": rmsea,
        "RMSEA_CI_lower": rmsea_ci[0],
        "RMSEA_CI_upper": rmsea_ci[1],
        "CFI": cfi,
        "TLI": tli,
        "SRMSR": srmsr,
        "M2": M2,
        "M2_df": df,
        "M2_p": m2_result["p_value"],
    }


def _compute_observed_margins(
    responses: NDArray[np.int_],
    valid_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute observed univariate and bivariate score moments."""
    values = np.where(valid_mask, responses, 0.0).astype(np.float64, copy=False)
    observed_uni, observed_bi, _, _, _ = _sample_score_moments(
        values,
        values**2,
        valid_mask,
    )
    return observed_uni, observed_bi


def _compute_expected_margins(
    model: BaseItemModel,
    n_quadpts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected score moments under the model using quadrature."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    _validate_quadrature_count(n_quadpts)
    quad = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=model.n_factors)
    weights = _normalized_weights(quad.weights)
    expected_scores, expected_squares, _ = _conditional_score_moments(model, quad.nodes)
    expected_uni, expected_bi, _ = _population_score_moments(
        expected_scores,
        expected_squares,
        weights,
    )
    return expected_uni, expected_bi


def _count_model_parameters(model: BaseItemModel) -> int:
    """Count number of estimated parameters in the model."""
    return int(model.n_parameters)


def _compute_baseline_m2(responses: NDArray[np.int_]) -> tuple[float, int]:
    """Compute M2 for baseline (independence) model."""
    valid_mask = np.isfinite(responses) & (responses >= 0)
    mask = valid_mask.astype(np.float64)
    values = np.where(valid_mask, responses, 0.0).astype(np.float64, copy=False)
    pair_counts = mask.T @ mask
    observed_bi = _safe_divide(values.T @ values, pair_counts)
    pair_sums = values.T @ mask
    pair_mean_left = _safe_divide(pair_sums, pair_counts)
    expected_bi = pair_mean_left * pair_mean_left.T

    upper = np.triu_indices(responses.shape[1], k=1)
    usable = pair_counts[upper] > 0
    residuals = observed_bi[upper][usable] - expected_bi[upper][usable]
    counts = pair_counts[upper][usable]
    return float(np.dot(counts, residuals**2)), int(np.sum(usable))


def _validate_diagnostic_inputs(
    model: BaseItemModel,
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Validate response data and return values plus an observed-data mask."""
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            "responses must be a two-dimensional matrix with at least "
            "2 persons and 1 item"
        )
    if values.shape[1] != model.n_items:
        raise ValueError(
            f"responses have {values.shape[1]} items, expected {model.n_items}"
        )
    if np.any(np.isinf(values)):
        raise ValueError("responses must not contain infinite values")

    valid_mask = np.isfinite(values) & (values >= 0)
    observed = values[valid_mask]
    if observed.size == 0:
        raise ValueError("responses contain no observed values")
    if np.any(observed != np.floor(observed)):
        raise ValueError("observed responses must be integer category codes")
    return values, valid_mask


def _prepare_theta(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_persons: int,
) -> NDArray[np.float64]:
    """Normalize and validate person ability values."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim == 1 and model.n_factors == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("theta must be a two-dimensional ability matrix")
    if values.shape != (n_persons, model.n_factors):
        raise ValueError(
            f"theta must have shape ({n_persons}, {model.n_factors}), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain only finite values")
    return values


def _validate_quadrature_count(n_quadpts: int) -> None:
    """Validate the requested quadrature resolution."""
    if isinstance(n_quadpts, bool) or not isinstance(n_quadpts, (int, np.integer)):
        raise ValueError("n_quadpts must be an integer")
    if n_quadpts < 2:
        raise ValueError("n_quadpts must be at least 2")


def _normalized_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return finite nonnegative integration weights summing to one."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(values))
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or total <= 0
    ):
        raise ValueError("quadrature weights must be finite and nonnegative")
    return values / total


def _conditional_score_moments(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Compute conditional item score means and second moments in one pass."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    n_rows = theta.shape[0]

    if probabilities.ndim == 1 and model.n_items == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.ndim == 2:
        if probabilities.shape != (n_rows, model.n_items):
            raise ValueError(
                f"model probability output must have shape ({n_rows}, {model.n_items})"
            )
        expected_scores = probabilities
        expected_squares = probabilities
        max_score = 1
    elif probabilities.ndim == 3:
        if probabilities.shape[:2] != (n_rows, model.n_items):
            raise ValueError(
                "polytomous probability output must start with shape "
                f"({n_rows}, {model.n_items})"
            )
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        expected_scores = probabilities @ category_scores
        expected_squares = probabilities @ category_scores**2
        max_score = probabilities.shape[2] - 1
        if not np.allclose(np.sum(probabilities, axis=2), 1.0, atol=1e-8):
            raise ValueError("polytomous probabilities must sum to one")
    else:
        raise ValueError("model probability output has an unsupported shape")

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -PROB_EPSILON)
        or np.any(probabilities > 1.0 + PROB_EPSILON)
    ):
        raise ValueError("model probabilities must be finite and between zero and one")
    return expected_scores, expected_squares, max_score


def _safe_divide(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Divide arrays while marking unsupported moments as missing."""
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def _sample_score_moments(
    conditional_means: NDArray[np.float64],
    conditional_seconds: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Aggregate score moments over persons with pairwise missingness."""
    mask = valid_mask.astype(np.float64)
    means = conditional_means * mask
    seconds = conditional_seconds * mask
    uni_counts = np.sum(mask, axis=0)
    pair_counts = mask.T @ mask

    univariate = _safe_divide(np.sum(means, axis=0), uni_counts)
    bivariate = _safe_divide(means.T @ means, pair_counts)

    pair_mean_left = _safe_divide(means.T @ mask, pair_counts)
    pair_second_left = _safe_divide(seconds.T @ mask, pair_counts)
    covariance = bivariate - pair_mean_left * pair_mean_left.T
    variance_left = np.maximum(pair_second_left - pair_mean_left**2, 0.0)
    denominator = np.sqrt(variance_left * variance_left.T)
    correlation = _safe_divide(covariance, denominator)
    return univariate, bivariate, correlation, uni_counts, pair_counts


def _population_score_moments(
    conditional_means: NDArray[np.float64],
    conditional_seconds: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Integrate score moments over a latent population distribution."""
    if weights.size != conditional_means.shape[0]:
        raise ValueError("quadrature weights and probability rows must have equal size")
    univariate = weights @ conditional_means
    bivariate = (conditional_means * weights[:, None]).T @ conditional_means
    second_moments = weights @ conditional_seconds
    covariance = bivariate - np.outer(univariate, univariate)
    variances = np.maximum(second_moments - univariate**2, 0.0)
    denominator = np.sqrt(np.outer(variances, variances))
    correlation = _safe_divide(covariance, denominator)
    return univariate, bivariate, correlation


def _prepare_fit_moments(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
) -> _FitMoments:
    """Compute observed and expected moments with one probability evaluation."""
    observed_scores = np.where(valid_mask, responses, 0.0)
    observed_uni, observed_bi, observed_corr, uni_counts, pair_counts = (
        _sample_score_moments(
            observed_scores,
            observed_scores**2,
            valid_mask,
        )
    )

    if theta is None:
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        _validate_quadrature_count(n_quadpts)
        quadrature = GaussHermiteQuadrature(
            n_points=n_quadpts,
            n_dimensions=model.n_factors,
        )
        expected_scores, expected_squares, max_score = _conditional_score_moments(
            model, quadrature.nodes
        )
        expected_uni, expected_bi, expected_corr = _population_score_moments(
            expected_scores,
            expected_squares,
            _normalized_weights(quadrature.weights),
        )
    else:
        theta_values = _prepare_theta(model, theta, responses.shape[0])
        expected_scores, expected_squares, max_score = _conditional_score_moments(
            model, theta_values
        )
        expected_uni, expected_bi, expected_corr, _, _ = _sample_score_moments(
            expected_scores,
            expected_squares,
            valid_mask,
        )

    if np.any(responses[valid_mask] > max_score):
        raise ValueError(
            f"observed response categories must be between 0 and {max_score}"
        )
    return _FitMoments(
        observed_uni=observed_uni,
        observed_bi=observed_bi,
        expected_uni=expected_uni,
        expected_bi=expected_bi,
        observed_corr=observed_corr,
        expected_corr=expected_corr,
        uni_counts=uni_counts,
        pair_counts=pair_counts,
    )


def _m2_from_moments(
    model: BaseItemModel,
    moments: _FitMoments,
) -> dict[str, float]:
    """Compute the limited-information statistic from prepared moments."""
    usable_uni = (
        (moments.uni_counts > 0)
        & np.isfinite(moments.observed_uni)
        & np.isfinite(moments.expected_uni)
    )
    upper = np.triu_indices(model.n_items, k=1)
    usable_pairs = (
        (moments.pair_counts[upper] > 0)
        & np.isfinite(moments.observed_bi[upper])
        & np.isfinite(moments.expected_bi[upper])
    )
    residuals = np.concatenate(
        [
            (moments.observed_uni - moments.expected_uni)[usable_uni],
            (moments.observed_bi[upper] - moments.expected_bi[upper])[usable_pairs],
        ]
    )
    counts = np.concatenate(
        [
            moments.uni_counts[usable_uni],
            moments.pair_counts[upper][usable_pairs],
        ]
    )
    if residuals.size == 0:
        raise ValueError(
            "responses contain no estimable first- or second-order moments"
        )

    statistic = float(np.dot(counts, residuals**2))
    degrees_of_freedom = max(residuals.size - _count_model_parameters(model), 1)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return {
        "M2": statistic,
        "df": degrees_of_freedom,
        "p_value": p_value,
        "M2_df_ratio": statistic / degrees_of_freedom,
    }


def _srmsr_from_moments(moments: _FitMoments) -> float:
    """Compute SRMSR from finite pairwise score correlations."""
    upper = np.triu_indices_from(moments.observed_corr, k=1)
    residuals = moments.observed_corr[upper] - moments.expected_corr[upper]
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(residuals**2)))


def _compute_rmsea(chi2: float, df: int, n: int) -> float:
    """Compute RMSEA."""
    if df <= 0:
        return np.nan

    rmsea_sq = max((chi2 / df - 1) / (n - 1), 0)
    return float(np.sqrt(rmsea_sq))


def _compute_rmsea_ci(
    chi2: float,
    df: int,
    n: int,
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Compute confidence interval for RMSEA."""
    if df <= 0:
        return (np.nan, np.nan)

    def rmsea_from_ncp(ncp: float) -> float:
        return np.sqrt(max(ncp / (df * (n - 1)), 0))

    from scipy.optimize import brentq

    try:
        central_survival = float(stats.chi2.sf(chi2, df))

        def solve_ncp(target_survival: float) -> float:
            if central_survival >= target_survival:
                return 0.0
            upper = max(float(chi2), float(df), 1.0)
            while stats.ncx2.sf(chi2, df, upper) < target_survival:
                upper *= 2.0
                if upper > 1e8:
                    raise RuntimeError("could not bracket RMSEA noncentrality")
            return float(
                brentq(
                    lambda ncp: stats.ncx2.sf(chi2, df, ncp) - target_survival,
                    0.0,
                    upper,
                )
            )

        ncp_lower = solve_ncp(alpha / 2)
        lower = rmsea_from_ncp(ncp_lower)
        ncp_upper = solve_ncp(1 - alpha / 2)
        upper = rmsea_from_ncp(ncp_upper)

    except (ValueError, RuntimeError):
        se = np.sqrt(2 / (n - 1))
        rmsea = _compute_rmsea(chi2, df, n)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = max(rmsea - z * se, 0)
        upper = rmsea + z * se

    return (float(lower), float(upper))


def _compute_cfi(chi2: float, df: int, chi2_0: float, df_0: int) -> float:
    """Compute Comparative Fit Index."""
    if df_0 <= 0:
        return np.nan

    numerator = max(chi2 - df, 0)
    denominator = max(chi2_0 - df_0, chi2 - df, 0)

    if denominator <= 0:
        return 1.0

    cfi = 1 - numerator / denominator
    return float(np.clip(cfi, 0, 1))


def _compute_tli(chi2: float, df: int, chi2_0: float, df_0: int) -> float:
    """Compute Tucker-Lewis Index (NNFI)."""
    if df_0 <= 0 or df <= 0:
        return np.nan

    ratio_0 = chi2_0 / df_0
    ratio = chi2 / df

    if ratio_0 <= 1:
        return 1.0

    tli = (ratio_0 - ratio) / (ratio_0 - 1)
    return float(tli)


def _compute_srmsr(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_quadpts: int,
    theta: NDArray[np.float64] | None = None,
) -> float:
    """Compute Standardized Root Mean Square Residual."""
    response_values, valid_mask = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        valid_mask,
        theta,
        n_quadpts,
    )
    return _srmsr_from_moments(moments)


def model_fit_summary(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> str:
    """Generate a formatted summary of model fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates

    Returns
    -------
    str
        Formatted summary string
    """
    fit = compute_fit_indices(model, responses, theta)

    lines = [
        "Model Fit Summary",
        "=" * 50,
        "",
        f"M2 statistic:     {fit['M2']:.3f}",
        f"Degrees of freedom: {fit['M2_df']}",
        f"P-value:          {fit['M2_p']:.4f}",
        "",
        f"RMSEA:            {fit['RMSEA']:.4f}",
        f"  90% CI:         [{fit['RMSEA_CI_lower']:.4f}, {fit['RMSEA_CI_upper']:.4f}]",
        f"CFI:              {fit['CFI']:.4f}",
        f"TLI:              {fit['TLI']:.4f}",
        f"SRMSR:            {fit['SRMSR']:.4f}",
        "",
        "Interpretation guidelines:",
        "  RMSEA < 0.05: Good fit",
        "  RMSEA < 0.08: Acceptable fit",
        "  CFI > 0.95: Good fit",
        "  TLI > 0.95: Good fit",
        "  SRMSR < 0.08: Good fit",
    ]

    return "\n".join(lines)
