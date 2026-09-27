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


_MOMENT_CHUNK_ELEMENTS = 262_144


@dataclass(frozen=True)
class _ScoreMoments:
    """Final score moments and pairwise means on the same observations."""

    univariate: NDArray[np.float64]
    bivariate: NDArray[np.float64]
    correlation: NDArray[np.float64]
    pair_means: NDArray[np.float64]


class _SampleMomentAccumulator:
    """Accumulate score moments with shared pairwise observation counts."""

    def __init__(self, n_items: int) -> None:
        self.sums = np.zeros(n_items)
        self.products = np.zeros((n_items, n_items))
        self.pair_sums = np.zeros_like(self.products)
        self.pair_seconds = np.zeros_like(self.products)

    def add(
        self,
        means: NDArray[np.float64],
        seconds: NDArray[np.float64],
        mask: NDArray[np.float64] | None,
    ) -> None:
        if mask is not None:
            means = means * mask
            seconds = seconds * mask
        sums = means.sum(axis=0)
        self.sums += sums
        self.products += means.T @ means
        if mask is None:
            self.pair_sums += sums[:, None]
            self.pair_seconds += seconds.sum(axis=0)[:, None]
        else:
            self.pair_sums += means.T @ mask
            self.pair_seconds += seconds.T @ mask

    def finish(self, pair_counts: NDArray[np.float64]) -> _ScoreMoments:
        univariate = _safe_divide(self.sums, pair_counts.diagonal())
        bivariate = _safe_divide(self.products, pair_counts)
        pair_means = _safe_divide(self.pair_sums, pair_counts)
        pair_seconds = _safe_divide(self.pair_seconds, pair_counts)
        return _ScoreMoments(
            univariate,
            bivariate,
            _score_correlations(bivariate, pair_means, pair_seconds),
            pair_means,
        )


def _score_correlations(
    bivariate: NDArray[np.float64],
    pair_means: NDArray[np.float64],
    pair_seconds: NDArray[np.float64],
) -> NDArray[np.float64]:
    covariance = bivariate - pair_means * pair_means.T
    variances = np.maximum(pair_seconds - pair_means**2, 0.0)
    return _safe_divide(covariance, np.sqrt(variances * variances.T))


@dataclass(frozen=True)
class _FitMoments:
    """Observed and model-implied first- and second-order score moments."""

    observed_uni: NDArray[np.float64]
    observed_bi: NDArray[np.float64]
    expected_uni: NDArray[np.float64]
    expected_bi: NDArray[np.float64]
    observed_corr: NDArray[np.float64]
    expected_corr: NDArray[np.float64]
    observed_pair_means: NDArray[np.float64]
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
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
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
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    n_persons = response_values.shape[0]
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
        theta,
        n_quadpts,
    )
    m2_result = _m2_from_moments(model, moments)
    M2 = m2_result["M2"]
    df = m2_result["df"]

    M2_0, df_0 = _baseline_m2(
        moments.observed_bi, moments.observed_pair_means, moments.pair_counts
    )

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


def _compute_expected_margins(
    model: BaseItemModel,
    n_quadpts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected score moments under the model using quadrature."""
    moments, _ = _integrate_model_moments(model, n_quadpts)
    return moments.univariate, moments.bivariate


def _count_model_parameters(model: BaseItemModel) -> int:
    """Count number of estimated parameters in the model."""
    return int(model.n_parameters)


def _baseline_m2(
    observed_bi: NDArray[np.float64],
    pair_means: NDArray[np.float64],
    pair_counts: NDArray[np.float64],
) -> tuple[float, int]:
    """Compute independence-model M2 from the existing observed moments."""
    expected_bi = pair_means * pair_means.T
    upper = np.triu_indices_from(pair_counts, k=1)
    usable = pair_counts[upper] > 0
    residuals = observed_bi[upper][usable] - expected_bi[upper][usable]
    counts = pair_counts[upper][usable]
    return float(np.dot(counts, residuals**2)), int(np.sum(usable))


def _validate_diagnostic_inputs(
    model: BaseItemModel,
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], float]:
    """Validate response blocks without retaining a full mask or float copy."""
    values = np.asarray(responses)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            "responses must be a two-dimensional matrix with at least "
            "2 persons and 1 item"
        )
    if values.shape[1] != model.n_items:
        raise ValueError(
            f"responses have {values.shape[1]} items, expected {model.n_items}"
        )
    rows_per_chunk = max(1, _MOMENT_CHUNK_ELEMENTS // model.n_items)
    max_observed = -1.0
    for start in range(0, values.shape[0], rows_per_chunk):
        block = np.asarray(values[start : start + rows_per_chunk], dtype=np.float64)
        if np.any(np.isinf(block)):
            raise ValueError("responses must not contain infinite values")
        observed = block[np.isfinite(block) & (block >= 0)]
        if np.any(observed != np.floor(observed)):
            raise ValueError("observed responses must be integer category codes")
        max_observed = max(max_observed, float(np.max(observed, initial=-1.0)))
    if max_observed < 0:
        raise ValueError("responses contain no observed values")
    return values, max_observed


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


def _moment_rows_per_chunk(model: BaseItemModel) -> int:
    """Budget probability storage by the widest item category count."""
    width = max(model.n_categories) if model.is_polytomous else 1
    return max(1, _MOMENT_CHUNK_ELEMENTS // (model.n_items * width))


def _integrate_model_moments(
    model: BaseItemModel, n_quadpts: int
) -> tuple[_ScoreMoments, int]:
    """Integrate model moments without retaining full-grid probability arrays."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    _validate_quadrature_count(n_quadpts)
    quadrature = GaussHermiteQuadrature(
        n_points=n_quadpts, n_dimensions=model.n_factors
    )
    weights = _normalized_weights(quadrature.weights)
    univariate = np.zeros(model.n_items)
    seconds = np.zeros(model.n_items)
    bivariate = np.zeros((model.n_items, model.n_items))
    rows_per_chunk = _moment_rows_per_chunk(model)
    max_score = 0
    for start in range(0, weights.size, rows_per_chunk):
        stop = start + rows_per_chunk
        means, conditional_seconds, max_score = _conditional_score_moments(
            model, quadrature.nodes[start:stop]
        )
        block_weights = weights[start:stop]
        univariate += block_weights @ means
        seconds += block_weights @ conditional_seconds
        bivariate += (means * block_weights[:, None]).T @ means
    pair_means = univariate[:, None]
    return _ScoreMoments(
        univariate,
        bivariate,
        _score_correlations(bivariate, pair_means, seconds[:, None]),
        pair_means,
    ), max_score


def _prepare_fit_moments(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    max_observed: float,
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
) -> _FitMoments:
    """Stream moments, sharing observation counts and complete-block shortcuts."""
    observed = _SampleMomentAccumulator(model.n_items)
    expected = _SampleMomentAccumulator(model.n_items) if theta is not None else None
    pair_counts = np.zeros((model.n_items, model.n_items))
    theta_values = (
        _prepare_theta(model, theta, responses.shape[0]) if theta is not None else None
    )
    if theta is None:
        expected_moments, max_score = _integrate_model_moments(model, n_quadpts)

    rows_per_chunk = (
        _moment_rows_per_chunk(model)
        if theta is not None
        else max(1, _MOMENT_CHUNK_ELEMENTS // model.n_items)
    )
    for start in range(0, responses.shape[0], rows_per_chunk):
        stop = start + rows_per_chunk
        block = np.asarray(responses[start:stop], dtype=np.float64)
        valid = np.isfinite(block) & (block >= 0)
        mask = None
        if np.all(valid):
            pair_counts += block.shape[0]
            scores = block
        else:
            mask = valid.astype(np.float64)
            pair_counts += mask.T @ mask
            scores = np.where(valid, block, 0.0)
        observed.add(scores, scores**2, mask)
        if expected is not None and theta_values is not None:
            means, seconds, max_score = _conditional_score_moments(
                model, theta_values[start:stop]
            )
            expected.add(means, seconds, mask)
        if max_observed > max_score:
            raise ValueError(
                f"observed response categories must be between 0 and {max_score}"
            )

    observed_moments = observed.finish(pair_counts)
    if expected is not None:
        expected_moments = expected.finish(pair_counts)
    return _FitMoments(
        observed_uni=observed_moments.univariate,
        observed_bi=observed_moments.bivariate,
        expected_uni=expected_moments.univariate,
        expected_bi=expected_moments.bivariate,
        observed_corr=observed_moments.correlation,
        expected_corr=expected_moments.correlation,
        observed_pair_means=observed_moments.pair_means,
        uni_counts=pair_counts.diagonal(),
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
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
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
