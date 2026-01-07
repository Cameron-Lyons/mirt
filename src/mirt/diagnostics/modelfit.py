"""Model fit statistics for IRT models.

This module provides limited-information goodness-of-fit statistics:
- M2 statistic (Maydeu-Olivares & Joe, 2005)
- RMSEA (Root Mean Square Error of Approximation)
- CFI (Comparative Fit Index)
- TLI (Tucker-Lewis Index)
- SRMSR (Standardized Root Mean Square Residual)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def compute_m2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> dict[str, float]:
    """Compute M2 limited-information fit statistic.

    The M2 statistic tests whether the model reproduces the first and
    second order margins (item proportions and pairwise associations).

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix (n_persons, n_items)
    theta : NDArray, optional
        Ability estimates (if None, computed via quadrature)
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
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    valid_mask = responses >= 0

    obs_uni, obs_bi = _compute_observed_margins(responses, valid_mask)

    exp_uni, exp_bi = _compute_expected_margins(model, n_quadpts)

    r_uni = obs_uni - exp_uni

    r_bi = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            r_bi.append(obs_bi[i, j] - exp_bi[i, j])
    r_bi = np.array(r_bi)

    r = np.concatenate([r_uni, r_bi])

    W = np.eye(len(r)) * n_persons

    M2 = float(r @ W @ r)

    n_observed = n_items + n_items * (n_items - 1) // 2
    n_params = _count_model_parameters(model)
    df = max(n_observed - n_params, 1)

    p_value = 1 - stats.chi2.cdf(M2, df)

    return {
        "M2": M2,
        "df": df,
        "p_value": float(p_value),
        "M2_df_ratio": M2 / df if df > 0 else np.nan,
    }


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
        Ability estimates
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
    responses = np.asarray(responses)
    n_persons = responses.shape[0]

    m2_result = compute_m2(model, responses, theta, n_quadpts)
    M2 = m2_result["M2"]
    df = m2_result["df"]

    M2_0, df_0 = _compute_baseline_m2(responses)

    rmsea = _compute_rmsea(M2, df, n_persons)
    rmsea_ci = _compute_rmsea_ci(M2, df, n_persons)

    cfi = _compute_cfi(M2, df, M2_0, df_0)

    tli = _compute_tli(M2, df, M2_0, df_0)

    srmsr = _compute_srmsr(model, responses, n_quadpts)

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
    """Compute observed univariate and bivariate proportions."""
    n_persons, n_items = responses.shape

    obs_uni = np.zeros(n_items)
    for j in range(n_items):
        valid_j = valid_mask[:, j]
        if valid_j.any():
            obs_uni[j] = responses[valid_j, j].mean()

    obs_bi = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid_ij = valid_mask[:, i] & valid_mask[:, j]
            if valid_ij.any():
                obs_bi[i, j] = (responses[valid_ij, i] * responses[valid_ij, j]).mean()
                obs_bi[j, i] = obs_bi[i, j]

    return obs_uni, obs_bi


def _compute_expected_margins(
    model: BaseItemModel,
    n_quadpts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected margins under the model using quadrature."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    n_items = model.n_items

    quad = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=model.n_factors)
    nodes = quad.nodes
    weights = quad.weights

    exp_uni = np.zeros(n_items)
    for j in range(n_items):
        probs = model.probability(nodes, j)
        exp_uni[j] = np.sum(weights * probs)

    exp_bi = np.zeros((n_items, n_items))
    for i in range(n_items):
        p_i = model.probability(nodes, i)
        for j in range(i + 1, n_items):
            p_j = model.probability(nodes, j)
            exp_bi[i, j] = np.sum(weights * p_i * p_j)
            exp_bi[j, i] = exp_bi[i, j]

    return exp_uni, exp_bi


def _count_model_parameters(model: BaseItemModel) -> int:
    """Count number of estimated parameters in the model."""
    n_params = 0
    for name, values in model.parameters.items():
        n_params += values.size
    return n_params


def _compute_baseline_m2(responses: NDArray[np.int_]) -> tuple[float, int]:
    """Compute M2 for baseline (independence) model."""
    n_persons, n_items = responses.shape
    valid_mask = responses >= 0

    obs_uni = np.zeros(n_items)
    for j in range(n_items):
        valid_j = valid_mask[:, j]
        if valid_j.any():
            obs_uni[j] = responses[valid_j, j].mean()

    r_bi = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid_ij = valid_mask[:, i] & valid_mask[:, j]
            if valid_ij.any():
                obs_bij = (responses[valid_ij, i] * responses[valid_ij, j]).mean()
                exp_bij = obs_uni[i] * obs_uni[j]
                r_bi.append(obs_bij - exp_bij)

    r_bi = np.array(r_bi)
    M2_0 = float(n_persons * np.sum(r_bi**2))
    df_0 = n_items * (n_items - 1) // 2

    return M2_0, df_0


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
        if chi2 <= df:
            lower = 0.0
        else:

            def f_lower(ncp: float) -> float:
                return stats.ncx2.sf(chi2, df, ncp) - (1 - alpha / 2)

            ncp_lower = brentq(f_lower, 0, max(chi2 * 3, 100))
            lower = rmsea_from_ncp(ncp_lower)

        def f_upper(ncp: float) -> float:
            return stats.ncx2.sf(chi2, df, ncp) - alpha / 2

        ncp_upper = brentq(f_upper, 0, max(chi2 * 5, 200))
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
) -> float:
    """Compute Standardized Root Mean Square Residual."""
    n_items = model.n_items

    obs_corr = np.corrcoef(responses.T)

    exp_uni, exp_bi = _compute_expected_margins(model, n_quadpts)

    exp_corr = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                exp_corr[i, j] = 1.0
            else:
                p_i = exp_uni[i]
                p_j = exp_uni[j]
                p_ij = exp_bi[i, j] if i < j else exp_bi[j, i]

                num = p_ij - p_i * p_j
                denom = np.sqrt(p_i * (1 - p_i) * p_j * (1 - p_j))
                if denom > 1e-10:
                    exp_corr[i, j] = num / denom

    residuals = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if not np.isnan(obs_corr[i, j]):
                residuals.append((obs_corr[i, j] - exp_corr[i, j]) ** 2)

    if len(residuals) == 0:
        return np.nan

    return float(np.sqrt(np.mean(residuals)))


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
