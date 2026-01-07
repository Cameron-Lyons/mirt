"""Model comparison tools for IRT models.

This module provides methods for comparing fitted IRT models:
- Likelihood ratio tests (LRT) for nested models
- Information criteria comparison (AIC, BIC)
- Akaike weights
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


def anova_irt(
    *results: FitResult,
    method: str = "LRT",
) -> Any:
    """Compare nested IRT models using likelihood ratio test.

    Models should be ordered from simplest (most constrained) to most
    complex (least constrained).

    Parameters
    ----------
    *results : FitResult
        Two or more fitted model results to compare
    method : str
        Comparison method ('LRT' for likelihood ratio test)

    Returns
    -------
    DataFrame
        Comparison table with model fit statistics and test results

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> result_1pl = fit_mirt(data, model="1PL")
    >>> result_2pl = fit_mirt(data, model="2PL")
    >>> anova_irt(result_1pl, result_2pl)
    """
    from mirt.utils.dataframe import create_dataframe

    if len(results) < 2:
        raise ValueError("At least two models required for comparison")

    model_names = []
    log_likelihoods = []
    n_params_list = []
    aics = []
    bics = []

    for i, result in enumerate(results):
        model_names.append(f"Model {i + 1}: {result.model.model_name}")
        log_likelihoods.append(result.log_likelihood)
        n_params = sum(v.size for v in result.model.parameters.values())
        n_params_list.append(n_params)
        aics.append(result.aic)
        bics.append(result.bic)

    chi_sq = [np.nan]
    df_diff = [np.nan]
    p_values = [np.nan]

    for i in range(1, len(results)):
        ll_diff = 2 * (log_likelihoods[i] - log_likelihoods[i - 1])
        param_diff = n_params_list[i] - n_params_list[i - 1]

        if param_diff <= 0:
            chi_sq.append(np.nan)
            df_diff.append(np.nan)
            p_values.append(np.nan)
        else:
            chi_sq.append(ll_diff)
            df_diff.append(param_diff)
            p_values.append(1 - stats.chi2.cdf(ll_diff, param_diff))

    data = {
        "Model": model_names,
        "LogLik": log_likelihoods,
        "npar": n_params_list,
        "AIC": aics,
        "BIC": bics,
        "Chi-sq": chi_sq,
        "df": df_diff,
        "p-value": p_values,
    }

    return create_dataframe(data)


def compare_models(
    results: list[FitResult],
    criteria: list[str] | None = None,
) -> Any:
    """Compare multiple IRT models using information criteria.

    Useful for comparing non-nested models.

    Parameters
    ----------
    results : list of FitResult
        Fitted model results to compare
    criteria : list of str, optional
        Information criteria to compute. Options:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'SABIC': Sample-size Adjusted BIC
        Default: ['AIC', 'BIC']

    Returns
    -------
    DataFrame
        Comparison table with information criteria and weights
    """
    from mirt.utils.dataframe import create_dataframe

    if criteria is None:
        criteria = ["AIC", "BIC"]

    n_models = len(results)
    model_names = []
    log_likelihoods = []
    n_params_list = []
    n_obs_list = []

    for i, result in enumerate(results):
        model_names.append(f"Model {i + 1}: {result.model.model_name}")
        log_likelihoods.append(result.log_likelihood)
        n_params = sum(v.size for v in result.model.parameters.values())
        n_params_list.append(n_params)
        n_obs_list.append(result.n_observations)

    data: dict[str, Any] = {
        "Model": model_names,
        "LogLik": log_likelihoods,
        "npar": n_params_list,
    }

    for criterion in criteria:
        values = []
        for i in range(n_models):
            ll = log_likelihoods[i]
            k = n_params_list[i]
            n = n_obs_list[i] if n_obs_list[i] > 0 else 1

            if criterion == "AIC":
                values.append(-2 * ll + 2 * k)
            elif criterion == "BIC":
                values.append(-2 * ll + k * np.log(n))
            elif criterion == "SABIC":
                values.append(-2 * ll + k * np.log((n + 2) / 24))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

        data[criterion] = values

        min_val = min(values)
        data[f"d{criterion}"] = [v - min_val for v in values]

        deltas = np.array(data[f"d{criterion}"])
        weights = np.exp(-0.5 * deltas)
        weights = weights / weights.sum()
        data[f"w{criterion}"] = weights.tolist()

    return create_dataframe(data)


def vuong_test(
    result1: FitResult,
    result2: FitResult,
    responses: NDArray[np.int_],
) -> dict[str, float]:
    """Vuong test for non-nested model comparison.

    Tests whether two models are equally close to the true data generating
    process versus one being closer.

    Parameters
    ----------
    result1, result2 : FitResult
        Two fitted model results
    responses : NDArray
        Response matrix used to fit the models

    Returns
    -------
    dict
        Dictionary with:
        - 'z': Vuong test statistic
        - 'p_value': Two-sided p-value
        - 'preferred': Name of preferred model (or 'neither')
    """
    responses = np.asarray(responses)
    n_persons = responses.shape[0]

    ll1 = _compute_person_loglik(result1.model, responses)
    ll2 = _compute_person_loglik(result2.model, responses)

    diff = ll1 - ll2

    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)

    if var_diff < 1e-10:
        return {
            "z": 0.0,
            "p_value": 1.0,
            "preferred": "neither",
        }

    z = np.sqrt(n_persons) * mean_diff / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    if p_value < 0.05:
        if z > 0:
            preferred = result1.model.model_name
        else:
            preferred = result2.model.model_name
    else:
        preferred = "neither"

    return {
        "z": float(z),
        "p_value": float(p_value),
        "preferred": preferred,
    }


def _compute_person_loglik(
    model: Any,
    responses: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute log-likelihood for each person."""
    from mirt.scoring import fscores

    n_persons, n_items = responses.shape

    scores = fscores(model, responses, method="EAP")
    theta = scores.theta
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    return model.log_likelihood(responses, theta)


def information_criteria(
    result: FitResult,
    n_obs: int | None = None,
) -> dict[str, float]:
    """Compute various information criteria for a fitted model.

    Parameters
    ----------
    result : FitResult
        Fitted model result
    n_obs : int, optional
        Number of observations (if not in result)

    Returns
    -------
    dict
        Dictionary with AIC, BIC, SABIC, AICc, CAIC
    """
    ll = result.log_likelihood
    k = sum(v.size for v in result.model.parameters.values())
    n = n_obs if n_obs is not None else result.n_observations

    if n <= 0:
        n = 1

    aic = -2 * ll + 2 * k

    bic = -2 * ll + k * np.log(n)

    sabic = -2 * ll + k * np.log((n + 2) / 24)

    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = np.inf

    caic = -2 * ll + k * (np.log(n) + 1)

    return {
        "AIC": float(aic),
        "BIC": float(bic),
        "SABIC": float(sabic),
        "AICc": float(aicc),
        "CAIC": float(caic),
        "-2LogLik": float(-2 * ll),
        "npar": k,
    }


def relative_fit(
    results: list[FitResult],
    criterion: str = "AIC",
) -> dict[str, Any]:
    """Compute relative fit measures across models.

    Parameters
    ----------
    results : list of FitResult
        Fitted models to compare
    criterion : str
        Information criterion to use

    Returns
    -------
    dict
        Dictionary with model rankings, evidence ratios, and probabilities
    """
    ic_values = []
    for result in results:
        ic = information_criteria(result)
        ic_values.append(ic[criterion])

    ic_values = np.array(ic_values)

    min_ic = ic_values.min()
    delta_ic = ic_values - min_ic

    weights = np.exp(-0.5 * delta_ic)
    weights = weights / weights.sum()

    best_idx = int(np.argmin(ic_values))
    evidence_ratios = weights[best_idx] / weights

    rankings = stats.rankdata(ic_values, method="ordinal").astype(int)

    return {
        "criterion_values": ic_values.tolist(),
        "delta": delta_ic.tolist(),
        "weights": weights.tolist(),
        "evidence_ratios": evidence_ratios.tolist(),
        "rankings": rankings.tolist(),
        "best_model_idx": best_idx,
    }
