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
from scipy import special, stats

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


def _parameter_count(result: FitResult) -> int:
    """Read fitted metadata, with compatibility for result-like objects."""
    count = getattr(result, "n_parameters", None)
    if count is not None:
        return int(count)

    model_count = getattr(result.model, "n_parameters", None)
    if model_count is not None:
        return int(model_count)
    return sum(np.asarray(values).size for values in result.model.parameters.values())


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
    if not isinstance(method, str) or method.upper() != "LRT":
        raise ValueError("method must be 'LRT'")

    model_names = []
    log_likelihoods = []
    n_params_list = []
    aics = []
    bics = []

    for i, result in enumerate(results):
        model_names.append(f"Model {i + 1}: {result.model.model_name}")
        log_likelihoods.append(result.log_likelihood)
        n_params = _parameter_count(result)
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
            p_values.append(stats.chi2.sf(ll_diff, param_diff))

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
        n_params = _parameter_count(result)
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
    *,
    alpha: float = 0.05,
    n_quadpts: int = 49,
) -> dict[str, float | str]:
    """Vuong test for non-nested model comparison.

    Tests whether two models are equally close to the true data generating
    process versus one being closer.

    Parameters
    ----------
    result1, result2 : FitResult
        Two fitted model results
    responses : NDArray
        Response matrix used to fit the models
    alpha : float, default=0.05
        Two-sided significance level used to select a preferred model
    n_quadpts : int, default=49
        Number of Gauss-Hermite points per latent dimension

    Returns
    -------
    dict
        Dictionary with:
        - 'z': Vuong test statistic
        - 'p_value': Two-sided p-value
        - 'preferred': Name of preferred model (or 'neither')
        - 'mean_log_likelihood_difference': Mean personwise difference
        - 'standard_error': Standard error of the mean difference

    Notes
    -----
    Personwise likelihoods are marginalized over a standard-normal latent
    distribution. This matches the default latent distribution used during
    model estimation.
    """
    from mirt.results._common import validate_alpha
    from mirt.scoring._common import validate_scoring_responses

    validated_alpha = validate_alpha(alpha)
    if (
        isinstance(n_quadpts, (bool, np.bool_))
        or not isinstance(n_quadpts, (int, np.integer))
        or n_quadpts < 5
    ):
        raise ValueError("n_quadpts must be an integer of at least 5")

    model1 = result1.model
    model2 = result2.model
    if model1.n_items != model2.n_items:
        raise ValueError("models must contain the same number of items")
    categories1 = (
        np.asarray(model1.n_categories, dtype=np.int_)
        if model1.is_polytomous
        else np.full(model1.n_items, 2, dtype=np.int_)
    )
    categories2 = (
        np.asarray(model2.n_categories, dtype=np.int_)
        if model2.is_polytomous
        else np.full(model2.n_items, 2, dtype=np.int_)
    )
    if not np.array_equal(categories1, categories2):
        raise ValueError("models must use the same response categories")

    validated_responses = validate_scoring_responses(model1, responses)
    # Validate category ranges against both candidate models. This matters for
    # comparisons between different polytomous model families.
    second_responses = validate_scoring_responses(model2, responses)
    if not np.array_equal(validated_responses, second_responses):
        raise ValueError("models must use the same response coding")

    n_persons = validated_responses.shape[0]
    if n_persons < 2:
        raise ValueError("responses must contain at least two persons")
    for result in (result1, result2):
        fitted_n = getattr(result, "n_observations", 0)
        if fitted_n > 0 and fitted_n != n_persons:
            raise ValueError(
                "responses must contain the observations used to fit both models"
            )

    ll1 = _compute_person_loglik(model1, validated_responses, n_quadpts)
    ll2 = _compute_person_loglik(model2, validated_responses, n_quadpts)

    diff = ll1 - ll2

    mean_diff = float(np.mean(diff))
    var_diff = float(np.var(diff, ddof=1))
    standard_error = float(np.sqrt(var_diff / n_persons))

    variance_tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.mean(np.square(diff)))
    )
    if var_diff <= variance_tolerance:
        return {
            "z": 0.0,
            "p_value": 1.0,
            "preferred": "neither",
            "mean_log_likelihood_difference": mean_diff,
            "standard_error": standard_error,
        }

    z = mean_diff / standard_error
    p_value = 2.0 * special.ndtr(-abs(z))

    if p_value < validated_alpha:
        if z > 0:
            preferred = model1.model_name
        else:
            preferred = model2.model_name
    else:
        preferred = "neither"

    return {
        "z": float(z),
        "p_value": float(p_value),
        "preferred": preferred,
        "mean_log_likelihood_difference": mean_diff,
        "standard_error": standard_error,
    }


def _compute_person_loglik(
    model: Any,
    responses: NDArray[np.int_],
    n_quadpts: int,
) -> NDArray[np.float64]:
    """Compute each person's marginal response-pattern log-likelihood."""
    from mirt.scoring._common import build_quadrature
    from mirt.utils.numeric import logsumexp_axis1

    quad_points, quad_weights = build_quadrature(
        n_quadpts=n_quadpts,
        n_factors=model.n_factors,
        prior_mean=None,
        prior_cov=None,
    )
    weights = np.asarray(quad_weights, dtype=np.float64)
    if (
        weights.ndim != 1
        or weights.shape[0] != quad_points.shape[0]
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("quadrature weights must be finite and positive")
    weights = weights / weights.sum()
    log_weights = np.log(weights)

    n_persons = responses.shape[0]
    n_nodes = quad_points.shape[0]
    # Bound the largest temporary likelihood matrix to roughly eight MiB.
    batch_size = max(1, 1_000_000 // n_nodes)
    marginal = np.empty(n_persons, dtype=np.float64)

    batch_method = getattr(model, "log_likelihood_batch", None)
    for start in range(0, n_persons, batch_size):
        stop = min(start + batch_size, n_persons)
        response_batch = responses[start:stop]

        if callable(batch_method):
            try:
                raw_log_likelihood = batch_method(response_batch, quad_points)
            except NotImplementedError:
                batch_method = None
                raw_log_likelihood = _conditional_loglik_grid(
                    model, response_batch, quad_points
                )
        else:
            raw_log_likelihood = _conditional_loglik_grid(
                model, response_batch, quad_points
            )

        log_likelihood = np.asarray(raw_log_likelihood, dtype=np.float64)
        expected_shape = (stop - start, n_nodes)
        if log_likelihood.shape != expected_shape:
            raise ValueError(
                "model.log_likelihood_batch returned shape "
                f"{log_likelihood.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(log_likelihood)):
            raise ValueError("model produced non-finite person log-likelihoods")

        marginal[start:stop] = logsumexp_axis1(log_likelihood + log_weights[None, :])

    if not np.all(np.isfinite(marginal)):
        raise ValueError("model produced non-finite marginal log-likelihoods")
    return marginal


def _conditional_loglik_grid(
    model: Any,
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a model without a vectorized quadrature-grid method."""
    n_persons = responses.shape[0]
    columns = []
    for point in quad_points:
        theta = np.broadcast_to(point, (n_persons, point.size))
        columns.append(np.asarray(model.log_likelihood(responses, theta)))
    return np.column_stack(columns)


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
    k = _parameter_count(result)
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
