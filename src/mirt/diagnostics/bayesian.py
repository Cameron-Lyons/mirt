"""Bayesian model diagnostics for IRT models.

This module provides:
- PSIS-LOO (Pareto-smoothed importance sampling leave-one-out cross-validation)
- WAIC (Widely Applicable Information Criterion)
- Posterior predictive checks
- DIC (Deviance Information Criterion)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.estimation.mcmc import MCMCResult
    from mirt.models.base import BaseItemModel


@dataclass
class PSISResult:
    """Result from PSIS-LOO cross-validation.

    Attributes
    ----------
    elpd_loo : float
        Expected log pointwise predictive density (sum over observations).
    p_loo : float
        Estimated effective number of parameters.
    looic : float
        LOO Information Criterion (-2 * elpd_loo).
    pointwise : ndarray of shape (n_observations,)
        Pointwise elpd_loo values.
    pareto_k : ndarray of shape (n_observations,)
        Pareto k diagnostic values for each observation.
    n_high_k : int
        Number of observations with k > 0.7 (potentially problematic).
    se_elpd : float
        Standard error of elpd_loo.
    """

    elpd_loo: float
    p_loo: float
    looic: float
    pointwise: NDArray[np.float64]
    pareto_k: NDArray[np.float64]
    n_high_k: int
    se_elpd: float

    def summary(self) -> str:
        """Generate summary of PSIS-LOO results."""
        lines = [
            "PSIS-LOO Cross-Validation",
            "=" * 40,
            f"elpd_loo: {self.elpd_loo:.2f} (SE = {self.se_elpd:.2f})",
            f"p_loo: {self.p_loo:.2f}",
            f"looic: {self.looic:.2f}",
            "",
            "Pareto k diagnostics:",
        ]

        k_thresholds = [0.5, 0.7, 1.0]
        for i, thresh in enumerate(k_thresholds):
            if i == 0:
                count = np.sum(self.pareto_k < thresh)
                lines.append(f"  k < {thresh}: {count} (good)")
            elif i == len(k_thresholds) - 1:
                count = np.sum(self.pareto_k >= k_thresholds[i - 1])
                lines.append(f"  k >= {k_thresholds[i - 1]}: {count} (problematic)")
            else:
                count = np.sum(
                    (self.pareto_k >= k_thresholds[i - 1]) & (self.pareto_k < thresh)
                )
                lines.append(f"  {k_thresholds[i - 1]} <= k < {thresh}: {count} (ok)")

        if self.n_high_k > 0:
            lines.append(
                f"\nWarning: {self.n_high_k} observations have k > 0.7. "
                "Consider using more posterior samples or K-fold CV."
            )

        return "\n".join(lines)


@dataclass
class WAICResult:
    """Result from WAIC computation.

    Attributes
    ----------
    waic : float
        WAIC value (deviance scale).
    elpd_waic : float
        Expected log pointwise predictive density.
    p_waic : float
        Estimated effective number of parameters.
    pointwise : ndarray of shape (n_observations,)
        Pointwise WAIC values.
    se_waic : float
        Standard error of WAIC.
    """

    waic: float
    elpd_waic: float
    p_waic: float
    pointwise: NDArray[np.float64]
    se_waic: float

    def summary(self) -> str:
        """Generate summary of WAIC results."""
        lines = [
            "WAIC (Widely Applicable Information Criterion)",
            "=" * 45,
            f"WAIC: {self.waic:.2f} (SE = {self.se_waic:.2f})",
            f"elpd_waic: {self.elpd_waic:.2f}",
            f"p_waic: {self.p_waic:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class PPCResult:
    """Result from posterior predictive check.

    Attributes
    ----------
    test_statistic_observed : float
        Test statistic computed on observed data.
    test_statistic_replicated : ndarray
        Test statistic computed on replicated datasets.
    p_value : float
        Posterior predictive p-value.
    summary_stats : dict
        Summary statistics of replicated test statistics.
    """

    test_statistic_observed: float
    test_statistic_replicated: NDArray[np.float64]
    p_value: float
    summary_stats: dict[str, float]

    def summary(self) -> str:
        """Generate summary of PPC results."""
        lines = [
            "Posterior Predictive Check",
            "=" * 40,
            f"Observed test statistic: {self.test_statistic_observed:.4f}",
            f"Replicated mean: {self.summary_stats['mean']:.4f}",
            f"Replicated SD: {self.summary_stats['std']:.4f}",
            f"Replicated 95% CI: [{self.summary_stats['q025']:.4f}, "
            f"{self.summary_stats['q975']:.4f}]",
            f"Posterior p-value: {self.p_value:.4f}",
        ]

        if self.p_value < 0.05 or self.p_value > 0.95:
            lines.append("\nWarning: Extreme p-value suggests model misfit.")

        return "\n".join(lines)


def _pareto_k_estimate(log_weights: NDArray[np.float64], min_tail: int = 10) -> float:
    """Estimate Pareto k parameter using the Zhang-Stephens method.

    Parameters
    ----------
    log_weights : ndarray
        Log importance weights.
    min_tail : int
        Minimum tail length for estimation.

    Returns
    -------
    float
        Estimated Pareto k parameter.
    """
    n = len(log_weights)
    if n < min_tail:
        return np.inf

    log_weights = np.sort(log_weights)

    log_cutoff = log_weights[-min_tail]
    tail = log_weights[log_weights > log_cutoff]

    if len(tail) < min_tail:
        tail = log_weights[-min_tail:]

    m = len(tail)
    if m < 2:
        return np.inf

    tail_shifted = tail - tail.min()
    if tail_shifted.max() == 0:
        return 0.0

    log_tail = np.log(tail_shifted[tail_shifted > 0] + PROB_EPSILON)

    k = np.mean(log_tail) - log_tail[0] if len(log_tail) > 1 else 0.0
    k = max(0.0, k)

    return k


def _pareto_smooth_weights(
    log_weights: NDArray[np.float64], k_threshold: float = 0.7
) -> tuple[NDArray[np.float64], float]:
    """Apply Pareto smoothing to importance weights.

    Parameters
    ----------
    log_weights : ndarray
        Log importance weights.
    k_threshold : float
        Threshold for Pareto k warning.

    Returns
    -------
    weights : ndarray
        Smoothed and normalized weights.
    k : float
        Estimated Pareto k value.
    """
    log_weights_centered = log_weights - np.max(log_weights)

    k = _pareto_k_estimate(log_weights_centered)

    weights = np.exp(log_weights_centered)
    weights = weights / weights.sum()

    return weights, k


def psis_loo(
    log_lik: NDArray[np.float64],
    k_threshold: float = 0.7,
) -> PSISResult:
    """Compute PSIS-LOO cross-validation.

    PSIS-LOO uses Pareto-smoothed importance sampling to estimate
    leave-one-out cross-validation without refitting the model.

    Parameters
    ----------
    log_lik : ndarray of shape (n_samples, n_observations)
        Pointwise log-likelihood matrix. Each row is a posterior sample,
        each column is an observation (person-item combination or person).
    k_threshold : float, default=0.7
        Threshold for Pareto k diagnostic warning.

    Returns
    -------
    PSISResult
        Results including elpd_loo, p_loo, looic, and diagnostics.

    Notes
    -----
    The implementation follows Vehtari et al. (2017) "Practical Bayesian
    model evaluation using leave-one-out cross-validation and WAIC".

    Pareto k values indicate reliability:
    - k < 0.5: Very reliable
    - 0.5 <= k < 0.7: Good
    - 0.7 <= k < 1.0: Unreliable, consider more samples
    - k >= 1.0: Very unreliable, use K-fold CV instead

    References
    ----------
    Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC.
    Statistics and Computing, 27(5), 1413-1432.
    """
    log_lik = np.asarray(log_lik)
    if log_lik.ndim == 1:
        log_lik = log_lik.reshape(-1, 1)

    n_samples, n_obs = log_lik.shape

    pointwise_elpd = np.zeros(n_obs)
    pareto_k = np.zeros(n_obs)

    for i in range(n_obs):
        log_ratios = -log_lik[:, i]

        weights, k = _pareto_smooth_weights(log_ratios, k_threshold)
        pareto_k[i] = k

        log_lik_i = log_lik[:, i]
        loo_elpd = np.log(np.sum(weights * np.exp(log_lik_i - np.max(log_lik_i))))
        loo_elpd += np.max(log_lik_i)
        pointwise_elpd[i] = loo_elpd

    elpd_loo = np.sum(pointwise_elpd)

    lppd = np.sum(np.log(np.mean(np.exp(log_lik), axis=0)))
    p_loo = lppd - elpd_loo

    looic = -2 * elpd_loo

    n_high_k = int(np.sum(pareto_k > k_threshold))

    se_elpd = float(np.sqrt(n_obs * np.var(pointwise_elpd)))

    return PSISResult(
        elpd_loo=float(elpd_loo),
        p_loo=float(p_loo),
        looic=float(looic),
        pointwise=pointwise_elpd,
        pareto_k=pareto_k,
        n_high_k=n_high_k,
        se_elpd=se_elpd,
    )


def waic(log_lik: NDArray[np.float64]) -> WAICResult:
    """Compute WAIC (Widely Applicable Information Criterion).

    WAIC uses the full posterior distribution to estimate out-of-sample
    predictive accuracy.

    Parameters
    ----------
    log_lik : ndarray of shape (n_samples, n_observations)
        Pointwise log-likelihood matrix.

    Returns
    -------
    WAICResult
        Results including WAIC, elpd_waic, and p_waic.

    Notes
    -----
    WAIC is computed as:

        WAIC = -2 * (lppd - p_waic)

    where:
    - lppd = sum over observations of log(mean over samples of likelihood)
    - p_waic = sum over observations of variance of log-likelihood

    References
    ----------
    Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation
    and widely applicable information criterion in singular learning theory.
    Journal of Machine Learning Research, 11, 3571-3594.
    """
    log_lik = np.asarray(log_lik)
    if log_lik.ndim == 1:
        log_lik = log_lik.reshape(-1, 1)

    n_samples, n_obs = log_lik.shape

    max_log_lik = np.max(log_lik, axis=0)
    lppd_i = np.log(np.mean(np.exp(log_lik - max_log_lik), axis=0)) + max_log_lik
    lppd = np.sum(lppd_i)

    p_waic_i = np.var(log_lik, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)

    elpd_waic = lppd - p_waic
    waic_val = -2 * elpd_waic

    pointwise = -2 * (lppd_i - p_waic_i)

    se_waic = float(np.sqrt(n_obs * np.var(pointwise)))

    return WAICResult(
        waic=float(waic_val),
        elpd_waic=float(elpd_waic),
        p_waic=float(p_waic),
        pointwise=pointwise,
        se_waic=se_waic,
    )


def dic(
    log_lik_at_mean: float,
    log_lik_samples: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute DIC (Deviance Information Criterion).

    Parameters
    ----------
    log_lik_at_mean : float
        Log-likelihood at posterior mean of parameters.
    log_lik_samples : ndarray of shape (n_samples,)
        Log-likelihood at each posterior sample.

    Returns
    -------
    dic : float
        DIC value.
    p_dic : float
        Effective number of parameters.
    """
    deviance_at_mean = -2 * log_lik_at_mean
    mean_deviance = -2 * np.mean(log_lik_samples)

    p_dic = mean_deviance - deviance_at_mean
    dic_val = deviance_at_mean + 2 * p_dic

    return float(dic_val), float(p_dic)


def posterior_predictive_check(
    mcmc_result: MCMCResult,
    responses: NDArray[np.int_],
    model: BaseItemModel,
    test_statistic: Callable[[NDArray[np.int_]], float] | str = "item_mean",
    n_rep: int | None = None,
    seed: int | None = None,
) -> PPCResult:
    """Perform posterior predictive check.

    Generates replicated data from the posterior and compares
    test statistics between observed and replicated data.

    Parameters
    ----------
    mcmc_result : MCMCResult
        Result from MCMC estimation containing chains.
    responses : ndarray of shape (n_persons, n_items)
        Observed response matrix.
    model : BaseItemModel
        IRT model used for simulation.
    test_statistic : callable or str, default='item_mean'
        Function computing test statistic from responses.
        Built-in options: 'item_mean', 'person_score', 'chi_square',
        'correlation', 'odds_ratio'.
    n_rep : int, optional
        Number of replications. Defaults to number of posterior samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PPCResult
        Results including observed/replicated statistics and p-value.
    """
    rng = np.random.default_rng(seed)
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if isinstance(test_statistic, str):
        test_statistic = _get_builtin_statistic(test_statistic, n_items)

    t_obs = test_statistic(responses)

    chains = mcmc_result.chains
    n_samples = len(next(iter(chains.values())))

    if n_rep is None:
        n_rep = n_samples

    sample_indices = rng.choice(n_samples, size=n_rep, replace=n_rep > n_samples)

    t_rep = np.zeros(n_rep)

    for rep_idx, sample_idx in enumerate(sample_indices):
        for param_name, chain in chains.items():
            if hasattr(model, f"_{param_name}"):
                setattr(model, f"_{param_name}", chain[sample_idx])
            elif param_name in model._parameters:
                model._parameters[param_name] = chain[sample_idx]

        theta_samples = chains.get(
            "theta", rng.standard_normal((n_persons, model.n_factors))
        )
        if theta_samples.ndim == 3:
            theta = theta_samples[sample_idx]
        else:
            theta = theta_samples

        probs = model.probability(theta)
        y_rep = (rng.random((n_persons, n_items)) < probs).astype(int)

        valid_mask = responses >= 0
        y_rep = np.where(valid_mask, y_rep, -1)

        t_rep[rep_idx] = test_statistic(y_rep)

    p_value = np.mean(t_rep >= t_obs)

    summary_stats = {
        "mean": float(np.mean(t_rep)),
        "std": float(np.std(t_rep)),
        "q025": float(np.percentile(t_rep, 2.5)),
        "q975": float(np.percentile(t_rep, 97.5)),
        "min": float(np.min(t_rep)),
        "max": float(np.max(t_rep)),
    }

    return PPCResult(
        test_statistic_observed=float(t_obs),
        test_statistic_replicated=t_rep,
        p_value=float(p_value),
        summary_stats=summary_stats,
    )


def _get_builtin_statistic(
    name: str, n_items: int
) -> Callable[[NDArray[np.int_]], float]:
    """Get built-in test statistic function."""

    def item_mean(responses: NDArray[np.int_]) -> float:
        valid = responses >= 0
        return float(np.sum(responses[valid]) / np.sum(valid))

    def person_score_variance(responses: NDArray[np.int_]) -> float:
        person_scores = np.sum(np.maximum(responses, 0), axis=1)
        return float(np.var(person_scores))

    def chi_square(responses: NDArray[np.int_]) -> float:
        valid = responses >= 0
        n_valid = np.sum(valid)
        if n_valid == 0:
            return 0.0
        p_obs = np.sum(responses[valid]) / n_valid
        expected = 0.5
        return float(n_valid * (p_obs - expected) ** 2 / (expected * (1 - expected)))

    def max_item_correlation(responses: NDArray[np.int_]) -> float:
        responses_filled = np.where(responses >= 0, responses, np.nan)
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(responses_filled.T)
        np.fill_diagonal(corr, 0)
        valid_corr = corr[~np.isnan(corr)]
        return float(np.max(np.abs(valid_corr))) if len(valid_corr) > 0 else 0.0

    def odds_ratio_range(responses: NDArray[np.int_]) -> float:
        eps = 0.5
        odds_ratios = []
        for i in range(min(n_items - 1, 5)):
            for j in range(i + 1, min(n_items, i + 6)):
                valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
                if np.sum(valid) < 10:
                    continue
                a = np.sum((responses[valid, i] == 1) & (responses[valid, j] == 1))
                b = np.sum((responses[valid, i] == 1) & (responses[valid, j] == 0))
                c = np.sum((responses[valid, i] == 0) & (responses[valid, j] == 1))
                d = np.sum((responses[valid, i] == 0) & (responses[valid, j] == 0))
                odds_ratio = ((a + eps) * (d + eps)) / ((b + eps) * (c + eps))
                odds_ratios.append(np.log(odds_ratio))

        if not odds_ratios:
            return 0.0
        return float(np.max(odds_ratios) - np.min(odds_ratios))

    statistics = {
        "item_mean": item_mean,
        "person_score": person_score_variance,
        "chi_square": chi_square,
        "correlation": max_item_correlation,
        "odds_ratio": odds_ratio_range,
    }

    if name not in statistics:
        raise ValueError(
            f"Unknown test statistic: {name}. Available: {list(statistics.keys())}"
        )

    return statistics[name]


def compare_models(
    *results: PSISResult | WAICResult,
    names: list[str] | None = None,
) -> str:
    """Compare multiple models using LOO or WAIC.

    Parameters
    ----------
    *results : PSISResult or WAICResult
        Model comparison results.
    names : list of str, optional
        Model names for display.

    Returns
    -------
    str
        Formatted comparison table.
    """
    n_models = len(results)
    if names is None:
        names = [f"Model {i + 1}" for i in range(n_models)]

    if len(names) != n_models:
        raise ValueError(f"Expected {n_models} names, got {len(names)}")

    if isinstance(results[0], PSISResult):
        ic_values = [r.looic for r in results]
        elpd_values = [r.elpd_loo for r in results]
        se_values = [r.se_elpd for r in results]
        ic_name = "LOOIC"
    else:
        ic_values = [r.waic for r in results]
        elpd_values = [r.elpd_waic for r in results]
        se_values = [r.se_waic for r in results]
        ic_name = "WAIC"

    sorted_idx = np.argsort(ic_values)
    best_ic = ic_values[sorted_idx[0]]

    lines = [
        "Model Comparison",
        "=" * 60,
        f"{'Model':<15} {ic_name:>10} {'elpd':>10} {'SE':>8} {'d' + ic_name:>10}",
        "-" * 60,
    ]

    for idx in sorted_idx:
        diff = ic_values[idx] - best_ic
        lines.append(
            f"{names[idx]:<15} {ic_values[idx]:>10.2f} "
            f"{elpd_values[idx]:>10.2f} {se_values[idx]:>8.2f} {diff:>10.2f}"
        )

    return "\n".join(lines)


def compute_pointwise_log_lik(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    chains: dict[str, NDArray[np.float64]],
    by: Literal["person", "observation"] = "person",
) -> NDArray[np.float64]:
    """Compute pointwise log-likelihood from MCMC chains.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    chains : dict
        MCMC chains for model parameters.
    by : {'person', 'observation'}, default='person'
        Whether to compute log-lik per person or per observation.

    Returns
    -------
    log_lik : ndarray
        If by='person': shape (n_samples, n_persons)
        If by='observation': shape (n_samples, n_persons * n_items)
    """
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    sample_chain = next(iter(chains.values()))
    n_samples = len(sample_chain)

    if by == "person":
        log_lik = np.zeros((n_samples, n_persons))
    else:
        log_lik = np.zeros((n_samples, n_persons * n_items))

    for s in range(n_samples):
        for param_name, chain in chains.items():
            if param_name in model._parameters:
                model._parameters[param_name] = chain[s]

        theta_chain = chains.get("theta", None)
        if theta_chain is not None:
            if theta_chain.ndim == 3:
                theta = theta_chain[s]
            else:
                theta = theta_chain
        else:
            theta = np.zeros((n_persons, model.n_factors))

        probs = model.probability(theta)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        if by == "person":
            for i in range(n_persons):
                ll = 0.0
                for j in range(n_items):
                    if responses[i, j] >= 0:
                        r = responses[i, j]
                        p = probs[i, j]
                        ll += r * np.log(p) + (1 - r) * np.log(1 - p)
                log_lik[s, i] = ll
        else:
            obs_idx = 0
            for i in range(n_persons):
                for j in range(n_items):
                    if responses[i, j] >= 0:
                        r = responses[i, j]
                        p = probs[i, j]
                        log_lik[s, obs_idx] = r * np.log(p) + (1 - r) * np.log(1 - p)
                    obs_idx += 1

    return log_lik
