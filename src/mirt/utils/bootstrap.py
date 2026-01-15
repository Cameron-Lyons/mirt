"""Bootstrap methods for standard errors and confidence intervals.

This module provides nonparametric bootstrap procedures for:
- Standard error estimation
- Confidence interval construction
- Parameter uncertainty quantification
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def bootstrap_se(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    n_bootstrap: int = 200,
    statistic: Literal["parameters", "theta"] | Callable = "parameters",
    seed: int | None = None,
    verbose: bool = False,
    warm_start: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Compute bootstrap standard errors.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted model or fit result
    responses : NDArray
        Response matrix (n_persons, n_items)
    n_bootstrap : int
        Number of bootstrap samples
    statistic : str or callable
        What to compute SE for:
        - 'parameters': Item parameter SEs
        - 'theta': Ability estimate SEs
        - callable: Custom function f(model, responses) -> dict
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Whether to print progress
    warm_start : bool
        Whether to use original parameter estimates as starting values
        for bootstrap samples. This significantly speeds up convergence.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and SE arrays as values
    """
    from mirt.estimation.em import EMEstimator
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        model = model.model

    rng = np.random.default_rng(seed)
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    boot_estimates: dict[str, list[NDArray]] = {}

    max_iter = 100 if warm_start else 200
    estimator = EMEstimator(max_iter=max_iter, tol=1e-3, verbose=False)

    original_params = {k: v.copy() for k, v in model.parameters.items()}

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 50 == 0:
            print(f"Bootstrap sample {b + 1}/{n_bootstrap}")

        indices = rng.integers(0, n_persons, size=n_persons)
        boot_responses = responses[indices]

        boot_model = model.copy()

        if warm_start:
            for name, values in original_params.items():
                boot_model._parameters[name] = values.copy()
            boot_model._is_fitted = False

        try:
            result = estimator.fit(boot_model, boot_responses)

            if statistic == "parameters":
                for name, values in result.model.parameters.items():
                    if name not in boot_estimates:
                        boot_estimates[name] = []
                    boot_estimates[name].append(values.copy())

            elif statistic == "theta":
                from mirt.scoring import fscores

                scores = fscores(result.model, boot_responses, method="EAP")
                if "theta" not in boot_estimates:
                    boot_estimates["theta"] = []
                boot_estimates["theta"].append(scores.theta.copy())

            elif callable(statistic):
                custom_result = statistic(result.model, boot_responses)
                for name, values in custom_result.items():
                    if name not in boot_estimates:
                        boot_estimates[name] = []
                    boot_estimates[name].append(np.asarray(values))

        except Exception:
            continue

    se_results: dict[str, NDArray[np.float64]] = {}
    for name, estimates in boot_estimates.items():
        if len(estimates) > 1:
            stacked = np.stack(estimates, axis=0)
            se_results[name] = np.std(stacked, axis=0, ddof=1)
        else:
            se_results[name] = np.full_like(estimates[0], np.nan)

    return se_results


def bootstrap_ci(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    method: Literal["percentile", "BCa", "basic"] = "percentile",
    statistic: Literal["parameters", "theta"] | Callable = "parameters",
    seed: int | None = None,
    verbose: bool = False,
    warm_start: bool = True,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Compute bootstrap confidence intervals.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted model or fit result
    responses : NDArray
        Response matrix
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    method : str
        CI method:
        - 'percentile': Simple percentile method
        - 'BCa': Bias-corrected and accelerated
        - 'basic': Basic bootstrap interval
    statistic : str or callable
        What to compute CI for ('parameters', 'theta', or callable)
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
    warm_start : bool
        Whether to use original parameter estimates as starting values
        for bootstrap samples. This significantly speeds up convergence.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and (lower, upper) CI tuples
    """
    from mirt.estimation.em import EMEstimator
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        original_model = model.model
    else:
        original_model = model

    rng = np.random.default_rng(seed)
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    original_estimates: dict[str, NDArray] = {}
    if statistic == "parameters":
        original_estimates = {k: v.copy() for k, v in original_model.parameters.items()}
    elif statistic == "theta":
        from mirt.scoring import fscores

        scores = fscores(original_model, responses, method="EAP")
        original_estimates["theta"] = scores.theta.copy()
    elif callable(statistic):
        original_estimates = {
            k: np.asarray(v) for k, v in statistic(original_model, responses).items()
        }

    boot_estimates: dict[str, list[NDArray]] = {k: [] for k in original_estimates}

    original_params = {k: v.copy() for k, v in original_model.parameters.items()}

    max_iter = 100 if warm_start else 200
    estimator = EMEstimator(max_iter=max_iter, tol=1e-3, verbose=False)

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 50 == 0:
            print(f"Bootstrap sample {b + 1}/{n_bootstrap}")

        indices = rng.integers(0, n_persons, size=n_persons)
        boot_responses = responses[indices]
        boot_model = original_model.copy()

        if warm_start:
            for name, values in original_params.items():
                boot_model._parameters[name] = values.copy()
            boot_model._is_fitted = False

        try:
            result = estimator.fit(boot_model, boot_responses)

            if statistic == "parameters":
                for name, values in result.model.parameters.items():
                    if name in boot_estimates:
                        boot_estimates[name].append(values.copy())

            elif statistic == "theta":
                from mirt.scoring import fscores

                scores = fscores(result.model, boot_responses, method="EAP")
                boot_estimates["theta"].append(scores.theta.copy())

            elif callable(statistic):
                custom_result = statistic(result.model, boot_responses)
                for name, values in custom_result.items():
                    if name in boot_estimates:
                        boot_estimates[name].append(np.asarray(values))

        except Exception:
            continue

    ci_results: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for name, estimates in boot_estimates.items():
        if len(estimates) < 10:
            original = original_estimates[name]
            ci_results[name] = (
                np.full_like(original, np.nan),
                np.full_like(original, np.nan),
            )
            continue

        stacked = np.stack(estimates, axis=0)
        original = original_estimates[name]

        if method == "percentile":
            lower = np.percentile(stacked, 100 * alpha / 2, axis=0)
            upper = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0)

        elif method == "basic":
            lower_pct = np.percentile(stacked, 100 * alpha / 2, axis=0)
            upper_pct = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0)
            lower = 2 * original - upper_pct
            upper = 2 * original - lower_pct

        elif method == "BCa":
            from scipy import stats

            prop_below = np.mean(stacked < original, axis=0)
            z0 = stats.norm.ppf(np.clip(prop_below, 0.001, 0.999))

            max_jack = min(20, n_persons)
            jack_indices = rng.choice(n_persons, size=max_jack, replace=False)

            jackknife_estimates = []
            for i in jack_indices:
                jack_responses = np.delete(responses, i, axis=0)
                jack_model = original_model.copy()
                try:
                    jack_result = estimator.fit(jack_model, jack_responses)
                    if statistic == "parameters":
                        jackknife_estimates.append(
                            jack_result.model.parameters[name].copy()
                        )
                    elif statistic == "theta":
                        from mirt.scoring import fscores

                        scores = fscores(
                            jack_result.model, jack_responses, method="EAP"
                        )
                        padded = np.full(n_persons, np.nan)
                        mask = np.ones(n_persons, dtype=bool)
                        mask[i] = False
                        padded[mask] = scores.theta.ravel()
                        jackknife_estimates.append(padded)
                except Exception:
                    pass

            if len(jackknife_estimates) > 10:
                jack_stacked = np.stack(jackknife_estimates, axis=0)
                jack_mean = np.nanmean(jack_stacked, axis=0)
                jack_diff = jack_mean - jack_stacked
                a = np.nansum(jack_diff**3, axis=0) / (
                    6 * np.nansum(jack_diff**2, axis=0) ** 1.5 + PROB_EPSILON
                )
            else:
                a = np.zeros_like(original)

            z_alpha = stats.norm.ppf(alpha / 2)
            z_1_alpha = stats.norm.ppf(1 - alpha / 2)

            adj_lower = stats.norm.cdf(
                z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha) + PROB_EPSILON)
            )
            adj_upper = stats.norm.cdf(
                z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha) + PROB_EPSILON)
            )

            adj_lower = np.clip(adj_lower, 0.001, 0.999)
            adj_upper = np.clip(adj_upper, 0.001, 0.999)

            lower = np.array(
                [
                    np.percentile(
                        stacked[:, i] if stacked.ndim > 1 else stacked,
                        100 * adj_lower.flat[i],
                    )
                    for i in range(original.size)
                ]
            ).reshape(original.shape)
            upper = np.array(
                [
                    np.percentile(
                        stacked[:, i] if stacked.ndim > 1 else stacked,
                        100 * adj_upper.flat[i],
                    )
                    for i in range(original.size)
                ]
            ).reshape(original.shape)

        else:
            raise ValueError(f"Unknown CI method: {method}")

        ci_results[name] = (lower.astype(np.float64), upper.astype(np.float64))

    return ci_results


def parametric_bootstrap(
    model: BaseItemModel | FitResult,
    n_bootstrap: int = 200,
    n_persons: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    warm_start: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Parametric bootstrap using model to generate data.

    Instead of resampling observed data, generates new data from the fitted model.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted model
    n_bootstrap : int
        Number of bootstrap samples
    n_persons : int, optional
        Number of persons to simulate (default: 500)
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
    warm_start : bool
        Whether to use original parameter estimates as starting values
        for bootstrap samples. This significantly speeds up convergence.

    Returns
    -------
    dict
        Standard errors for each parameter
    """
    from mirt.estimation.em import EMEstimator
    from mirt.results.fit_result import FitResult
    from mirt.utils.simulation import simdata

    if isinstance(model, FitResult):
        model = model.model

    if n_persons is None:
        n_persons = 500

    rng = np.random.default_rng(seed)
    boot_estimates: dict[str, list[NDArray]] = {}

    max_iter = 100 if warm_start else 200
    estimator = EMEstimator(max_iter=max_iter, tol=1e-3, verbose=False)

    params = model.parameters
    model_name = model.model_name
    original_params = {k: v.copy() for k, v in params.items()}

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 50 == 0:
            print(f"Parametric bootstrap {b + 1}/{n_bootstrap}")

        theta = rng.standard_normal(n_persons)

        if model_name in ("1PL", "2PL", "3PL", "4PL", "Rasch"):
            discrimination = params.get("discrimination", np.ones(model.n_items))
            difficulty = params["difficulty"]
            guessing = params.get("guessing", np.zeros(model.n_items))

            sim_data = simdata(
                model=model_name,
                n_persons=n_persons,
                n_items=model.n_items,
                discrimination=discrimination,
                difficulty=difficulty,
                guessing=guessing,
                theta=theta,
                seed=rng.integers(0, 2**31),
            )
        else:
            probs = model.probability(theta.reshape(-1, 1))
            if probs.ndim == 1:
                probs = probs.reshape(-1, model.n_items)
            sim_data = (rng.random((n_persons, model.n_items)) < probs).astype(np.int_)

        boot_model = model.copy()

        if warm_start:
            for name, values in original_params.items():
                boot_model._parameters[name] = values.copy()
            boot_model._is_fitted = False

        try:
            result = estimator.fit(boot_model, sim_data)

            for name, values in result.model.parameters.items():
                if name not in boot_estimates:
                    boot_estimates[name] = []
                boot_estimates[name].append(values.copy())
        except Exception:
            continue

    se_results: dict[str, NDArray[np.float64]] = {}
    for name, estimates in boot_estimates.items():
        if len(estimates) > 1:
            stacked = np.stack(estimates, axis=0)
            se_results[name] = np.std(stacked, axis=0, ddof=1)

    return se_results
