"""Utilities for generating starting values and null models.

This module provides functions for:
- Generating random starting values for IRT parameters
- Computing null models for baseline comparisons
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def gen_random_pars(
    model: BaseItemModel,
    n_sets: int = 1,
    seed: int | None = None,
    discrimination_range: tuple[float, float] = (0.5, 2.0),
    difficulty_range: tuple[float, float] = (-2.0, 2.0),
    guessing_range: tuple[float, float] = (0.1, 0.3),
    upper_range: tuple[float, float] = (0.9, 1.0),
) -> list[dict[str, NDArray[np.float64]]]:
    """Generate random starting values for IRT model parameters.

    This function generates multiple sets of random parameter starting
    values, which can be useful for checking sensitivity to starting
    values or for multi-start optimization strategies.

    Parameters
    ----------
    model : BaseItemModel
        The IRT model for which to generate starting values.
    n_sets : int
        Number of random parameter sets to generate.
    seed : int, optional
        Random seed for reproducibility.
    discrimination_range : tuple of float
        Range for discrimination parameters (min, max).
    difficulty_range : tuple of float
        Range for difficulty parameters (min, max).
    guessing_range : tuple of float
        Range for guessing parameters (min, max).
    upper_range : tuple of float
        Range for upper asymptote parameters (min, max).

    Returns
    -------
    list of dict
        List of parameter dictionaries, each containing random starting
        values for all model parameters.

    Examples
    --------
    >>> from mirt import TwoParameterLogistic
    >>> from mirt.utils.starting import gen_random_pars
    >>> model = TwoParameterLogistic(n_items=10)
    >>> random_starts = gen_random_pars(model, n_sets=5, seed=42)
    >>> len(random_starts)
    5
    >>> 'discrimination' in random_starts[0]
    True
    """
    rng = np.random.default_rng(seed)

    param_names = list(model.parameters.keys())

    random_sets = []

    for _ in range(n_sets):
        params = {}

        for name in param_names:
            shape = model.parameters[name].shape

            if name in ("discrimination", "slopes", "loadings"):
                low, high = discrimination_range
                params[name] = rng.uniform(low, high, shape)

            elif name in ("difficulty", "intercepts", "location"):
                low, high = difficulty_range
                params[name] = rng.uniform(low, high, shape)

            elif name in ("thresholds", "steps"):
                n_thresh = shape[-1]
                base_thresh = np.linspace(
                    difficulty_range[0], difficulty_range[1], n_thresh
                )
                noise = rng.uniform(-0.5, 0.5, shape)
                if len(shape) == 2:
                    params[name] = base_thresh[None, :] + noise
                else:
                    params[name] = base_thresh + noise

            elif name in ("guessing", "lower"):
                low, high = guessing_range
                params[name] = rng.uniform(low, high, shape)

            elif name in ("upper", "slipping"):
                low, high = upper_range
                params[name] = rng.uniform(low, high, shape)

            elif name == "asymmetry":
                params[name] = rng.uniform(0.5, 2.0, shape)

            else:
                current = model.parameters[name]
                noise = rng.normal(0, 0.1 * (np.abs(current).mean() + 0.1), shape)
                params[name] = current + noise

        random_sets.append(params)

    return random_sets


def calc_null(
    responses: NDArray[np.int_],
    model_type: Literal["independence", "intercept_only"] = "independence",
) -> dict[str, float]:
    """Calculate null model statistics for baseline comparisons.

        The null model serves as a baseline for computing incremental fit
        indices such as CFI and TLI. It represents a model with no item
        discrimination (all items equally uninformative).

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.
        model_type : str
            Type of null model:
            - 'independence': Items are independent, difficulty-only model
            - 'intercept_only': Uses observed proportions only

        Returns
        -------
        dict
            Dictionary containing:
            - log_likelihood: Null model log-likelihood
            - n_parameters: Number of parameters in null model
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion

        Examples
        --------
        >>> import numpy as np

    from mirt.constants import PROB_EPSILON
        >>> from mirt.utils.starting import calc_null
        >>> data = np.random.randint(0, 2, (100, 10))
        >>> null_stats = calc_null(data)
        >>> print(f"Null LL: {null_stats['log_likelihood']:.2f}")
    """
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    valid_mask = responses >= 0

    valid_responses = np.where(valid_mask, responses, 0).astype(np.float64)
    n_valid_per_item = np.sum(valid_mask, axis=0)
    n_correct_per_item = np.sum(valid_responses, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_items = np.where(
            n_valid_per_item > 0,
            n_correct_per_item / n_valid_per_item,
            0.5,
        )
    p_items = np.clip(p_items, PROB_EPSILON, 1 - PROB_EPSILON)

    log_p = np.log(p_items)
    log_1_minus_p = np.log(1 - p_items)

    ll_per_item = (
        n_correct_per_item * log_p
        + (n_valid_per_item - n_correct_per_item) * log_1_minus_p
    )
    ll = float(np.sum(ll_per_item))

    n_params = n_items

    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n_persons) * n_params

    return {
        "log_likelihood": ll,
        "n_parameters": n_params,
        "aic": aic,
        "bic": bic,
    }


def _fit_single_start(args: tuple) -> tuple[float, FitResult] | tuple[float, None]:
    """Helper function for parallel fitting with a single start."""
    from mirt.estimation.em import EMEstimator

    model, responses, start_params, fit_kwargs = args
    trial_model = model.copy()
    trial_model.set_parameters(**start_params)

    estimator = EMEstimator(**fit_kwargs)

    try:
        result = estimator.fit(trial_model, responses)
        return (result.log_likelihood, result)
    except Exception:
        return (-np.inf, None)


def multi_start_fit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_starts: int = 5,
    seed: int | None = None,
    verbose: bool = False,
    n_jobs: int = 1,
    **fit_kwargs: Any,
) -> FitResult:
    """Fit model with multiple random starting values.

    Performs multiple fits with different random starting values and
    returns the best result based on log-likelihood.

    Parameters
    ----------
    model : BaseItemModel
        The IRT model to fit.
    responses : ndarray
        Response matrix.
    n_starts : int
        Number of random starts to try.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    n_jobs : int
        Number of parallel jobs. Use -1 for all CPUs, 1 for sequential.
    **fit_kwargs
        Additional arguments passed to the estimator.

    Returns
    -------
    FitResult
        Best fit result across all starts.

    Examples
    --------
    >>> from mirt import TwoParameterLogistic, simdata
    >>> from mirt.utils.starting import multi_start_fit
    >>> data = simdata(n_persons=500, n_items=10)
    >>> model = TwoParameterLogistic(n_items=10)
    >>> result = multi_start_fit(model, data, n_starts=3, seed=42)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from mirt.estimation.em import EMEstimator

    random_starts = gen_random_pars(model, n_sets=n_starts, seed=seed)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, n_starts)

    if n_jobs == 1:
        best_result = None
        best_ll = -np.inf

        for i, start_params in enumerate(random_starts):
            trial_model = model.copy()
            trial_model.set_parameters(**start_params)

            estimator = EMEstimator(**fit_kwargs)

            try:
                result = estimator.fit(trial_model, responses)

                if verbose:
                    print(f"Start {i + 1}/{n_starts}: LL = {result.log_likelihood:.4f}")

                if result.log_likelihood > best_ll:
                    best_ll = result.log_likelihood
                    best_result = result

            except Exception as e:
                if verbose:
                    print(f"Start {i + 1}/{n_starts}: Failed ({e})")

        if best_result is None:
            raise RuntimeError("All starting value sets failed to converge")

        return best_result

    args_list = [
        (model, responses, start_params, fit_kwargs) for start_params in random_starts
    ]

    best_result = None
    best_ll = -np.inf
    completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_fit_single_start, args): i
            for i, args in enumerate(args_list)
        }

        for future in as_completed(futures):
            completed += 1
            ll, result = future.result()

            if verbose:
                if result is not None:
                    print(f"Start {completed}/{n_starts}: LL = {ll:.4f}")
                else:
                    print(f"Start {completed}/{n_starts}: Failed")

            if result is not None and ll > best_ll:
                best_ll = ll
                best_result = result

    if best_result is None:
        raise RuntimeError("All starting value sets failed to converge")

    return best_result
