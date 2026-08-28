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
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_FIXED_PARAMETERS_BY_MODEL = {
    "1PL": frozenset({"discrimination"}),
    "PCM": frozenset({"discrimination"}),
}


def _validate_positive_integer(value: int, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise MirtValidationError(
            f"{name} must be a positive integer",
            parameter=name,
            value=value,
        )
    return int(value)


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise MirtValidationError(
            "seed must be a non-negative integer or None",
            parameter="seed",
            value=seed,
        )
    return int(seed)


def _validate_parameter_range(
    values: tuple[float, float],
    *,
    name: str,
    positive: bool = False,
    probability: bool = False,
) -> tuple[float, float]:
    try:
        bounds = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{name} must contain two finite numbers", parameter=name
        ) from exc

    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise MirtValidationError(
            f"{name} must contain two finite numbers", parameter=name
        )
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower >= upper:
        raise MirtValidationError(
            f"{name} lower bound must be less than its upper bound",
            parameter=name,
            value=(lower, upper),
        )
    if positive and lower <= 0:
        raise MirtValidationError(f"{name} must be strictly positive", parameter=name)
    if probability and not 0 <= lower < upper <= 1:
        raise MirtValidationError(f"{name} must lie between 0 and 1", parameter=name)
    return lower, upper


def _normalize_responses(
    responses: NDArray[np.int_],
    *,
    n_items: int | None = None,
) -> NDArray[np.int_]:
    """Validate responses while accepting NaN as a missing-value alias."""
    try:
        response_array = np.asarray(responses)
    except (TypeError, ValueError) as exc:
        raise MirtDataError("responses must be a rectangular numeric array") from exc
    if response_array.dtype.kind == "f" and np.any(np.isnan(response_array)):
        response_array = response_array.copy()
        response_array[np.isnan(response_array)] = -1
    return validate_responses(response_array, n_items=n_items)


def _random_category_parameter(
    model: BaseItemModel,
    current: NDArray[np.float64],
    rng: np.random.Generator,
    bounds: tuple[float, float],
    *,
    ordered: bool,
) -> NDArray[np.float64]:
    """Randomize active category parameters while retaining padding."""
    values = current.copy()
    categories = getattr(model, "n_categories", None)
    if (
        current.ndim >= 2
        and categories is not None
        and len(categories) == model.n_items
    ):
        for item_index, n_categories in enumerate(categories):
            n_active = min(n_categories - 1, current.shape[1])
            draws = rng.uniform(*bounds, size=n_active)
            values[item_index, :n_active] = np.sort(draws) if ordered else draws
        return values

    draws = rng.uniform(*bounds, size=current.shape)
    return np.sort(draws, axis=-1) if ordered else draws


def _random_ggum_thresholds(
    model: BaseItemModel,
    current: NDArray[np.float64],
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Randomize free GGUM thresholds and construct the symmetric scale."""
    values = current.copy()
    if len(set(model.n_categories)) == 1:
        n_independent = model.n_categories[0] - 1
        n_active = 2 * n_independent + 1
        independent = np.sort(
            rng.uniform(*bounds, size=(model.n_items, n_independent)), axis=1
        )
        values[:, :n_independent] = independent
        values[:, n_independent] = 0.0
        values[:, n_independent + 1 : n_active] = -independent[:, ::-1]
        return values

    for item_index, n_categories in enumerate(model.n_categories):
        n_independent = n_categories - 1
        n_active = 2 * n_independent + 1
        independent = np.sort(rng.uniform(*bounds, size=n_independent))
        values[item_index, :n_independent] = independent
        values[item_index, n_independent] = 0.0
        values[item_index, n_independent + 1 : n_active] = -independent[::-1]
    return values


def _random_nominal_parameter(
    model: BaseItemModel,
    current: NDArray[np.float64],
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Randomize free NRM categories without moving its reference category."""
    values = current.copy()
    for item_index, n_categories in enumerate(model.n_categories):
        target_shape = (n_categories - 1, *current.shape[2:])
        values[item_index, 1:n_categories] = rng.uniform(*bounds, size=target_shape)
    return values


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
        Range for upper asymptote and ideal-point peak-height parameters
        (min, max).

    Returns
    -------
    list of dict
        List of parameter dictionaries containing random starting values for
        all free model parameters. Fixed and reference-category parameters are
        omitted or retained at their identifying values.

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
    n_sets = _validate_positive_integer(n_sets, name="n_sets")
    seed = _validate_seed(seed)
    discrimination_range = _validate_parameter_range(
        discrimination_range,
        name="discrimination_range",
        positive=True,
    )
    difficulty_range = _validate_parameter_range(
        difficulty_range, name="difficulty_range"
    )
    guessing_range = _validate_parameter_range(
        guessing_range,
        name="guessing_range",
        probability=True,
    )
    upper_range = _validate_parameter_range(
        upper_range,
        name="upper_range",
        probability=True,
    )

    base_parameters = model.parameters
    if (
        any(name in base_parameters for name in ("guessing", "lower"))
        and "upper" in base_parameters
        and guessing_range[1] >= upper_range[0]
    ):
        raise MirtValidationError(
            "guessing_range must remain below upper_range",
            parameter="guessing_range",
        )

    rng = np.random.default_rng(seed)
    fixed_parameters = _FIXED_PARAMETERS_BY_MODEL.get(
        getattr(model, "model_name", ""), frozenset()
    )
    random_sets: list[dict[str, NDArray[np.float64]]] = []

    for _ in range(n_sets):
        params: dict[str, NDArray[np.float64]] = {}

        for name, current in base_parameters.items():
            if name in fixed_parameters:
                continue

            if getattr(model, "model_name", "") == "NRM" and name in (
                "slopes",
                "intercepts",
            ):
                bounds = discrimination_range if name == "slopes" else difficulty_range
                params[name] = _random_nominal_parameter(model, current, rng, bounds)

            elif name in ("discrimination", "slopes", "loadings"):
                params[name] = rng.uniform(*discrimination_range, size=current.shape)

            elif name in ("difficulty", "intercepts", "location"):
                params[name] = rng.uniform(*difficulty_range, size=current.shape)

            elif name == "thresholds" and getattr(model, "model_name", "") == "GGUM":
                params[name] = _random_ggum_thresholds(
                    model,
                    current,
                    rng,
                    difficulty_range,
                )

            elif name in ("thresholds", "steps"):
                params[name] = _random_category_parameter(
                    model,
                    current,
                    rng,
                    difficulty_range,
                    ordered=name == "thresholds",
                )

            elif name in ("guessing", "lower", "guess", "slipping", "slip"):
                params[name] = rng.uniform(*guessing_range, size=current.shape)

            elif name == "upper":
                params[name] = rng.uniform(*upper_range, size=current.shape)

            elif name == "peak_height":
                values = rng.uniform(*upper_range, size=current.shape)
                params[name] = np.maximum(values, np.nextafter(0.0, 1.0))

            elif name == "asymmetry":
                params[name] = rng.uniform(0.5, 2.0, size=current.shape)

            else:
                noise = rng.normal(
                    0,
                    0.1 * (np.abs(current).mean() + 0.1),
                    size=current.shape,
                )
                params[name] = current + noise

        random_sets.append(params)

    return random_sets


def calc_null(
    responses: NDArray[np.int_],
    model_type: Literal["independence", "intercept_only"] = "independence",
) -> dict[str, float | int]:
    """Calculate null model statistics for baseline comparisons.

    ``independence`` estimates one response probability per item, while
    ``intercept_only`` estimates one probability shared by every item. These
    models provide baselines for incremental fit indices such as CFI and TLI.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Dichotomous response matrix. Negative values and NaN are treated as
        missing.
    model_type : {"independence", "intercept_only"}
        Baseline model to calculate.

    Returns
    -------
    dict
        Log-likelihood, parameter count, AIC, and BIC for the baseline model.

    Examples
    --------
    >>> import numpy as np
    >>> from mirt.utils.starting import calc_null
    >>> data = np.array([[1, 0], [1, 1], [0, 1]])
    >>> null_stats = calc_null(data)
    >>> null_stats["n_parameters"]
    2
    """
    if model_type not in ("independence", "intercept_only"):
        raise MirtValidationError(
            "model_type must be 'independence' or 'intercept_only'",
            parameter="model_type",
            value=model_type,
        )

    validated = _normalize_responses(responses)
    observed_mask = validated >= 0
    if np.any(validated[observed_mask] > 1):
        raise MirtDataError("null models require dichotomous responses coded 0 or 1")

    n_persons, n_items = validated.shape
    observed_per_item = observed_mask.sum(axis=0)
    if np.any(observed_per_item == 0):
        missing_items = np.flatnonzero(observed_per_item == 0).tolist()
        raise MirtDataError(
            "each item must contain at least one observed response",
            items=missing_items,
        )

    observed_values = np.where(observed_mask, validated, 0)
    correct_per_item = observed_values.sum(axis=0, dtype=np.float64)

    if model_type == "independence":
        probabilities = correct_per_item / observed_per_item
        n_params = n_items
    else:
        pooled_probability = float(correct_per_item.sum() / observed_mask.sum())
        probabilities = np.full(n_items, pooled_probability)
        n_params = 1

    probabilities = np.clip(probabilities, PROB_EPSILON, 1 - PROB_EPSILON)
    ll_per_item = correct_per_item * np.log(probabilities) + (
        observed_per_item - correct_per_item
    ) * np.log1p(-probabilities)
    ll = float(ll_per_item.sum())
    aic = float(-2 * ll + 2 * n_params)
    bic = float(-2 * ll + np.log(n_persons) * n_params)

    return {
        "log_likelihood": ll,
        "n_parameters": n_params,
        "aic": aic,
        "bic": bic,
    }


def _validate_model_responses(
    model: BaseItemModel,
    responses: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Validate response codes against a model's category structure."""
    validated = _normalize_responses(responses, n_items=model.n_items)
    observed = validated >= 0

    if not model.is_polytomous:
        if np.any(validated[observed] > 1):
            raise MirtDataError("dichotomous responses must be coded as 0 or 1")
        return validated

    for item_index, n_categories in enumerate(model.n_categories):
        item_observed = observed[:, item_index]
        if np.any(validated[item_observed, item_index] >= n_categories):
            raise MirtDataError(
                "polytomous response codes must be below n_categories",
                item=item_index,
                n_categories=n_categories,
            )
    return validated


def _fit_single_start(
    args: tuple[
        int,
        BaseItemModel,
        NDArray[np.int_],
        dict[str, NDArray[np.float64]],
        dict[str, Any],
    ],
) -> tuple[int, float, FitResult | None, str | None]:
    """Fit one starting-value set and preserve its original ordering."""
    from mirt.estimation.em import EMEstimator

    start_index, model, responses, start_params, fit_kwargs = args
    try:
        trial_model = model.copy()
        trial_model.set_parameters(**start_params)
        # EM initialization otherwise replaces the supplied starting values.
        trial_model._is_fitted = True
        estimator = EMEstimator(**fit_kwargs)
        result = estimator.fit(trial_model, responses)
        log_likelihood = float(result.log_likelihood)
        if not np.isfinite(log_likelihood):
            raise ArithmeticError("fit returned a non-finite log-likelihood")
        return start_index, log_likelihood, result, None
    except (
        TypeError,
        ValueError,
        RuntimeError,
        ArithmeticError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as exc:
        return start_index, -np.inf, None, str(exc)


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
    >>> data = simdata(n_persons=30, n_items=3, seed=42)
    >>> model = TwoParameterLogistic(n_items=3)
    >>> result = multi_start_fit(
    ...     model, data, n_starts=2, seed=42, max_iter=1, n_quadpts=5,
    ...     use_gpu=False,
    ... )
    >>> bool(np.isfinite(result.log_likelihood))
    True
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_starts = _validate_positive_integer(n_starts, name="n_starts")
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise MirtValidationError(
            "n_jobs must be a positive integer or -1",
            parameter="n_jobs",
            value=n_jobs,
        )
    n_jobs = int(n_jobs)
    validated = _validate_model_responses(model, responses)
    random_starts = gen_random_pars(model, n_sets=n_starts, seed=seed)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, n_starts)

    args_list = [
        (index, model, validated, start_params, fit_kwargs)
        for index, start_params in enumerate(random_starts)
    ]
    outcomes: list[tuple[int, float, FitResult | None, str | None]] = []

    if n_jobs == 1:
        outcomes = [_fit_single_start(args) for args in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_fit_single_start, args): args[0] for args in args_list
            }
            for future in as_completed(futures):
                start_index = futures[future]
                try:
                    outcomes.append(future.result())
                except Exception as exc:  # Process and serialization failures.
                    outcomes.append((start_index, -np.inf, None, str(exc)))

    outcomes.sort(key=lambda outcome: outcome[0])
    if verbose:
        for start_index, ll, result, error in outcomes:
            if result is None:
                print(f"Start {start_index + 1}/{n_starts}: Failed ({error})")
            else:
                print(f"Start {start_index + 1}/{n_starts}: LL = {ll:.4f}")

    successful = [outcome for outcome in outcomes if outcome[2] is not None]
    if not successful:
        first_error = next((outcome[3] for outcome in outcomes if outcome[3]), None)
        detail = f": {first_error}" if first_error else ""
        raise RuntimeError(f"All {n_starts} starting value sets failed{detail}")

    best = max(successful, key=lambda outcome: (outcome[1], -outcome[0]))
    best_result = best[2]
    if best_result is None:  # Narrow the optional type for static checkers.
        raise RuntimeError("No successful starting value set was retained")
    return best_result
