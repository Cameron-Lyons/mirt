"""Bootstrap methods for standard errors and confidence intervals.

This module provides nonparametric bootstrap procedures for:
- Standard error estimation
- Confidence interval construction
- Parameter uncertainty quantification
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

_BOOTSTRAP_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    FloatingPointError,
    np.linalg.LinAlgError,
)
_CI_METHODS = ("percentile", "BCa", "basic")
_STATISTICS = ("parameters", "theta")
_TaskInput = TypeVar("_TaskInput")
_TaskResult = TypeVar("_TaskResult")


@dataclass(slots=True)
class _StatisticFitTask:
    model: BaseItemModel
    original_params: dict[str, NDArray[np.float64]]
    warm_start: bool
    max_iter: int
    responses: NDArray[np.int_]
    sample_indices: list[NDArray[np.int64]]
    statistic: Literal["parameters", "theta"] | Callable[..., Any]


@dataclass(slots=True)
class _ParametricFitTask:
    model: BaseItemModel
    original_params: dict[str, NDArray[np.float64]]
    warm_start: bool
    max_iter: int
    n_persons: int
    rng_states: list[dict[str, Any]]


def _validate_resample_count(n_bootstrap: int) -> None:
    if (
        not isinstance(n_bootstrap, (int, np.integer))
        or isinstance(n_bootstrap, (bool, np.bool_))
        or n_bootstrap < 2
    ):
        raise MirtValidationError(
            "n_bootstrap must be an integer of at least 2",
            parameter="n_bootstrap",
            value=n_bootstrap,
        )


def _validate_n_jobs(n_jobs: int) -> int:
    """Validate and resolve bootstrap worker counts."""
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise MirtValidationError(
            "n_jobs must be -1 or a positive integer",
            parameter="n_jobs",
            value=n_jobs,
        )
    if n_jobs == -1:
        import os

        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def _run_bootstrap_tasks(
    function: Callable[[_TaskInput], _TaskResult],
    inputs: list[_TaskInput],
    n_jobs: int,
) -> list[_TaskResult]:
    """Run independent bootstrap tasks in deterministic input order."""
    if n_jobs == 1 or len(inputs) < 2:
        return [function(value) for value in inputs]

    import pickle
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    try:
        pickle.dumps((function, inputs[0]))
    except (AttributeError, pickle.PickleError, TypeError) as exc:
        raise MirtValidationError(
            "parallel bootstrap inputs must be picklable; use n_jobs=1 for "
            "locally defined models or statistics",
            parameter="n_jobs",
            value=n_jobs,
        ) from exc

    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(inputs)),
        mp_context=get_context("spawn"),
    ) as executor:
        return list(executor.map(function, inputs))


def _chunk_values(values: list[_TaskInput], n_chunks: int) -> list[list[_TaskInput]]:
    """Split ordered inputs into balanced contiguous worker chunks."""
    if not values:
        return []
    chunk_count = min(n_chunks, len(values))
    quotient, remainder = divmod(len(values), chunk_count)
    chunks: list[list[_TaskInput]] = []
    start = 0
    for chunk_index in range(chunk_count):
        stop = start + quotient + (chunk_index < remainder)
        chunks.append(values[start:stop])
        start = stop
    return chunks


def _validate_statistic(statistic: str | Callable[..., Any]) -> None:
    if isinstance(statistic, str):
        if statistic not in _STATISTICS:
            raise MirtValidationError(
                "Unknown bootstrap statistic",
                parameter="statistic",
                value=statistic,
                expected="'parameters', 'theta', or a callable",
            )
    elif not callable(statistic):
        raise MirtValidationError(
            "statistic must be 'parameters', 'theta', or a callable",
            parameter="statistic",
            value=statistic,
        )


def _validate_ci_configuration(alpha: float, method: str) -> None:
    if method not in _CI_METHODS:
        raise MirtValidationError(
            "Unknown bootstrap confidence interval method",
            parameter="method",
            value=method,
            expected=", ".join(_CI_METHODS),
        )
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0 < float(alpha) < 1
    ):
        raise MirtValidationError(
            "alpha must be a finite number between 0 and 1",
            parameter="alpha",
            value=alpha,
        )


def _prepare_bootstrap_model(
    model: BaseItemModel,
    original_params: Mapping[str, NDArray[np.float64]],
    warm_start: bool,
) -> BaseItemModel:
    boot_model = model.copy()
    if warm_start:
        boot_model._parameters = {
            name: values.copy() for name, values in original_params.items()
        }
    else:
        boot_model._parameters.clear()
        boot_model._initialize_parameters()
    boot_model._is_fitted = False
    return boot_model


def _as_statistic_mapping(result: Any) -> dict[str, NDArray[np.float64]]:
    if not isinstance(result, Mapping) or not result:
        raise MirtValidationError(
            "A custom bootstrap statistic must return a non-empty mapping"
        )

    converted: dict[str, NDArray[np.float64]] = {}
    for name, values in result.items():
        if not isinstance(name, str):
            raise MirtValidationError("Bootstrap statistic names must be strings")
        try:
            converted[name] = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"Bootstrap statistic {name!r} must be numeric"
            ) from exc
    return converted


def _fit_statistic_task(
    task: _StatisticFitTask,
) -> list[tuple[dict[str, NDArray[np.float64]] | None, str | None]]:
    """Fit one worker chunk and extract each requested statistic."""
    from mirt.estimation.em import EMEstimator

    task_results: list[tuple[dict[str, NDArray[np.float64]] | None, str | None]] = []
    for indices in task.sample_indices:
        fit_responses = task.responses[indices]
        boot_model = _prepare_bootstrap_model(
            task.model,
            task.original_params,
            task.warm_start,
        )
        try:
            estimator = EMEstimator(max_iter=task.max_iter, tol=1e-3, verbose=False)
            result = estimator.fit(boot_model, fit_responses)

            if task.statistic == "parameters":
                values_by_name = {
                    name: np.asarray(values, dtype=np.float64).copy()
                    for name, values in result.model.parameters.items()
                }
            elif task.statistic == "theta":
                from mirt.scoring import fscores

                scores = fscores(result.model, task.responses, method="EAP")
                values_by_name = {
                    "theta": np.asarray(scores.theta, dtype=np.float64).copy()
                }
            else:
                values_by_name = _as_statistic_mapping(
                    task.statistic(result.model, fit_responses)
                )
            task_results.append((values_by_name, None))
        except _BOOTSTRAP_EXCEPTIONS as exc:
            task_results.append((None, f"{type(exc).__name__}: {exc}"))
    return task_results


def _elementwise_percentile(
    samples: NDArray[np.float64], quantiles: NDArray[np.float64]
) -> NDArray[np.float64]:
    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one bootstrap replicate")
    flat_samples = samples.reshape(samples.shape[0], -1)
    flat_quantiles = np.asarray(quantiles, dtype=np.float64).reshape(-1)
    if flat_quantiles.size != flat_samples.shape[1]:
        raise ValueError("quantiles must contain one value per sample element")
    if not np.all(np.isfinite(flat_quantiles)) or np.any(
        (flat_quantiles < 0.0) | (flat_quantiles > 1.0)
    ):
        raise ValueError("quantiles must be finite values in [0, 1]")

    ordered = np.sort(flat_samples, axis=0)
    positions = (flat_samples.shape[0] - 1) * flat_quantiles
    lower_indices = np.floor(positions).astype(np.intp)
    upper_indices = np.ceil(positions).astype(np.intp)
    columns = np.arange(flat_quantiles.size)
    lower = ordered[lower_indices, columns]
    upper = ordered[upper_indices, columns]
    values = lower + (upper - lower) * (positions - lower_indices)
    return values.reshape(samples.shape[1:])


def _bca_interval(
    samples: NDArray[np.float64],
    original: NDArray[np.float64],
    jackknife: list[NDArray[np.float64]],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from scipy import stats

    prop_below = np.mean(samples < original, axis=0)
    z0 = stats.norm.ppf(np.clip(prop_below, 0.001, 0.999))

    acceleration = np.zeros_like(original, dtype=np.float64)
    if len(jackknife) >= 3:
        jack_stacked = np.stack(jackknife, axis=0)
        jack_mean = jack_stacked.mean(axis=0)
        jack_diff = jack_mean - jack_stacked
        numerator = np.sum(jack_diff**3, axis=0)
        denominator = 6 * np.sum(jack_diff**2, axis=0) ** 1.5
        acceleration = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > PROB_EPSILON,
        )

    z_lower = stats.norm.ppf(alpha / 2)
    z_upper = stats.norm.ppf(1 - alpha / 2)

    def adjusted_quantile(z_alpha: float) -> NDArray[np.float64]:
        numerator = z0 + z_alpha
        denominator = 1 - acceleration * numerator
        denominator = np.where(
            np.abs(denominator) < PROB_EPSILON,
            np.copysign(PROB_EPSILON, denominator),
            denominator,
        )
        return np.clip(stats.norm.cdf(z0 + numerator / denominator), 0.001, 0.999)

    lower = _elementwise_percentile(samples, adjusted_quantile(z_lower))
    upper = _elementwise_percentile(samples, adjusted_quantile(z_upper))
    return lower, upper


def _simulate_model_responses(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    n_persons = theta.shape[0]

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.ndim == 2:
        expected_shape = (n_persons, model.n_items)
        if probabilities.shape != expected_shape:
            raise MirtModelError(
                "Binary probability output has an unexpected shape",
                model_type=model.model_name,
                value=probabilities.shape,
                expected=str(expected_shape),
            )
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < -PROB_EPSILON) | (probabilities > 1 + PROB_EPSILON)
        ):
            raise MirtModelError(
                "Binary probabilities must be finite and within [0, 1]"
            )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return (rng.random(expected_shape) < probabilities).astype(np.int_)

    if probabilities.ndim != 3 or probabilities.shape[:2] != (
        n_persons,
        model.n_items,
    ):
        raise MirtModelError(
            "Categorical probability output has an unexpected shape",
            model_type=model.model_name,
            value=probabilities.shape,
            expected=f"({n_persons}, {model.n_items}, n_categories)",
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -PROB_EPSILON):
        raise MirtModelError("Categorical probabilities must be finite and nonnegative")

    probabilities = np.maximum(probabilities, 0.0)
    totals = probabilities.sum(axis=2, keepdims=True)
    if np.any(totals <= PROB_EPSILON):
        raise MirtModelError("Each categorical probability row must have positive mass")
    normalized = probabilities / totals
    cumulative = np.cumsum(normalized, axis=2)

    category_counts = getattr(model, "n_categories", None)
    if category_counts is None:
        category_counts = [probabilities.shape[2]] * model.n_items
    elif isinstance(category_counts, int):
        category_counts = [category_counts] * model.n_items
    if len(category_counts) != model.n_items:
        raise MirtModelError("Category counts must match the number of items")
    for item_idx, n_categories in enumerate(category_counts):
        if (
            not isinstance(n_categories, (int, np.integer))
            or n_categories < 2
            or n_categories > probabilities.shape[2]
        ):
            raise MirtModelError(
                "Category counts must be valid for the probability output"
            )
        cumulative[:, item_idx, n_categories - 1 :] = 1.0

    uniforms = rng.random((n_persons, model.n_items, 1))
    return (uniforms > cumulative).sum(axis=2).astype(np.int_)


def _fit_parametric_replicate(
    task: _ParametricFitTask,
    replicate_rng: np.random.Generator,
) -> tuple[dict[str, NDArray[np.float64]] | None, str | None]:
    """Simulate and fit one parametric-bootstrap replicate."""
    from mirt.estimation.em import EMEstimator

    theta = replicate_rng.standard_normal((task.n_persons, task.model.n_factors))
    sim_data = _simulate_model_responses(task.model, theta, replicate_rng)
    boot_model = _prepare_bootstrap_model(
        task.model,
        task.original_params,
        task.warm_start,
    )
    try:
        estimator = EMEstimator(max_iter=task.max_iter, tol=1e-3, verbose=False)
        result = estimator.fit(boot_model, sim_data)
        return (
            {
                name: np.asarray(values, dtype=np.float64).copy()
                for name, values in result.model.parameters.items()
            },
            None,
        )
    except _BOOTSTRAP_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _fit_parametric_task(
    task: _ParametricFitTask,
) -> list[tuple[dict[str, NDArray[np.float64]] | None, str | None]]:
    """Simulate and fit one deterministic worker chunk."""
    task_results = []
    for rng_state in task.rng_states:
        replicate_rng = np.random.default_rng()
        replicate_rng.bit_generator.state = rng_state
        task_results.append(_fit_parametric_replicate(task, replicate_rng))
    return task_results


def _native_2pl_bootstrap_samples(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_bootstrap: int,
    seed: int | None,
    warm_start: bool,
) -> dict[str, NDArray[np.float64]] | None:
    """Return native parallel parameter samples for eligible 2PL models."""
    from mirt.backends.rust._helpers import rust_enabled
    from mirt.backends.rust.estimation import bootstrap_fit_2pl
    from mirt.models.dichotomous import TwoParameterLogistic

    if (
        not rust_enabled()
        or not isinstance(model, TwoParameterLogistic)
        or model.model_name != "2PL"
    ):
        return None
    if model.n_factors != 1:
        return None

    parameters = model.parameters
    initial_discrimination = parameters["discrimination"] if warm_start else None
    initial_difficulty = parameters["difficulty"] if warm_start else None
    rng = np.random.default_rng(seed)
    native_seed = int(rng.integers(0, 2**31))
    discrimination, difficulty = bootstrap_fit_2pl(
        responses,
        n_bootstrap=n_bootstrap,
        n_quadpts=21,
        max_iter=100 if warm_start else 200,
        tol=1e-3,
        seed=native_seed,
        initial_discrimination=initial_discrimination,
        initial_difficulty=initial_difficulty,
    )

    expected_shape = (n_bootstrap, model.n_items)
    samples = {
        "discrimination": np.asarray(discrimination, dtype=np.float64),
        "difficulty": np.asarray(difficulty, dtype=np.float64),
    }
    if any(values.shape != expected_shape for values in samples.values()):
        raise MirtModelError(
            "Native bootstrap returned an unexpected parameter shape",
            model_type=model.model_name,
            expected=str(expected_shape),
        )
    if any(not np.all(np.isfinite(values)) for values in samples.values()):
        raise MirtModelError("Native bootstrap returned non-finite parameters")
    return samples


def bootstrap_se(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    n_bootstrap: int = 200,
    statistic: Literal["parameters", "theta"] | Callable = "parameters",
    seed: int | None = None,
    verbose: bool = False,
    warm_start: bool = True,
    n_jobs: int = 1,
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
    n_jobs : int
        Number of process workers for the general Python implementation.
        Use ``-1`` for all available CPU cores. The default ``1`` preserves
        serial execution and is preferable for small fits. The native 2PL path
        manages its own parallelism.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and SE arrays as values

    Notes
    -----
    Parameter bootstraps for unidimensional 2PL models use the native parallel
    implementation when that backend is enabled. Other models and statistics
    retain the general Python implementation. Parallel custom models and
    statistic callables must be picklable; define them at module scope. Seeded
    results are deterministic and retain input order across worker counts.
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        model = model.model

    _validate_resample_count(n_bootstrap)
    _validate_statistic(statistic)
    n_jobs = _validate_n_jobs(n_jobs)

    responses = validate_responses(responses, n_items=model.n_items)
    n_persons = responses.shape[0]
    if statistic == "parameters":
        native_samples = _native_2pl_bootstrap_samples(
            model, responses, n_bootstrap, seed, warm_start
        )
        if native_samples is not None:
            return {
                name: np.std(values, axis=0, ddof=1)
                for name, values in native_samples.items()
            }

    rng = np.random.default_rng(seed)

    boot_estimates: dict[str, list[NDArray]] = {}

    max_iter = 100 if warm_start else 200
    original_params = {k: v.copy() for k, v in model.parameters.items()}

    sample_indices = [
        rng.integers(0, n_persons, size=n_persons) for _ in range(n_bootstrap)
    ]
    replicate_tasks = [
        _StatisticFitTask(
            model=model,
            original_params=original_params,
            warm_start=warm_start,
            max_iter=max_iter,
            responses=responses,
            sample_indices=index_chunk,
            statistic=statistic,
        )
        for index_chunk in _chunk_values(sample_indices, n_jobs)
    ]
    chunk_results = _run_bootstrap_tasks(
        _fit_statistic_task,
        replicate_tasks,
        n_jobs,
    )
    replicate_results = [result for chunk in chunk_results for result in chunk]
    for b, (values_by_name, error) in enumerate(replicate_results, start=1):
        if verbose and b % 50 == 0:
            print(f"Bootstrap sample {b}/{n_bootstrap}")
        if error is not None:
            if verbose:
                warnings.warn(
                    f"Bootstrap replicate failed and was skipped: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        assert values_by_name is not None
        for name, values in values_by_name.items():
            boot_estimates.setdefault(name, []).append(values)

    se_results: dict[str, NDArray[np.float64]] = {}
    for name, estimates in boot_estimates.items():
        if len(estimates) > 1:
            stacked = np.stack(estimates, axis=0)
            se_results[name] = np.std(stacked, axis=0, ddof=1)
        else:
            se_results[name] = np.full_like(estimates[0], np.nan, dtype=np.float64)

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
    n_jobs: int = 1,
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
    n_jobs : int
        Number of process workers for bootstrap and jackknife fits. Use ``-1``
        for all available CPU cores. The default is serial execution and is
        preferable for small fits. The native 2PL path manages its own
        parallelism.

    Returns
    -------
    dict
        Dictionary with parameter names as keys and (lower, upper) CI tuples

    Notes
    -----
    Parameter bootstraps for unidimensional 2PL models use the native parallel
    implementation when that backend is enabled. Other models and statistics
    retain the general Python implementation. Parallel custom models and
    statistic callables must be picklable; define them at module scope. Seeded
    results are deterministic and retain input order across worker counts.
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        original_model = model.model
    else:
        original_model = model

    _validate_resample_count(n_bootstrap)
    _validate_statistic(statistic)
    _validate_ci_configuration(alpha, method)
    n_jobs = _validate_n_jobs(n_jobs)

    rng = np.random.default_rng(seed)
    responses = validate_responses(responses, n_items=original_model.n_items)
    n_persons = responses.shape[0]

    original_estimates: dict[str, NDArray[np.float64]] = {}
    if statistic == "parameters":
        original_estimates = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in original_model.parameters.items()
        }
    elif statistic == "theta":
        from mirt.scoring import fscores

        scores = fscores(original_model, responses, method="EAP")
        original_estimates["theta"] = np.asarray(scores.theta, dtype=np.float64)
    elif callable(statistic):
        original_estimates = _as_statistic_mapping(statistic(original_model, responses))

    boot_estimates: dict[str, list[NDArray[np.float64]]] = {
        name: [] for name in original_estimates
    }

    original_params = {k: v.copy() for k, v in original_model.parameters.items()}

    max_iter = 100 if warm_start else 200

    native_samples = None
    if statistic == "parameters":
        native_samples = _native_2pl_bootstrap_samples(
            original_model, responses, n_bootstrap, seed, warm_start
        )
    if native_samples is not None:
        for name, samples in native_samples.items():
            if (
                name in boot_estimates
                and samples.shape[1:] == original_estimates[name].shape
            ):
                boot_estimates[name].extend(samples)
    else:
        sample_indices = [
            rng.integers(0, n_persons, size=n_persons) for _ in range(n_bootstrap)
        ]
        replicate_tasks = [
            _StatisticFitTask(
                model=original_model,
                original_params=original_params,
                warm_start=warm_start,
                max_iter=max_iter,
                responses=responses,
                sample_indices=index_chunk,
                statistic=statistic,
            )
            for index_chunk in _chunk_values(sample_indices, n_jobs)
        ]
        chunk_results = _run_bootstrap_tasks(
            _fit_statistic_task,
            replicate_tasks,
            n_jobs,
        )
        replicate_results = [result for chunk in chunk_results for result in chunk]
        for b, (values_by_name, error) in enumerate(replicate_results, start=1):
            if verbose and b % 50 == 0:
                print(f"Bootstrap sample {b}/{n_bootstrap}")
            if error is not None:
                if verbose:
                    warnings.warn(
                        f"Bootstrap CI replicate failed and was skipped: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue
            assert values_by_name is not None
            for name, values in values_by_name.items():
                if (
                    name in boot_estimates
                    and values.shape == original_estimates[name].shape
                ):
                    boot_estimates[name].append(values)

    jackknife_estimates: dict[str, list[NDArray[np.float64]]] = {
        name: [] for name in original_estimates
    }
    if method == "BCa" and any(
        len(estimates) >= 10 for estimates in boot_estimates.values()
    ):
        max_jack = min(20, n_persons)
        jack_indices = rng.choice(n_persons, size=max_jack, replace=False).tolist()
        all_indices = np.arange(n_persons, dtype=np.int64)
        jackknife_sample_indices = [
            np.delete(all_indices, index) for index in jack_indices
        ]
        jackknife_tasks = [
            _StatisticFitTask(
                model=original_model,
                original_params=original_params,
                warm_start=warm_start,
                max_iter=max_iter,
                responses=responses,
                sample_indices=index_chunk,
                statistic=statistic,
            )
            for index_chunk in _chunk_values(jackknife_sample_indices, n_jobs)
        ]
        jackknife_chunk_results = _run_bootstrap_tasks(
            _fit_statistic_task,
            jackknife_tasks,
            n_jobs,
        )
        jackknife_results = [
            result for chunk in jackknife_chunk_results for result in chunk
        ]
        for values_by_name, error in jackknife_results:
            if error is not None:
                continue
            assert values_by_name is not None
            for name, values in values_by_name.items():
                if (
                    name in jackknife_estimates
                    and values.shape == original_estimates[name].shape
                ):
                    jackknife_estimates[name].append(values)

    ci_results: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for name, estimates in boot_estimates.items():
        if len(estimates) < 10:
            original = original_estimates[name]
            ci_results[name] = (
                np.full_like(original, np.nan, dtype=np.float64),
                np.full_like(original, np.nan, dtype=np.float64),
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

        else:  # method == "BCa", validated before fitting
            lower, upper = _bca_interval(
                stacked,
                original,
                jackknife_estimates[name],
                alpha,
            )

        ci_results[name] = (lower.astype(np.float64), upper.astype(np.float64))

    return ci_results


def parametric_bootstrap(
    model: BaseItemModel | FitResult,
    n_bootstrap: int = 200,
    n_persons: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
    warm_start: bool = True,
    n_jobs: int = 1,
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
    n_jobs : int
        Number of process workers. Use ``-1`` for all available CPU cores. The
        default ``1`` preserves serial execution and is preferable for small
        fits. Custom models must be picklable when using multiple workers.

    Returns
    -------
    dict
        Standard errors for each parameter

    Notes
    -----
    Seeded simulations are deterministic and retain replicate order across
    worker counts.
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        model = model.model

    _validate_resample_count(n_bootstrap)
    n_jobs = _validate_n_jobs(n_jobs)
    if n_persons is None:
        n_persons = 500
    if (
        not isinstance(n_persons, (int, np.integer))
        or isinstance(n_persons, (bool, np.bool_))
        or n_persons < 1
    ):
        raise MirtValidationError(
            "n_persons must be a positive integer",
            parameter="n_persons",
            value=n_persons,
        )

    rng = np.random.default_rng(seed)
    boot_estimates: dict[str, list[NDArray]] = {}

    max_iter = 100 if warm_start else 200
    original_params = {k: v.copy() for k, v in model.parameters.items()}

    task_context = _ParametricFitTask(
        model=model,
        original_params=original_params,
        warm_start=warm_start,
        max_iter=max_iter,
        n_persons=int(n_persons),
        rng_states=[],
    )
    if n_jobs == 1:
        replicate_results = [
            _fit_parametric_replicate(task_context, rng) for _ in range(n_bootstrap)
        ]
    else:
        replicate_rng_states: list[dict[str, Any]] = []
        for _ in range(n_bootstrap):
            replicate_rng_states.append(deepcopy(rng.bit_generator.state))
            rng.standard_normal((n_persons, model.n_factors))
            rng.random((n_persons, model.n_items))
        replicate_tasks = [
            _ParametricFitTask(
                model=model,
                original_params=original_params,
                warm_start=warm_start,
                max_iter=max_iter,
                n_persons=int(n_persons),
                rng_states=state_chunk,
            )
            for state_chunk in _chunk_values(replicate_rng_states, n_jobs)
        ]
        chunk_results = _run_bootstrap_tasks(
            _fit_parametric_task,
            replicate_tasks,
            n_jobs,
        )
        replicate_results = [result for chunk in chunk_results for result in chunk]
    for b, (values_by_name, error) in enumerate(replicate_results, start=1):
        if verbose and b % 50 == 0:
            print(f"Parametric bootstrap {b}/{n_bootstrap}")
        if error is not None:
            if verbose:
                warnings.warn(
                    f"Parametric bootstrap replicate failed and was skipped: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        assert values_by_name is not None
        for name, values in values_by_name.items():
            boot_estimates.setdefault(name, []).append(values)

    se_results: dict[str, NDArray[np.float64]] = {}
    for name, estimates in boot_estimates.items():
        if len(estimates) > 1:
            stacked = np.stack(estimates, axis=0)
            se_results[name] = np.std(stacked, axis=0, ddof=1)
        else:
            se_results[name] = np.full_like(estimates[0], np.nan, dtype=np.float64)

    return se_results
