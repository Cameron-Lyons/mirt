"""Batch model fitting utilities.

This module provides functions for fitting multiple IRT models
and comparing them efficiently.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


ModelType = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]
OnError = Literal["raise", "skip"]
ParallelBackend = Literal["thread", "process"]

_SUPPORTED_MODELS = frozenset({"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"})
_UNIDIMENSIONAL_MODELS = frozenset({"1PL", "3PL", "4PL", "PCM"})
_FIT_FAILURES = (
    TypeError,
    ValueError,
    RuntimeError,
    ArithmeticError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class _FitTask:
    key: str
    model: ModelType
    n_categories: int | None
    n_factors: int
    n_quadpts: int
    max_iter: int
    tol: float


_PROCESS_RESPONSES: NDArray[np.int_] | None = None


def _initialize_process_worker(responses: NDArray[np.int_]) -> None:
    """Retain one response matrix per process instead of one copy per task."""
    global _PROCESS_RESPONSES
    _PROCESS_RESPONSES = responses


def _validate_models(models: Sequence[ModelType]) -> list[ModelType]:
    """Validate a non-empty, unique model sequence."""
    if isinstance(models, (str, bytes)):
        raise MirtValidationError(
            "models must be a non-empty sequence of model names",
            parameter="models",
        )

    validated = list(models)
    if not validated:
        raise MirtValidationError(
            "models must contain at least one model",
            parameter="models",
        )

    invalid = [model for model in validated if model not in _SUPPORTED_MODELS]
    if invalid:
        raise MirtValidationError(
            f"unsupported model type: {invalid[0]}",
            parameter="models",
            value=invalid[0],
        )

    if len(set(validated)) != len(validated):
        duplicate = next(
            model for index, model in enumerate(validated) if model in validated[:index]
        )
        raise MirtValidationError(
            f"duplicate model type: {duplicate}",
            parameter="models",
            value=duplicate,
        )

    return [cast(ModelType, model) for model in validated]


def _validate_integer_grid(
    values: Sequence[int] | None,
    *,
    name: str,
    default: int,
    minimum: int,
) -> list[int]:
    """Validate unique positive integer grid values."""
    validated = [default] if values is None else list(values)
    if not validated:
        raise MirtValidationError(
            f"{name} must contain at least one value",
            parameter=name,
        )
    for value in validated:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise MirtValidationError(
                f"{name} values must be integers greater than or equal to {minimum}",
                parameter=name,
                value=value,
            )
    normalized = [int(value) for value in validated]
    if len(set(normalized)) != len(normalized):
        raise MirtValidationError(
            f"{name} must not contain duplicate values",
            parameter=name,
        )
    return normalized


def _resolve_worker_count(n_jobs: int, n_tasks: int) -> int:
    """Normalize the requested worker count and cap it to useful work."""
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
    requested = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
    return min(requested, n_tasks)


def _validate_on_error(on_error: OnError) -> OnError:
    if on_error not in ("raise", "skip"):
        raise MirtValidationError(
            "on_error must be 'raise' or 'skip'",
            parameter="on_error",
            value=on_error,
        )
    return on_error


def _validate_parallel_backend(backend: ParallelBackend) -> ParallelBackend:
    if backend not in ("thread", "process"):
        raise MirtValidationError(
            "parallel_backend must be 'thread' or 'process'",
            parameter="parallel_backend",
            value=backend,
        )
    return backend


def _fit_task(task: _FitTask, responses: NDArray[np.int_]) -> FitResult:
    """Fit one batch task without importing the public API at module load."""
    from mirt import fit_mirt

    if task.n_factors != 1 and task.model in _UNIDIMENSIONAL_MODELS:
        raise MirtValidationError(
            f"{task.model} only supports n_factors=1",
            parameter="n_factors",
            value=task.n_factors,
        )

    result = fit_mirt(
        responses,
        model=task.model,
        n_categories=task.n_categories,
        n_factors=task.n_factors,
        n_quadpts=task.n_quadpts,
        max_iter=task.max_iter,
        tol=task.tol,
        verbose=False,
    )
    fit_statistics = (result.log_likelihood, result.aic, result.bic)
    if not np.all(np.isfinite(fit_statistics)):
        raise ArithmeticError("fit returned non-finite likelihood statistics")
    return result


def _fit_process_task(task: _FitTask) -> FitResult:
    """Fit a task against the matrix installed by the process initializer."""
    if _PROCESS_RESPONSES is None:
        raise RuntimeError("process worker response data were not initialized")
    return _fit_task(task, _PROCESS_RESPONSES)


def _execute_tasks(
    tasks: Sequence[_FitTask],
    responses: NDArray[np.int_],
    *,
    n_jobs: int,
    on_error: OnError,
    parallel_backend: ParallelBackend,
) -> tuple[dict[str, FitResult], dict[str, str]]:
    """Execute independent fits and retain deterministic result ordering."""
    worker_count = _resolve_worker_count(n_jobs, len(tasks))
    completed: dict[str, FitResult] = {}
    failures: dict[str, str] = {}

    def record_failure(task: _FitTask, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if on_error == "raise":
            raise RuntimeError(f"fit failed for {task.key}: {message}") from exc
        failures[task.key] = message

    if worker_count == 1:
        for task in tasks:
            try:
                completed[task.key] = _fit_task(task, responses)
            except _FIT_FAILURES as exc:
                record_failure(task, exc)
    else:
        try:
            executor: ThreadPoolExecutor | ProcessPoolExecutor
            if parallel_backend == "thread":
                executor = ThreadPoolExecutor(max_workers=worker_count)
            else:
                # Forking after the native backend initializes its thread pool can
                # deadlock the child. Spawn workers from a clean interpreter instead.
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=get_context("spawn"),
                    initializer=_initialize_process_worker,
                    initargs=(responses,),
                )
        except OSError as exc:
            raise RuntimeError(
                f"{parallel_backend} parallel backend is unavailable; "
                "use parallel_backend='thread' or n_jobs=1"
            ) from exc
        with executor:
            if parallel_backend == "thread":
                future_tasks = {
                    executor.submit(_fit_task, task, responses): task for task in tasks
                }
            else:
                future_tasks = {
                    executor.submit(_fit_process_task, task): task for task in tasks
                }
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    completed[task.key] = future.result()
                except _FIT_FAILURES as exc:
                    record_failure(task, exc)

    ordered_results = {
        task.key: completed[task.key] for task in tasks if task.key in completed
    }
    ordered_failures = {
        task.key: failures[task.key] for task in tasks if task.key in failures
    }
    return ordered_results, ordered_failures


@dataclass
class BatchFitResult:
    """Result of batch model fitting.

    Attributes
    ----------
    results : dict[str, FitResult]
        Fitted results keyed by model type.
    comparison : Any
        DataFrame comparing all models when pandas or polars is available;
        otherwise a dependency-free list of summary records.
    best_model : str
        Name of best model by BIC.
    failures : dict[str, str]
        Failed model names and their error details when ``on_error="skip"``.

    Examples
    --------
    >>> batch_result = fit_models(["1PL", "2PL", "3PL"], responses)
    >>> print(batch_result.summary())
    >>> best = batch_result[batch_result.best_model]
    """

    results: dict[str, FitResult]
    comparison: Any
    best_model: str
    failures: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, model: str) -> FitResult:
        """Get result for a specific model.

        Parameters
        ----------
        model : str
            Model name (e.g., "2PL").

        Returns
        -------
        FitResult
            The fit result for that model.
        """
        return self.results[model]

    def summary(self) -> str:
        """Generate a text summary of batch fitting results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["Batch Model Fitting Results", "=" * 60]
        lines.append(f"Models fitted: {len(self.results)}")
        if self.failures:
            lines.append(f"Models failed: {len(self.failures)}")
        lines.append(f"Best model (BIC): {self.best_model}")
        lines.append("-" * 60)
        lines.append(
            f"{'Model':<10} {'LogLik':>12} {'AIC':>12} {'BIC':>12} {'Conv':>8}"
        )
        lines.append("-" * 60)
        for name, result in self.results.items():
            marker = " *" if name == self.best_model else ""
            lines.append(
                f"{name:<10} {result.log_likelihood:>12.2f} "
                f"{result.aic:>12.2f} {result.bic:>12.2f} "
                f"{str(result.converged):>8}{marker}"
            )
        lines.append("-" * 60)
        lines.append("* = best model by BIC")
        return "\n".join(lines)

    def get_best_result(self) -> FitResult:
        """Get the result for the best model.

        Returns
        -------
        FitResult
            The fit result for the best model by BIC.
        """
        return self.results[self.best_model]


class GridFitResult(dict[str, "FitResult"]):
    """Dictionary-compatible grid results with retained failure details.

    ``GridFitResult`` behaves like the dictionary returned by earlier
    versions while making skipped combinations inspectable through
    :attr:`failures`.
    """

    def __init__(
        self,
        results: dict[str, FitResult] | None = None,
        *,
        failures: dict[str, str] | None = None,
    ) -> None:
        super().__init__(results or {})
        self.failures = failures or {}


def _comparison_records(
    results: dict[str, FitResult],
) -> list[dict[str, str | float | bool]]:
    """Build a dependency-free comparison when no dataframe backend exists."""
    return [
        {
            "Model": name,
            "LogLik": float(result.log_likelihood),
            "AIC": float(result.aic),
            "BIC": float(result.bic),
            "Converged": bool(result.converged),
        }
        for name, result in results.items()
    ]


def fit_models(
    models: Sequence[ModelType],
    responses: NDArray[np.int_],
    n_categories: int | None = None,
    n_factors: int = 1,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    n_jobs: int = 1,
    on_error: OnError = "raise",
    parallel_backend: ParallelBackend = "thread",
) -> BatchFitResult:
    """Fit multiple IRT models to the same data.

    This function fits multiple model types to the same response data
    and returns a comparison of all models with information criteria.

    Parameters
    ----------
    models : sequence of ModelType
        Unique model types to fit (e.g., ["1PL", "2PL", "3PL"]).
    responses : NDArray
        Response matrix (n_persons, n_items).
    n_categories : int, optional
        Number of categories for polytomous models.
    n_factors : int, default=1
        Number of latent factors.
    n_quadpts : int, default=21
        Quadrature points for EM.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print progress.
    n_jobs : int, default=1
        Number of concurrent fits. Use -1 for all available CPUs.
    on_error : {"raise", "skip"}, default="raise"
        Raise on the first failed fit or retain failures in the result and
        continue with successful models.
    parallel_backend : {"thread", "process"}, default="thread"
        Standard-library executor used when ``n_jobs`` is greater than one.
        Threads minimize transfer overhead; processes can improve CPU-bound
        grids. Process workers receive one response-matrix copy at startup
        and reuse it for every assigned fit.

    Returns
    -------
    BatchFitResult
        Results containing all fitted models and comparison.

    Examples
    --------
    >>> from mirt import load_dataset
    >>> from mirt.utils.batch import fit_models
    >>> data = load_dataset("LSAT6")
    >>> batch_result = fit_models(
    ...     models=["1PL", "2PL", "3PL"],
    ...     responses=data["data"],
    ... )
    >>> print(batch_result.summary())
    >>> best_result = batch_result[batch_result.best_model]

    Notes
    -----
    The best model is determined by BIC (Bayesian Information Criterion),
    which balances model fit with model complexity.
    """
    from mirt.diagnostics.comparison import compare_models

    validated_models = _validate_models(models)
    validated_responses = validate_responses(responses)
    on_error = _validate_on_error(on_error)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    tasks = [
        _FitTask(
            key=model,
            model=model,
            n_categories=n_categories,
            n_factors=n_factors,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
        )
        for model in validated_models
    ]

    for model_type in validated_models:
        if verbose:
            print(f"Fitting {model_type}...")

    results, failures = _execute_tasks(
        tasks,
        validated_responses,
        n_jobs=n_jobs,
        on_error=on_error,
        parallel_backend=parallel_backend,
    )
    if not results:
        details = "; ".join(f"{key}: {error}" for key, error in failures.items())
        raise RuntimeError(f"all requested model fits failed: {details}")

    if verbose:
        for model_type in validated_models:
            if model_type in failures:
                print(f"  {model_type} failed: {failures[model_type]}")
                continue
            result = results[model_type]
            print(
                f"  {model_type}: LL={result.log_likelihood:.2f}, "
                f"converged={result.converged}"
            )

    result_list = list(results.values())
    try:
        comparison = compare_models(result_list)
    except ImportError:
        comparison = _comparison_records(results)

    bic_values = {name: r.bic for name, r in results.items()}
    best_model = min(bic_values, key=lambda k: bic_values[k])

    return BatchFitResult(
        results=results,
        comparison=comparison,
        best_model=best_model,
        failures=failures,
    )


def fit_model_grid(
    models: Sequence[ModelType],
    responses: NDArray[np.int_],
    n_factors_range: Sequence[int] | None = None,
    n_quadpts_range: Sequence[int] | None = None,
    n_categories: int | None = None,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    n_jobs: int = 1,
    on_error: OnError = "skip",
    parallel_backend: ParallelBackend = "thread",
) -> GridFitResult:
    """Fit models across a grid of hyperparameters.

    This function performs a grid search over model types and
    hyperparameters, returning all fitted models.

    Parameters
    ----------
    models : sequence of ModelType
        Unique model types to fit.
    responses : ndarray
        Response matrix (n_persons, n_items).
    n_factors_range : sequence of int, optional
        Range of factor counts to try (for MIRT). Default: [1].
    n_quadpts_range : sequence of int, optional
        Range of quadrature points to try. Default: [21].
    n_categories : int, optional
        Number of categories for polytomous models.
    max_iter : int, default=500
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print progress.
    n_jobs : int, default=1
        Number of concurrent fits. Use -1 for all available CPUs.
    on_error : {"raise", "skip"}, default="skip"
        Raise when a combination fails or retain the error in
        ``result.failures`` and continue.
    parallel_backend : {"thread", "process"}, default="thread"
        Standard-library executor used when ``n_jobs`` is greater than one.

    Returns
    -------
    GridFitResult
        Dictionary-compatible results keyed by
        ``"model_f{n_factors}_q{n_quadpts}"``. Skipped combination details
        are available through ``result.failures``.

    Examples
    --------
    >>> results = fit_model_grid(
    ...     models=["2PL"],
    ...     responses=data,
    ...     n_factors_range=[1, 2, 3],
    ...     n_quadpts_range=[11, 21, 31],
    ... )
    >>> for key, result in results.items():
    ...     print(f"{key}: BIC={result.bic:.2f}")
    """
    validated_models = _validate_models(models)
    factors = _validate_integer_grid(
        n_factors_range,
        name="n_factors_range",
        default=1,
        minimum=1,
    )
    quadrature_points = _validate_integer_grid(
        n_quadpts_range,
        name="n_quadpts_range",
        default=21,
        minimum=5,
    )
    validated_responses = validate_responses(responses)
    on_error = _validate_on_error(on_error)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    tasks = [
        _FitTask(
            key=f"{model}_f{n_factors}_q{n_quadpts}",
            model=model,
            n_categories=n_categories,
            n_factors=n_factors,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
        )
        for model in validated_models
        for n_factors in factors
        for n_quadpts in quadrature_points
    ]

    if verbose:
        for task in tasks:
            print(f"Fitting {task.key}...")

    results, failures = _execute_tasks(
        tasks,
        validated_responses,
        n_jobs=n_jobs,
        on_error=on_error,
        parallel_backend=parallel_backend,
    )

    if verbose:
        for task in tasks:
            if task.key in failures:
                print(f"  {task.key} failed: {failures[task.key]}")
            else:
                result = results[task.key]
                print(
                    f"  {task.key}: BIC={result.bic:.2f}, converged={result.converged}"
                )

    return GridFitResult(results, failures=failures)


__all__ = [
    "fit_models",
    "fit_model_grid",
    "BatchFitResult",
    "GridFitResult",
]
