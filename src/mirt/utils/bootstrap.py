"""Bootstrap methods for standard errors, confidence intervals and tests.

This module provides bootstrap procedures for:
- Standard error estimation
- Confidence interval construction
- Parameter uncertainty quantification
- Likelihood-ratio tests of nested models
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from mirt._categorical import draw_item_responses
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.utils._parallel import _process_pool, ensure_picklable, resolve_n_jobs
from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.estimation._refit import RefitRecipe
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
# Bootstrap-based diagnostics resolve their worker counts through this name.
_validate_n_jobs = resolve_n_jobs
# Convergence tolerance of parameter bootstrap replicates.
_REPLICATE_TOL = 1e-3


@dataclass(slots=True)
class _StatisticFitTask:
    model: BaseItemModel
    original_params: dict[str, NDArray[np.float64]]
    warm_start: bool
    max_iter: int
    responses: NDArray[np.int_]
    statistic: Literal["parameters", "theta"] | Callable[..., Any]
    sample_indices: list[NDArray[np.int64]] | None = None
    omitted_indices: list[int] | None = None
    resample_rng_state: dict[str, Any] | None = None
    n_resamples: int = 0
    statistic_shapes: dict[str, tuple[int, ...]] | None = None
    recipe: RefitRecipe | None = None


def _refit_source(
    model_or_result: BaseItemModel | FitResult,
) -> tuple[BaseItemModel, RefitRecipe | None]:
    """Return the item model and the recipe its refits follow, if recorded.

    A fit without a recipe, for example one rebuilt by ``FitResult.from_dict``,
    is refitted by the default estimator of its model family; a warning says
    so when that drops the fit's item priors or latent population.
    """
    from mirt.results.fit_result import FitResult

    if not isinstance(model_or_result, FitResult):
        return model_or_result, None
    recipe = model_or_result.refit_recipe
    if recipe is None and (
        model_or_result.log_posterior is not None
        or model_or_result.latent_mean is not None
        or model_or_result.latent_covariance is not None
    ):
        warnings.warn(
            "the fit records no refit settings, so refits use the default "
            "estimator of its model family, without item priors and with "
            "standard normal abilities",
            RuntimeWarning,
            stacklevel=3,
        )
    return model_or_result.model, recipe


def _uses_default_estimator(recipe: RefitRecipe | None) -> bool:
    """Whether refits estimate what the native 2PL bootstrap estimates."""
    from mirt.estimation.em import EMEstimator

    return recipe is None or (
        recipe.estimator is EMEstimator
        and recipe.latent_density is None
        and recipe.options.get("item_priors") is None
        and not recipe.options.get("constraints")
    )


def _replicate_estimator(
    model: BaseItemModel, recipe: RefitRecipe | None, max_iter: int
) -> Any:
    """Return the estimator of one parameter replicate.

    Replicates keep the original fit's estimator and settings but use their
    own iteration limit and tolerance, and skip standard errors, which no
    bootstrap statistic uses.
    """
    from mirt.estimation._refit import em_estimator_for

    return em_estimator_for(
        model,
        recipe=recipe,
        max_iter=max_iter,
        tol=_REPLICATE_TOL,
        verbose=False,
        compute_standard_errors=False,
    )


def _population_factors(
    model_or_result: BaseItemModel | FitResult,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Return the mean and Cholesky factor of a fit's latent population.

    ``None`` entries mean the standard-normal default.
    """
    from mirt.results._common import resolve_latent_prior

    _, mean, cov = resolve_latent_prior(model_or_result)
    return (
        None if mean is None else np.asarray(mean, dtype=np.float64),
        None if cov is None else np.linalg.cholesky(np.asarray(cov, dtype=np.float64)),
    )


def _draw_abilities(
    rng: np.random.Generator,
    n_persons: int,
    n_factors: int,
    mean: NDArray[np.float64] | None,
    cholesky: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Draw abilities from ``N(mean, cholesky @ cholesky.T)``.

    The standard-normal draws are the same for every population, so seeded
    simulations of a standard-normal population keep their random stream.
    """
    theta = rng.standard_normal((n_persons, n_factors))
    if cholesky is not None:
        theta = theta @ cholesky.T
    if mean is not None:
        theta += mean
    return theta


@dataclass(slots=True)
class _JackknifeMoments:
    """Mergeable central moments using scaled differences from a fixed origin.

    Normalizing before powers keeps acceleration independent of statistic
    units. Subtracting the origin first preserves differences at large offsets;
    only coordinates whose subtraction overflows use a scaled subtraction.
    Storage depends on statistic shape, never on the number of omitted people.
    """

    count: int
    reference: NDArray[np.float64]
    scale: NDArray[np.float64]
    mean: NDArray[np.float64]
    m2: NDArray[np.float64]
    m3: NDArray[np.float64]

    @classmethod
    def from_value(cls, value: NDArray[np.float64]) -> _JackknifeMoments:
        value = np.asarray(value, dtype=np.float64)
        return cls(1, value.copy(), *(np.zeros_like(value) for _ in range(4)))

    def _normalized_difference(
        self, value: NDArray[np.float64], other_scale: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = value - self.reference
        overflow = ~np.isfinite(difference)
        distance = np.where(
            overflow,
            np.maximum(np.abs(value), np.abs(self.reference)),
            np.abs(difference),
        )
        scale = np.maximum(np.maximum(self.scale, other_scale), distance)
        normalized = np.divide(
            difference,
            scale,
            out=np.zeros_like(difference),
            where=(scale > 0.0) & ~overflow,
        )
        if np.any(overflow):
            # Applying this fallback only to overflowing coordinates prevents
            # one extreme parameter from losing precision in another parameter.
            normalized = np.where(
                overflow,
                np.divide(value, scale, out=np.zeros_like(value), where=scale > 0.0)
                - np.divide(
                    self.reference,
                    scale,
                    out=np.zeros_like(value),
                    where=scale > 0.0,
                ),
                normalized,
            )
        return normalized, scale

    def add(self, value: NDArray[np.float64]) -> None:
        """Accumulate one statistic without retaining the input array."""
        normalized, scale = self._normalized_difference(value, self.scale)
        ratio = np.divide(
            self.scale, scale, out=np.zeros_like(scale), where=scale > 0.0
        )
        self.mean *= ratio
        self.m2 *= ratio**2
        self.m3 *= ratio**3
        self.scale = scale

        previous_count = self.count
        self.count += 1
        delta = normalized - self.mean
        mean_delta = delta / self.count
        term = delta * mean_delta * previous_count
        self.m3 += term * mean_delta * (self.count - 2) - 3 * mean_delta * self.m2
        self.m2 += term
        self.mean += mean_delta

    def merge(self, other: _JackknifeMoments) -> None:
        """Combine worker summaries in deterministic task order."""
        origin_difference, scale = self._normalized_difference(
            other.reference, other.scale
        )
        left_ratio = np.divide(
            self.scale, scale, out=np.zeros_like(scale), where=scale > 0.0
        )
        right_ratio = np.divide(
            other.scale, scale, out=np.zeros_like(scale), where=scale > 0.0
        )
        left_mean = self.mean * left_ratio
        right_mean = origin_difference + other.mean * right_ratio
        left_m2, right_m2 = self.m2 * left_ratio**2, other.m2 * right_ratio**2
        left_m3, right_m3 = self.m3 * left_ratio**3, other.m3 * right_ratio**3
        delta = right_mean - left_mean
        left_count, right_count = self.count, other.count
        count = left_count + right_count
        cross_weight = left_count * right_count / count
        self.m3 = (
            left_m3
            + right_m3
            + delta**3 * cross_weight * (left_count - right_count) / count
            + 3 * delta * (left_count * right_m2 - right_count * left_m2) / count
        )
        self.m2 = left_m2 + right_m2 + delta**2 * cross_weight
        self.mean = left_mean + delta * right_count / count
        self.scale = scale
        self.count = count

    def acceleration(self) -> NDArray[np.float64]:
        denominator = 6 * np.maximum(self.m2, 0.0) ** 1.5
        return np.divide(
            -self.m3,
            denominator,
            out=np.zeros_like(self.m3),
            where=denominator > 0.0,
        )


@dataclass(slots=True)
class _ParametricFitTask:
    model: BaseItemModel
    original_params: dict[str, NDArray[np.float64]]
    warm_start: bool
    max_iter: int
    n_persons: int
    rng_states: list[dict[str, Any]]
    recipe: RefitRecipe | None = None
    latent_mean: NDArray[np.float64] | None = None
    latent_cholesky: NDArray[np.float64] | None = None


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


def _run_bootstrap_tasks(
    function: Callable[[_TaskInput], _TaskResult],
    inputs: list[_TaskInput],
    n_jobs: int,
) -> list[_TaskResult]:
    """Run independent bootstrap tasks in deterministic input order."""
    if n_jobs == 1 or len(inputs) < 2:
        return [function(value) for value in inputs]

    ensure_picklable(function, inputs[0], n_jobs=n_jobs)
    with _process_pool(min(n_jobs, len(inputs))) as executor:
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


def _chunk_sizes(n_values: int, n_chunks: int) -> list[int]:
    """Return balanced sizes for contiguous chunks without materializing values."""
    chunk_count = min(n_chunks, n_values)
    quotient, remainder = divmod(n_values, chunk_count)
    return [quotient + (chunk_index < remainder) for chunk_index in range(chunk_count)]


def _resample_rng_chunks(
    rng: np.random.Generator,
    n_resamples: int,
    n_persons: int,
    n_chunks: int,
) -> list[tuple[dict[str, Any], int]]:
    """Capture compact worker states while preserving the seeded random stream."""
    chunks: list[tuple[dict[str, Any], int]] = []
    for chunk_size in _chunk_sizes(n_resamples, n_chunks):
        chunks.append((deepcopy(rng.bit_generator.state), chunk_size))
        for _ in range(chunk_size):
            rng.integers(0, n_persons, size=n_persons)
    return chunks


def _iter_sample_indices(task: _StatisticFitTask) -> Iterator[NDArray[np.int64]]:
    """Yield explicit, jackknife, or generated indices for one worker task."""
    if task.sample_indices is not None:
        yield from task.sample_indices
        return

    if task.omitted_indices is not None:
        all_indices = np.arange(task.responses.shape[0], dtype=np.int64)
        for omitted in task.omitted_indices:
            yield np.delete(all_indices, omitted)
        return

    if task.resample_rng_state is None:
        raise RuntimeError("A bootstrap task requires indices or random state")
    rng = np.random.default_rng()
    rng.bit_generator.state = task.resample_rng_state
    n_persons = task.responses.shape[0]
    for _ in range(task.n_resamples):
        yield np.asarray(
            rng.integers(0, n_persons, size=n_persons),
            dtype=np.int64,
        )


def _resample_fit_tasks(
    model: BaseItemModel,
    original_params: dict[str, NDArray[np.float64]],
    warm_start: bool,
    max_iter: int,
    responses: NDArray[np.int_],
    statistic: Literal["parameters", "theta"] | Callable[..., Any],
    rng: np.random.Generator,
    n_resamples: int,
    n_jobs: int,
    recipe: RefitRecipe | None = None,
) -> list[_StatisticFitTask]:
    """Build constant-size worker tasks for nonparametric resampling."""
    return [
        _StatisticFitTask(
            model=model,
            original_params=original_params,
            warm_start=warm_start,
            max_iter=max_iter,
            responses=responses,
            statistic=statistic,
            resample_rng_state=rng_state,
            n_resamples=chunk_size,
            recipe=recipe,
        )
        for rng_state, chunk_size in _resample_rng_chunks(
            rng,
            n_resamples,
            responses.shape[0],
            n_jobs,
        )
    ]


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
    """Copy a model for refitting from its original estimates.

    A warm start keeps every estimate as a starting value. A cold start lets
    EM reinitialize free coordinates while fixed coordinates keep their values.
    """
    boot_model = model.copy()
    boot_model._parameters = {
        name: values.copy() for name, values in original_params.items()
    }
    # EM reinitializes the free coordinates of unfitted models only.
    boot_model._is_fitted = bool(warm_start)
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
    return list(_iter_statistic_fits(task))


def _iter_statistic_fits(
    task: _StatisticFitTask,
) -> Iterator[tuple[dict[str, NDArray[np.float64]] | None, str | None]]:
    """Yield fit results so jackknife statistics can be reduced immediately."""
    for indices in _iter_sample_indices(task):
        fit_responses = task.responses[indices]
        boot_model = _prepare_bootstrap_model(
            task.model,
            task.original_params,
            task.warm_start,
        )
        try:
            estimator = _replicate_estimator(boot_model, task.recipe, task.max_iter)
            result = estimator.fit(boot_model, fit_responses)

            if task.statistic == "parameters":
                values_by_name = {
                    name: np.asarray(values, dtype=np.float64).copy()
                    for name, values in result.model.parameters.items()
                }
            elif task.statistic == "theta":
                from mirt.scoring import fscores

                # Each replicate scores against its own estimated population.
                scores = fscores(result, task.responses, method="EAP")
                values_by_name = {
                    "theta": np.asarray(scores.theta, dtype=np.float64).copy()
                }
            else:
                values_by_name = _as_statistic_mapping(
                    task.statistic(result.model, fit_responses)
                )
            yield values_by_name, None
        except _BOOTSTRAP_EXCEPTIONS as exc:
            yield None, f"{type(exc).__name__}: {exc}"


def _fit_jackknife_task(task: _StatisticFitTask) -> dict[str, _JackknifeMoments]:
    """Return fixed-size moments, rather than all leave-one-out statistics."""
    assert task.statistic_shapes is not None
    summaries: dict[str, _JackknifeMoments] = {}
    for values_by_name, error in _iter_statistic_fits(task):
        if error is not None:
            continue
        assert values_by_name is not None
        for name, values in values_by_name.items():
            if (
                name not in task.statistic_shapes
                or values.shape != task.statistic_shapes[name]
                or not np.all(np.isfinite(values))
            ):
                continue
            if name in summaries:
                summaries[name].add(values)
            else:
                summaries[name] = _JackknifeMoments.from_value(values)
    return summaries


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
    jackknife: Iterable[NDArray[np.float64]],
    alpha: float,
    *,
    acceleration: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from scipy import stats

    # Mid-ranks handle discrete statistics without treating ties as bias.
    percentile = np.mean(samples < original, axis=0) + 0.5 * np.mean(
        samples == original, axis=0
    )
    half_replicate = 0.5 / samples.shape[0]
    z0 = stats.norm.ppf(np.clip(percentile, half_replicate, 1 - half_replicate))

    if acceleration is None:
        summary = None
        for values in jackknife:
            if summary is None:
                summary = _JackknifeMoments.from_value(values)
            else:
                summary.add(values)
        acceleration = (
            np.zeros_like(original, dtype=np.float64)
            if summary is None
            else summary.acceleration()
        )

    z_lower = stats.norm.ppf(alpha / 2)
    z_upper = stats.norm.ppf(1 - alpha / 2)

    def adjusted_quantile(z_alpha: float) -> NDArray[np.float64]:
        numerator = z0 + z_alpha
        denominator = 1 - acceleration * numerator
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return stats.norm.cdf(z0 + numerator / denominator)

    lower = _elementwise_percentile(samples, adjusted_quantile(z_lower))
    upper = _elementwise_percentile(samples, adjusted_quantile(z_upper))
    return lower, upper


def _fit_parametric_replicate(
    task: _ParametricFitTask,
    replicate_rng: np.random.Generator,
) -> tuple[dict[str, NDArray[np.float64]] | None, str | None]:
    """Simulate and fit one parametric-bootstrap replicate."""
    theta = _draw_abilities(
        replicate_rng,
        task.n_persons,
        task.model.n_factors,
        task.latent_mean,
        task.latent_cholesky,
    )
    sim_data = draw_item_responses(task.model, theta, replicate_rng)
    boot_model = _prepare_bootstrap_model(
        task.model,
        task.original_params,
        task.warm_start,
    )
    try:
        estimator = _replicate_estimator(boot_model, task.recipe, task.max_iter)
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
    recipe: RefitRecipe | None = None,
) -> dict[str, NDArray[np.float64]] | None:
    """Return native parallel parameter samples for eligible 2PL models.

    The native kernel fits plain marginal maximum likelihood under a standard
    normal population, so fits with item priors, another latent density or
    equality constraints are refitted by their own estimator instead.
    """
    from mirt.backends.rust._helpers import rust_enabled
    from mirt.backends.rust.estimation import bootstrap_fit_2pl
    from mirt.models.dichotomous import TwoParameterLogistic

    if (
        not rust_enabled()
        or not isinstance(model, TwoParameterLogistic)
        or model.model_name != "2PL"
        or not _uses_default_estimator(recipe)
    ):
        return None
    # The native kernel estimates every parameter, so fixed ones need EM.
    if model.n_factors != 1 or model._free_parameter_restrictions:
        return None

    parameters = model.parameters
    initial_discrimination = parameters["discrimination"] if warm_start else None
    initial_difficulty = parameters["difficulty"] if warm_start else None
    rng = np.random.default_rng(seed)
    native_seed = int(rng.integers(0, 2**31))
    discrimination, difficulty = bootstrap_fit_2pl(
        responses,
        n_bootstrap=n_bootstrap,
        n_quadpts=21 if recipe is None else recipe.options.get("n_quadpts", 21),
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
        Fitted model or fit result. Replicates of a ``FitResult`` are refitted
        like the original fit (see Notes); a bare model is refitted by the
        default EM estimator of its family.
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
        Whether each replicate fit starts from the original parameter
        estimates, which significantly speeds up convergence. Otherwise EM
        reinitializes the free parameters; fixed parameters keep their values
        either way.
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
    Each replicate of a ``FitResult`` is refitted by the estimator that
    produced the fit (``FitResult.refit_recipe``), with the same item priors,
    latent density, equality constraints and quadrature, so the replicates
    estimate the same quantities as the original fit; for example, a
    ``bfactor`` fit is refitted by ``BifactorEMEstimator`` and a Bayes modal
    fit keeps its priors. Replicates use their own iteration limit and a
    tolerance of 1e-3 and skip standard errors. ``statistic='theta'`` scores
    with each replicate's estimated latent population.

    Parameter bootstraps for unidimensional 2PL models fitted by plain
    marginal maximum likelihood use the native parallel implementation when
    that backend is enabled. Other models and statistics retain the general
    Python implementation. Parallel custom models and statistic callables
    must be picklable; define them at module scope. Seeded results are
    deterministic and retain input order across worker counts.
    """
    model, recipe = _refit_source(model)

    _validate_resample_count(n_bootstrap)
    _validate_statistic(statistic)
    n_jobs = resolve_n_jobs(n_jobs)

    responses = validate_responses(responses, n_items=model.n_items)
    if statistic == "parameters":
        native_samples = _native_2pl_bootstrap_samples(
            model, responses, n_bootstrap, seed, warm_start, recipe
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

    replicate_tasks = _resample_fit_tasks(
        model,
        original_params,
        warm_start,
        max_iter,
        responses,
        statistic,
        rng,
        n_bootstrap,
        n_jobs,
        recipe,
    )
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
        Fitted model or fit result. Replicates of a ``FitResult`` are refitted
        like the original fit, as in :func:`bootstrap_se`.
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
        Whether each replicate fit starts from the original parameter
        estimates, which significantly speeds up convergence. Otherwise EM
        reinitializes the free parameters; fixed parameters keep their values
        either way.
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
    Bootstrap and jackknife fits of a ``FitResult`` use the estimator, item
    priors, latent density, equality constraints and quadrature of the
    original fit (see :func:`bootstrap_se`), and skip standard errors.
    Parameter bootstraps for unidimensional 2PL models fitted by plain
    marginal maximum likelihood use the native parallel implementation when
    that backend is enabled. Other models and statistics retain the general
    Python implementation. Parallel custom models and statistic callables
    must be picklable; define them at module scope. Seeded results are
    deterministic and retain input order across worker counts.
    BCa acceleration uses the full leave-one-person-out jackknife. It requires
    one additional fit per person; process workers share those fits. Jackknife
    samples are generated lazily, and workers reduce jackknife statistics into
    fixed-size central moments. The jackknife uses storage proportional to
    workers times statistic size, including when scoring every person's theta.
    """
    source = model
    original_model, recipe = _refit_source(model)

    _validate_resample_count(n_bootstrap)
    _validate_statistic(statistic)
    _validate_ci_configuration(alpha, method)
    n_jobs = resolve_n_jobs(n_jobs)

    rng = np.random.default_rng(seed)
    responses = validate_responses(responses, n_items=original_model.n_items)
    n_persons = responses.shape[0]
    if method == "BCa" and n_persons < 2:
        raise MirtValidationError("BCa intervals require at least two people")

    original_estimates: dict[str, NDArray[np.float64]] = {}
    if statistic == "parameters":
        original_estimates = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in original_model.parameters.items()
        }
    elif statistic == "theta":
        from mirt.scoring import fscores

        scores = fscores(source, responses, method="EAP")
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
            original_model, responses, n_bootstrap, seed, warm_start, recipe
        )
    if native_samples is not None:
        for name, samples in native_samples.items():
            if (
                name in boot_estimates
                and samples.shape[1:] == original_estimates[name].shape
            ):
                boot_estimates[name].extend(samples)
    else:
        replicate_tasks = _resample_fit_tasks(
            original_model,
            original_params,
            warm_start,
            max_iter,
            responses,
            statistic,
            rng,
            n_bootstrap,
            n_jobs,
            recipe,
        )
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

    jackknife_summaries: dict[str, _JackknifeMoments] = {}
    if method == "BCa" and any(
        len(estimates) >= 10 for estimates in boot_estimates.values()
    ):
        jackknife_tasks = [
            _StatisticFitTask(
                model=original_model,
                original_params=original_params,
                warm_start=warm_start,
                max_iter=max_iter,
                responses=responses,
                statistic=statistic,
                omitted_indices=index_chunk,
                statistic_shapes={
                    name: values.shape for name, values in original_estimates.items()
                },
                recipe=recipe,
            )
            for index_chunk in _chunk_values(list(range(n_persons)), n_jobs)
        ]
        jackknife_chunk_results = _run_bootstrap_tasks(
            _fit_jackknife_task,
            jackknife_tasks,
            n_jobs,
        )
        for chunk_summary in jackknife_chunk_results:
            for name, summary in chunk_summary.items():
                if name in jackknife_summaries:
                    jackknife_summaries[name].merge(summary)
                else:
                    jackknife_summaries[name] = summary

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

        if method == "BCa" and (
            name not in jackknife_summaries
            or jackknife_summaries[name].count != n_persons
        ):
            warnings.warn(
                f"BCa interval for {name!r} could not be computed: "
                "the full jackknife did not succeed",
                RuntimeWarning,
                stacklevel=2,
            )
            ci_results[name] = (
                np.full_like(original, np.nan, dtype=np.float64),
                np.full_like(original, np.nan, dtype=np.float64),
            )
            continue

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
                (),
                alpha,
                acceleration=jackknife_summaries[name].acceleration(),
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
        Fitted model. Abilities are drawn from the latent population of a
        ``FitResult`` (``latent_mean`` and ``latent_covariance``) and from
        the standard normal for a bare model.
    n_bootstrap : int
        Number of bootstrap samples
    n_persons : int, optional
        Number of persons to simulate (default: 500)
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
    warm_start : bool
        Whether each replicate fit starts from the original parameter
        estimates, which significantly speeds up convergence. Otherwise EM
        reinitializes the free parameters; fixed parameters keep their values
        either way.
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
    Replicates of a ``FitResult`` are refitted like the original fit, with
    its estimator, item priors, latent density, equality constraints and
    quadrature (see :func:`bootstrap_se`), and skip standard errors. Seeded
    simulations are deterministic and retain replicate order across worker
    counts.
    """
    latent_mean, latent_cholesky = _population_factors(model)
    model, recipe = _refit_source(model)

    _validate_resample_count(n_bootstrap)
    n_jobs = resolve_n_jobs(n_jobs)
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
        recipe=recipe,
        latent_mean=latent_mean,
        latent_cholesky=latent_cholesky,
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
            replace(task_context, rng_states=state_chunk)
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


@dataclass(frozen=True)
class BootstrapLRResult:
    """Parametric bootstrap likelihood-ratio test of nested models.

    Attributes
    ----------
    statistic : float
        Observed likelihood-ratio statistic ``2 * (ll_full - ll_reduced)``,
        floored at zero, from refits of both models under the replicate
        estimator settings.
    p_value : float
        Bootstrap p-value ``(1 + #{null >= statistic}) / (n_successful + 1)``.
        NaN when every replicate failed.
    null_statistics : ndarray
        Likelihood-ratio statistics of the successful replicates, which were
        simulated from the refitted reduced model.
    n_failed : int
        Number of replicates whose data could not be fitted by both models.
    df : int
        Difference in the numbers of free parameters.
    asymptotic_p_value : float
        Upper tail of the chi-square distribution with ``df`` degrees of
        freedom at ``statistic``. It is unreliable when the reduced model lies
        on the boundary of the full model, for example 2PL versus 3PL.
    reduced_log_likelihood : float
        Marginal log-likelihood of the refitted reduced model.
    full_log_likelihood : float
        Marginal log-likelihood of the refitted full model.
    """

    statistic: float
    p_value: float
    null_statistics: NDArray[np.float64]
    n_failed: int
    df: int
    asymptotic_p_value: float
    reduced_log_likelihood: float
    full_log_likelihood: float


@dataclass(slots=True)
class _LRFitTask:
    reduced: BaseItemModel
    full: BaseItemModel
    missing: NDArray[np.bool_]
    warm_start: bool
    estimator_options: dict[str, Any]
    seeds: list[np.random.SeedSequence]
    reduced_recipe: RefitRecipe | None = None
    full_recipe: RefitRecipe | None = None
    latent_mean: NDArray[np.float64] | None = None
    latent_cholesky: NDArray[np.float64] | None = None


def _refit_log_likelihood(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    warm_start: bool,
    estimator_options: Mapping[str, Any],
    recipe: RefitRecipe | None = None,
) -> FitResult:
    """Refit a model copy from its current estimates.

    Raises
    ------
    ArithmeticError
        If the refit has a non-finite log-likelihood.
    """
    from mirt.estimation._refit import em_estimator_for

    start = _prepare_bootstrap_model(model, model.parameters, warm_start)
    estimator = em_estimator_for(start, recipe=recipe, **estimator_options)
    result = estimator.fit(start, responses)
    if not np.isfinite(result.log_likelihood):
        raise ArithmeticError("fit returned a non-finite log-likelihood")
    return result


def _free_parameter_count(
    model_or_result: BaseItemModel | FitResult, recipe: RefitRecipe | None
) -> int:
    """Return the number of parameters that refits estimate.

    A fit refitted with its recipe counts what its estimator counted:
    estimated latent (co)variances, and each equality-constraint group once.
    Otherwise refits use the default estimator, which estimates the model's
    free item parameters.
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model_or_result, FitResult):
        if recipe is not None:
            return model_or_result.n_parameters
        return model_or_result.model.n_parameters
    return model_or_result.n_parameters


def _fit_lr_task(task: _LRFitTask) -> list[tuple[float, str | None]]:
    """Simulate from the reduced model and refit both models per replicate."""
    n_persons = task.missing.shape[0]
    outcomes: list[tuple[float, str | None]] = []
    for seed in task.seeds:
        rng = np.random.default_rng(seed)
        theta = _draw_abilities(
            rng,
            n_persons,
            task.reduced.n_factors,
            task.latent_mean,
            task.latent_cholesky,
        )
        simulated = draw_item_responses(task.reduced, theta, rng)
        simulated[task.missing] = -1
        try:
            reduced_ll = _refit_log_likelihood(
                task.reduced,
                simulated,
                task.warm_start,
                task.estimator_options,
                task.reduced_recipe,
            ).log_likelihood
            full_ll = _refit_log_likelihood(
                task.full,
                simulated,
                task.warm_start,
                task.estimator_options,
                task.full_recipe,
            ).log_likelihood
        except _BOOTSTRAP_EXCEPTIONS as exc:
            outcomes.append((np.nan, f"{type(exc).__name__}: {exc}"))
            continue
        outcomes.append((max(0.0, 2.0 * (full_ll - reduced_ll)), None))
    return outcomes


def bootstrap_lr(
    reduced: BaseItemModel | FitResult,
    full: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    *,
    n_bootstrap: int = 200,
    seed: int | None = None,
    n_quadpts: int = 21,
    tol: float = 1e-5,
    max_iter: int = 500,
    warm_start: bool = True,
    n_jobs: int = 1,
) -> BootstrapLRResult:
    """Parametric bootstrap likelihood-ratio test for nested models.

    The chi-square reference distribution of the likelihood-ratio statistic
    fails when the reduced model fixes a parameter on the boundary of the
    full model, such as 2PL versus 3PL (zero guessing) or ``k`` versus
    ``k + 1`` factors. This test instead simulates the statistic's null
    distribution from the fitted reduced model, like ``boot.LR`` in the R
    package mirt.

    Parameters
    ----------
    reduced : BaseItemModel or FitResult
        Fitted model of the null hypothesis.
    full : BaseItemModel or FitResult
        Fitted model that nests ``reduced`` and has more free parameters. The
        parameter count of a ``FitResult`` with a ``refit_recipe``
        (``n_parameters``) includes its estimated latent (co)variances and
        counts each equality-constraint group once; a bare model, or a fit
        without a recipe, counts its model's free item parameters.
    responses : ndarray of shape (n_persons, n_items)
        Observed responses used to fit both models. Negative codes are
        missing.
    n_bootstrap : int, default=200
        Number of simulated data sets, at least 2.
    seed : int, optional
        Random seed for reproducible simulations.
    n_quadpts : int, default=21
        Quadrature points per latent dimension for every refit (per dimension
        of the two-dimensional grids of a bifactor fit).
    tol : float, default=1e-5
        EM convergence tolerance for every refit. Statistics near zero need
        tighter tolerances than parameter bootstraps.
    max_iter : int, default=500
        Maximum EM iterations for every refit.
    warm_start : bool, default=True
        Whether replicate fits start from the observed-data estimates.
        Otherwise EM reinitializes the free parameters.
    n_jobs : int, default=1
        Number of process workers. Use ``-1`` for all available CPU cores.
        Custom models must be picklable when using multiple workers.

    Returns
    -------
    BootstrapLRResult
        Observed statistic, bootstrap and asymptotic p-values, and the null
        statistics.

    Notes
    -----
    Both models are first refitted to ``responses``, starting from their
    supplied estimates, with exactly the estimator settings used for the
    replicates. The observed statistic comes from these refits rather than
    from the supplied fits, so differing quadrature or tolerance settings
    cannot bias the comparison. Each replicate draws every person's
    abilities from the latent population of the refitted reduced model
    (standard normal unless it estimates a mean or covariance), simulates
    responses from that model, applies the observed missing-data pattern,
    and refits both models. A ``FitResult`` is refitted by the estimator that
    produced it, with its item priors, latent density and equality
    constraints (see :func:`bootstrap_se`), and a bare model by the default
    EM estimator of its model family, so observed and replicate statistics
    are computed alike. Replicates that fail to fit are excluded and counted
    in ``n_failed``. Seeded results are identical for every worker count.

    Examples
    --------
    >>> from mirt import bootstrap_lr, fit_mirt, simdata
    >>> data = simdata("2PL", n_persons=300, n_items=8, seed=1)
    >>> reduced = fit_mirt(data, model="2PL", max_iter=50)
    >>> full = fit_mirt(data, model="3PL", max_iter=50)
    >>> test = bootstrap_lr(reduced, full, data, n_bootstrap=5, seed=1)
    >>> 0.0 < test.p_value <= 1.0
    True
    """
    from scipy import stats

    from mirt.estimation.em import EMEstimator
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

    for name, candidate in (("reduced", reduced), ("full", full)):
        if isinstance(candidate, FitResult):
            candidate = candidate.model
        if not isinstance(candidate, BaseItemModel):
            raise MirtValidationError(
                f"{name} must be an item model or a FitResult wrapping one",
                parameter=name,
                value=type(candidate).__name__,
            )
    reduced_model, reduced_recipe = _refit_source(reduced)
    full_model, full_recipe = _refit_source(full)
    _validate_resample_count(n_bootstrap)
    n_jobs = resolve_n_jobs(n_jobs)
    if reduced_model.n_items != full_model.n_items:
        raise MirtValidationError(
            "reduced and full models must contain the same items",
            parameter="full",
            value=full_model.n_items,
            expected=str(reduced_model.n_items),
        )
    reduced_count = _free_parameter_count(reduced, reduced_recipe)
    full_count = _free_parameter_count(full, full_recipe)
    df = full_count - reduced_count
    if df < 1:
        raise MirtValidationError(
            "the full model must have more free parameters than the reduced model",
            parameter="full",
            value=full_count,
            expected=f"> {reduced_count}",
        )
    responses = validate_responses(responses, n_items=reduced_model.n_items)
    estimator_options: dict[str, Any] = {
        "n_quadpts": n_quadpts,
        "max_iter": max_iter,
        "tol": tol,
        "compute_standard_errors": False,
        "verbose": False,
    }
    EMEstimator(**estimator_options)  # Validate settings before any fitting.

    reduced_refit = _refit_log_likelihood(
        reduced_model,
        responses,
        reduced_model.is_fitted,
        estimator_options,
        reduced_recipe,
    )
    full_refit = _refit_log_likelihood(
        full_model, responses, full_model.is_fitted, estimator_options, full_recipe
    )
    reduced_ll = float(reduced_refit.log_likelihood)
    full_ll = float(full_refit.log_likelihood)
    statistic = max(0.0, 2.0 * (full_ll - reduced_ll))
    latent_mean, latent_cholesky = _population_factors(reduced_refit)

    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
    tasks = [
        _LRFitTask(
            reduced=reduced_refit.model,
            full=full_refit.model,
            missing=responses < 0,
            warm_start=warm_start,
            estimator_options=estimator_options,
            seeds=seed_chunk,
            reduced_recipe=reduced_recipe,
            full_recipe=full_recipe,
            latent_mean=latent_mean,
            latent_cholesky=latent_cholesky,
        )
        for seed_chunk in _chunk_values(seeds, n_jobs)
    ]
    outcomes = [
        outcome
        for chunk in _run_bootstrap_tasks(_fit_lr_task, tasks, n_jobs)
        for outcome in chunk
    ]
    null_statistics = np.array(
        [value for value, error in outcomes if error is None], dtype=np.float64
    )
    n_failed = len(outcomes) - null_statistics.size
    if null_statistics.size == 0:
        warnings.warn(
            "every bootstrap likelihood-ratio replicate failed",
            RuntimeWarning,
            stacklevel=2,
        )
        p_value = np.nan
    else:
        exceedances = np.count_nonzero(null_statistics >= statistic)
        p_value = (1.0 + exceedances) / (null_statistics.size + 1.0)

    return BootstrapLRResult(
        statistic=statistic,
        p_value=float(p_value),
        null_statistics=null_statistics,
        n_failed=int(n_failed),
        df=int(df),
        asymptotic_p_value=float(stats.chi2.sf(statistic, df)),
        reduced_log_likelihood=reduced_ll,
        full_log_likelihood=full_ll,
    )
