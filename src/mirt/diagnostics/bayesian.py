"""Bayesian model diagnostics for IRT models.

This module provides:
- PSIS-LOO (Pareto-smoothed importance sampling leave-one-out cross-validation)
- WAIC (Widely Applicable Information Criterion)
- Posterior predictive checks
- DIC (Deviance Information Criterion)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.estimation.mcmc import MCMCResult
    from mirt.models.base import BaseItemModel


_CHAIN_METADATA = {"log_likelihood", "theta"}


def _copy_state_value(value: Any) -> Any:
    """Copy mutable model state without assuming it is an ndarray."""
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)


@contextmanager
def _preserve_model_state(
    model: BaseItemModel,
    parameter_names: set[str],
) -> Iterator[None]:
    """Restore model parameters and mirrored private attributes on exit."""
    original_parameters = {
        name: value.copy() for name, value in model._parameters.items()
    }
    attribute_names = {
        f"_{name}" for name in parameter_names if hasattr(model, f"_{name}")
    }
    original_attributes = {
        name: _copy_state_value(getattr(model, name)) for name in attribute_names
    }

    try:
        yield
    finally:
        model._parameters.clear()
        model._parameters.update(original_parameters)
        for name, value in original_attributes.items():
            setattr(model, name, value)


def _sampled_parameter_chains(
    model: BaseItemModel,
    chains: dict[str, NDArray[np.float64]],
    *,
    n_persons: int,
) -> tuple[int, dict[str, NDArray[np.float64]]]:
    """Validate posterior chains and return sample-varying model parameters."""
    parameter_chains: dict[str, NDArray[np.float64]] = {}
    sample_counts: list[int] = []

    for name, chain in chains.items():
        if name in _CHAIN_METADATA:
            continue

        attribute_name = f"_{name}"
        if name in model._parameters:
            expected_shape = model._parameters[name].shape
        elif hasattr(model, attribute_name):
            expected_shape = np.asarray(getattr(model, attribute_name)).shape
        else:
            continue

        values = np.asarray(chain, dtype=np.float64)
        if values.ndim != len(expected_shape) + 1 or values.shape[1:] != expected_shape:
            raise ValueError(
                f"chain '{name}' must have shape (n_samples, {expected_shape}), "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"chain '{name}' must contain only finite values")
        parameter_chains[name] = values
        sample_counts.append(values.shape[0])

    log_likelihood_chain = chains.get("log_likelihood")
    if log_likelihood_chain is not None:
        log_likelihood_values = np.asarray(log_likelihood_chain)
        if log_likelihood_values.ndim != 1:
            raise ValueError("chain 'log_likelihood' must be one-dimensional")
        sample_counts.append(log_likelihood_values.shape[0])

    theta_chain = chains.get("theta")
    if theta_chain is not None:
        theta_values = np.asarray(theta_chain, dtype=np.float64)
        if not np.all(np.isfinite(theta_values)):
            raise ValueError("chain 'theta' must contain only finite values")
        if theta_values.ndim == 3:
            sample_counts.append(theta_values.shape[0])
        elif (
            theta_values.ndim == 2
            and model.n_factors == 1
            and theta_values.shape[1] == n_persons
            and theta_values.shape != (n_persons, model.n_factors)
        ):
            sample_counts.append(theta_values.shape[0])

    if not sample_counts:
        return 1, parameter_chains

    n_samples = sample_counts[0]
    if n_samples == 0:
        raise ValueError("posterior chains must contain at least one sample")
    if any(count != n_samples for count in sample_counts[1:]):
        raise ValueError("posterior chains must contain the same number of samples")

    return n_samples, parameter_chains


def _apply_parameter_sample(
    model: BaseItemModel,
    parameter_chains: dict[str, NDArray[np.float64]],
    sample_idx: int,
) -> None:
    """Apply one posterior parameter sample to a model."""
    for name, chain in parameter_chains.items():
        value = np.asarray(chain[sample_idx], dtype=np.float64)
        if name in model._parameters:
            model._parameters[name] = value
        attribute_name = f"_{name}"
        if hasattr(model, attribute_name):
            setattr(model, attribute_name, value)


def _theta_for_sample(
    chains: dict[str, NDArray[np.float64]],
    *,
    sample_idx: int,
    n_samples: int,
    n_persons: int,
    n_factors: int,
) -> NDArray[np.float64]:
    """Resolve sampled or fixed latent traits to a two-dimensional matrix."""
    theta_chain = chains.get("theta")
    if theta_chain is None:
        return np.zeros((n_persons, n_factors), dtype=np.float64)

    theta = np.asarray(theta_chain, dtype=np.float64)
    if theta.ndim == 3 and theta.shape == (n_samples, n_persons, n_factors):
        return theta[sample_idx]
    if theta.ndim == 2 and theta.shape == (n_persons, n_factors):
        return theta
    if theta.ndim == 2 and n_factors == 1 and theta.shape == (n_samples, n_persons):
        return theta[sample_idx, :, None]
    if theta.ndim == 1 and n_factors == 1 and theta.shape == (n_persons,):
        return theta[:, None]

    raise ValueError(
        "chain 'theta' must be fixed (n_persons, n_factors), sampled "
        "(n_samples, n_persons, n_factors), or sampled unidimensional "
        "(n_samples, n_persons)"
    )


def _supports_batched_pointwise_2pl(model: BaseItemModel) -> bool:
    """Return whether a model has the exact batched 2PL contract."""
    from mirt.models.dichotomous import TwoParameterLogistic

    return type(model) is TwoParameterLogistic and model.n_factors == 1


def _batched_2pl_pointwise_inputs(
    model: BaseItemModel,
    chains: dict[str, NDArray[np.float64]],
    parameter_chains: dict[str, NDArray[np.float64]],
    *,
    n_samples: int,
    n_persons: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Materialize fixed or sampled values for the batched 2PL backend."""
    discrimination = parameter_chains.get("discrimination")
    if discrimination is None:
        discrimination = np.broadcast_to(
            model.parameters["discrimination"],
            (n_samples, model.n_items),
        )
    difficulty = parameter_chains.get("difficulty")
    if difficulty is None:
        difficulty = np.broadcast_to(
            model.parameters["difficulty"],
            (n_samples, model.n_items),
        )

    theta = np.empty((n_samples, n_persons), dtype=np.float64)
    for sample_idx in range(n_samples):
        theta[sample_idx] = _theta_for_sample(
            chains,
            sample_idx=sample_idx,
            n_samples=n_samples,
            n_persons=n_persons,
            n_factors=1,
        )[:, 0]
    return discrimination, difficulty, theta


def _validate_response_matrix(
    responses: NDArray[np.int_],
    model: BaseItemModel,
) -> NDArray[np.int_]:
    """Validate response shape and observed category codes."""
    values = np.asarray(responses)
    if values.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    if values.shape[1] != model.n_items:
        raise ValueError(
            f"responses has {values.shape[1]} items, expected {model.n_items}"
        )

    is_boolean = np.issubdtype(values.dtype, np.bool_)
    if not is_boolean and not np.issubdtype(values.dtype, np.number):
        raise ValueError("responses must contain finite integer category codes")
    if not is_boolean and (
        not np.all(np.isfinite(values)) or not np.all(values == np.floor(values))
    ):
        raise ValueError("responses must contain finite integer category codes")

    integer_values = values.astype(np.int64, copy=False)
    observed = integer_values >= 0
    if model.is_polytomous:
        category_limits = np.asarray(model.n_categories, dtype=np.int64)[None, :]
        invalid = observed & (integer_values >= category_limits)
    else:
        invalid = observed & (integer_values > 1)
    if np.any(invalid):
        raise ValueError("responses contain categories unsupported by the model")

    return integer_values


def _pointwise_log_likelihood(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute one posterior sample's person-item log likelihood matrix."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    observed = responses >= 0

    if probabilities.shape[:2] != responses.shape:
        raise ValueError(
            "model probability shape does not match the response matrix: "
            f"{probabilities.shape[:2]} != {responses.shape}"
        )

    if probabilities.ndim == 2:
        probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)
        log_likelihood = np.where(
            responses == 1,
            np.log(probabilities),
            np.log1p(-probabilities),
        )
    elif probabilities.ndim == 3:
        safe_responses = np.where(observed, responses, 0)
        selected = np.take_along_axis(
            probabilities,
            safe_responses[..., None],
            axis=2,
        )[..., 0]
        log_likelihood = np.log(np.clip(selected, PROB_EPSILON, 1.0))
    else:
        raise ValueError(
            "model probabilities must have shape (n_persons, n_items) or "
            "(n_persons, n_items, n_categories)"
        )

    return np.where(observed, log_likelihood, 0.0)


def _simulate_response_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Draw a dichotomous or polytomous response matrix from a model."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    expected_shape = (theta.shape[0], model.n_items)

    if probabilities.shape[:2] != expected_shape:
        raise ValueError(
            "model probability shape does not match theta and item dimensions: "
            f"{probabilities.shape[:2]} != {expected_shape}"
        )

    if probabilities.ndim == 2:
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("model probabilities must be finite")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return (rng.random(probabilities.shape) < probabilities).astype(np.int64)

    if probabilities.ndim != 3:
        raise ValueError(
            "model probabilities must have shape (n_persons, n_items) or "
            "(n_persons, n_items, n_categories)"
        )

    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=2, keepdims=True)
    if not np.all(np.isfinite(probabilities)) or np.any(totals <= 0):
        raise ValueError("category probabilities must be finite with positive totals")
    probabilities = probabilities / totals
    cumulative = np.cumsum(probabilities, axis=2)
    draws = rng.random(probabilities.shape[:2])
    replicated = np.sum(draws[..., None] > cumulative, axis=2)
    category_max = np.asarray(model.n_categories, dtype=np.int64)[None, :] - 1
    return np.minimum(replicated, category_max).astype(np.int64)


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


def _validate_log_likelihood(log_lik: ArrayLike) -> NDArray[np.float64]:
    """Return a finite sample-by-observation log-likelihood matrix."""
    values = np.asarray(log_lik, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("log_lik must be a one- or two-dimensional array")
    if values.shape[0] < 2:
        raise ValueError("log_lik must contain at least two posterior samples")
    if values.shape[1] == 0:
        raise ValueError("log_lik must contain at least one observation")
    if not np.all(np.isfinite(values)):
        raise ValueError("log_lik must contain only finite values")
    return values


def _validate_relative_efficiency(
    relative_eff: ArrayLike,
    n_observations: int,
) -> NDArray[np.float64]:
    """Broadcast positive relative efficiencies across observations."""
    raw_values = np.asarray(relative_eff)
    if np.issubdtype(raw_values.dtype, np.bool_):
        raise ValueError("relative_eff must contain positive finite values")
    try:
        values = raw_values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("relative_eff must contain positive finite values") from exc
    if values.ndim == 0:
        values = np.full(n_observations, float(values), dtype=np.float64)
    elif values.shape != (n_observations,):
        raise ValueError(
            "relative_eff must be a scalar or contain one value per observation"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("relative_eff must contain positive finite values")
    return values


def _fit_generalized_pareto(
    excesses: NDArray[np.float64],
) -> tuple[float, float]:
    """Fit a generalized Pareto tail with the Zhang-Stephens estimator."""
    sorted_excesses = np.sort(np.asarray(excesses, dtype=np.float64))
    n_tail = sorted_excesses.size
    if n_tail <= 4 or sorted_excesses[0] < 0.0:
        return np.inf, np.nan
    if sorted_excesses[-1] <= 0.0:
        return 0.0, 0.0

    n_candidates = 30 + int(np.sqrt(n_tail))
    candidate_index = np.arange(1, n_candidates + 1, dtype=np.float64) - 0.5
    inverse_scales = 1.0 - np.sqrt(n_candidates / candidate_index)
    quartile = sorted_excesses[int(n_tail / 4.0 + 0.5) - 1]
    if quartile <= 0.0:
        return np.inf, np.nan
    inverse_scales /= 3.0 * quartile
    inverse_scales += 1.0 / sorted_excesses[-1]

    log_terms = np.log1p(-inverse_scales[:, None] * sorted_excesses[None, :])
    shape_candidates = np.mean(log_terms, axis=1)
    objective = n_tail * (
        np.log(-(inverse_scales / shape_candidates)) - shape_candidates - 1.0
    )
    posterior_weights = np.exp(objective - np.max(objective))
    keep = posterior_weights >= 10.0 * np.finfo(np.float64).eps
    posterior_weights = posterior_weights[keep]
    inverse_scales = inverse_scales[keep]
    posterior_weights /= np.sum(posterior_weights)

    inverse_scale = float(np.sum(inverse_scales * posterior_weights))
    shape = float(np.mean(np.log1p(-inverse_scale * sorted_excesses)))
    scale = float(-shape / inverse_scale)

    prior_weight = 10.0
    shape = (n_tail * shape + prior_weight * 0.5) / (n_tail + prior_weight)
    return shape, scale


def _generalized_pareto_quantile(
    probabilities: NDArray[np.float64],
    shape: float,
    scale: float,
) -> NDArray[np.float64]:
    """Return generalized Pareto quantiles for probabilities in ``(0, 1)``."""
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full(probabilities.shape, np.nan, dtype=np.float64)
    log_survival = np.log1p(-probabilities)
    if abs(shape) < np.finfo(np.float64).eps:
        return -scale * log_survival
    return scale * np.expm1(-shape * log_survival) / shape


def _pareto_smooth_log_weights(
    log_weights: NDArray[np.float64],
    relative_eff: float,
) -> tuple[NDArray[np.float64], float]:
    """Return normalized Pareto-smoothed log importance weights."""
    values = np.asarray(log_weights, dtype=np.float64).copy()
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("log_weights must contain at least two finite values")

    values -= np.max(values)
    if np.all(values == values[0]):
        values.fill(-np.log(values.size))
        return values, 0.0

    tail_size = int(
        np.ceil(min(0.2 * values.size, 3.0 * np.sqrt(values.size / relative_eff)))
    )
    cutoff_index = -tail_size - 1
    order = np.argsort(values)
    cutoff = max(values[order[cutoff_index]], np.log(np.finfo(np.float64).tiny))
    tail_indices = np.flatnonzero(values > cutoff)

    if tail_indices.size <= 4:
        shape = np.inf
    else:
        tail_order = np.argsort(values[tail_indices])
        cutoff_weight = np.exp(cutoff)
        excesses = np.exp(values[tail_indices][tail_order]) - cutoff_weight
        shape, scale = _fit_generalized_pareto(excesses)
        if shape >= 1.0 / 3.0 and np.isfinite(shape):
            probabilities = (
                np.arange(tail_indices.size, dtype=np.float64) + 0.5
            ) / tail_indices.size
            smoothed = _generalized_pareto_quantile(probabilities, shape, scale)
            smoothed += cutoff_weight
            values[tail_indices[tail_order]] = np.log(smoothed)

    np.minimum(values, 0.0, out=values)
    values -= float(logsumexp(values))
    return values, float(shape)


def _pareto_smooth_weights(
    log_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return normalized Pareto-smoothed importance weights and tail shape."""
    smoothed, shape = _pareto_smooth_log_weights(log_weights, relative_eff=1.0)
    return np.exp(smoothed), shape


def psis_loo(
    log_lik: ArrayLike,
    k_threshold: float = 0.7,
    relative_eff: ArrayLike = 1.0,
    n_jobs: int = 1,
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
    relative_eff : float or array-like, default=1.0
        Relative effective sample size of the inverse importance ratios. A
        scalar is shared across observations; an array supplies one value per
        observation. Independent posterior draws may use the default.
    n_jobs : int, default=1
        Number of threads used to smooth independent observation tails. Use
        ``-1`` for all available processors. The serial default avoids thread
        overhead for small log-likelihood matrices.

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
    log_lik = _validate_log_likelihood(log_lik)
    if isinstance(k_threshold, (bool, np.bool_)):
        raise ValueError("k_threshold must be a positive finite number")
    try:
        k_threshold = float(k_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_threshold must be a positive finite number") from exc
    if not np.isfinite(k_threshold) or k_threshold <= 0.0:
        raise ValueError("k_threshold must be a positive finite number")
    n_samples, n_obs = log_lik.shape
    relative_efficiency = _validate_relative_efficiency(relative_eff, n_obs)
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise ValueError("n_jobs must be -1 or a positive integer")
    if n_jobs == -1:
        import os

        worker_count = os.cpu_count() or 1
    else:
        worker_count = int(n_jobs)

    pointwise_elpd = np.empty(n_obs, dtype=np.float64)
    pareto_k = np.empty(n_obs, dtype=np.float64)

    def smooth_observation(observation: int) -> tuple[float, float]:
        column = log_lik[:, observation]
        log_weights, k = _pareto_smooth_log_weights(
            -column,
            relative_eff=float(relative_efficiency[observation]),
        )
        return float(logsumexp(column + log_weights)), k

    if worker_count == 1 or n_obs == 1:
        smoothed_results = map(smooth_observation, range(n_obs))
        for observation, (elpd, k) in enumerate(smoothed_results):
            pointwise_elpd[observation] = elpd
            pareto_k[observation] = k
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(worker_count, n_obs)) as executor:
            smoothed_results = executor.map(smooth_observation, range(n_obs))
            for observation, (elpd, k) in enumerate(smoothed_results):
                pointwise_elpd[observation] = elpd
                pareto_k[observation] = k

    elpd_loo = np.sum(pointwise_elpd)

    lppd_i = logsumexp(log_lik, axis=0) - np.log(n_samples)
    lppd = np.sum(lppd_i)
    p_loo = lppd - elpd_loo

    looic = -2 * elpd_loo

    n_high_k = int(np.sum(pareto_k > k_threshold))

    se_elpd = (
        0.0 if n_obs == 1 else float(np.sqrt(n_obs * np.var(pointwise_elpd, ddof=1)))
    )

    return PSISResult(
        elpd_loo=float(elpd_loo),
        p_loo=float(p_loo),
        looic=float(looic),
        pointwise=pointwise_elpd,
        pareto_k=pareto_k,
        n_high_k=n_high_k,
        se_elpd=se_elpd,
    )


def waic(log_lik: ArrayLike) -> WAICResult:
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
    log_lik = _validate_log_likelihood(log_lik)
    n_samples, n_obs = log_lik.shape

    lppd_i = logsumexp(log_lik, axis=0) - np.log(n_samples)
    lppd = np.sum(lppd_i)

    p_waic_i = np.var(log_lik, axis=0, ddof=1)
    p_waic = np.sum(p_waic_i)

    elpd_waic = lppd - p_waic
    waic_val = -2 * elpd_waic

    pointwise = -2 * (lppd_i - p_waic_i)

    se_waic = 0.0 if n_obs == 1 else float(np.sqrt(n_obs * np.var(pointwise, ddof=1)))

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

    Notes
    -----
    Dichotomous and polytomous response models are supported. Posterior samples
    are applied temporarily; the supplied model is restored before return, even
    when a custom statistic raises an exception.
    """
    return posterior_predictive_checks(
        mcmc_result,
        responses,
        model,
        {"result": test_statistic},
        n_rep=n_rep,
        seed=seed,
    )["result"]


def posterior_predictive_checks(
    mcmc_result: MCMCResult,
    responses: NDArray[np.int_],
    model: BaseItemModel,
    test_statistics: Mapping[str, Callable[[NDArray[np.int_]], float] | str]
    | Iterable[str],
    *,
    n_rep: int | None = None,
    seed: int | None = None,
) -> dict[str, PPCResult]:
    """Evaluate several posterior predictive statistics in one simulation run.

    Parameters
    ----------
    mcmc_result : MCMCResult
        Result from MCMC estimation containing chains.
    responses : ndarray of shape (n_persons, n_items)
        Observed response matrix.
    model : BaseItemModel
        IRT model used for simulation.
    test_statistics : mapping or iterable of str
        A mapping from result labels to built-in statistic names or callables, or
        an iterable of built-in names. Available built-ins are ``'item_mean'``,
        ``'person_score'``, ``'chi_square'``, ``'correlation'``, and
        ``'odds_ratio'``.
    n_rep : int, optional
        Number of replications. Defaults to number of posterior samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict of str to PPCResult
        One result per requested statistic, preserving input order.

    Notes
    -----
    Each posterior replicate is simulated once and shared by every statistic.
    This produces a directly comparable diagnostic suite while avoiding repeated
    probability evaluation. Posterior samples are applied temporarily and the
    supplied model is restored before return, including when a statistic raises.
    """
    rng = np.random.default_rng(seed)
    responses = _validate_response_matrix(responses, model)
    n_persons, n_items = responses.shape

    if not np.any(responses >= 0):
        raise ValueError("responses must contain at least one observed value")

    statistics = _resolve_test_statistics(test_statistics, n_items)
    observed_statistics = {
        name: float(statistic(responses)) for name, statistic in statistics.items()
    }

    chains = mcmc_result.chains
    n_samples, parameter_chains = _sampled_parameter_chains(
        model,
        chains,
        n_persons=n_persons,
    )

    if n_rep is None:
        n_rep = n_samples
    elif not isinstance(n_rep, (int, np.integer)) or isinstance(n_rep, bool):
        raise ValueError("n_rep must be a positive integer")
    if n_rep <= 0:
        raise ValueError("n_rep must be a positive integer")

    sample_indices = rng.choice(n_samples, size=n_rep, replace=n_rep > n_samples)
    replicated_statistics = {
        name: np.zeros(n_rep, dtype=np.float64) for name in statistics
    }
    observed = responses >= 0
    parameter_names = set(parameter_chains)

    with _preserve_model_state(model, parameter_names):
        for rep_idx, sample_idx in enumerate(sample_indices):
            _apply_parameter_sample(model, parameter_chains, int(sample_idx))
            if "theta" in chains:
                theta = _theta_for_sample(
                    chains,
                    sample_idx=int(sample_idx),
                    n_samples=n_samples,
                    n_persons=n_persons,
                    n_factors=model.n_factors,
                )
            else:
                theta = rng.standard_normal((n_persons, model.n_factors))
            replicated = _simulate_response_matrix(model, theta, rng)
            replicated = np.where(observed, replicated, -1)
            for name, statistic in statistics.items():
                replicated_statistics[name][rep_idx] = statistic(replicated)

    return {
        name: _make_ppc_result(observed_statistics[name], replicated)
        for name, replicated in replicated_statistics.items()
    }


def _make_ppc_result(
    observed: float,
    replicated: NDArray[np.float64],
) -> PPCResult:
    """Summarize one observed statistic and its posterior replicates."""
    summary_stats = {
        "mean": float(np.mean(replicated)),
        "std": float(np.std(replicated)),
        "q025": float(np.percentile(replicated, 2.5)),
        "q975": float(np.percentile(replicated, 97.5)),
        "min": float(np.min(replicated)),
        "max": float(np.max(replicated)),
    }
    return PPCResult(
        test_statistic_observed=observed,
        test_statistic_replicated=replicated,
        p_value=float(np.mean(replicated >= observed)),
        summary_stats=summary_stats,
    )


def _resolve_test_statistics(
    test_statistics: Mapping[str, Callable[[NDArray[np.int_]], float] | str]
    | Iterable[str],
    n_items: int,
) -> dict[str, Callable[[NDArray[np.int_]], float]]:
    """Validate labels and resolve built-in posterior predictive statistics."""
    if isinstance(test_statistics, Mapping):
        entries = list(test_statistics.items())
    else:
        if isinstance(test_statistics, (str, bytes)):
            raise ValueError(
                "test_statistics must be a mapping or an iterable of names"
            )
        try:
            names = list(test_statistics)
        except TypeError as exc:
            raise ValueError(
                "test_statistics must be a mapping or an iterable of names"
            ) from exc
        if any(not isinstance(name, str) for name in names):
            raise ValueError("iterable test_statistics must contain only names")
        if len(names) != len(set(names)):
            raise ValueError("test statistic names must be unique")
        entries = [(name, name) for name in names]

    if not entries:
        raise ValueError("test_statistics must contain at least one statistic")

    resolved: dict[str, Callable[[NDArray[np.int_]], float]] = {}
    for label, specification in entries:
        if not isinstance(label, str) or not label:
            raise ValueError("test statistic labels must be non-empty strings")
        if isinstance(specification, str):
            resolved[label] = _get_builtin_statistic(specification, n_items)
        elif callable(specification):
            resolved[label] = specification
        else:
            raise ValueError(
                "test statistics must be built-in names or callable functions"
            )
    return resolved


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
    by: Literal["person", "observation", "observed"] = "person",
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
    by : {'person', 'observation', 'observed'}, default='person'
        Aggregation level. ``'observation'`` preserves the flattened response
        layout and assigns zero to missing cells. ``'observed'`` returns only
        non-missing person-item cells, which is convenient for WAIC or PSIS-LOO.

    Returns
    -------
    log_lik : ndarray
        If by='person': shape (n_samples, n_persons)
        If by='observation': shape (n_samples, n_persons * n_items)
        If by='observed': shape (n_samples, n_observed_responses)

    Notes
    -----
    Dichotomous and polytomous response models are supported. Posterior samples
    are applied temporarily and the supplied model is restored before return.
    """
    if by not in {"person", "observation", "observed"}:
        raise ValueError("by must be 'person', 'observation', or 'observed'")

    responses = _validate_response_matrix(responses, model)
    n_persons, n_items = responses.shape
    n_samples, parameter_chains = _sampled_parameter_chains(
        model,
        chains,
        n_persons=n_persons,
    )

    if _supports_batched_pointwise_2pl(model):
        from mirt.backends.rust.diagnostics import compute_pointwise_loglik_2pl

        discrimination, difficulty, theta = _batched_2pl_pointwise_inputs(
            model,
            chains,
            parameter_chains,
            n_samples=n_samples,
            n_persons=n_persons,
        )
        return compute_pointwise_loglik_2pl(
            responses,
            discrimination,
            difficulty,
            theta,
            aggregation=by,
        )

    if by == "person":
        log_lik = np.zeros((n_samples, n_persons), dtype=np.float64)
    elif by == "observation":
        log_lik = np.zeros((n_samples, n_persons * n_items), dtype=np.float64)
    else:
        log_lik = np.zeros((n_samples, int(np.sum(responses >= 0))), dtype=np.float64)

    parameter_names = set(parameter_chains)
    observed = responses >= 0
    with _preserve_model_state(model, parameter_names):
        for sample_idx in range(n_samples):
            _apply_parameter_sample(model, parameter_chains, sample_idx)
            theta = _theta_for_sample(
                chains,
                sample_idx=sample_idx,
                n_samples=n_samples,
                n_persons=n_persons,
                n_factors=model.n_factors,
            )
            pointwise = _pointwise_log_likelihood(model, responses, theta)

            if by == "person":
                log_lik[sample_idx] = pointwise.sum(axis=1)
            elif by == "observation":
                log_lik[sample_idx] = pointwise.ravel()
            else:
                log_lik[sample_idx] = pointwise[observed]

    return log_lik
