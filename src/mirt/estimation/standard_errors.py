"""Matrix-based standard errors for marginal item-response models.

The routines in this module differentiate the person-level marginal
log-likelihood. Keeping that objective in one place makes the observed,
cross-product, and sandwich estimators consistent and ensures that a fitted
latent density is not silently replaced by the quadrature's default mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.utils.numeric import logsumexp, logsumexp_axis1

if TYPE_CHECKING:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.models.base import BaseItemModel


@dataclass(frozen=True)
class _ParameterLayout:
    """Mapping between stored parameters and the free parameter vector."""

    shape: tuple[int, ...]
    free_indices: NDArray[np.int_]
    template: NDArray[np.float64]


def _validate_step_size(h: float) -> float:
    step = float(h)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("h must be finite and positive")
    return step


def _validate_posterior(
    posterior_weights: NDArray[np.float64],
    n_persons: int,
    n_quadpts: int,
) -> NDArray[np.float64]:
    posterior = np.asarray(posterior_weights, dtype=np.float64)
    if posterior.shape != (n_persons, n_quadpts):
        raise ValueError(
            "posterior_weights must have shape "
            f"({n_persons}, {n_quadpts}), got {posterior.shape}"
        )
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
        raise ValueError("posterior_weights must contain finite non-negative values")
    row_sums = posterior.sum(axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("each posterior_weights row must have positive mass")
    return posterior


def _validate_prior_mass(
    prior_mass: NDArray[np.float64],
    n_quadpts: int,
) -> NDArray[np.float64]:
    mass = np.asarray(prior_mass, dtype=np.float64)
    if mass.shape != (n_quadpts,):
        raise ValueError(f"prior_mass must have shape ({n_quadpts},), got {mass.shape}")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("prior_mass must contain finite non-negative values")
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("prior_mass must contain positive total mass")
    return mass / total


def _infer_prior_mass(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
) -> NDArray[np.float64]:
    """Recover normalized prior mass from a final E-step posterior."""
    nodes = quadrature.nodes
    posterior = _validate_posterior(
        posterior_weights,
        responses.shape[0],
        nodes.shape[0],
    )
    usable_rows = np.flatnonzero(np.all(posterior > 0.0, axis=1))
    if usable_rows.size == 0:
        raise ValueError(
            "cannot infer quadrature prior mass from zero posterior cells; "
            "pass prior_mass explicitly"
        )

    log_likelihood = np.asarray(
        model.log_likelihood_batch(responses, nodes),
        dtype=np.float64,
    )
    if log_likelihood.shape != posterior.shape or not np.all(
        np.isfinite(log_likelihood)
    ):
        raise ValueError("model returned invalid log likelihoods at quadrature nodes")

    row = int(usable_rows[0])
    log_mass = np.log(posterior[row]) - log_likelihood[row]
    log_mass -= float(logsumexp(log_mass))
    mass = np.exp(log_mass)

    # Posterior rows may be scaled, but they must imply the same normalized
    # prior. Checking a small sample catches stale or unrelated posteriors.
    for other_row in usable_rows[1:9]:
        candidate = np.log(posterior[other_row]) - log_likelihood[other_row]
        candidate -= float(logsumexp(candidate))
        if not np.allclose(candidate, log_mass, rtol=0.0, atol=5e-8):
            # Some advanced callers provide working weights rather than a
            # final E-step posterior. Preserve their historical default.
            return _validate_prior_mass(quadrature.weights, nodes.shape[0])
    return mass


def _resolve_prior_mass(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    n_quadpts = quadrature.nodes.shape[0]
    _validate_posterior(posterior_weights, responses.shape[0], n_quadpts)
    if prior_mass is not None:
        return _validate_prior_mass(prior_mass, n_quadpts)
    return _infer_prior_mass(model, responses, posterior_weights, quadrature)


def _posterior_from_model(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    log_prior_mass: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute posterior quadrature weights for testing and advanced use."""
    response_array = np.asarray(responses)
    nodes = quadrature.nodes
    if response_array.ndim != 2 or response_array.shape[0] == 0:
        raise ValueError("responses must be a non-empty two-dimensional array")

    if log_prior_mass is None:
        mass = _validate_prior_mass(quadrature.weights, nodes.shape[0])
        log_mass = np.log(mass)
    else:
        log_mass = np.asarray(log_prior_mass, dtype=np.float64)
        if log_mass.shape != (nodes.shape[0],):
            raise ValueError(
                f"log_prior_mass must have shape ({nodes.shape[0]},), "
                f"got {log_mass.shape}"
            )
        if np.any(np.isnan(log_mass)) or np.any(np.isposinf(log_mass)):
            raise ValueError("log_prior_mass must contain finite values or -inf")
        if not np.any(np.isfinite(log_mass)):
            raise ValueError("log_prior_mass must contain positive total mass")
        log_mass = log_mass - float(logsumexp(log_mass))

    log_joint = (
        np.asarray(
            model.log_likelihood_batch(response_array, nodes),
            dtype=np.float64,
        )
        + log_mass[None, :]
    )
    log_norm = logsumexp_axis1(log_joint)
    return np.exp(log_joint - log_norm[:, None])


def _flatten_parameters(
    model: BaseItemModel,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    """Flatten statistically free model parameters into a single vector."""
    chunks: list[NDArray[np.float64]] = []
    layouts: dict[str, _ParameterLayout] = {}
    free_masks = model.free_parameter_masks

    for name, values in model.parameters.items():
        canonical = model._canonical_parameter_values(name, values)
        free_mask = np.asarray(free_masks[name], dtype=np.bool_)
        if free_mask.shape != values.shape:
            raise RuntimeError(
                f"free-parameter mask for {name} has shape {free_mask.shape}, "
                f"expected {values.shape}"
            )
        free_indices = np.flatnonzero(free_mask.ravel())
        layouts[name] = _ParameterLayout(
            shape=values.shape,
            free_indices=free_indices,
            template=canonical,
        )
        chunks.append(canonical.ravel()[free_indices])

    flattened = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return flattened, layouts


def _set_flat_parameters(
    model: BaseItemModel,
    params_flat: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
) -> None:
    """Set the model from a vector while keeping fixed storage canonical."""
    offset = 0
    for name, layout in layouts.items():
        size = layout.free_indices.size
        values = layout.template.copy().ravel()
        values[layout.free_indices] = params_flat[offset : offset + size]
        model._parameters[name] = values.reshape(layout.shape)
        offset += size


def _restore_parameters(
    model: BaseItemModel,
    parameters: dict[str, NDArray[np.float64]],
) -> None:
    model._parameters = {name: values.copy() for name, values in parameters.items()}


def _unflatten_se(
    se_flat: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    _model: BaseItemModel | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Restore standard errors to the model's stored parameter shapes."""
    result: dict[str, NDArray[np.float64]] = {}
    offset = 0
    for name, layout in layouts.items():
        size = layout.free_indices.size
        values = np.zeros(layout.shape, dtype=np.float64)
        values.ravel()[layout.free_indices] = se_flat[offset : offset + size]
        result[name] = values
        offset += size
    return result


def _marginal_log_likelihoods(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
) -> NDArray[np.float64]:
    log_mass = np.full(prior_mass.shape, -np.inf, dtype=np.float64)
    positive = prior_mass > 0.0
    log_mass[positive] = np.log(prior_mass[positive])
    log_joint = (
        np.asarray(
            model.log_likelihood_batch(responses, quadrature.nodes),
            dtype=np.float64,
        )
        + log_mass[None, :]
    )
    result = logsumexp_axis1(log_joint)
    if not np.all(np.isfinite(result)):
        raise ValueError("marginal log likelihood must be finite")
    return result


def _finite_difference_values(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
    *,
    include_cross_terms: bool,
) -> tuple[
    NDArray[np.float64],
    dict[str, _ParameterLayout],
    dict[tuple[float, ...], NDArray[np.float64]],
]:
    params_flat, layouts = _flatten_parameters(model)
    original = model.parameters
    cache: dict[tuple[float, ...], NDArray[np.float64]] = {}

    def evaluate(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
        key = tuple(float(value) for value in candidate)
        if key not in cache:
            _set_flat_parameters(model, candidate, layouts)
            cache[key] = _marginal_log_likelihoods(
                model, responses, quadrature, prior_mass
            )
        return cache[key]

    try:
        evaluate(params_flat)
        for index in range(params_flat.size):
            plus = params_flat.copy()
            minus = params_flat.copy()
            plus[index] += h
            minus[index] -= h
            evaluate(plus)
            evaluate(minus)
        if include_cross_terms:
            for row in range(params_flat.size):
                for column in range(row + 1, params_flat.size):
                    for row_sign, column_sign in (
                        (1.0, 1.0),
                        (1.0, -1.0),
                        (-1.0, 1.0),
                        (-1.0, -1.0),
                    ):
                        candidate = params_flat.copy()
                        candidate[row] += row_sign * h
                        candidate[column] += column_sign * h
                        evaluate(candidate)
    finally:
        _restore_parameters(model, original)

    return params_flat, layouts, cache


def _cached_value(
    cache: dict[tuple[float, ...], NDArray[np.float64]],
    candidate: NDArray[np.float64],
) -> NDArray[np.float64]:
    return cache[tuple(float(value) for value in candidate)]


def _finite_difference_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
    person_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    params, layouts, cache = _finite_difference_values(
        model,
        responses,
        quadrature,
        prior_mass,
        h,
        include_cross_terms=True,
    )
    n_persons = responses.shape[0]
    weights = (
        np.ones(n_persons, dtype=np.float64)
        if person_weights is None
        else np.asarray(person_weights, dtype=np.float64)
    )
    center = _cached_value(cache, params)
    information = np.zeros((params.size, params.size), dtype=np.float64)

    for row in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[row] += h
        minus[row] -= h
        second = _cached_value(cache, plus) - 2.0 * center + _cached_value(cache, minus)
        information[row, row] = -float(weights @ second) / h**2

        for column in range(row + 1, params.size):
            plus_plus = params.copy()
            plus_minus = params.copy()
            minus_plus = params.copy()
            minus_minus = params.copy()
            plus_plus[row] += h
            plus_plus[column] += h
            plus_minus[row] += h
            plus_minus[column] -= h
            minus_plus[row] -= h
            minus_plus[column] += h
            minus_minus[row] -= h
            minus_minus[column] -= h
            cross = (
                _cached_value(cache, plus_plus)
                - _cached_value(cache, plus_minus)
                - _cached_value(cache, minus_plus)
                + _cached_value(cache, minus_minus)
            )
            value = -float(weights @ cross) / (4.0 * h**2)
            information[row, column] = value
            information[column, row] = value
    return information, layouts


def _finite_difference_scores(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    params, layouts, cache = _finite_difference_values(
        model,
        responses,
        quadrature,
        prior_mass,
        h,
        include_cross_terms=False,
    )
    scores = np.empty((responses.shape[0], params.size), dtype=np.float64)
    for column in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[column] += h
        minus[column] -= h
        scores[:, column] = (
            _cached_value(cache, plus) - _cached_value(cache, minus)
        ) / (2.0 * h)
    return scores, layouts


def _se_from_information(
    information: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
) -> dict[str, NDArray[np.float64]]:
    if information.size == 0:
        return _unflatten_se(np.empty(0, dtype=np.float64), layouts)
    information = (information + information.T) / 2.0
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    variances = np.diag(covariance)
    se = np.full(variances.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(variances) & (variances >= 0.0)
    se[valid] = np.sqrt(variances[valid])
    return _unflatten_se(se, layouts)


def compute_observed_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the negative Hessian of the marginal log-likelihood."""
    step = _validate_step_size(h)
    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    information, _ = _finite_difference_information(
        model, response_array, quadrature, mass, step
    )
    return information


def compute_crossprod_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute standard errors from the outer product of person scores."""
    step = _validate_step_size(h)
    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    scores, layouts = _finite_difference_scores(
        model, response_array, quadrature, mass, step
    )
    return _se_from_information(scores.T @ scores, layouts)


def compute_sandwich_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    survey_weights: NDArray[np.float64] | None = None,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute robust bread-meat-bread standard errors."""
    step = _validate_step_size(h)
    response_array = np.asarray(responses)
    n_persons = response_array.shape[0]
    if survey_weights is None:
        weights = np.ones(n_persons, dtype=np.float64)
    else:
        weights = np.asarray(survey_weights, dtype=np.float64)
        if weights.shape != (n_persons,):
            raise ValueError(
                f"survey_weights must have shape ({n_persons},), got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("survey_weights must contain finite non-negative values")
        if not np.any(weights > 0.0):
            raise ValueError("survey_weights must contain positive total weight")

    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    bread, layouts = _finite_difference_information(
        model,
        response_array,
        quadrature,
        mass,
        step,
        person_weights=weights,
    )
    scores, _ = _finite_difference_scores(model, response_array, quadrature, mass, step)
    weighted_scores = weights[:, None] * scores
    meat = weighted_scores.T @ weighted_scores
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)
    covariance = bread_inv @ meat @ bread_inv.T
    variances = np.diag(covariance)
    se = np.full(variances.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(variances) & (variances >= 0.0)
    se[valid] = np.sqrt(variances[valid])
    return _unflatten_se(se, layouts)


def compute_oakes_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute observed-information standard errors at an EM solution."""
    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    information, layouts = _finite_difference_information(
        model,
        response_array,
        quadrature,
        mass,
        _validate_step_size(h),
    )
    return _se_from_information(information, layouts)


def compute_sem_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    n_bootstrap: int = 50,
    seed: int | None = None,
    *,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute deterministic observed-information SEs for an EM solution.

    ``n_bootstrap`` and ``seed`` remain accepted for API compatibility with
    the former stochastic approximation.
    """
    del n_bootstrap, seed
    return compute_oakes_se(
        model,
        responses,
        posterior_weights,
        quadrature,
        h=h,
        prior_mass=prior_mass,
    )
