"""Matrix-based standard errors for marginal item-response models.

The routines in this module differentiate the person-level marginal
log-likelihood. Keeping that objective in one place makes the observed,
cross-product, and sandwich estimators consistent and ensures that a fitted
latent density is not silently replaced by the quadrature's default mass.

Built-in item models whose parameters each belong to one item, or to all
items as rating-scale thresholds do, use the exact Louis observed
information and marginal scores from
:mod:`mirt.estimation._louis_information`. Other models difference the
marginal log-likelihood, which costs O(P^2) likelihood evaluations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.utils.numeric import logsumexp, logsumexp_axis1

if TYPE_CHECKING:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.models.base import BaseItemModel

StandardErrorMethod = Literal["auto", "oakes", "crossprod", "sandwich", "complete_data"]
_STANDARD_ERROR_METHODS = ("auto", "oakes", "crossprod", "sandwich", "complete_data")
# Coordinates this close to an optimizer bound are held fixed.
_BOUND_TOLERANCE = 1e-6
# Relative curvature below which a coordinate carries no information.
_INFORMATION_TOLERANCE = 1e-12


def validate_se_method(value: object) -> StandardErrorMethod:
    """Validate a fitted-model standard-error method name."""
    if not isinstance(value, str) or value not in _STANDARD_ERROR_METHODS:
        expected = ", ".join(repr(name) for name in _STANDARD_ERROR_METHODS)
        raise MirtValidationError(
            f"se_method must be one of {expected}",
            parameter="se_method",
            value=value,
            expected=expected,
        )
    return value


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
        model._parameters[name] = model._canonical_parameter_values(
            name, values.reshape(layout.shape)
        )
        offset += size


def _restore_parameters(
    model: BaseItemModel,
    parameters: dict[str, NDArray[np.float64]],
) -> None:
    model._parameters = {name: values.copy() for name, values in parameters.items()}


def _unflatten_se(
    se_flat: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    model: BaseItemModel | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Restore standard errors to the model's stored parameter shapes."""
    result: dict[str, NDArray[np.float64]] = {}
    offset = 0
    for name, layout in layouts.items():
        size = layout.free_indices.size
        values = np.zeros(layout.shape, dtype=np.float64)
        values.ravel()[layout.free_indices] = se_flat[offset : offset + size]
        result[name] = (
            values
            if model is None
            else model._expand_parameter_standard_errors(name, values)
        )
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


def _finite_difference_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
    person_weights: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    """Stream perturbations with O(N + P²) retained storage.

    Difference person-level values before summing to preserve the original
    numerical cancellation behavior, including survey-weighted information.
    """
    params, layouts = _flatten_parameters(model)
    original = model.parameters
    weights = (
        np.ones(responses.shape[0], dtype=np.float64)
        if person_weights is None
        else np.asarray(person_weights, dtype=np.float64)
    )
    information = np.zeros((params.size, params.size), dtype=np.float64)

    def evaluate(candidate: NDArray[np.float64]) -> NDArray[np.float64]:
        _set_flat_parameters(model, candidate, layouts)
        return _marginal_log_likelihoods(model, responses, quadrature, prior_mass)

    try:
        center = evaluate(params)
        for row in range(params.size):
            plus, minus = params.copy(), params.copy()
            plus[row] += h
            minus[row] -= h
            second = evaluate(plus) - 2.0 * center + evaluate(minus)
            information[row, row] = -float(weights @ second) / h**2
            for column in range(row + 1, params.size):
                cross = None
                for row_sign, column_sign, coefficient in (
                    (1, 1, 1),
                    (1, -1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1),
                ):
                    candidate = params.copy()
                    candidate[row] += row_sign * h
                    candidate[column] += column_sign * h
                    values = evaluate(candidate)
                    if cross is None:
                        cross = values
                    elif coefficient == 1:
                        cross += values
                    else:
                        cross -= values
                value = -float(weights @ cross) / (4.0 * h**2)
                information[row, column] = value
                information[column, row] = value
    finally:
        _restore_parameters(model, original)
    return information, layouts


def _finite_difference_scores(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    """Retain only the output score matrix and the current perturbation pair."""
    params, layouts = _flatten_parameters(model)
    original = model.parameters
    scores = np.empty((responses.shape[0], params.size), dtype=np.float64)
    try:
        for column in range(params.size):
            plus, minus = params.copy(), params.copy()
            plus[column] += h
            minus[column] -= h
            _set_flat_parameters(model, plus, layouts)
            values = _marginal_log_likelihoods(model, responses, quadrature, prior_mass)
            _set_flat_parameters(model, minus, layouts)
            values -= _marginal_log_likelihoods(
                model, responses, quadrature, prior_mass
            )
            scores[:, column] = values / (2.0 * h)
    finally:
        _restore_parameters(model, original)
    return scores, layouts


def _information_and_meat(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    h: float,
    *,
    person_weights: NDArray[np.float64] | None = None,
    meat_weights: NDArray[np.float64] | None = None,
    observed: bool = True,
) -> tuple[
    NDArray[np.float64] | None, NDArray[np.float64], dict[str, _ParameterLayout]
]:
    """Return the weighted observed information and score cross-product.

    The information is ``sum_i w_i`` times each person's negative Hessian and
    the cross-product ``sum_i m_i s_i s_i'``, with ``m`` defaulting to the
    person weights ``w``. The information is omitted when ``observed`` is
    false.
    """
    from mirt.estimation._louis_information import (
        louis_information,
        supports_louis_information,
    )

    _, layouts = _flatten_parameters(model)
    if supports_louis_information(model):
        terms = louis_information(
            model,
            responses,
            quadrature.nodes,
            prior_mass,
            layouts,
            person_weights=person_weights,
            meat_weights=meat_weights,
            observed=observed,
        )
        information = terms.information if observed else None
        return information, terms.score_crossproduct, layouts

    information = None
    if observed:
        information, _ = _finite_difference_information(
            model, responses, quadrature, prior_mass, h, person_weights=person_weights
        )
    scores, _ = _finite_difference_scores(model, responses, quadrature, prior_mass, h)
    weights = meat_weights if meat_weights is not None else person_weights
    weighted = scores if weights is None else scores * weights[:, None]
    return information, weighted.T @ scores, layouts


def _symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return (matrix + matrix.T) / 2.0


def _inverse(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        return np.linalg.inv(_symmetric(matrix))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(_symmetric(matrix))


def _covariance(
    method: Literal["oakes", "crossprod", "sandwich"],
    information: NDArray[np.float64] | None,
    meat: NDArray[np.float64] | None,
    active: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Invert the active block; held or non-estimable coordinates become NaN.

    Coordinates without positive curvature, such as those of an item with no
    observed responses, carry no information and are excluded rather than
    given the zero variance of a pseudo-inverse.
    """
    size = active.size
    covariance = np.full((size, size), np.nan, dtype=np.float64)
    source = meat if method == "crossprod" else information
    assert source is not None
    curvature = np.diag(source)
    if np.any(active):
        scale = float(np.max(np.abs(curvature[active])))
        active = active & (curvature > _INFORMATION_TOLERANCE * scale)
    block = np.ix_(active, active)
    if not np.any(active):
        return covariance
    if method == "crossprod":
        assert meat is not None
        active_covariance = _inverse(meat[block])
    else:
        assert information is not None
        bread = _inverse(information[block])
        if method == "sandwich":
            assert meat is not None
            bread = bread @ meat[block] @ bread.T
        active_covariance = bread
    covariance[block] = _symmetric(active_covariance)
    variances = np.diag(covariance)
    invalid = ~(np.isfinite(variances) & (variances >= 0.0))
    covariance[invalid, :] = np.nan
    covariance[:, invalid] = np.nan
    return covariance


def _se_from_covariance(
    covariance: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    model: BaseItemModel | None = None,
) -> dict[str, NDArray[np.float64]]:
    variances = np.diag(covariance)
    se = np.full(variances.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(variances) & (variances >= 0.0)
    se[valid] = np.sqrt(variances[valid])
    return _unflatten_se(se, layouts, model)


def _se_from_information(
    information: NDArray[np.float64],
    layouts: dict[str, _ParameterLayout],
    model: BaseItemModel | None = None,
) -> dict[str, NDArray[np.float64]]:
    if information.size == 0:
        return _unflatten_se(np.empty(0, dtype=np.float64), layouts, model)
    active = np.ones(information.shape[0], dtype=np.bool_)
    covariance = _covariance("oakes", information, None, active)
    return _se_from_covariance(covariance, layouts, model)


def compute_observed_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the negative Hessian of the marginal log-likelihood.

    Built-in item models use the exact Louis identity, for which ``h`` is
    unused; other models use central differences with step ``h``. Rows and
    columns follow the free parameters in ``model.parameters`` order.
    """
    step = _validate_step_size(h)
    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    information, _, _ = _information_and_meat(
        model, response_array, quadrature, mass, step
    )
    assert information is not None
    return information


def compute_crossprod_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute standard errors from the outer product of person scores.

    Built-in item models use exact marginal scores; others use central
    differences with step ``h``.
    """
    step = _validate_step_size(h)
    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    _, meat, layouts = _information_and_meat(
        model, response_array, quadrature, mass, step, observed=False
    )
    active = np.ones(meat.shape[0], dtype=np.bool_)
    return _se_from_covariance(
        _covariance("crossprod", None, meat, active), layouts, model
    )


def compute_sandwich_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    survey_weights: NDArray[np.float64] | None = None,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute robust bread-meat-bread standard errors.

    The bread is the survey-weighted observed information and the meat the
    cross-product of survey-weighted person scores. Built-in item models use
    exact Louis terms; others use central differences with step ``h``.
    """
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
    bread, meat, layouts = _information_and_meat(
        model,
        response_array,
        quadrature,
        mass,
        step,
        person_weights=weights,
        meat_weights=weights**2,
    )
    active = np.ones(meat.shape[0], dtype=np.bool_)
    covariance = _covariance("sandwich", bread, meat, active)
    return _se_from_covariance(covariance, layouts, model)


def compute_oakes_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute observed-information standard errors at an EM solution.

    Built-in item models use the exact Louis observed information, which is
    the target of the Oakes (1999) identity; other models difference the
    marginal log-likelihood with step ``h``.
    """
    information = compute_observed_information(
        model, responses, posterior_weights, quadrature, h=h, prior_mass=prior_mass
    )
    _, layouts = _flatten_parameters(model)
    return _se_from_information(information, layouts, model)


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


def compute_expected_information(
    model: BaseItemModel,
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    n_persons: int,
    h: float = 1e-5,
) -> tuple[NDArray[np.float64], dict[str, _ParameterLayout]]:
    """Return the marginal expected (Fisher) information of ``n_persons``.

    The information ``n_persons * sum_y P(y) s(y) s(y)'`` enumerates every
    complete response pattern ``y`` with its exact marginal score ``s(y)``.

    Raises
    ------
    MirtValidationError
        If the model has more than ``2**16`` response patterns.
    """
    from mirt.estimation._louis_information import (
        MAX_ENUMERATED_PATTERNS,
        enumerated_patterns,
        expected_louis_information,
        supports_louis_information,
    )

    patterns = enumerated_patterns(model)
    if patterns is None:
        raise MirtValidationError(
            "expected (Fisher) information enumerates every response pattern, "
            f"which exceeds {MAX_ENUMERATED_PATTERNS} for this model; use "
            "method='oakes' for observed information",
            parameter="method",
            value="fisher",
            expected="'oakes'",
        )
    mass = _validate_prior_mass(prior_mass, quadrature.nodes.shape[0])
    _, layouts = _flatten_parameters(model)
    if supports_louis_information(model):
        information = expected_louis_information(
            model, quadrature.nodes, mass, layouts, patterns
        )
    else:
        step = _validate_step_size(h)
        scores, _ = _finite_difference_scores(model, patterns, quadrature, mass, step)
        probability = np.exp(
            _marginal_log_likelihoods(model, patterns, quadrature, mass)
        )
        information = (scores * probability[:, None]).T @ scores
    return float(n_persons) * information, layouts


@dataclass(frozen=True)
class CovarianceEstimate:
    """Standard errors of a fitted model with their parameter covariance.

    Attributes
    ----------
    standard_errors : dict
        Standard errors in the stored parameter shapes.
    covariance : ndarray of shape (P, P)
        Covariance of the free parameters in ``model.parameters`` order.
        Rows and columns of coordinates held at an optimizer bound, or whose
        variance is not estimable, are NaN.
    method : str
        Estimator that produced the covariance.
    """

    standard_errors: dict[str, NDArray[np.float64]]
    covariance: NDArray[np.float64]
    method: str


def _free_values(
    arrays: Mapping[str, NDArray[np.float64]],
    layouts: dict[str, _ParameterLayout],
) -> NDArray[np.float64]:
    """Gather full-shape arrays at the free coordinates; missing names are 0."""
    chunks = [
        (
            np.asarray(arrays[name], dtype=np.float64).ravel()[layout.free_indices]
            if name in arrays
            else np.zeros(layout.free_indices.size)
        )
        for name, layout in layouts.items()
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _coordinates_at_bounds(
    layouts: dict[str, _ParameterLayout],
    bounds: Callable[[str], tuple[float, float]],
) -> NDArray[np.bool_]:
    flags = []
    for name, layout in layouts.items():
        values = layout.template.ravel()[layout.free_indices]
        low, high = bounds(name)
        flags.append(
            (np.abs(values - low) <= _BOUND_TOLERANCE)
            | (np.abs(values - high) <= _BOUND_TOLERANCE)
        )
    return np.concatenate(flags) if flags else np.empty(0, dtype=np.bool_)


def estimate_covariance(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    prior_mass: NDArray[np.float64],
    method: Literal["oakes", "crossprod", "sandwich"],
    *,
    frequencies: NDArray[np.float64] | None = None,
    h: float = 1e-5,
    bounds: Callable[[str], tuple[float, float]] | None = None,
    prior_information: Mapping[str, NDArray[np.float64]] | None = None,
) -> CovarianceEstimate:
    """Estimate the free-parameter covariance of a fitted marginal model.

    Parameters
    ----------
    model : BaseItemModel
        Fitted model.
    responses : ndarray of shape (n_rows, n_items)
        Responses, possibly compressed to unique patterns.
    quadrature : GaussHermiteQuadrature
        Quadrature used in fitting.
    prior_mass : ndarray of shape (n_points,)
        Latent prior mass at the nodes; latent-density parameters are
        treated as fixed.
    method : {"oakes", "crossprod", "sandwich"}
        Observed information, score cross-product, or their sandwich.
    frequencies : ndarray of shape (n_rows,), optional
        Number of persons sharing each response row.
    h : float, default=1e-5
        Finite-difference step for models without exact derivatives.
    bounds : callable, optional
        Maps a parameter name to its optimizer box. Free coordinates on a
        bound are held fixed and receive NaN standard errors.
    prior_information : mapping of str to ndarray, optional
        Negative second derivative of an independent item log-prior, in the
        stored parameter shapes, for Bayes modal estimates. It is added to the
        observed information, to the bread of the sandwich, and to the score
        cross-product that ``"crossprod"`` inverts.

    Returns
    -------
    CovarianceEstimate
        Standard errors and the free-parameter covariance.
    """
    mass = _validate_prior_mass(prior_mass, quadrature.nodes.shape[0])
    information, meat, layouts = _information_and_meat(
        model,
        np.asarray(responses),
        quadrature,
        mass,
        _validate_step_size(h),
        person_weights=frequencies,
        observed=method != "crossprod",
    )
    if prior_information is not None:
        curvature = np.diag(_free_values(prior_information, layouts))
        if information is not None:
            information = information + curvature
        if method == "crossprod":
            meat = meat + curvature
    active = np.ones(meat.shape[0], dtype=np.bool_)
    if bounds is not None:
        active &= ~_coordinates_at_bounds(layouts, bounds)
    covariance = _covariance(method, information, meat, active)
    return CovarianceEstimate(
        _se_from_covariance(covariance, layouts, model), covariance, method
    )
