"""Standard error computation methods for IRT models.

This module provides multiple methods for computing standard errors:

- Numerical, central, forward and Richardson: itemwise finite differences of
  the expected complete-data log-likelihood (diagonal complete-data curvature,
  which ignores the missing information and understates uncertainty).
  Parameters shared by all items, such as rating-scale thresholds, are
  differenced in the expected log-likelihood summed over items.
- Louis, Oakes and SEM: the observed information of the marginal likelihood
- Crossprod: the outer product of marginal person scores
- Sandwich: observed information bread around the score cross-product
- Fisher: the marginal expected information, by response-pattern enumeration

The marginal information and scores are exact (Louis, 1982) for the item
models accepted by
:func:`~mirt.estimation._louis_information.supports_louis_information`:
1PL-4PL with any number of factors, GRM, GPCM, PCM, NRM, RSM, GRSM,
multidimensional and bifactor models, and the other built-in families whose
likelihood is a product of item curves. Other models, such as custom,
testlet or mixture models, use central differences of the marginal
log-likelihood, which cost O(P^2) likelihood evaluations. As in fitted
results, the matrix methods hold coordinates on an EM optimizer bound fixed
with ``NaN`` standard errors.

References
----------
Louis, T. A. (1982). Finding the observed information matrix when using
    the EM algorithm. Journal of the Royal Statistical Society B, 44, 226-233.

Oakes, D. (1999). Direct calculation of the information matrix via the EM
    algorithm. Journal of the Royal Statistical Society B, 61, 479-482.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.models.base import BaseItemModel


SEMethod = Literal[
    "numerical",
    "richardson",
    "forward",
    "central",
    "louis",
    "sandwich",
    "oakes",
    "crossprod",
    "sem",
    "fisher",
]
# Methods built from each item's own expected complete-data likelihood.
_ITEMWISE_METHODS = frozenset({"numerical", "central", "forward", "richardson"})


def _valid_second_derivative(
    log_likelihood_at_offset: Callable[[float], float],
    h: float,
    *,
    scheme: Literal["central", "forward"],
    center: float | None = None,
) -> float:
    """Evaluate a second derivative without crossing parameter boundaries."""
    if center is None:
        center = log_likelihood_at_offset(0.0)
    if scheme == "central":
        stencils = ((-1.0, 0.0, 1.0), (0.0, 1.0, 2.0), (0.0, -1.0, -2.0))
    else:
        stencils = ((0.0, 1.0, 2.0), (0.0, -1.0, -2.0))

    last_error: MirtValidationError | None = None
    step = h
    for _ in range(20):
        for offsets in stencils:
            try:
                evaluations = [
                    center if offset == 0.0 else log_likelihood_at_offset(offset * step)
                    for offset in offsets
                ]
            except MirtValidationError as exc:
                last_error = exc
                continue
            return (evaluations[0] - 2.0 * evaluations[1] + evaluations[2]) / (step**2)
        step *= 0.5

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to construct a valid finite-difference stencil")


def _diagonal_item_standard_errors(
    log_likelihood: Callable[[float | NDArray[np.float64]], float],
    current: float | NDArray[np.float64],
    h: float,
    *,
    scheme: Literal["central", "forward"],
    free_mask: NDArray[np.bool_] | np.bool_ | bool | None = None,
) -> float | NDArray[np.float64]:
    """Perturb each item coordinate independently, preserving its array shape."""
    mask = None if free_mask is None else np.asarray(free_mask, dtype=np.bool_)
    if mask is not None and not np.any(mask):
        return np.zeros_like(current) if isinstance(current, np.ndarray) else 0.0
    center = log_likelihood(current)
    if isinstance(current, np.ndarray):
        standard_errors = np.zeros_like(current)
        for index in np.ndindex(current.shape):
            if mask is not None and not mask[index]:
                continue

            def at_offset(offset: float) -> float:
                candidate = current.copy()
                candidate[index] += offset
                return log_likelihood(candidate)

            curvature = _valid_second_derivative(
                at_offset, h, scheme=scheme, center=center
            )
            standard_errors[index] = (
                np.sqrt(-1.0 / curvature) if curvature < 0 else np.nan
            )
        return standard_errors

    curvature = _valid_second_derivative(
        lambda offset: log_likelihood(current + offset), h, scheme=scheme, center=center
    )
    return np.sqrt(-1.0 / curvature) if curvature < 0 else np.nan


def _em_bounds(model: BaseItemModel) -> Callable[[str], tuple[float, float]]:
    """Return the optimizer boxes that EM fits hold coordinates fixed at."""
    from mirt.estimation.base import _parameter_bounds

    return lambda name: _parameter_bounds(model, name)


def compute_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    method: SEMethod = "numerical",
    step_size: float = 1e-5,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute standard errors using specified method.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    responses : ndarray
        Response matrix.
    quadrature : GaussHermiteQuadrature
        Quadrature object for integration.
    posterior_weights : ndarray
        Posterior weights from final E-step.
    method : str
        Method for SE computation. The default ``"numerical"`` is itemwise
        complete-data curvature; ``"oakes"`` (or its aliases ``"louis"`` and
        ``"sem"``) gives observed-information standard errors. The matrix
        methods (``"oakes"``, ``"louis"``, ``"sem"``, ``"crossprod"``,
        ``"sandwich"`` and ``"fisher"``) hold coordinates on an EM optimizer
        bound, such as a guessing parameter of 0, fixed with ``NaN`` standard
        errors, so they match the fitted ``se_method`` of an EM result
        without item priors or equality constraints.
    step_size : float
        Step size for numerical differentiation. Matrix-based methods use it
        only for models without exact item derivatives.
    n_jobs : int
        Number of parallel jobs for item-wise computation.
        Use -1 for all CPUs, 1 for sequential.
    prior_mass : ndarray, optional
        Quadrature prior mass used by matrix-based methods. When omitted,
        it is recovered from the final posterior weights.

    Returns
    -------
    dict
        Standard errors for each parameter.

    Notes
    -----
    For a :class:`~mirt.models.mixed_format.MixedItemModel`, the itemwise
    methods difference each component on its own items and report its
    errors under qualified names such as ``"3PL.guessing"``.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel) and method in _ITEMWISE_METHODS:
        responses = np.asarray(responses)
        return {
            f"{prefix}.{name}": errors
            for prefix, (component, items) in zip(
                model.component_names, model.components, strict=True
            )
            for name, errors in compute_se(
                component,
                responses[:, items],
                quadrature,
                posterior_weights,
                method,
                step_size,
                n_jobs,
            ).items()
        }
    if method in ("numerical", "central"):
        return _se_numerical_central(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "forward":
        return _se_numerical_forward(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "richardson":
        return _se_richardson(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "louis":
        return _se_louis(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    elif method == "sandwich":
        return _se_sandwich(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    elif method == "oakes":
        return _se_oakes(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    elif method == "crossprod":
        return _se_crossprod(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    elif method == "sem":
        return _se_sem(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    elif method == "fisher":
        return _se_fisher(
            model,
            responses,
            quadrature,
            posterior_weights,
            step_size,
            n_jobs,
            prior_mass,
        )
    else:
        raise ValueError(f"Unknown SE method: {method}")


def _se_numerical_central(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Central difference numerical Hessian."""
    return _se_itemwise_numerical(
        model,
        responses,
        quadrature,
        posterior_weights,
        h,
        n_jobs,
        scheme="central",
    )


def _se_itemwise_numerical(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int,
    *,
    scheme: Literal["central", "forward"],
    richardson: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Compute diagonal item-wise curvature without sharing mutable models."""
    from copy import deepcopy

    from mirt.scoring._common import resolve_n_jobs

    params = model.parameters
    free_masks = model.free_parameter_masks
    result = {name: np.zeros_like(values) for name, values in params.items()}
    workers = resolve_n_jobs(n_jobs)
    shared = any(np.any(free_masks[name]) for name in model._shared_parameters)
    names_by_item = {
        item: tuple(
            name
            for name, mask in free_masks.items()
            if name not in model._shared_parameters and np.any(mask[item])
        )
        for item in range(model.n_items)
    }
    items = [item for item, names in names_by_item.items() if names]
    if not items and not shared:
        return result

    with EMFitContext(responses) as context:
        correct = observed = None
        category_counts: dict[int, NDArray[np.float64]] = {}
        if model.is_polytomous:
            for item in range(model.n_items) if shared else items:
                category_counts[item] = context.expected_category_counts(
                    item, model.n_categories[item], posterior_weights
                )
        else:
            correct, observed = context.expected_counts(
                posterior_weights, cache_components=False
            )
        if shared:
            from mirt.estimation._shared_step import binary_category_counts

            counts = (
                [category_counts[item] for item in range(model.n_items)]
                if model.is_polytomous
                else binary_category_counts(correct, observed)
            )
            for step in (h, h / 2) if richardson else (h,):
                errors = shared_parameter_standard_errors(
                    model, quadrature.nodes, counts, step, scheme=scheme
                )
                for name, values in errors.items():
                    # Richardson combines the full and half steps.
                    result[name] = (
                        values if step == h else (4 * values - result[name]) / 3
                    )

        def compute_item(
            item: int,
        ) -> tuple[int, dict[str, float | NDArray[np.float64]]]:
            # Preserve constructor state and bound instance methods as well as
            # parameters. One isolated model serves all of this item's fields.
            local = model if workers == 1 else deepcopy(model)
            counts = category_counts.get(item)
            item_correct = None if correct is None else correct[item]
            item_observed = None if observed is None else observed[item]
            item_result = {}
            for name in names_by_item[item]:
                first = _compute_item_se_curvature(
                    local,
                    item,
                    name,
                    responses,
                    quadrature,
                    posterior_weights,
                    h,
                    scheme=scheme,
                    r_k=item_correct,
                    n_k_valid=item_observed,
                    r_kc=counts,
                )
                if richardson:
                    second = _compute_item_se_curvature(
                        local,
                        item,
                        name,
                        responses,
                        quadrature,
                        posterior_weights,
                        h / 2,
                        scheme=scheme,
                        r_k=item_correct,
                        n_k_valid=item_observed,
                        r_kc=counts,
                    )
                    first = (4 * second - first) / 3
                item_result[name] = first
            return item, item_result

        if workers == 1 or len(items) <= 1:
            results = map(compute_item, items)
        else:
            results = context.executor(min(workers, len(items))).map(
                compute_item, items
            )
        for item, values in results:
            for name, value in values.items():
                result[name][item] = value

    return {
        name: model._expand_parameter_standard_errors(name, errors)
        for name, errors in result.items()
    }


def shared_parameter_standard_errors(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    counts: Sequence[NDArray[np.float64]],
    h: float,
    *,
    scheme: Literal["central", "forward"] = "central",
    epsilon: float = PROB_EPSILON,
    exact: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Diagonal complete-data curvature of parameters shared by all items.

    Each free shared coordinate is differenced in the expected complete-data
    log-likelihood summed over items, matching the itemwise convention for
    item parameters.

    Parameters
    ----------
    model : BaseItemModel
        Fitted model. Its parameters are restored before returning.
    nodes : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    counts : sequence of ndarray
        Each item's expected ``(n_points, n_categories)`` counts; a binary
        item has incorrect and correct columns.
    h : float
        Finite-difference step.
    scheme : {"central", "forward"}, default="central"
        Difference stencil.
    epsilon : float, optional
        Probability clipping bound.
    exact : bool, default=False
        Use closed-form second derivatives when the model has them, in place
        of differences with ``h`` and ``scheme``.

    Returns
    -------
    dict
        Standard errors of each shared parameter, zero at fixed coordinates.
    """
    from mirt.estimation._louis_information import (
        has_analytic_item_derivatives,
        item_terms,
    )
    from mirt.estimation._shared_step import SharedParameters, _numerical_objective
    from mirt.estimation.standard_errors import _flatten_parameters

    layout = SharedParameters(model)
    result = {
        name: np.zeros_like(model._parameters[name])
        for name in model._shared_parameters
    }
    if not layout.size:
        return result
    if exact and has_analytic_item_derivatives(model):
        # Shared coordinates trail every item's derivative terms.
        curvature = np.zeros(layout.size)
        for term, item_counts in zip(
            item_terms(model, nodes, _flatten_parameters(model)[1]), counts, strict=True
        ):
            assert term is not None
            shared = term.second[-layout.size :, -layout.size :]
            curvature += np.einsum("aaqc,qc->a", shared, item_counts)
        errors = np.sqrt(
            np.divide(
                -1.0,
                curvature,
                out=np.full(layout.size, np.nan),
                where=curvature < 0,
            )
        )
    else:
        loss = _numerical_objective(model, layout, nodes, counts, epsilon)
        original = {name: model._parameters[name] for name in layout.masks}
        try:
            errors = np.asarray(
                _diagonal_item_standard_errors(
                    lambda vector: -loss(vector), layout.get(model), h, scheme=scheme
                )
            )
        finally:
            model._parameters.update(original)
    offset = 0
    for name, mask in layout.masks.items():
        count = int(np.count_nonzero(mask))
        result[name][mask] = errors[offset : offset + count]
        offset += count
    return result


def _compute_item_se_curvature(
    model: BaseItemModel,
    item_idx: int,
    param_name: str,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    *,
    scheme: Literal["central", "forward"],
    r_k: NDArray[np.float64] | None = None,
    n_k_valid: NDArray[np.float64] | None = None,
    r_kc: NDArray[np.float64] | None = None,
    epsilon: float = PROB_EPSILON,
) -> float | NDArray[np.float64]:
    """Compute a single item's diagonal finite-difference curvature."""
    quad_points = quadrature.nodes

    values = model.parameters[param_name]
    if values.ndim == 1:
        current = float(values[item_idx])
    else:
        current = values[item_idx].copy()

    def set_parameter(param_val):
        candidate = values.copy()
        candidate[item_idx] = param_val
        canonical = model._canonical_parameter_values(param_name, candidate)
        row = np.asarray(canonical[item_idx])
        model.set_item_parameter(
            item_idx, param_name, float(row) if row.ndim == 0 else row
        )

    if model.is_polytomous:
        if r_kc is None:
            r_kc = EMFitContext(responses).expected_category_counts(
                item_idx, model.n_categories[item_idx], posterior_weights
            )

        def log_likelihood(param_val):
            set_parameter(param_val)
            try:
                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, epsilon, 1 - epsilon)
                return float(np.sum(r_kc * np.log(probs)))
            finally:
                model.set_item_parameter(item_idx, param_name, current)
    else:
        if r_k is None or n_k_valid is None:
            correct, observed = EMFitContext(
                responses[:, item_idx : item_idx + 1]
            ).expected_counts(posterior_weights, cache_components=False)
            if r_k is None:
                r_k = correct[0]
            if n_k_valid is None:
                n_k_valid = observed[0]

        def log_likelihood(param_val):
            set_parameter(param_val)
            try:
                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, epsilon, 1 - epsilon)
                return float(
                    np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))
                )
            finally:
                model.set_item_parameter(item_idx, param_name, current)

    return _diagonal_item_standard_errors(
        log_likelihood,
        current,
        h,
        scheme=scheme,
        free_mask=model.free_parameter_masks[param_name][item_idx],
    )


def _se_numerical_forward(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Forward difference numerical Hessian (less accurate but faster)."""
    return _se_itemwise_numerical(
        model,
        responses,
        quadrature,
        posterior_weights,
        h,
        n_jobs,
        scheme="forward",
    )


def _se_richardson(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Richardson extrapolation for improved numerical accuracy.

    Uses two step sizes and extrapolates for higher accuracy.
    """
    return _se_itemwise_numerical(
        model,
        responses,
        quadrature,
        posterior_weights,
        h,
        n_jobs,
        scheme="central",
        richardson=True,
    )


def _se_louis(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Louis-equivalent observed-information standard errors."""
    return _se_oakes(
        model,
        responses,
        quadrature,
        posterior_weights,
        h,
        n_jobs,
        prior_mass,
    )


def _se_sandwich(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Sandwich (robust) standard errors.

    Computes SE as: sqrt(diag(H^-1 * B * H^-1))
    where H is the Hessian and B is the outer product of gradients.

    This provides consistent SEs even under model misspecification.
    """
    del n_jobs
    from mirt.estimation.standard_errors import compute_sandwich_se

    return compute_sandwich_se(
        model,
        responses,
        posterior_weights,
        quadrature,
        h=h,
        prior_mass=prior_mass,
        bounds=_em_bounds(model),
    )


def _se_oakes(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Oakes information method.

    Evaluate the observed-information target of the Oakes (1999) identity
    directly from the marginal likelihood at the converged EM solution.
    """
    del n_jobs
    from mirt.estimation.standard_errors import compute_oakes_se

    return compute_oakes_se(
        model,
        responses,
        posterior_weights,
        quadrature,
        h=h,
        prior_mass=prior_mass,
        bounds=_em_bounds(model),
    )


def _se_crossprod(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Cross-product of scores standard errors.

    Estimates information from the outer product of score vectors:
        I ≈ sum_i s_i * s_i'
    """
    del n_jobs
    from mirt.estimation.standard_errors import compute_crossprod_se

    return compute_crossprod_se(
        model,
        responses,
        posterior_weights,
        quadrature,
        h=h,
        prior_mass=prior_mass,
        bounds=_em_bounds(model),
    )


def _se_sem(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Supplemented EM (SEM) standard errors.

    Evaluate the observed-information target of supplemented EM directly.
    This deterministic form avoids a noisy, seed-dependent rate estimate.
    """
    del n_jobs
    from mirt.estimation.standard_errors import compute_sem_se

    return compute_sem_se(
        model,
        responses,
        posterior_weights,
        quadrature,
        h=h,
        prior_mass=prior_mass,
        bounds=_em_bounds(model),
    )


def _se_fisher(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
    prior_mass: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Marginal expected (Fisher) information standard errors.

    The information ``N * sum_y P(y) s(y) s(y)'`` sums the outer products of
    exact marginal scores over every complete response pattern, weighted by
    the pattern probabilities under the fitted model. It assumes the model is
    correctly specified and is available for at most ``2**16`` patterns.
    """
    del n_jobs
    from mirt.estimation.standard_errors import (
        _resolve_prior_mass,
        _se_from_information,
        compute_expected_information,
    )

    response_array = np.asarray(responses)
    mass = _resolve_prior_mass(
        model, response_array, posterior_weights, quadrature, prior_mass
    )
    information, layouts = compute_expected_information(
        model, quadrature, mass, response_array.shape[0], h
    )
    return _se_from_information(information, layouts, model, _em_bounds(model))
