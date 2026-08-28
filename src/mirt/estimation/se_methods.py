"""Standard error computation methods for IRT models.

This module provides multiple methods for computing standard errors:
- Numerical (finite difference Hessian)
- Louis (missing information principle)
- Sandwich (robust standard errors)
- Oakes (cross-product of scores)
- Crossprod (observed information from scores)
- SEM (supplemented EM)

References
----------
Louis, T. A. (1982). Finding the observed information matrix when using
    the EM algorithm. Journal of the Royal Statistical Society B, 44, 226-233.

Oakes, D. (1999). Direct calculation of the information matrix via the EM
    algorithm. Journal of the Royal Statistical Society B, 61, 479-482.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
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
        Method for SE computation.
    step_size : float
        Step size for numerical differentiation.
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
    """
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
            model, responses, quadrature, posterior_weights, step_size, n_jobs
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
) -> dict[str, NDArray[np.float64]]:
    """Compute diagonal item-wise curvature without sharing mutable models."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    se_dict = {}
    free_masks = model.free_parameter_masks

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    for param_name, values in model.parameters.items():
        free_mask = free_masks[param_name]
        if not np.any(free_mask):
            se_dict[param_name] = np.zeros_like(values)
            continue

        se = np.zeros_like(values)

        def item_standard_error(
            item_model: BaseItemModel,
            item_idx: int,
        ) -> float | NDArray[np.float64]:
            return _compute_item_se_curvature(
                item_model,
                item_idx,
                param_name,
                responses,
                quadrature,
                posterior_weights,
                h,
                scheme=scheme,
            )

        if n_jobs == 1:
            for item_idx in range(model.n_items):
                se[item_idx] = item_standard_error(model, item_idx)
        else:

            def compute_item(
                item_idx: int,
            ) -> tuple[int, float | NDArray[np.float64]]:
                return item_idx, item_standard_error(model.copy(), item_idx)

            with ThreadPoolExecutor(max_workers=min(n_jobs, model.n_items)) as executor:
                results = list(executor.map(compute_item, range(model.n_items)))

            for item_idx, item_se in results:
                se[item_idx] = item_se

        se[~free_mask] = 0.0
        se_dict[param_name] = se

    return se_dict


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
) -> float | NDArray[np.float64]:
    """Compute a single item's diagonal finite-difference curvature."""
    quad_points = quadrature.nodes
    item_responses = responses[:, item_idx]
    valid_mask = item_responses >= 0

    values = model.parameters[param_name]
    if values.ndim == 1:
        current = float(values[item_idx])
        is_scalar = True
    else:
        current = values[item_idx].copy()
        is_scalar = False

    n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)

    if model.is_polytomous:
        n_categories = model._n_categories[item_idx]
        n_quad = quad_points.shape[0]
        r_kc = np.zeros((n_quad, n_categories))
        for c in range(n_categories):
            cat_mask = valid_mask & (item_responses == c)
            r_kc[:, c] = np.sum(posterior_weights[cat_mask, :], axis=0)

        def log_likelihood(param_val):
            model.set_item_parameter(item_idx, param_name, param_val)
            probs = model.probability(quad_points, item_idx)
            probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)
            ll = float(np.sum(r_kc * np.log(probs)))
            model.set_item_parameter(item_idx, param_name, current)
            return ll
    else:
        r_k = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )

        def log_likelihood(param_val):
            model.set_item_parameter(item_idx, param_name, param_val)
            probs = model.probability(quad_points, item_idx)
            probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)
            ll = float(
                np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))
            )
            model.set_item_parameter(item_idx, param_name, current)
            return ll

    ll_center = log_likelihood(current)
    if is_scalar:
        hessian = _valid_second_derivative(
            lambda offset: log_likelihood(current + offset),
            h,
            scheme=scheme,
            center=ll_center,
        )
        return np.sqrt(-1.0 / hessian) if hessian < 0 else np.nan
    else:
        n_params = len(current)
        se = np.zeros(n_params)
        for i in range(n_params):

            def log_likelihood_at_offset(offset: float) -> float:
                candidate = current.copy()
                candidate[i] += offset
                return log_likelihood(candidate)

            hessian = _valid_second_derivative(
                log_likelihood_at_offset,
                h,
                scheme=scheme,
                center=ll_center,
            )
            se[i] = np.sqrt(-1.0 / hessian) if hessian < 0 else np.nan
        return se


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
    se1 = _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
    )
    se2 = _se_numerical_central(
        model, responses, quadrature, posterior_weights, h / 2, n_jobs
    )

    se_dict = {}
    for name in se1:
        se_dict[name] = (4 * se2[name] - se1[name]) / 3

    return se_dict


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
    )


def _se_fisher(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Expected (Fisher) information standard errors.

    Uses the expected information matrix computed from the model.
    This assumes the model is correctly specified.
    """
    quad_points = quadrature.nodes
    quad_weights = quadrature.weights
    n_persons = responses.shape[0]

    se_dict = {}
    free_masks = model.free_parameter_masks

    for param_name, values in model.parameters.items():
        free_mask = free_masks[param_name]
        if not np.any(free_mask):
            se_dict[param_name] = np.zeros_like(values)
            continue

        se = np.zeros_like(values)

        for item_idx in range(model.n_items):
            info = model.information(quad_points, item_idx)
            expected_info = n_persons * np.sum(quad_weights * info)

            if values.ndim == 1:
                se[item_idx] = (
                    1.0 / np.sqrt(expected_info) if expected_info > 0 else np.nan
                )
            else:
                se[item_idx] = np.full(
                    values.shape[1],
                    1.0 / np.sqrt(expected_info) if expected_info > 0 else np.nan,
                )

        se[~free_mask] = 0.0
        se_dict[param_name] = se

    return se_dict
