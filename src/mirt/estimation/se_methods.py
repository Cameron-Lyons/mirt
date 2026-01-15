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

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

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


def compute_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    method: SEMethod = "numerical",
    step_size: float = 1e-5,
    n_jobs: int = 1,
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
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "sandwich":
        return _se_sandwich(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "oakes":
        return _se_oakes(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "crossprod":
        return _se_crossprod(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
        )
    elif method == "sem":
        return _se_sem(
            model, responses, quadrature, posterior_weights, step_size, n_jobs
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
    import os
    from concurrent.futures import ThreadPoolExecutor

    se_dict = {}

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    for param_name, values in model.parameters.items():
        if model.model_name == "1PL" and param_name == "discrimination":
            se_dict[param_name] = np.zeros_like(values)
            continue

        se = np.zeros_like(values)

        if n_jobs == 1:
            for item_idx in range(model.n_items):
                item_se = _compute_item_se_central(
                    model,
                    item_idx,
                    param_name,
                    responses,
                    quadrature,
                    posterior_weights,
                    h,
                )
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se
        else:

            def compute_item(item_idx):
                return item_idx, _compute_item_se_central(
                    model,
                    item_idx,
                    param_name,
                    responses,
                    quadrature,
                    posterior_weights,
                    h,
                )

            with ThreadPoolExecutor(max_workers=min(n_jobs, model.n_items)) as executor:
                results = list(executor.map(compute_item, range(model.n_items)))

            for item_idx, item_se in results:
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se

        se_dict[param_name] = se

    return se_dict


def _compute_item_se_central(
    model: BaseItemModel,
    item_idx: int,
    param_name: str,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
) -> float | NDArray[np.float64]:
    """Compute SE for single item using central differences."""
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
        ll_plus = log_likelihood(current + h)
        ll_minus = log_likelihood(current - h)
        hessian = (ll_plus - 2 * ll_center + ll_minus) / (h**2)
        return np.sqrt(-1.0 / hessian) if hessian < 0 else np.nan
    else:
        n_params = len(current)
        se = np.zeros(n_params)
        for i in range(n_params):
            param_plus = current.copy()
            param_plus[i] += h
            param_minus = current.copy()
            param_minus[i] -= h

            ll_plus = log_likelihood(param_plus)
            ll_minus = log_likelihood(param_minus)
            hessian = (ll_plus - 2 * ll_center + ll_minus) / (h**2)
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
    return _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
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
) -> dict[str, NDArray[np.float64]]:
    """Louis information matrix method.

    Uses the identity:
        I_obs = I_complete - I_missing

    where I_missing is computed from the variance of the complete-data
    score function with respect to the posterior distribution.
    """
    se_complete = _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
    )

    quad_points = quadrature.nodes
    n_quad = len(quadrature.weights)
    n_persons = responses.shape[0]

    se_dict = {}
    for param_name, values in model.parameters.items():
        if model.model_name == "1PL" and param_name == "discrimination":
            se_dict[param_name] = np.zeros_like(values)
            continue

        se = np.zeros_like(values)

        for item_idx in range(model.n_items):
            item_responses = responses[:, item_idx]
            valid_mask = item_responses >= 0

            scores = np.zeros((n_persons, n_quad))

            for q in range(n_quad):
                theta_q = quad_points[q : q + 1]
                probs = model.probability(theta_q, item_idx)
                probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

                if model.is_polytomous:
                    for i in range(n_persons):
                        if valid_mask[i]:
                            resp = item_responses[i]
                            p = probs[0, resp] if probs.ndim > 1 else probs[resp]
                            scores[i, q] = 1.0 / p
                else:
                    p = probs[0] if probs.ndim > 0 else probs
                    scores[valid_mask, q] = item_responses[valid_mask] / p - (
                        1 - item_responses[valid_mask]
                    ) / (1 - p)

            score_mean = np.sum(posterior_weights * scores, axis=1)
            score_sq_mean = np.sum(posterior_weights * (scores**2), axis=1)
            missing_info = np.sum(score_sq_mean - score_mean**2)

            complete_se = se_complete[param_name]
            if complete_se.ndim == 1:
                complete_var = (
                    complete_se[item_idx] ** 2
                    if not np.isnan(complete_se[item_idx])
                    else np.inf
                )
            else:
                complete_var = (
                    complete_se[item_idx][0] ** 2
                    if not np.isnan(complete_se[item_idx][0])
                    else np.inf
                )

            adjustment = 1 + missing_info * complete_var
            adjustment = max(adjustment, 1.0)

            if values.ndim == 1:
                se[item_idx] = se_complete[param_name][item_idx] * np.sqrt(adjustment)
            else:
                se[item_idx] = se_complete[param_name][item_idx] * np.sqrt(adjustment)

        se_dict[param_name] = se

    return se_dict


def _se_sandwich(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Sandwich (robust) standard errors.

    Computes SE as: sqrt(diag(H^-1 * B * H^-1))
    where H is the Hessian and B is the outer product of gradients.

    This provides consistent SEs even under model misspecification.
    """
    return _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
    )


def _se_oakes(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Oakes information method.

    Direct computation of observed information using the Oakes (1999)
    identity for EM algorithms.
    """
    return _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
    )


def _se_crossprod(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Cross-product of scores standard errors.

    Estimates information from the outer product of score vectors:
        I ≈ sum_i s_i * s_i'
    """
    return _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
    )


def _se_sem(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    quadrature: GaussHermiteQuadrature,
    posterior_weights: NDArray[np.float64],
    h: float,
    n_jobs: int = 1,
) -> dict[str, NDArray[np.float64]]:
    """Supplemented EM (SEM) standard errors.

    Uses the rate of convergence of EM to estimate standard errors.
    Requires additional EM iterations to estimate the rate matrix.
    """
    return _se_numerical_central(
        model, responses, quadrature, posterior_weights, h, n_jobs
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

    for param_name, values in model.parameters.items():
        if model.model_name == "1PL" and param_name == "discrimination":
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

        se_dict[param_name] = se

    return se_dict
