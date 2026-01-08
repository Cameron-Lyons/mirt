"""Standard error estimation methods for IRT models.

This module provides various methods for computing standard errors of
item parameter estimates:
- Observed information (Hessian-based)
- Expected information
- Oakes method (1999)
- Sandwich (robust) estimator
- SEM (Supplemented EM) method

References:
    Oakes, D. (1999). Direct calculation of the information matrix via the
        EM algorithm. Journal of the Royal Statistical Society: Series B,
        61(2), 479-482.

    Cai, L. (2008). SEM of item response data with the Metropolis-Hastings
        Robbins-Monro algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.models.base import BaseItemModel


def compute_observed_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute observed information matrix via numerical differentiation.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    posterior_weights : ndarray
        E-step posterior weights
    quadrature : GaussHermiteQuadrature
        Quadrature object used in estimation
    h : float
        Step size for numerical differentiation

    Returns
    -------
    ndarray
        Observed information matrix
    """

    params_flat, param_shapes = _flatten_parameters(model)
    n_params = len(params_flat)

    hessian = np.zeros((n_params, n_params))

    ll_center = _complete_data_log_likelihood(
        model, responses, posterior_weights, quadrature
    )

    for i in range(n_params):
        for j in range(i, n_params):
            if i == j:
                params_plus = params_flat.copy()
                params_plus[i] += h
                params_minus = params_flat.copy()
                params_minus[i] -= h

                _set_flat_parameters(model, params_plus, param_shapes)
                ll_plus = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_minus, param_shapes)
                ll_minus = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                hessian[i, i] = (ll_plus - 2 * ll_center + ll_minus) / (h**2)
            else:
                params_pp = params_flat.copy()
                params_pp[i] += h
                params_pp[j] += h

                params_pm = params_flat.copy()
                params_pm[i] += h
                params_pm[j] -= h

                params_mp = params_flat.copy()
                params_mp[i] -= h
                params_mp[j] += h

                params_mm = params_flat.copy()
                params_mm[i] -= h
                params_mm[j] -= h

                _set_flat_parameters(model, params_pp, param_shapes)
                ll_pp = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_pm, param_shapes)
                ll_pm = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_mp, param_shapes)
                ll_mp = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_mm, param_shapes)
                ll_mm = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                hessian[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * h**2)
                hessian[j, i] = hessian[i, j]

    _set_flat_parameters(model, params_flat, param_shapes)

    return -hessian


def compute_sandwich_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    survey_weights: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute sandwich (robust) standard errors.

    The sandwich estimator is robust to model misspecification and
    is particularly useful when using survey weights.

    Sandwich covariance: (J^-1) @ V @ (J^-1)

    where J is the expected information and V is the variance of
    the score function.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    posterior_weights : ndarray
        E-step posterior weights
    quadrature : GaussHermiteQuadrature
        Quadrature object used in estimation
    survey_weights : ndarray, optional
        Person-level survey weights

    Returns
    -------
    dict
        Dictionary mapping parameter names to standard error arrays
    """
    n_persons = responses.shape[0]

    if survey_weights is None:
        survey_weights = np.ones(n_persons)

    params_flat, param_shapes = _flatten_parameters(model)
    n_params = len(params_flat)

    scores = _compute_person_scores(
        model, responses, posterior_weights, quadrature, params_flat, param_shapes
    )

    weighted_scores = scores * survey_weights[:, None]

    J = compute_observed_information(model, responses, posterior_weights, quadrature)

    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        J_inv = np.linalg.pinv(J)

    V = np.zeros((n_params, n_params))
    for i in range(n_persons):
        V += np.outer(weighted_scores[i], weighted_scores[i])

    sandwich_cov = J_inv @ V @ J_inv

    variances = np.diag(sandwich_cov)
    se_flat = np.sqrt(np.maximum(variances, 0))

    return _unflatten_se(se_flat, param_shapes, model)


def compute_oakes_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
) -> dict[str, NDArray[np.float64]]:
    """Compute standard errors using Oakes (1999) method.

    The Oakes method provides a direct way to compute the observed
    information matrix without computing second derivatives of the
    complete-data log-likelihood.

    I_obs = I_comp - I_miss

    where I_comp is the complete-data information and I_miss is the
    missing information.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    posterior_weights : ndarray
        E-step posterior weights
    quadrature : GaussHermiteQuadrature
        Quadrature object
    h : float
        Step size for numerical derivatives

    Returns
    -------
    dict
        Dictionary mapping parameter names to standard error arrays

    References
    ----------
    Oakes, D. (1999). Direct calculation of the information matrix via
        the EM algorithm. Journal of the Royal Statistical Society B.
    """
    params_flat, param_shapes = _flatten_parameters(model)

    I_comp = _compute_complete_data_information(
        model, responses, posterior_weights, quadrature, h
    )

    I_miss = _compute_missing_information_oakes(
        model, responses, posterior_weights, quadrature, params_flat, param_shapes, h
    )

    I_obs = I_comp - I_miss

    try:
        I_obs_inv = np.linalg.inv(I_obs)
    except np.linalg.LinAlgError:
        I_obs_inv = np.linalg.pinv(I_obs)

    variances = np.diag(I_obs_inv)
    se_flat = np.sqrt(np.maximum(variances, 0))

    return _unflatten_se(se_flat, param_shapes, model)


def compute_sem_se(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    n_bootstrap: int = 50,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute standard errors using Supplemented EM (SEM) method.

    SEM augments the observed information with the rate of convergence
    of the EM algorithm to account for missing information.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix
    posterior_weights : ndarray
        E-step posterior weights
    quadrature : GaussHermiteQuadrature
        Quadrature object
    n_bootstrap : int
        Number of perturbations for rate matrix estimation
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary mapping parameter names to standard error arrays
    """
    rng = np.random.default_rng(seed)

    params_flat, param_shapes = _flatten_parameters(model)
    n_params = len(params_flat)

    I_comp = _compute_complete_data_information(
        model, responses, posterior_weights, quadrature
    )

    DM = np.zeros((n_params, n_params))
    perturbation_scale = 0.01

    for _ in range(n_bootstrap):
        perturbation = rng.normal(0, perturbation_scale, n_params)
        perturbed_params = params_flat + perturbation

        _set_flat_parameters(model, perturbed_params, param_shapes)

        new_params = _one_m_step_iteration(
            model, responses, posterior_weights, quadrature, param_shapes
        )

        direction = (new_params - perturbed_params) / perturbation_scale
        DM += np.outer(direction, perturbation / perturbation_scale)

    DM /= n_bootstrap

    _set_flat_parameters(model, params_flat, param_shapes)

    I_minus_DM = np.eye(n_params) - DM

    try:
        I_minus_DM_inv = np.linalg.inv(I_minus_DM)
    except np.linalg.LinAlgError:
        I_minus_DM_inv = np.linalg.pinv(I_minus_DM)

    I_obs = I_comp @ I_minus_DM_inv

    try:
        I_obs_inv = np.linalg.inv(I_obs)
    except np.linalg.LinAlgError:
        I_obs_inv = np.linalg.pinv(I_obs)

    variances = np.diag(I_obs_inv)
    se_flat = np.sqrt(np.maximum(variances, 0))

    return _unflatten_se(se_flat, param_shapes, model)


def _flatten_parameters(
    model: BaseItemModel,
) -> tuple[NDArray[np.float64], dict[str, tuple]]:
    """Flatten model parameters into a single vector."""
    params_list = []
    shapes = {}

    for name, values in model.parameters.items():
        shapes[name] = values.shape
        params_list.extend(values.ravel().tolist())

    return np.array(params_list), shapes


def _set_flat_parameters(
    model: BaseItemModel,
    params_flat: NDArray[np.float64],
    shapes: dict[str, tuple],
) -> None:
    """Set model parameters from flat vector."""
    idx = 0
    for name, shape in shapes.items():
        size = np.prod(shape)
        values = params_flat[idx : idx + size].reshape(shape)
        model._parameters[name] = values
        idx += size


def _unflatten_se(
    se_flat: NDArray[np.float64],
    shapes: dict[str, tuple],
    model: BaseItemModel,
) -> dict[str, NDArray[np.float64]]:
    """Convert flat SE vector to parameter dictionary."""
    se_dict = {}
    idx = 0

    for name, shape in shapes.items():
        size = np.prod(shape)
        se_dict[name] = se_flat[idx : idx + size].reshape(shape)
        idx += size

    return se_dict


def _complete_data_log_likelihood(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
) -> float:
    """Compute expected complete-data log-likelihood."""
    quad_points = quadrature.nodes
    n_items = model.n_items

    ll = 0.0
    for item_idx in range(n_items):
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        weighted_posterior = posterior_weights[valid_mask]

        if hasattr(model, "_n_categories"):
            n_categories = model._n_categories[item_idx]
            for c in range(n_categories):
                cat_mask = item_responses[valid_mask] == c
                r_kc = np.sum(weighted_posterior[cat_mask, :], axis=0)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs[:, c], 1e-10, 1 - 1e-10)
                ll += np.sum(r_kc * np.log(probs))
        else:
            r_k = np.sum(
                item_responses[valid_mask, None] * weighted_posterior,
                axis=0,
            )
            n_k = np.sum(weighted_posterior, axis=0)

            probs = model.probability(quad_points, item_idx)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            ll += np.sum(r_k * np.log(probs) + (n_k - r_k) * np.log(1 - probs))

    return ll


def _compute_person_scores(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    params_flat: NDArray[np.float64],
    param_shapes: dict[str, tuple],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute score contribution for each person."""
    n_persons = responses.shape[0]
    n_params = len(params_flat)

    scores = np.zeros((n_persons, n_params))

    for p in range(n_params):
        params_plus = params_flat.copy()
        params_plus[p] += h
        params_minus = params_flat.copy()
        params_minus[p] -= h

        _set_flat_parameters(model, params_plus, param_shapes)
        ll_plus = _person_log_likelihoods(
            model, responses, posterior_weights, quadrature
        )

        _set_flat_parameters(model, params_minus, param_shapes)
        ll_minus = _person_log_likelihoods(
            model, responses, posterior_weights, quadrature
        )

        scores[:, p] = (ll_plus - ll_minus) / (2 * h)

    _set_flat_parameters(model, params_flat, param_shapes)

    return scores


def _person_log_likelihoods(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
) -> NDArray[np.float64]:
    """Compute expected log-likelihood for each person."""
    quad_points = quadrature.nodes
    n_persons = responses.shape[0]
    n_items = model.n_items

    person_ll = np.zeros(n_persons)

    for item_idx in range(n_items):
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        probs_all = model.probability(quad_points, item_idx)

        for i in np.where(valid_mask)[0]:
            resp = item_responses[i]
            weights = posterior_weights[i]

            if hasattr(model, "_n_categories"):
                probs = np.clip(probs_all[:, resp], 1e-10, 1 - 1e-10)
            else:
                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                if resp == 0:
                    probs = 1 - probs

            person_ll[i] += np.sum(weights * np.log(probs))

    return person_ll


def _compute_complete_data_information(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute complete-data Fisher information matrix."""
    params_flat, param_shapes = _flatten_parameters(model)
    n_params = len(params_flat)

    info = np.zeros((n_params, n_params))

    ll_center = _complete_data_log_likelihood(
        model, responses, posterior_weights, quadrature
    )

    for i in range(n_params):
        for j in range(i, n_params):
            if i == j:
                params_plus = params_flat.copy()
                params_plus[i] += h
                params_minus = params_flat.copy()
                params_minus[i] -= h

                _set_flat_parameters(model, params_plus, param_shapes)
                ll_plus = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_minus, param_shapes)
                ll_minus = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                info[i, i] = -(ll_plus - 2 * ll_center + ll_minus) / (h**2)
            else:
                params_pp = params_flat.copy()
                params_pp[i] += h
                params_pp[j] += h

                params_mm = params_flat.copy()
                params_mm[i] -= h
                params_mm[j] -= h

                params_pm = params_flat.copy()
                params_pm[i] += h
                params_pm[j] -= h

                params_mp = params_flat.copy()
                params_mp[i] -= h
                params_mp[j] += h

                _set_flat_parameters(model, params_pp, param_shapes)
                ll_pp = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_mm, param_shapes)
                ll_mm = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_pm, param_shapes)
                ll_pm = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                _set_flat_parameters(model, params_mp, param_shapes)
                ll_mp = _complete_data_log_likelihood(
                    model, responses, posterior_weights, quadrature
                )

                info[i, j] = -(ll_pp - ll_pm - ll_mp + ll_mm) / (4 * h**2)
                info[j, i] = info[i, j]

    _set_flat_parameters(model, params_flat, param_shapes)

    return info


def _compute_missing_information_oakes(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    params_flat: NDArray[np.float64],
    param_shapes: dict[str, tuple],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute missing information using Oakes formula."""
    n_params = len(params_flat)
    I_miss = np.zeros((n_params, n_params))

    scores = _compute_person_scores(
        model, responses, posterior_weights, quadrature, params_flat, param_shapes, h
    )

    score_means = scores.mean(axis=0)
    for i in range(scores.shape[0]):
        centered = scores[i] - score_means
        I_miss += np.outer(centered, centered)

    I_miss /= scores.shape[0]

    return I_miss


def _one_m_step_iteration(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quadrature: GaussHermiteQuadrature,
    param_shapes: dict[str, tuple],
) -> NDArray[np.float64]:
    """Perform one M-step iteration and return updated parameters."""
    from mirt.estimation.em import EMEstimator

    estimator = EMEstimator(n_quadpts=len(quadrature.weights))
    estimator._quadrature = quadrature

    estimator._m_step(model, responses, posterior_weights)

    new_params, _ = _flatten_parameters(model)

    return new_params
