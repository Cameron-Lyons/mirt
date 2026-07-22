"""Rust backend: cat."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
)
from mirt.constants import PROB_EPSILON


def cat_compute_item_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute Fisher information for all items at a given theta.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters, shape (n_items,).
    difficulty : ndarray
        Item difficulty parameters, shape (n_items,).

    Returns
    -------
    ndarray or None
        Fisher information for each item, shape (n_items,).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.cat_compute_item_info(
            float(theta),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    z = discrimination * (theta - difficulty)
    p = sigmoid(z)
    q = 1.0 - p
    return (discrimination**2) * p * q


def cat_select_max_info(
    theta: float,
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    available_mask: NDArray[np.bool_],
) -> int:
    """Select item with maximum Fisher information.

    Parameters
    ----------
    theta : float
        Current ability estimate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    available_mask : ndarray
        Boolean mask of available items.

    Returns
    -------
    int
        Index of selected item, or -1 if no items available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.cat_select_max_info(
            float(theta),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            available_mask.astype(np.bool_),
        )

    info = cat_compute_item_info(theta, discrimination, difficulty)
    info = np.where(available_mask, info, -np.inf)
    return int(np.argmax(info))


def cat_eap_update(
    administered_items: NDArray[np.int32],
    responses: NDArray[np.int32],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Incremental EAP update after responses.

    Parameters
    ----------
    administered_items : ndarray
        Indices of administered items.
    responses : ndarray
        Responses to administered items.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points.
    quad_weights : ndarray
        Quadrature weights.

    Returns
    -------
    tuple[float, float]
        (theta_estimate, standard_error).
    """
    if RUST_AVAILABLE:
        theta, se = mirt_rs.cat_eap_update(
            _ensure_i32(administered_items),
            _ensure_i32(responses),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
        )
        return float(theta[0]), float(se[0])

    n_quad = len(quad_points)

    log_likes = np.zeros(n_quad)
    for q in range(n_quad):
        theta_q = quad_points[q]
        ll = 0.0
        for i, item_idx in enumerate(administered_items):
            j = int(item_idx)
            r = responses[i]
            if r >= 0:
                z = discrimination[j] * (theta_q - difficulty[j])
                p = sigmoid(z)
                p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                if r == 1:
                    ll += np.log(p)
                else:
                    ll += np.log(1 - p)
        log_likes[q] = ll

    log_prior = -0.5 * quad_points**2 - 0.5 * np.log(2 * np.pi)
    log_posterior = log_likes + log_prior + np.log(quad_weights + 1e-300)

    log_norm = np.max(log_posterior) + np.log(
        np.sum(np.exp(log_posterior - np.max(log_posterior)))
    )
    posterior = np.exp(log_posterior - log_norm)

    theta_eap = np.sum(posterior * quad_points)
    variance = np.sum(posterior * (quad_points - theta_eap) ** 2)
    se = np.sqrt(variance)

    return float(theta_eap), float(se)


def cat_simulate_batch(
    true_thetas: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    se_threshold: float,
    max_items: int,
    min_items: int,
    n_replications: int,
    seed: int | None = None,
) -> (
    tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]
    ]
    | None
):
    """Run batch CAT simulations in parallel using Rust.

    Parameters
    ----------
    true_thetas : ndarray
        True ability values to simulate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points for EAP scoring.
    quad_weights : ndarray
        Quadrature weights.
    se_threshold : float
        SE stopping threshold.
    max_items : int
        Maximum items per test.
    min_items : int
        Minimum items before stopping.
    n_replications : int
        Number of replications per theta.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple or None
        (theta_estimates, se_estimates, n_items, true_thetas_expanded).
        Returns None if Rust backend not available.
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.cat_simulate_batch(
            _ensure_f64(true_thetas),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            float(se_threshold),
            int(max_items),
            int(min_items),
            int(n_replications),
            int(seed),
        )

    return None


def cat_conditional_mse(
    eval_thetas: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    se_threshold: float,
    max_items: int,
    min_items: int,
    n_replications: int,
    seed: int | None = None,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]
    | None
):
    """Compute conditional MSE at specified theta values using Rust.

    Parameters
    ----------
    eval_thetas : ndarray
        Theta values to evaluate.
    discrimination : ndarray
        Item discrimination parameters.
    difficulty : ndarray
        Item difficulty parameters.
    quad_points : ndarray
        Quadrature points for EAP scoring.
    quad_weights : ndarray
        Quadrature weights.
    se_threshold : float
        SE stopping threshold.
    max_items : int
        Maximum items per test.
    min_items : int
        Minimum items before stopping.
    n_replications : int
        Number of replications per theta.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple or None
        (thetas, biases, mses, avg_items).
        Returns None if Rust backend not available.
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.cat_conditional_mse(
            _ensure_f64(eval_thetas),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            float(se_threshold),
            int(max_items),
            int(min_items),
            int(n_replications),
            int(seed),
        )

    return None
