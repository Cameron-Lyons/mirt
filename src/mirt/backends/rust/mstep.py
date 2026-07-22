"""Rust backend: mstep."""

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


def m_step_dichotomous_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    max_iter: int = 10,
    tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    damping: float = 0.5,
    regularization: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parallel M-step optimization for dichotomous items.

    Uses Newton-Raphson optimization for each item in parallel using Rayon.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    posterior_weights : NDArray
        Posterior weights from E-step (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Initial discrimination parameters (n_items,)
    difficulty : NDArray
        Initial difficulty parameters (n_items,)
    max_iter : int
        Maximum Newton-Raphson iterations per item
    tol : float
        Convergence tolerance
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max)
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max)
    damping : float
        Damping factor for Newton-Raphson updates
    regularization : float
        Regularization for Hessian diagonal

    Returns
    -------
    tuple
        (new_discrimination, new_difficulty)
    """
    if RUST_AVAILABLE:
        qp = _ensure_f64(quad_points)
        disc = _ensure_f64(discrimination)
        diff = _ensure_f64(difficulty)
        return mirt_rs.m_step_dichotomous_parallel(
            _ensure_i32(responses),
            _ensure_f64(posterior_weights),
            qp.ravel() if qp is not None else quad_points.astype(np.float64).ravel(),
            disc.ravel()
            if disc is not None
            else discrimination.astype(np.float64).ravel(),
            diff.ravel() if diff is not None else difficulty.astype(np.float64).ravel(),
            max_iter,
            tol,
            disc_bounds,
            diff_bounds,
            damping,
            regularization,
        )

    from scipy.optimize import minimize

    n_items = responses.shape[1]
    new_disc = np.zeros(n_items)
    new_diff = np.zeros(n_items)

    for j in range(n_items):
        item_responses = responses[:, j]
        valid_mask = item_responses >= 0

        r_k = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )
        n_k = np.sum(posterior_weights[valid_mask], axis=0)

        def neg_ll(params):
            a, b = params
            z = a * (quad_points - b)
            p = sigmoid(z)
            p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
            ll = np.sum(r_k * np.log(p) + (n_k - r_k) * np.log(1 - p))
            return -ll

        result = minimize(
            neg_ll,
            x0=[discrimination[j], difficulty[j]],
            method="L-BFGS-B",
            bounds=[disc_bounds, diff_bounds],
        )
        new_disc[j], new_diff[j] = result.x

    return new_disc, new_diff
