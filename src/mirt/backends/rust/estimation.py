"""Rust backend: estimation.

Fallback mode: mixed. em_fit_2pl / gibbs_sample_2pl / mhrm_fit_2pl / bootstrap_fit_2pl are required; em_iteration_2pl / em_iteration_3pl are optional.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
    rust_enabled,
    rust_required,
)

FALLBACK_MODE = "mixed"


def em_fit_2pl(
    responses: NDArray[np.int_],
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int, bool]:
    """Fit 2PL model using EM algorithm in Rust.

    Returns
    -------
    tuple
        (discrimination, difficulty, log_likelihood, n_iterations, converged)
    """
    if rust_enabled():
        return mirt_rs.em_fit_2pl(
            _ensure_i32(responses),
            n_quadpts,
            max_iter,
            tol,
        )

    rust_required("em_fit_2pl")


def gibbs_sample_2pl(
    responses: NDArray[np.int_],
    n_iter: int = 5000,
    burnin: int = 1000,
    thin: int = 1,
    seed: int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Run Gibbs sampler for 2PL model in Rust.

    Returns
    -------
    tuple
        (disc_chain, diff_chain, theta_chain, ll_chain)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if rust_enabled():
        return mirt_rs.gibbs_sample_2pl(
            _ensure_i32(responses),
            n_iter,
            burnin,
            thin,
            int(seed),
        )

    rust_required("gibbs_sample_2pl")


def mhrm_fit_2pl(
    responses: NDArray[np.int_],
    n_cycles: int = 2000,
    burnin: int = 500,
    proposal_sd: float = 0.5,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit 2PL model using MHRM algorithm in Rust.

    Returns
    -------
    tuple
        (discrimination, difficulty, log_likelihood)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if rust_enabled():
        return mirt_rs.mhrm_fit_2pl(
            _ensure_i32(responses),
            n_cycles,
            burnin,
            proposal_sd,
            int(seed),
        )

    rust_required("mhrm_fit_2pl")


def bootstrap_fit_2pl(
    responses: NDArray[np.int_],
    n_bootstrap: int = 100,
    n_quadpts: int = 21,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run parallel bootstrap for 2PL model in Rust.

    Returns
    -------
    tuple
        (disc_samples, diff_samples) - arrays of shape (n_bootstrap, n_items)
    """
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if rust_enabled():
        return mirt_rs.bootstrap_fit_2pl(
            _ensure_i32(responses),
            n_bootstrap,
            n_quadpts,
            max_iter,
            tol,
            int(seed),
        )

    rust_required("bootstrap_fit_2pl")


def em_iteration_2pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    max_m_iter: int = 10,
    m_tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    damping: float = 0.5,
    regularization: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float] | None:
    """Single EM iteration for 2PL model (batched E+M step).

    Performs both E-step and M-step in a single call to reduce FFI overhead.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    discrimination : NDArray
        Current discrimination parameters (n_items,)
    difficulty : NDArray
        Current difficulty parameters (n_items,)
    prior_mean : float
        Prior mean for theta distribution
    prior_var : float
        Prior variance for theta distribution
    max_m_iter : int
        Maximum Newton-Raphson iterations in M-step
    m_tol : float
        Convergence tolerance for M-step
    disc_bounds : tuple
        (min, max) bounds for discrimination
    diff_bounds : tuple
        (min, max) bounds for difficulty
    damping : float
        Damping factor for parameter updates
    regularization : float
        Regularization strength

    Returns
    -------
    tuple or None
        (new_discrimination, new_difficulty, posterior_weights, log_likelihood)
        Returns None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.em_iteration_2pl(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            float(prior_mean),
            float(prior_var),
            max_m_iter,
            m_tol,
            disc_bounds,
            diff_bounds,
            damping,
            regularization,
        )

    return None


def em_iteration_3pl(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    max_m_iter: int = 10,
    m_tol: float = 1e-4,
    disc_bounds: tuple[float, float] = (0.1, 5.0),
    diff_bounds: tuple[float, float] = (-6.0, 6.0),
    guess_bounds: tuple[float, float] = (0.0, 0.35),
    damping_ab: float = 0.5,
    damping_c: float = 0.3,
    regularization: float = 0.01,
    regularization_c: float = 0.1,
) -> (
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]
    | None
):
    """Single EM iteration for 3PL model (batched E+M step).

    Performs both E-step and M-step in a single call to reduce FFI overhead.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    quad_weights : NDArray
        Quadrature weights (n_quad,)
    discrimination : NDArray
        Current discrimination parameters (n_items,)
    difficulty : NDArray
        Current difficulty parameters (n_items,)
    guessing : NDArray
        Current guessing parameters (n_items,)
    prior_mean : float
        Prior mean for theta distribution
    prior_var : float
        Prior variance for theta distribution
    max_m_iter : int
        Maximum Newton-Raphson iterations in M-step
    m_tol : float
        Convergence tolerance for M-step
    disc_bounds : tuple
        (min, max) bounds for discrimination
    diff_bounds : tuple
        (min, max) bounds for difficulty
    guess_bounds : tuple
        (min, max) bounds for guessing
    damping_ab : float
        Damping factor for a and b updates
    damping_c : float
        Damping factor for c updates
    regularization : float
        Regularization strength for a and b
    regularization_c : float
        Regularization strength for c

    Returns
    -------
    tuple or None
        (new_discrimination, new_difficulty, new_guessing, posterior_weights, log_likelihood)
        Returns None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.em_iteration_3pl(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(guessing),
            float(prior_mean),
            float(prior_var),
            max_m_iter,
            m_tol,
            disc_bounds,
            diff_bounds,
            guess_bounds,
            damping_ab,
            damping_c,
            regularization,
            regularization_c,
        )

    return None
