"""Rust backend: eapsum."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
    _ensure_f64,
)

def lord_wingersky_recursion(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute sum score distribution using Lord-Wingersky recursion in Rust.

    For dichotomous items, computes log P(sum_score = s | theta) for all s and theta.

    Parameters
    ----------
    theta : ndarray
        Quadrature points, shape (n_quad,)
    discrimination : ndarray
        Item discriminations, shape (n_items,)
    difficulty : ndarray
        Item difficulties, shape (n_items,)

    Returns
    -------
    ndarray or None
        Log probability distribution, shape (max_score + 1, n_quad).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.lord_wingersky_recursion(
            _ensure_f64(theta),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    return None

def lord_wingersky_polytomous(
    item_probs: NDArray[np.float64],
    max_score: int,
) -> NDArray[np.float64] | None:
    """Compute sum score distribution for polytomous items using Rust.

    Parameters
    ----------
    item_probs : ndarray
        Item probabilities, shape (n_items, n_quad, max_categories)
    max_score : int
        Maximum possible sum score

    Returns
    -------
    ndarray or None
        Log probability distribution, shape (max_score + 1, n_quad).
        Returns None if Rust backend not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.lord_wingersky_polytomous(
            _ensure_f64(item_probs),
            max_score,
        )

    return None

def eapsum_from_distribution(
    log_p_score_theta: NDArray[np.float64],
    log_prior: NDArray[np.float64],
    sum_scores: NDArray[np.int_],
    theta_points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Compute EAPsum scores from pre-computed distribution using Rust.

    Parameters
    ----------
    log_p_score_theta : ndarray
        Log P(score | theta), shape (max_score + 1, n_quad)
    log_prior : ndarray
        Log prior weights, shape (n_quad,)
    sum_scores : ndarray
        Observed sum scores, shape (n_persons,)
    theta_points : ndarray
        Quadrature points, shape (n_quad,)

    Returns
    -------
    tuple or None
        (theta_estimates, standard_errors) or None if Rust not available.
    """
    if RUST_AVAILABLE:
        return mirt_rs.eapsum_from_distribution(
            log_p_score_theta.astype(np.float64),
            log_prior.astype(np.float64),
            sum_scores.astype(np.int32),
            theta_points.astype(np.float64),
        )

    return None
