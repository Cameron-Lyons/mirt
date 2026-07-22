"""Rust backend: simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from mirt._core import sigmoid

from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
)

def simulate_grm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Graded Response Model."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_grm(
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = theta.shape[0]
    n_items = len(discrimination)
    n_categories = thresholds.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        cum_probs = np.ones((n_persons, n_categories))
        for k in range(n_categories - 1):
            z = discrimination[i] * (theta[:, 0] - thresholds[i, k])
            cum_probs[:, k + 1] = sigmoid(z)

        cat_probs = np.diff(
            np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1
        )
        cat_probs = np.maximum(cat_probs, 0)
        cat_probs = cat_probs / cat_probs.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses

def simulate_gpcm(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate responses from Generalized Partial Credit Model."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_gpcm(
            theta.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = theta.shape[0]
    n_items = len(discrimination)
    n_categories = thresholds.shape[1] + 1

    responses = np.zeros((n_persons, n_items), dtype=np.int_)

    for i in range(n_items):
        numerators = np.zeros((n_persons, n_categories))
        for k in range(n_categories):
            cumsum = 0.0
            for v in range(k):
                cumsum += discrimination[i] * (theta[:, 0] - thresholds[i, v])
            numerators[:, k] = np.exp(cumsum)

        cat_probs = numerators / numerators.sum(axis=1, keepdims=True)

        for p in range(n_persons):
            responses[p, i] = rng.choice(n_categories, p=cat_probs[p])

    return responses

def simulate_dichotomous(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64] | None = None,
    seed: int | None = None,
) -> NDArray[np.int_]:
    """Simulate dichotomous responses (2PL/3PL)."""
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**31)

    if RUST_AVAILABLE:
        return mirt_rs.simulate_dichotomous(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
            guessing.astype(np.float64) if guessing is not None else None,
            int(seed),
        )

    rng = np.random.default_rng(seed)
    n_persons = len(theta)
    n_items = len(discrimination)

    if guessing is None:
        guessing = np.zeros(n_items)

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    p_star = sigmoid(z)
    probs = guessing[None, :] + (1 - guessing[None, :]) * p_star

    u = rng.random((n_persons, n_items))
    return (u < probs).astype(np.int_)
