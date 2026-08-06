"""Rust backend: simulation.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._categorical import sample_categorical_rows
from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "numpy"


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

    if rust_enabled():
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

        cat_probs = -np.diff(
            np.column_stack([cum_probs, np.zeros((n_persons, 1))]), axis=1
        )
        cat_probs = np.maximum(cat_probs, 0)
        totals = cat_probs.sum(axis=1, keepdims=True)
        cat_probs = np.divide(
            cat_probs,
            totals,
            out=np.zeros_like(cat_probs),
            where=totals > 0,
        )

        responses[:, i] = sample_categorical_rows(cat_probs, rng)

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

    if rust_enabled():
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
        step_logits = discrimination[i] * (theta[:, :1] - thresholds[i][None, :])
        logits = np.zeros((n_persons, n_categories))
        logits[:, 1:] = np.cumsum(step_logits, axis=1)
        logits -= np.max(logits, axis=1, keepdims=True)
        numerators = np.exp(logits)

        cat_probs = numerators / numerators.sum(axis=1, keepdims=True)

        responses[:, i] = sample_categorical_rows(cat_probs, rng)

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

    if rust_enabled():
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
