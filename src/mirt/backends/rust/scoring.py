"""Rust backend: scoring."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON

from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
    _ensure_f64,
    _ensure_i32,
)

def compute_eap_scores(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    quad_weights: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute EAP scores with standard errors."""
    if RUST_AVAILABLE:
        return mirt_rs.compute_eap_scores(
            _ensure_i32(responses),
            _ensure_f64(quad_points),
            _ensure_f64(quad_weights),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    n_persons = responses.shape[0]
    n_quad = len(quad_points)

    log_weights = np.log(quad_weights + 1e-300)
    theta = np.zeros(n_persons)
    se = np.zeros(n_persons)

    for i in range(n_persons):
        log_likes = np.zeros(n_quad)
        for q in range(n_quad):
            ll = 0.0
            t = quad_points[q]
            for j in range(responses.shape[1]):
                if responses[i, j] >= 0:
                    z = discrimination[j] * (t - difficulty[j])
                    p = sigmoid(z)
                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                    if responses[i, j] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
            log_likes[q] = ll

        log_posterior = log_likes + log_weights
        log_posterior = log_posterior - np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()

        theta[i] = np.sum(posterior * quad_points)
        se[i] = np.sqrt(np.sum(posterior * (quad_points - theta[i]) ** 2))

    return theta, se
