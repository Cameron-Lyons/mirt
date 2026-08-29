"""Rust backend: explanatory IRT marginal likelihoods.

Fallback mode: numpy. The fallback streams over quadrature points so working
memory does not grow with the number of points.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"


def _prepare_marginal_likelihood_inputs(
    responses: ArrayLike,
    prior_means: ArrayLike,
    residual_std: float,
    quad_nodes: ArrayLike,
    quad_weights: ArrayLike,
    discrimination: ArrayLike,
    difficulty: ArrayLike,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate and normalize one explanatory marginal-likelihood batch."""
    response_values = np.asarray(responses)
    if response_values.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")
    if response_values.ndim != 2:
        raise ValueError("responses must be two-dimensional")
    n_persons, n_items = response_values.shape
    if not np.all(np.isfinite(response_values)):
        raise ValueError("responses must contain only finite values")
    observed = response_values >= 0
    if np.any(observed & (response_values != 0) & (response_values != 1)):
        raise ValueError("observed responses must contain only 0 and 1")

    mean_values = np.asarray(prior_means, dtype=np.float64)
    node_values = np.asarray(quad_nodes, dtype=np.float64)
    weight_values = np.asarray(quad_weights, dtype=np.float64)
    discrimination_values = np.asarray(discrimination, dtype=np.float64)
    difficulty_values = np.asarray(difficulty, dtype=np.float64)
    if mean_values.shape != (n_persons,):
        raise ValueError(f"prior_means must have shape ({n_persons},)")
    if node_values.ndim != 1 or node_values.size == 0:
        raise ValueError("quad_nodes must be a non-empty one-dimensional array")
    if weight_values.shape != node_values.shape:
        raise ValueError("quad_weights must have the same shape as quad_nodes")
    if discrimination_values.shape != (n_items,):
        raise ValueError(f"discrimination must have shape ({n_items},)")
    if difficulty_values.shape != (n_items,):
        raise ValueError(f"difficulty must have shape ({n_items},)")

    if not np.all(np.isfinite(mean_values)):
        raise ValueError("prior_means must contain only finite values")
    if not np.all(np.isfinite(node_values)):
        raise ValueError("quad_nodes must contain only finite values")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
        raise ValueError("quad_weights must contain finite positive values")
    if not np.all(np.isfinite(discrimination_values)):
        raise ValueError("discrimination must contain only finite values")
    if not np.all(np.isfinite(difficulty_values)):
        raise ValueError("difficulty must contain only finite values")
    if (
        isinstance(residual_std, (bool, np.bool_))
        or not isinstance(residual_std, Real)
        or not np.isfinite(residual_std)
        or residual_std <= 0.0
    ):
        raise ValueError("residual_std must be finite and positive")

    responses_i32 = _ensure_i32(
        response_values
        if response_values.dtype.kind in "biu"
        else np.where(observed, response_values, -1.0)
    )
    means_f64 = _ensure_f64(mean_values)
    nodes_f64 = _ensure_f64(node_values)
    weights_f64 = _ensure_f64(weight_values)
    discrimination_f64 = _ensure_f64(discrimination_values)
    difficulty_f64 = _ensure_f64(difficulty_values)
    assert responses_i32 is not None
    assert means_f64 is not None
    assert nodes_f64 is not None
    assert weights_f64 is not None
    assert discrimination_f64 is not None
    assert difficulty_f64 is not None
    if np.any(responses_i32 < -1):
        responses_i32 = responses_i32.copy()
        responses_i32[responses_i32 < 0] = -1
    return (
        responses_i32,
        means_f64,
        float(residual_std),
        nodes_f64,
        weights_f64,
        discrimination_f64,
        difficulty_f64,
    )


def compute_explanatory_marginal_log_likelihood(
    responses: ArrayLike,
    prior_means: ArrayLike,
    residual_std: float,
    quad_nodes: ArrayLike,
    quad_weights: ArrayLike,
    discrimination: ArrayLike,
    difficulty: ArrayLike,
) -> NDArray[np.float64]:
    """Integrate binary response-pattern likelihoods over residual ability."""
    (
        responses_i32,
        means_f64,
        residual_std_value,
        nodes_f64,
        weights_f64,
        discrimination_f64,
        difficulty_f64,
    ) = _prepare_marginal_likelihood_inputs(
        responses,
        prior_means,
        residual_std,
        quad_nodes,
        quad_weights,
        discrimination,
        difficulty,
    )

    if rust_enabled():
        return np.asarray(
            mirt_rs.explanatory_marginal_log_likelihood(
                responses_i32,
                means_f64,
                residual_std_value,
                nodes_f64,
                weights_f64,
                discrimination_f64,
                difficulty_f64,
            ),
            dtype=np.float64,
        )

    correct = (responses_i32 == 1).astype(np.float64)
    incorrect = (responses_i32 == 0).astype(np.float64)
    integrated = np.full(responses_i32.shape[0], -np.inf, dtype=np.float64)
    for node, weight in zip(nodes_f64, weights_f64, strict=True):
        theta = means_f64 + residual_std_value * node
        probabilities = np.asarray(
            sigmoid(
                discrimination_f64[None, :] * (theta[:, None] - difficulty_f64[None, :])
            ),
            dtype=np.float64,
        )
        np.clip(
            probabilities,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
            out=probabilities,
        )
        log_probabilities = np.log(probabilities)
        np.log1p(-probabilities, out=probabilities)
        conditional = np.einsum(
            "ij,ij->i",
            correct,
            log_probabilities,
            optimize=True,
        ) + np.einsum(
            "ij,ij->i",
            incorrect,
            probabilities,
            optimize=True,
        )
        np.logaddexp(integrated, np.log(weight) + conditional, out=integrated)
    return integrated
