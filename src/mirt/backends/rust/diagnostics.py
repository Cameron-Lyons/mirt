"""Rust backend: diagnostics.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import _entry_chunk_size, mirt_rs, rust_enabled
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"

_POINTWISE_AGGREGATION_CODES = {
    "person": 0,
    "observation": 1,
    "observed": 2,
}


def _prepare_pointwise_loglik_inputs(
    responses: NDArray[np.int_],
    discrimination_chain: NDArray[np.float64],
    difficulty_chain: NDArray[np.float64],
    theta_chain: NDArray[np.float64],
    aggregation: str,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
]:
    """Validate the batched 2PL pointwise log-likelihood contract."""
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    if raw_responses.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")
    if not np.all(np.isfinite(raw_responses)):
        raise ValueError("responses must contain only finite values")
    if np.any(raw_responses != np.floor(raw_responses)):
        raise ValueError("responses must contain integer-valued category codes")
    observed = raw_responses >= 0
    if np.any(observed & (raw_responses > 1)):
        raise ValueError("observed responses must contain only 0 or 1")
    response_values = np.where(observed, raw_responses, -1).astype(
        np.int32,
        copy=False,
    )

    discrimination_values = np.asarray(discrimination_chain, dtype=np.float64)
    difficulty_values = np.asarray(difficulty_chain, dtype=np.float64)
    theta_values = np.asarray(theta_chain, dtype=np.float64)
    n_persons, n_items = response_values.shape
    if discrimination_values.ndim != 2:
        raise ValueError("discrimination_chain must have shape (n_samples, n_items)")
    if discrimination_values.shape[0] == 0:
        raise ValueError("parameter chains must contain at least one sample")
    n_samples = discrimination_values.shape[0]
    expected_item_shape = (n_samples, n_items)
    if discrimination_values.shape != expected_item_shape:
        raise ValueError(f"discrimination_chain must have shape {expected_item_shape}")
    if difficulty_values.shape != expected_item_shape:
        raise ValueError(f"difficulty_chain must have shape {expected_item_shape}")
    expected_theta_shape = (n_samples, n_persons)
    if theta_values.shape != expected_theta_shape:
        raise ValueError(f"theta_chain must have shape {expected_theta_shape}")
    if (
        not np.all(np.isfinite(discrimination_values))
        or not np.all(np.isfinite(difficulty_values))
        or not np.all(np.isfinite(theta_values))
    ):
        raise ValueError("chains must contain only finite values")

    try:
        aggregation_code = _POINTWISE_AGGREGATION_CODES[aggregation]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "aggregation must be 'person', 'observation', or 'observed'"
        ) from exc

    return (
        np.ascontiguousarray(response_values),
        discrimination_values
        if discrimination_values.flags.aligned
        else discrimination_values.copy(),
        difficulty_values
        if difficulty_values.flags.aligned
        else difficulty_values.copy(),
        theta_values if theta_values.flags.aligned else theta_values.copy(),
        aggregation_code,
    )


def _pointwise_loglik_2pl_numpy(
    responses: NDArray[np.int32],
    discrimination_chain: NDArray[np.float64],
    difficulty_chain: NDArray[np.float64],
    theta_chain: NDArray[np.float64],
    aggregation_code: int,
) -> NDArray[np.float64]:
    """Compute 2PL pointwise log likelihood one posterior sample at a time."""
    n_samples = discrimination_chain.shape[0]
    n_persons, n_items = responses.shape
    if aggregation_code == 0:
        output_width = n_persons
    elif aggregation_code == 1:
        output_width = n_persons * n_items
    else:
        output_width = int(np.count_nonzero(responses >= 0))
    result = np.empty((n_samples, output_width), dtype=np.float64)
    chunk_size = _entry_chunk_size(n_persons, n_items)
    for sample in range(n_samples):
        offset = 0
        for start in range(0, n_persons, chunk_size):
            stop = min(start + chunk_size, n_persons)
            response_chunk = responses[start:stop]
            probabilities = np.asarray(
                sigmoid(
                    discrimination_chain[sample, None, :]
                    * (
                        theta_chain[sample, start:stop, None]
                        - difficulty_chain[sample, None, :]
                    )
                ),
                dtype=np.float64,
            )
            np.clip(
                probabilities,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
                out=probabilities,
            )
            values = np.where(
                response_chunk == 1,
                np.log(probabilities),
                np.log1p(-probabilities),
            )
            observed = response_chunk >= 0
            values[~observed] = 0.0

            if aggregation_code == 0:
                output_values = np.sum(values, axis=1)
            elif aggregation_code == 1:
                output_values = values.ravel()
            else:
                output_values = values[observed]
            result[sample, offset : offset + output_values.size] = output_values
            offset += output_values.size

    return result


def compute_pointwise_loglik_2pl(
    responses: NDArray[np.int_],
    discrimination_chain: NDArray[np.float64],
    difficulty_chain: NDArray[np.float64],
    theta_chain: NDArray[np.float64],
    aggregation: Literal["person", "observation", "observed"] = "person",
) -> NDArray[np.float64]:
    """Compute batched 2PL posterior pointwise log likelihoods."""
    (
        response_values,
        discrimination_values,
        difficulty_values,
        theta_values,
        aggregation_code,
    ) = _prepare_pointwise_loglik_inputs(
        responses,
        discrimination_chain,
        difficulty_chain,
        theta_chain,
        aggregation,
    )
    if rust_enabled():
        return mirt_rs.compute_pointwise_loglik_2pl(
            response_values,
            discrimination_values,
            difficulty_values,
            theta_values,
            aggregation_code,
        )
    return _pointwise_loglik_2pl_numpy(
        response_values,
        discrimination_values,
        difficulty_values,
        theta_values,
        aggregation_code,
    )
