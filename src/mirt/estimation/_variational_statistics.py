"""Bounded sufficient statistics for Gaussian variational item updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_MAX_WORKING_ELEMENTS = 1_000_000


@dataclass
class VariationalItemStatistics:
    """Item-level reductions, independent of the loading penalty or solver."""

    curvature: NDArray[np.float64] | None
    score: NDArray[np.float64] | None
    weighted_mean: NDArray[np.float64]
    weight_sum: NDArray[np.float64]
    response_sum: NDArray[np.float64]
    observed: NDArray[np.bool_]


def variational_item_statistics(
    responses: NDArray[np.int_],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    lambda_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    estimate_loadings: bool = True,
) -> VariationalItemStatistics:
    """Accumulate item curvature and scores without retaining respondent arrays.

    A loading update uses sum(2 * lambda * E[theta theta']) and
    sum((y - 1/2 - 2 * lambda * intercept) * E[theta]). Intercepts then use
    sum(y - 1/2) - loading' * sum(2 * lambda * E[theta]), divided by sum(2 * lambda).
    Missing observations contribute zero. Fixed loadings need only the intercept
    statistics, so their path avoids covariance calculations and loading scores.
    """
    n_persons, n_items = responses.shape
    n_factors = mu.shape[1]
    curvature = np.zeros((n_items, n_factors, n_factors)) if estimate_loadings else None
    score = np.zeros((n_items, n_factors)) if estimate_loadings else None
    weighted_mean = np.zeros((n_items, n_factors))
    weight_sum = np.zeros(n_items)
    response_sum = np.zeros(n_items)
    observed = np.zeros(n_items, dtype=np.bool_)
    entries_per_person = 8 * n_items + 2 * n_factors**2 + 2 * n_factors + 4
    block_size = max(1, _MAX_WORKING_ELEMENTS // max(1, entries_per_person))

    for start in range(0, n_persons, block_size):
        stop = min(start + block_size, n_persons)
        data, block_mu = responses[start:stop], mu[start:stop]
        valid = data >= 0
        observed |= np.any(valid, axis=0)
        weights = 2.0 * lambda_function(xi[start:stop])
        np.copyto(weights, 0.0, where=~valid)
        coefficients = np.where(valid, data - 0.5, 0.0)
        response_sum += np.sum(coefficients, axis=0)
        weight_sum += np.sum(weights, axis=0)
        weighted_mean += weights.T @ block_mu

        if curvature is not None and score is not None:
            second_moments = np.einsum("if,ig->ifg", block_mu, block_mu)
            second_moments += sigma[start:stop]
            curvature += np.einsum(
                "ij,ifg->jfg", weights, second_moments, optimize=True
            )
            del second_moments
            # The raw weights are no longer needed. Reuse them for the loading
            # score, avoiding operations on intercepts of missing observations.
            np.multiply(weights, intercepts, out=weights, where=valid)
            coefficients -= weights
            score += coefficients.T @ block_mu
        del weights, coefficients

    return VariationalItemStatistics(
        curvature, score, weighted_mean, weight_sum, response_sum, observed
    )
