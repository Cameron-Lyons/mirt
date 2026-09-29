"""Shared NumPy updates for Gaussian variational item-response estimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

_MAX_WORKING_ELEMENTS = 1_000_000


def jaakkola_lambda(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate tanh(xi / 2) / (4 * xi), with its 1/8 limit near zero."""
    xi = np.abs(xi)
    result = np.empty_like(xi)
    small = xi < 1e-6
    result[small] = 0.125
    large = ~small
    xi_large = xi[large]
    result[large] = np.tanh(xi_large / 2) / (4 * xi_large)
    return result


def variational_e_step(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_precision: NDArray[np.float64],
    xi: NDArray[np.float64],
    n_inner_iter: int,
    *,
    xi_floor: float = 0.0,
    lambda_function: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ] = jaakkola_lambda,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Update independent respondent blocks without changing input arrays.

    The returned means, covariances, and local bounds grow with sample size.
    Scratch space is sized by item and factor counts, with at least one full
    respondent per block. All inner iterations finish within a block because
    respondents interact only through the fixed item parameters and prior.
    """
    n_persons, n_items = responses.shape
    n_factors = loadings.shape[1]
    prior_natural_mean = prior_precision @ prior_mean

    def update(data, block_xi):
        valid = data >= 0
        for _ in range(n_inner_iter):
            lam = lambda_function(block_xi)
            weights = np.where(valid, 2.0 * lam, 0.0)
            precision = prior_precision + np.einsum(
                "ij,jf,jg->ifg", weights, loadings, loadings, optimize=True
            )
            block_sigma = np.linalg.inv(precision)
            coefficients = np.where(valid, data - 0.5 - 2.0 * lam * intercepts, 0.0)
            natural_mean = coefficients @ loadings + prior_natural_mean
            block_mu = np.einsum("ifg,ig->if", block_sigma, natural_mean, optimize=True)
            eta_mean = block_mu @ loadings.T + intercepts
            eta_variance = np.einsum(
                "jf,ifg,jg->ij", loadings, block_sigma, loadings, optimize=True
            )
            candidate_xi = np.sqrt(np.maximum(eta_variance + eta_mean**2, xi_floor))
            block_xi = np.where(valid, candidate_xi, block_xi)
        return block_mu, block_sigma, block_xi

    # Include response-wide buffers, covariance matrices, and possible einsum
    # intermediates when choosing the block size. Small samples return the block
    # outputs directly, without allocating and copying a second result set.
    entries_per_person = n_items * (n_factors + 10) + 4 * n_factors**2
    block_size = max(1, _MAX_WORKING_ELEMENTS // max(1, entries_per_person))
    if n_persons <= block_size:
        return update(responses, xi)

    mu = np.empty((n_persons, n_factors), dtype=np.float64)
    sigma = np.empty((n_persons, n_factors, n_factors), dtype=np.float64)
    updated_xi = np.empty((n_persons, n_items), dtype=np.float64)
    for start in range(0, n_persons, block_size):
        stop = min(start + block_size, n_persons)
        mu[start:stop], sigma[start:stop], updated_xi[start:stop] = update(
            responses[start:stop], xi[start:stop]
        )

    return mu, sigma, updated_xi
