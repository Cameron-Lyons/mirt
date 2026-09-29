"""Shared, bounded NumPy evaluation of the Gaussian variational logistic bound."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

_MAX_WORKING_ELEMENTS = 1_000_000


def variational_elbo(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    lambda_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> float:
    """Sum the logistic bound and Gaussian prior/entropy terms in row blocks.

    Scratch space accounts for item and factor counts, with at least one full
    respondent per block. Inputs are never modified. Missing responses contribute
    no likelihood, but every respondent contributes a Gaussian KL term.
    """
    n_persons, n_items = responses.shape
    n_factors = loadings.shape[1]
    prior_precision = np.linalg.solve(prior_cov, np.eye(n_factors))
    log_det_prior = float(np.linalg.slogdet(prior_cov)[1])
    entries_per_person = (
        n_items * (n_factors + 8) + 3 * n_factors**2 + 3 * n_factors + 8
    )
    block_size = max(1, _MAX_WORKING_ELEMENTS // max(1, entries_per_person))

    expected_log_likelihood = 0.0
    divergence = 0.0
    for start in range(0, n_persons, block_size):
        stop = min(start + block_size, n_persons)
        data, block_xi = responses[start:stop], xi[start:stop]
        block_mu, block_sigma = mu[start:stop], sigma[start:stop]

        lam = lambda_function(block_xi)
        eta_mean = block_mu @ loadings.T
        eta_mean += intercepts
        eta_second = np.einsum(
            "jf,ifg,jg->ij", loadings, block_sigma, loadings, optimize=True
        )
        eta_second += eta_mean**2

        terms = np.negative(block_xi)
        np.logaddexp(0.0, terms, out=terms)
        np.negative(terms, out=terms)
        eta_mean *= data - 0.5
        terms += eta_mean
        # Reuse the predictor buffer after adding its linear contribution.
        np.multiply(block_xi, 0.5, out=eta_mean)
        terms -= eta_mean
        np.square(block_xi, out=eta_mean)
        eta_second -= eta_mean
        eta_second *= lam
        terms -= eta_second
        # Zero missing entries in owned storage so the reduction can use NumPy's
        # pairwise summation without allocating a gathered observation array.
        np.copyto(terms, 0.0, where=~(data >= 0))
        expected_log_likelihood += float(np.sum(terms))
        del lam, eta_mean, eta_second, terms

        diff = block_mu - prior_mean
        kl = np.einsum("if,fg,ig->i", diff, prior_precision, diff, optimize=True)
        kl *= 0.5
        kl += 0.5 * np.einsum("fg,igf->i", prior_precision, block_sigma, optimize=True)
        kl += 0.5 * (log_det_prior - np.linalg.slogdet(block_sigma)[1])
        kl -= 0.5 * n_factors
        divergence += float(np.sum(kl))

    return expected_log_likelihood - divergence
