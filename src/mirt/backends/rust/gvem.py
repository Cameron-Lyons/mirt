"""Rust backend: gvem.

Fallback mode: optional. Returns None when Rust is unavailable; callers own Python paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import (
    mirt_rs,
    rust_enabled,
)

FALLBACK_MODE = "optional"


def gvem_e_step(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_cov_inv: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    n_inner_iter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    """GVEM E-step: update variational parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)
    prior_cov_inv : NDArray
        Inverse of prior covariance (n_factors, n_factors)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    n_inner_iter : int
        Number of inner iterations

    Returns
    -------
    tuple or None
        (mu, sigma, xi) updated arrays, or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.gvem_e_step(
            responses.astype(np.int32),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
            prior_cov_inv.astype(np.float64),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            n_inner_iter,
        )

    return None


def gvem_m_step(
    responses: NDArray[np.int_],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """GVEM M-step: update item parameters.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)

    Returns
    -------
    tuple or None
        (loadings, intercepts) updated arrays, or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.gvem_m_step(
            responses.astype(np.int32),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
        )

    return None


def gvem_compute_elbo(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
) -> float | None:
    """Compute ELBO for GVEM.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    loadings : NDArray
        Factor loadings (n_items, n_factors)
    intercepts : NDArray
        Item intercepts (n_items,)
    mu : NDArray
        Variational means (n_persons, n_factors)
    sigma : NDArray
        Variational covariances (n_persons, n_factors, n_factors)
    xi : NDArray
        Variational parameters (n_persons, n_items)
    prior_mean : NDArray
        Prior mean (n_factors,)
    prior_cov : NDArray
        Prior covariance (n_factors, n_factors)

    Returns
    -------
    float or None
        ELBO value, or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.gvem_compute_elbo(
            responses.astype(np.int32),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
            mu.astype(np.float64),
            sigma.astype(np.float64),
            xi.astype(np.float64),
            prior_mean.astype(np.float64),
            prior_cov.astype(np.float64),
        )

    return None
