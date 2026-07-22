"""Rust backend: regularized.

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


def coordinate_descent_mstep_regularized(
    r_k: NDArray[np.float64],
    n_k: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    adaptive_weights: NDArray[np.float64],
    lambda_val: float,
    alpha: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Coordinate descent M-step with regularization for MIRT.

    Parameters
    ----------
    r_k : NDArray
        Expected correct responses (n_items, n_quad)
    n_k : NDArray
        Expected total responses (n_items, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad, n_factors)
    loadings : NDArray
        Current factor loadings (n_items, n_factors)
    intercepts : NDArray
        Current intercepts (n_items,)
    adaptive_weights : NDArray
        Adaptive LASSO weights (n_items, n_factors)
    lambda_val : float
        Regularization strength
    alpha : float
        Elastic net mixing (1 = LASSO, 0 = ridge)
    max_iter : int
        Maximum coordinate descent iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    tuple or None
        (new_loadings, new_intercepts) or None if Rust unavailable
    """
    if rust_enabled():
        return mirt_rs.coordinate_descent_mstep_regularized(
            r_k.astype(np.float64),
            n_k.astype(np.float64),
            quad_points.astype(np.float64),
            loadings.astype(np.float64),
            intercepts.astype(np.float64),
            adaptive_weights.astype(np.float64),
            lambda_val,
            alpha,
            max_iter,
            tol,
        )

    return None
