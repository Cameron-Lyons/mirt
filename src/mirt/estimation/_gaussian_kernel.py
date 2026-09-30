"""Bounded Gaussian prior kernels for Monte Carlo draws."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

_MAX_GAUSSIAN_KERNEL_ELEMENTS = 131_072


def gaussian_log_kernel(
    theta: NDArray[np.float64],
    mean: NDArray[np.float64],
    factor: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the quadratic prior term with owned, bounded solve buffers."""
    theta = np.asarray(theta)
    mean = np.asarray(mean, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.float64)
    n_factors = theta.shape[-1]
    if n_factors == 0:
        raise ValueError("Gaussian samples must have at least one factor")
    if mean.shape != (n_factors,) or factor.shape != (n_factors, n_factors):
        raise ValueError("Gaussian prior dimensions must match the sample factors")
    diagonal = np.diag(factor)
    diagonal_only = np.count_nonzero(factor) == np.count_nonzero(diagonal)
    triangular = not np.any(np.triu(factor, k=1))
    if diagonal_only and np.any(diagonal == 0.0):
        raise np.linalg.LinAlgError("Singular matrix")

    shape = theta.shape[:-1]
    result = np.empty(shape)
    values = result.reshape(-1)
    points = theta.reshape(-1, n_factors) if theta.flags.c_contiguous else None
    block_size = max(1, _MAX_GAUSSIAN_KERNEL_ELEMENTS // n_factors)
    for start in range(0, values.size, block_size):
        stop = min(start + block_size, values.size)
        if points is None:
            if theta.ndim == 1:
                centered = np.array(theta[None, :], dtype=np.float64, copy=True)
            else:
                indices = np.unravel_index(np.arange(start, stop), shape)
                # Advanced indexing owns the gather; conversion stays within it.
                centered = np.asarray(theta[indices], dtype=np.float64)
        else:
            centered = np.array(points[start:stop], dtype=np.float64, copy=True)
        centered -= mean
        if diagonal_only:
            centered /= diagonal
        elif triangular:
            centered = solve_triangular(
                factor, centered.T, lower=True, check_finite=False, overwrite_b=True
            ).T
        else:
            # Preserve direct callers that supply a general invertible factor.
            centered = np.linalg.solve(factor, centered.T).T
        with np.errstate(over="ignore", invalid="ignore"):
            squared = np.einsum("ij,ij->i", centered, centered)
            np.multiply(squared, -0.5, out=values[start:stop])
            overflow = np.isinf(squared)
            if np.any(overflow):
                # Rescale only overflowing reductions, preserving tiny terms
                # while keeping representable tail kernels finite.
                tails = centered[overflow]
                tails *= 0.5
                values[start:stop][overflow] = -2.0 * np.einsum(
                    "ij,ij->i", tails, tails
                )
                del tails
        del centered, squared
    return result
