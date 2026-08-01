"""Normalized prior masses for standard-normal Gauss-Hermite quadrature."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validated_quadrature(
    theta: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate quadrature inputs and return points with log weights."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError("quadrature points must be a finite 1D or 2D array")

    weights = np.asarray(quadrature_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (len(points),):
        raise ValueError("quadrature weights must match the number of points")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError("quadrature weights must be finite, non-negative, and nonzero")

    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0.0
    log_weights[positive] = np.log(weights[positive])
    return points, log_weights


def normalize_log_mass(log_mass: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a finite-or-negative-infinity vector in log space."""
    values = np.asarray(log_mass, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("prior mass contains invalid values")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ValueError("prior mass must contain at least one positive value")
    log_total = maximum + np.log(np.exp(values - maximum).sum())
    return values - log_total


def log_density_quadrature_mass(
    theta: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
    log_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert target log densities to normalized masses on normal GH nodes."""
    points, log_weights = _validated_quadrature(theta, quadrature_weights)
    target = np.asarray(log_density, dtype=np.float64).reshape(-1)
    if target.shape != (len(points),):
        raise ValueError("log density must return one value per quadrature point")

    n_dimensions = points.shape[1]
    log_reference = -0.5 * (
        n_dimensions * np.log(2.0 * np.pi) + np.sum(points**2, axis=1)
    )
    return normalize_log_mass(log_weights + target - log_reference)


def gaussian_log_quadrature_mass(
    theta: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build normalized Gaussian masses on standard-normal GH nodes."""
    points = np.asarray(theta, dtype=np.float64)
    if points.ndim == 1:
        points = points[:, None]
    mean_array = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov_array = np.asarray(cov, dtype=np.float64)
    n_dimensions = points.shape[1]
    if mean_array.shape != (n_dimensions,):
        raise ValueError(f"mean must have shape ({n_dimensions},)")
    if cov_array.shape != (n_dimensions, n_dimensions):
        raise ValueError(f"cov must have shape ({n_dimensions}, {n_dimensions})")
    try:
        np.linalg.cholesky(cov_array)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc
    sign, log_determinant = np.linalg.slogdet(cov_array)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("cov must be positive definite")

    difference = points - mean_array
    try:
        solved = np.linalg.solve(cov_array, difference.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov must be positive definite") from exc
    mahalanobis = np.sum(difference * solved, axis=1)
    log_density = -0.5 * (
        n_dimensions * np.log(2.0 * np.pi) + log_determinant + mahalanobis
    )
    return log_density_quadrature_mass(theta, quadrature_weights, log_density)
