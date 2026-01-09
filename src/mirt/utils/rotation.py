"""Factor rotation methods for exploratory MIRT.

This module provides orthogonal and oblique rotation methods for
interpreting multidimensional IRT factor loadings.

References
----------
- Browne, M.W. (2001). An overview of analytic rotation in exploratory
  factor analysis. Multivariate Behavioral Research, 36, 111-150.
- Jennrich, R.I. (2002). A simple general method for oblique rotation.
  Psychometrika, 67, 7-19.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.optimize import minimize


def rotate_loadings(
    loadings: NDArray[np.float64],
    method: Literal[
        "varimax", "quartimax", "equamax", "oblimin", "promax", "geomin", "none"
    ] = "varimax",
    gamma: float | None = None,
    kappa: float = 4.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Rotate factor loadings for interpretability.

    Parameters
    ----------
    loadings : ndarray
        Unrotated loading matrix, shape (n_items, n_factors)
    method : str
        Rotation method:
        - 'varimax': Orthogonal rotation maximizing variance of squared loadings
        - 'quartimax': Orthogonal rotation simplifying rows
        - 'equamax': Compromise between varimax and quartimax
        - 'oblimin': Oblique rotation (allows correlated factors)
        - 'promax': Oblique rotation starting from varimax
        - 'geomin': Oblique rotation for simple structure
        - 'none': No rotation (returns original loadings)
    gamma : float, optional
        Parameter for oblimin rotation. Default depends on method.
    kappa : float
        Power parameter for promax rotation. Default 4.
    max_iter : int
        Maximum iterations for iterative methods
    tol : float
        Convergence tolerance
    normalize : bool
        Kaiser normalization before rotation. Default True.

    Returns
    -------
    rotated_loadings : ndarray
        Rotated loading matrix, shape (n_items, n_factors)
    rotation_matrix : ndarray
        Rotation matrix T such that rotated = loadings @ T
    factor_correlation : ndarray or None
        Factor correlation matrix for oblique rotations, None for orthogonal
    """
    loadings = np.asarray(loadings, dtype=np.float64)
    n_items, n_factors = loadings.shape

    if n_factors == 1:
        return loadings.copy(), np.eye(1), None

    if method == "none":
        return loadings.copy(), np.eye(n_factors), None

    if normalize:
        h2 = np.sum(loadings**2, axis=1, keepdims=True)
        h2 = np.where(h2 > 0, h2, 1.0)
        h = np.sqrt(h2)
        normalized = loadings / h
    else:
        normalized = loadings
        h = np.ones((n_items, 1))

    if method == "varimax":
        rotated, T = _varimax(normalized, max_iter, tol)
        factor_corr = None

    elif method == "quartimax":
        rotated, T = _quartimax(normalized, max_iter, tol)
        factor_corr = None

    elif method == "equamax":
        rotated, T = _equamax(normalized, max_iter, tol)
        factor_corr = None

    elif method == "oblimin":
        if gamma is None:
            gamma = 0.0
        rotated, T, factor_corr = _oblimin(normalized, gamma, max_iter, tol)

    elif method == "promax":
        rotated, T, factor_corr = _promax(normalized, kappa, max_iter, tol)

    elif method == "geomin":
        rotated, T, factor_corr = _geomin(normalized, max_iter, tol)

    else:
        raise ValueError(f"Unknown rotation method: {method}")

    if normalize:
        rotated = rotated * h

    return rotated, T, factor_corr


def _varimax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Varimax rotation (orthogonal).

    Maximizes variance of squared loadings within factors.
    """
    n, p = A.shape
    T = np.eye(p)

    for _ in range(max_iter):
        B = A @ T

        B2 = B**2
        col_means = B2.mean(axis=0, keepdims=True)
        U = B * (B2 - col_means)

        U_tilde, _, Vt = svd(A.T @ U, full_matrices=False)
        T_new = U_tilde @ Vt

        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break

        T = T_new

    return A @ T, T


def _quartimax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quartimax rotation (orthogonal).

    Simplifies rows (items) rather than columns (factors).
    """
    n, p = A.shape
    T = np.eye(p)

    for _ in range(max_iter):
        B = A @ T

        B2 = B**2
        U = B * B2

        U_tilde, _, Vt = svd(A.T @ U, full_matrices=False)
        T_new = U_tilde @ Vt

        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break

        T = T_new

    return A @ T, T


def _equamax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equamax rotation (orthogonal).

    Compromise between varimax and quartimax with gamma = p/2.
    """
    n, p = A.shape
    gamma = p / 2

    T = np.eye(p)

    for _ in range(max_iter):
        B = A @ T

        B2 = B**2
        col_means = B2.mean(axis=0, keepdims=True)

        U = B * (B2 - gamma * col_means / n)

        U_tilde, _, Vt = svd(A.T @ U, full_matrices=False)
        T_new = U_tilde @ Vt

        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break

        T = T_new

    return A @ T, T


def _oblimin(
    A: NDArray[np.float64],
    gamma: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Oblimin rotation (oblique).

    gamma = 0: Direct quartimin
    gamma = 0.5: Biquartimin
    gamma = 1: Covarimin
    """
    n, p = A.shape

    T = np.eye(p)
    alpha = 1.0

    N = np.ones((p, p)) - np.eye(p)

    for _ in range(max_iter):
        Ti = np.linalg.inv(T)
        L = A @ Ti.T

        L2 = L**2

        C = L2.T @ L2 / n

        if gamma == 0:
            G = L * (L2 @ N)
        else:
            G = L * (L2 @ N - gamma * np.diag(C).reshape(1, -1) * L2 / n)

        grad = -A.T @ G @ Ti.T

        T_new = T - alpha * grad

        U, _, Vt = svd(T_new, full_matrices=False)
        T_new = U @ Vt

        if np.max(np.abs(T_new - T)) < tol:
            T = T_new
            break

        T = T_new

    Ti = np.linalg.inv(T)
    rotated = A @ Ti.T

    factor_corr = Ti @ Ti.T

    return rotated, Ti.T, factor_corr


def _promax(
    A: NDArray[np.float64],
    kappa: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Promax rotation (oblique).

    Starts from varimax, then applies power transformation.
    """
    varimax_rotated, T_varimax = _varimax(A, max_iter, tol)

    P = varimax_rotated.copy()
    signs = np.sign(P)
    P = signs * (np.abs(P) ** kappa)

    try:
        T_oblique = np.linalg.lstsq(varimax_rotated, P, rcond=None)[0]
    except np.linalg.LinAlgError:
        T_oblique = np.linalg.pinv(varimax_rotated) @ P

    D = np.diag(1 / np.sqrt(np.diag(T_oblique.T @ T_oblique)))
    T_oblique = T_oblique @ D

    rotated = A @ T_varimax @ T_oblique

    Ti = np.linalg.inv(T_oblique)
    factor_corr = Ti @ Ti.T

    return rotated, T_varimax @ T_oblique, factor_corr


def _geomin(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
    epsilon: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Geomin rotation (oblique).

    Minimizes sum of geometric means of squared loadings.
    """
    n, p = A.shape

    def geomin_criterion(t_flat: NDArray[np.float64]) -> float:
        T = t_flat.reshape(p, p)
        try:
            Ti = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            return 1e10

        L = A @ Ti.T
        L2 = L**2 + epsilon

        log_L2 = np.log(L2)
        geo_means = np.exp(log_L2.mean(axis=1))

        return np.sum(geo_means)

    T0 = np.eye(p).flatten()
    result = minimize(
        geomin_criterion,
        T0,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol},
    )

    T = result.x.reshape(p, p)

    U, _, Vt = svd(T, full_matrices=False)
    T = U @ Vt

    Ti = np.linalg.inv(T)
    rotated = A @ Ti.T

    factor_corr = Ti @ Ti.T

    return rotated, Ti.T, factor_corr


def apply_rotation_to_model(
    model: BaseItemModel,
    rotation_matrix: NDArray[np.float64],
    factor_correlation: NDArray[np.float64] | None = None,
) -> None:
    """Apply rotation to a fitted MIRT model in-place.

    Parameters
    ----------
    model : MultidimensionalModel
        A fitted exploratory MIRT model
    rotation_matrix : ndarray
        Rotation matrix from rotate_loadings()
    factor_correlation : ndarray, optional
        Factor correlation matrix for oblique rotations
    """
    params = model.parameters

    if "loadings" in params:
        loadings = params["loadings"]
        rotated_loadings = loadings @ rotation_matrix
        model.set_parameters(loadings=rotated_loadings)

    elif "general_loadings" in params:
        gen_loadings = params["general_loadings"]
        rotated = gen_loadings @ rotation_matrix
        model.set_parameters(general_loadings=rotated)

    model._rotation_matrix = rotation_matrix
    model._factor_correlation = factor_correlation


def get_rotated_loadings(
    model: BaseItemModel,
    method: str = "varimax",
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Get rotated loadings from a fitted MIRT model.

    Parameters
    ----------
    model : MultidimensionalModel
        A fitted exploratory MIRT model
    method : str
        Rotation method
    **kwargs
        Additional arguments for rotate_loadings()

    Returns
    -------
    rotated_loadings : ndarray
        Rotated loading matrix
    factor_correlation : ndarray or None
        Factor correlation matrix for oblique rotations
    """
    params = model.parameters

    if "loadings" in params:
        loadings = params["loadings"]
    elif "general_loadings" in params:
        loadings = params["general_loadings"]
    else:
        raise ValueError("Model does not have loadings to rotate")

    rotated, _, factor_corr = rotate_loadings(loadings, method=method, **kwargs)

    return rotated, factor_corr


def varimax(loadings: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]:
    """Convenience function for varimax rotation."""
    rotated, _, _ = rotate_loadings(loadings, method="varimax", **kwargs)
    return rotated


def promax(loadings: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]:
    """Convenience function for promax rotation."""
    rotated, _, _ = rotate_loadings(loadings, method="promax", **kwargs)
    return rotated


def oblimin(loadings: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]:
    """Convenience function for oblimin rotation."""
    rotated, _, _ = rotate_loadings(loadings, method="oblimin", **kwargs)
    return rotated
