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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


RotationMethod = Literal[
    "varimax", "quartimax", "equamax", "oblimin", "promax", "geomin", "none"
]
RotationObjective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
_VALID_ROTATIONS = frozenset(
    {"varimax", "quartimax", "equamax", "oblimin", "promax", "geomin", "none"}
)
_MAX_ROTATION_CONDITION = 1.0 / np.sqrt(np.finfo(np.float64).eps)


def _is_finite_real(value: object) -> bool:
    """Return whether a runtime control is a finite, non-boolean real scalar."""
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
    )


def _kaiser_normalize(
    loadings: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Normalize rows without squaring values at their original scale."""
    row_scales = np.max(np.abs(loadings), axis=1, keepdims=True)
    row_scales = np.where(row_scales > 0.0, row_scales, 1.0)
    scaled = loadings / row_scales
    scaled_norms = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    scaled_norms = np.where(scaled_norms > 0.0, scaled_norms, 1.0)
    return scaled / scaled_norms, row_scales, scaled_norms


def rotate_loadings(
    loadings: NDArray[np.float64],
    method: RotationMethod = "varimax",
    gamma: float | None = None,
    kappa: float = 4.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: bool = True,
    geomin_epsilon: float = 0.01,
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
    geomin_epsilon : float
        Positive stabilization constant for geomin rotation. Default 0.01.

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
    if loadings.ndim != 2:
        raise ValueError("loadings must be a 2D matrix")
    n_items, n_factors = loadings.shape
    if n_items == 0 or n_factors == 0:
        raise ValueError("loadings must contain at least one item and one factor")
    if n_items < n_factors:
        raise ValueError("loadings must contain at least as many items as factors")
    if not np.all(np.isfinite(loadings)):
        raise ValueError("loadings must contain only finite values")

    if not isinstance(method, str):
        raise ValueError("method must be a string")
    normalized_method = method.lower()
    if normalized_method not in _VALID_ROTATIONS:
        raise ValueError(f"Unknown rotation method: {method}")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError("max_iter must be an integer")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not _is_finite_real(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive value")
    tol = float(tol)
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be a boolean")

    if gamma is not None and not _is_finite_real(gamma):
        raise ValueError("gamma must be a finite value")
    if normalized_method == "promax" and (not _is_finite_real(kappa) or kappa <= 1):
        raise ValueError("kappa must be a finite value greater than 1")
    if normalized_method == "geomin" and (
        not _is_finite_real(geomin_epsilon) or geomin_epsilon <= 0
    ):
        raise ValueError("geomin_epsilon must be a finite positive value")

    if n_factors == 1:
        return loadings.copy(), np.eye(1), None

    if normalized_method == "none":
        return loadings.copy(), np.eye(n_factors), None

    if normalize:
        normalized, row_scales, scaled_norms = _kaiser_normalize(loadings)
    else:
        normalized = loadings
        row_scales = np.ones((n_items, 1))
        scaled_norms = np.ones((n_items, 1))

    if normalized_method == "varimax":
        rotated, T = _varimax(normalized, max_iter, tol)
        factor_corr = None

    elif normalized_method == "quartimax":
        rotated, T = _quartimax(normalized, max_iter, tol)
        factor_corr = None

    elif normalized_method == "equamax":
        rotated, T = _equamax(normalized, max_iter, tol)
        factor_corr = None

    elif normalized_method == "oblimin":
        if gamma is None:
            gamma = 0.0
        rotated, T, factor_corr = _oblimin(normalized, float(gamma), max_iter, tol)

    elif normalized_method == "promax":
        rotated, T, factor_corr = _promax(normalized, float(kappa), max_iter, tol)

    elif normalized_method == "geomin":
        rotated, T, factor_corr = _geomin(
            normalized, max_iter, tol, float(geomin_epsilon)
        )

    else:  # pragma: no cover - guarded by validation above
        raise AssertionError("unreachable rotation method")

    if normalize:
        with np.errstate(over="ignore", invalid="ignore"):
            rotated = (rotated * scaled_norms) * row_scales
        if not np.all(np.isfinite(rotated)):
            raise ValueError("rotated loadings exceed the finite floating-point range")

    return rotated, T, factor_corr


def _varimax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Varimax rotation (orthogonal).

    Maximizes variance of squared loadings within factors.
    """
    return _orthomax(A, gamma=1.0, max_iter=max_iter, tol=tol)


def _quartimax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quartimax rotation (orthogonal).

    Simplifies rows (items) rather than columns (factors).
    """
    return _orthomax(A, gamma=0.0, max_iter=max_iter, tol=tol)


def _equamax(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equamax rotation (orthogonal).

    Compromise between varimax and quartimax with gamma = p/2.
    """
    return _orthomax(A, gamma=A.shape[1] / 2, max_iter=max_iter, tol=tol)


def _orthomax(
    A: NDArray[np.float64],
    gamma: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Optimize the orthomax family with an SVD fixed-point update."""
    n_items, n_factors = A.shape
    rotation = np.eye(n_factors)
    objective = 0.0
    gamma_per_item = gamma / n_items

    for _ in range(max_iter):
        basis = A @ rotation
        squared = basis * basis
        column_sums = np.sum(squared, axis=0, keepdims=True)
        transformed = A.T @ (basis * (squared - gamma_per_item * column_sums))
        left, singular_values, right_transpose = np.linalg.svd(
            transformed, full_matrices=False
        )
        updated_rotation = left @ right_transpose
        updated_objective = float(np.sum(singular_values))
        rotation = updated_rotation

        if objective > 0 and updated_objective <= objective * (1 + tol):
            break
        objective = updated_objective

    return A @ rotation, rotation


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

    def objective(
        loadings: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        return _oblimin_objective(loadings, gamma)

    return _oblique_gradient_projection(A, objective, max_iter, tol)


def _oblimin_objective(
    loadings: NDArray[np.float64],
    gamma: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the oblimin-family criterion and loading gradient."""
    squared = loadings * loadings
    cross_products = squared.sum(axis=1, keepdims=True) - squared
    if gamma != 0:
        cross_products = cross_products - (
            gamma / loadings.shape[0]
        ) * cross_products.sum(axis=0, keepdims=True)

    gradient = loadings * cross_products
    criterion = float(np.sum(squared * cross_products) / 4)
    return criterion, gradient


def _geomin_objective(
    loadings: NDArray[np.float64],
    epsilon: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the geomin criterion and loading gradient."""
    stabilized = loadings * loadings + epsilon
    geometric_means = np.exp(np.mean(np.log(stabilized), axis=1))
    gradient = (
        (2.0 / loadings.shape[1]) * (loadings / stabilized) * geometric_means[:, None]
    )
    return float(np.sum(geometric_means)), gradient


def _oblique_gradient_projection(
    A: NDArray[np.float64],
    objective: RotationObjective,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Minimize an oblique criterion using gradient projection."""
    n_factors = A.shape[1]
    factor_transform = np.eye(n_factors)
    inverse_transform = np.eye(n_factors)
    rotated = A.copy()
    criterion, loading_gradient = objective(rotated)
    gradient = -(rotated.T @ loading_gradient @ inverse_transform).T
    step_size = 1.0

    for _ in range(max_iter):
        column_inner_products = np.sum(factor_transform * gradient, axis=0)
        projected_gradient = gradient - factor_transform * column_inner_products
        gradient_norm = float(np.linalg.norm(projected_gradient))
        if gradient_norm < tol:
            break

        step_size *= 2.0
        accepted: (
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                NDArray[np.float64],
            ]
            | None
        ) = None
        fallback: (
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                NDArray[np.float64],
            ]
            | None
        ) = None

        for _ in range(12):
            candidate_transform = factor_transform - step_size * projected_gradient
            column_norms = np.linalg.norm(candidate_transform, axis=0)
            if np.any(column_norms <= np.finfo(np.float64).eps):
                step_size /= 2.0
                continue
            candidate_transform = candidate_transform / column_norms

            try:
                candidate_inverse = np.linalg.inv(candidate_transform)
            except np.linalg.LinAlgError:
                step_size /= 2.0
                continue

            candidate_loadings = A @ candidate_inverse.T
            candidate_criterion, candidate_loading_gradient = objective(
                candidate_loadings
            )
            if not np.isfinite(candidate_criterion) or not np.all(
                np.isfinite(candidate_loading_gradient)
            ):
                step_size /= 2.0
                continue

            proposal = (
                candidate_transform,
                candidate_inverse,
                candidate_loadings,
                candidate_criterion,
                candidate_loading_gradient,
            )
            if candidate_criterion < criterion:
                fallback = proposal
            improvement = criterion - candidate_criterion
            if improvement > 0.5 * gradient_norm**2 * step_size:
                accepted = proposal
                break
            step_size /= 2.0

        if accepted is None:
            accepted = fallback
        if accepted is None:
            break

        (
            factor_transform,
            inverse_transform,
            rotated,
            criterion,
            loading_gradient,
        ) = accepted
        gradient = -(rotated.T @ loading_gradient @ inverse_transform).T

    direct_rotation = inverse_transform.T
    return _standardize_oblique_solution(A, direct_rotation)


def _standardize_oblique_solution(
    A: NDArray[np.float64],
    rotation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale an oblique solution so its factor covariance is a correlation."""
    condition_number = float(np.linalg.cond(rotation))
    if not np.isfinite(condition_number) or condition_number >= _MAX_ROTATION_CONDITION:
        raise ValueError("oblique rotation produced an ill-conditioned transform")
    try:
        inverse_rotation = np.linalg.inv(rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError("oblique rotation produced a singular transform") from exc

    factor_correlation = inverse_rotation @ inverse_rotation.T
    scales = np.sqrt(np.diag(factor_correlation))
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("oblique rotation produced invalid factor scales")

    standardized_rotation = rotation * scales[None, :]
    factor_correlation = factor_correlation / np.outer(scales, scales)
    factor_correlation = (factor_correlation + factor_correlation.T) / 2
    np.fill_diagonal(factor_correlation, 1.0)
    return A @ standardized_rotation, standardized_rotation, factor_correlation


def _promax(
    A: NDArray[np.float64],
    kappa: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Promax rotation (oblique).

    Starts from varimax, then applies power transformation. When the loading
    matrix is rank deficient, retain the identified varimax solution and an
    identity factor correlation instead of constructing a singular transform.
    """
    varimax_rotated, T_varimax = _varimax(A, max_iter, tol)

    target = np.sign(varimax_rotated) * np.abs(varimax_rotated) ** kappa
    try:
        oblique_transform, _, rank, _ = np.linalg.lstsq(
            varimax_rotated, target, rcond=None
        )
    except np.linalg.LinAlgError:
        oblique_transform = np.linalg.pinv(varimax_rotated) @ target
        rank = np.linalg.matrix_rank(varimax_rotated)

    if rank < A.shape[1]:
        return varimax_rotated, T_varimax, np.eye(A.shape[1])

    rotation = T_varimax @ oblique_transform
    try:
        return _standardize_oblique_solution(A, rotation)
    except ValueError:
        return varimax_rotated, T_varimax, np.eye(A.shape[1])


def _geomin(
    A: NDArray[np.float64],
    max_iter: int,
    tol: float,
    epsilon: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Geomin rotation (oblique).

    Minimizes sum of geometric means of squared loadings.
    """

    def objective(
        loadings: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        return _geomin_objective(loadings, epsilon)

    return _oblique_gradient_projection(A, objective, max_iter, tol)


def apply_rotation_to_model(
    model: BaseItemModel,
    rotation_matrix: NDArray[np.float64],
    factor_correlation: NDArray[np.float64] | None = None,
) -> None:
    """Apply rotation to a fitted MIRT model in-place.

    Parameters
    ----------
    model : BaseItemModel
        A fitted exploratory model with a freely rotatable two-dimensional
        loading or slope matrix. Structured bifactor and confirmatory models
        cannot be updated in place.
    rotation_matrix : ndarray
        Rotation matrix from rotate_loadings()
    factor_correlation : ndarray, optional
        Factor correlation matrix for oblique rotations
    """
    if getattr(model, "model_type", None) == "confirmatory":
        raise ValueError("rotation can only be applied to exploratory models")

    params = model.parameters
    if "loadings" in params:
        parameter_name = "loadings"
    elif "slopes" in params:
        parameter_name = "slopes"
    elif "general_loadings" in params:
        raise ValueError(
            "structured bifactor loadings cannot be updated by a dense rotation"
        )
    else:
        raise ValueError("Model does not have freely rotatable loadings")

    loadings = np.asarray(params[parameter_name], dtype=np.float64)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if loadings.ndim != 2:
        raise ValueError("model loadings must be a 2D matrix")
    expected_shape = (loadings.shape[1], loadings.shape[1])
    if rotation.shape != expected_shape:
        raise ValueError(
            f"rotation_matrix must have shape {expected_shape}, got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_matrix must contain only finite values")
    condition_number = float(np.linalg.cond(rotation))
    if not np.isfinite(condition_number) or condition_number >= _MAX_ROTATION_CONDITION:
        raise ValueError("rotation_matrix must be well-conditioned and nonsingular")

    correlation: NDArray[np.float64] | None
    if factor_correlation is None:
        correlation = None
    else:
        correlation = np.asarray(factor_correlation, dtype=np.float64)
        if correlation.shape != expected_shape:
            raise ValueError(
                f"factor_correlation must have shape {expected_shape}, "
                f"got {correlation.shape}"
            )
        if not np.all(np.isfinite(correlation)):
            raise ValueError("factor_correlation must contain only finite values")
        if not np.allclose(correlation, correlation.T, atol=1e-8, rtol=1e-8):
            raise ValueError("factor_correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("factor_correlation must have ones on its diagonal")
        if np.min(np.linalg.eigvalsh(correlation)) <= 0:
            raise ValueError("factor_correlation must be positive definite")
        inverse_rotation = np.linalg.inv(rotation)
        implied_correlation = inverse_rotation @ inverse_rotation.T
        if not np.allclose(correlation, implied_correlation, atol=1e-7, rtol=1e-7):
            raise ValueError("factor_correlation is inconsistent with rotation_matrix")

    rotated_loadings = loadings @ rotation
    model.set_parameters(**{parameter_name: rotated_loadings})
    model._rotation_matrix = rotation.copy()
    model._factor_correlation = None if correlation is None else correlation.copy()


def get_rotated_loadings(
    model: BaseItemModel,
    method: str = "varimax",
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Get rotated loadings from a fitted MIRT model.

    Parameters
    ----------
    model : BaseItemModel
        A fitted model exposing a loading matrix. Structured bifactor models
        are supported for reading and rotation without in-place mutation.
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
    elif "slopes" in params:
        loadings = params["slopes"]
    elif callable(getattr(model, "get_loading_matrix", None)):
        loadings = np.asarray(model.get_loading_matrix(), dtype=np.float64)
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
