"""Multidimensional IRT linking using Procrustes rotation.

This module provides linking methods for multidimensional IRT models
using orthogonal and oblique Procrustes rotations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ProcrustesResult:
    """Result of Procrustes rotation for MIRT linking.

    Attributes
    ----------
    rotation_matrix : NDArray[np.float64]
        Orthogonal or oblique rotation matrix R.
    translation : NDArray[np.float64]
        Intercept shift vector.
    scaling : float
        Uniform scaling factor.
    rmse : float
        Root mean square error of transformed loadings.
    transformed_loadings : NDArray[np.float64]
        Loadings after transformation.
    """

    rotation_matrix: NDArray[np.float64]
    translation: NDArray[np.float64]
    scaling: float
    rmse: float
    transformed_loadings: NDArray[np.float64]


def link_mirt(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    rotation: Literal["orthogonal", "oblique"] = "orthogonal",
    scaling: bool = True,
    translation: bool = True,
    gamma: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ProcrustesResult:
    """Link multidimensional IRT models using Procrustes rotation.

    Finds transformation: A_new * R + t ≈ A_old
    where R is a rotation matrix and t is a translation vector.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference MIRT model (target).
    model_new : BaseItemModel
        New MIRT model (source).
    anchor_items_old : list[int]
        Anchor item indices in old model.
    anchor_items_new : list[int]
        Anchor item indices in new model.
    rotation : str
        Type of rotation: "orthogonal" or "oblique".
    scaling : bool
        Whether to include uniform scaling.
    translation : bool
        Whether to include translation.
    gamma : float
        Obliqueness parameter for oblique Procrustes (0-1).
    max_iter : int
        Maximum iterations for oblique Procrustes.
    tol : float
        Convergence tolerance.

    Returns
    -------
    ProcrustesResult
        Rotation matrix, translation, scaling, and fit statistics.
    """
    if len(anchor_items_old) != len(anchor_items_new):
        raise ValueError("Anchor lists must have same length")

    disc_old = np.asarray(model_old.discrimination)
    disc_new = np.asarray(model_new.discrimination)

    if disc_old.ndim == 1:
        disc_old = disc_old.reshape(-1, 1)
    if disc_new.ndim == 1:
        disc_new = disc_new.reshape(-1, 1)

    A_target = disc_old[anchor_items_old]
    A_source = disc_new[anchor_items_new]

    if A_target.shape[1] != A_source.shape[1]:
        raise ValueError(
            f"Dimension mismatch: {A_target.shape[1]} vs {A_source.shape[1]}"
        )

    if rotation == "orthogonal":
        R, s, rmse = orthogonal_procrustes_rotation(A_target, A_source, scaling)
    else:
        R, s, rmse = oblique_procrustes_rotation(
            A_target, A_source, gamma, max_iter, tol
        )

    if translation:
        t = np.mean(A_target, axis=0) - s * np.mean(A_source @ R, axis=0)
    else:
        t = np.zeros(A_target.shape[1])

    full_disc_new = disc_new
    full_transformed = s * full_disc_new @ R + t

    return ProcrustesResult(
        rotation_matrix=R,
        translation=t,
        scaling=s,
        rmse=rmse,
        transformed_loadings=full_transformed,
    )


def orthogonal_procrustes_rotation(
    A_target: NDArray[np.float64],
    A_source: NDArray[np.float64],
    scaling: bool = True,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute orthogonal Procrustes rotation.

    Finds orthogonal matrix R that minimizes ||A_source @ R - A_target||_F.

    Parameters
    ----------
    A_target : NDArray
        Target loading matrix (n_items x n_factors).
    A_source : NDArray
        Source loading matrix to transform.
    scaling : bool
        Whether to include uniform scaling.

    Returns
    -------
    tuple[NDArray, float, float]
        Rotation matrix R, scaling factor s, and RMSE.
    """
    n_items, n_factors = A_target.shape

    M = A_source.T @ A_target

    U, S, Vt = linalg.svd(M)
    R = U @ Vt

    if linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    if scaling:
        numer = np.trace(A_source @ R @ A_target.T)
        denom = np.trace(A_source @ A_source.T)
        s = numer / max(denom, 1e-10)
    else:
        s = 1.0

    A_transformed = s * A_source @ R
    diff = A_target - A_transformed
    rmse = float(np.sqrt(np.mean(diff**2)))

    return R, float(s), rmse


def oblique_procrustes_rotation(
    A_target: NDArray[np.float64],
    A_source: NDArray[np.float64],
    gamma: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute oblique Procrustes rotation.

    Uses gradient projection algorithm for oblique factor matching.

    Parameters
    ----------
    A_target : NDArray
        Target loading matrix.
    A_source : NDArray
        Source loading matrix.
    gamma : float
        Obliqueness parameter (0=orthogonal, 1=full oblique).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple[NDArray, float, float]
        Rotation matrix T, scaling factor, and RMSE.
    """
    n_items, n_factors = A_target.shape

    R, s, _ = orthogonal_procrustes_rotation(A_target, A_source, True)
    T = R.copy()

    for _ in range(max_iter):
        G = A_source.T @ (s * A_source @ T - A_target)

        T_new = T - 0.1 * G

        if gamma < 1.0:
            U_t, S_t, Vt_t = linalg.svd(T_new)
            T_orth = U_t @ Vt_t
            T_new = gamma * T_new + (1 - gamma) * T_orth

        numer = np.trace(A_source @ T_new @ A_target.T)
        denom = np.trace(A_source @ T_new @ T_new.T @ A_source.T)
        s = numer / max(denom, 1e-10)

        diff_T = np.max(np.abs(T_new - T))
        T = T_new

        if diff_T < tol:
            break

    A_transformed = s * A_source @ T
    diff = A_target - A_transformed
    rmse = float(np.sqrt(np.mean(diff**2)))

    return T, float(s), rmse


def transform_mirt_parameters(
    model: "BaseItemModel",
    R: NDArray[np.float64],
    t: NDArray[np.float64] | None = None,
    s: float = 1.0,
    in_place: bool = False,
) -> "BaseItemModel":
    """Transform MIRT model parameters using Procrustes result.

    Parameters
    ----------
    model : BaseItemModel
        Multidimensional model to transform.
    R : NDArray
        Rotation matrix.
    t : NDArray | None
        Translation vector.
    s : float
        Scaling factor.
    in_place : bool
        Modify in place or return copy.

    Returns
    -------
    BaseItemModel
        Transformed model.
    """
    if not in_place:
        model = model.copy()

    disc = np.asarray(model.discrimination)
    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    new_disc = s * disc @ R

    if t is not None:
        diff = np.asarray(model.difficulty)
        adjustment = np.dot(t, np.linalg.pinv(R).T) / s
        new_diff = diff - adjustment.mean()
        model.set_parameters(discrimination=new_disc, difficulty=new_diff)
    else:
        model.set_parameters(discrimination=new_disc)

    return model


def factor_congruence_coefficient(
    A1: NDArray[np.float64],
    A2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Tucker's congruence coefficient between factor solutions.

    Parameters
    ----------
    A1 : NDArray
        First loading matrix (n_items x n_factors).
    A2 : NDArray
        Second loading matrix (n_items x n_factors).

    Returns
    -------
    NDArray
        Matrix of congruence coefficients (n_factors x n_factors).
    """
    n_factors = A1.shape[1]
    phi = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        for j in range(n_factors):
            numer = np.sum(A1[:, i] * A2[:, j])
            denom = np.sqrt(np.sum(A1[:, i] ** 2) * np.sum(A2[:, j] ** 2))
            phi[i, j] = numer / max(denom, 1e-10)

    return phi


def match_factors(
    A_target: NDArray[np.float64],
    A_source: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[int]]:
    """Match factors between two loading matrices.

    Uses Hungarian algorithm to find optimal factor matching.

    Parameters
    ----------
    A_target : NDArray
        Target loading matrix.
    A_source : NDArray
        Source loading matrix.

    Returns
    -------
    tuple[NDArray, list[int]]
        Reordered source matrix and permutation indices.
    """
    from scipy.optimize import linear_sum_assignment

    phi = factor_congruence_coefficient(A_target, A_source)

    cost_matrix = 1 - np.abs(phi)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    A_matched = A_source[:, col_ind]

    for i, j in enumerate(col_ind):
        if phi[i, j] < 0:
            A_matched[:, i] *= -1

    return A_matched, list(col_ind)


def compute_mirt_linking_fit(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    procrustes_result: ProcrustesResult,
) -> dict[str, float]:
    """Compute fit statistics for MIRT linking.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference model.
    model_new : BaseItemModel
        Transformed new model.
    anchor_items_old : list[int]
        Anchor indices in old model.
    anchor_items_new : list[int]
        Anchor indices in new model.
    procrustes_result : ProcrustesResult
        Procrustes rotation result.

    Returns
    -------
    dict[str, float]
        Fit statistics including RMSE, congruence, and variance explained.
    """
    disc_old = np.asarray(model_old.discrimination)
    if disc_old.ndim == 1:
        disc_old = disc_old.reshape(-1, 1)

    A_target = disc_old[anchor_items_old]
    A_transformed = procrustes_result.transformed_loadings[anchor_items_new]

    diff = A_target - A_transformed
    rmse = float(np.sqrt(np.mean(diff**2)))

    phi = factor_congruence_coefficient(A_target, A_transformed)
    mean_congruence = float(np.mean(np.diag(phi)))

    ss_total = np.sum((A_target - np.mean(A_target, axis=0)) ** 2)
    ss_resid = np.sum(diff**2)
    r_squared = 1 - ss_resid / max(ss_total, 1e-10)

    return {
        "rmse": rmse,
        "mean_congruence": mean_congruence,
        "r_squared": float(r_squared),
        "scaling": procrustes_result.scaling,
    }


def target_rotation(
    A: NDArray[np.float64],
    T: NDArray[np.float64],
    rotation_type: Literal["orthogonal", "oblique"] = "orthogonal",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate loading matrix toward target pattern.

    Parameters
    ----------
    A : NDArray
        Loading matrix to rotate.
    T : NDArray
        Target pattern matrix.
    rotation_type : str
        "orthogonal" or "oblique".
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple[NDArray, NDArray]
        Rotated loadings and rotation matrix.
    """
    n_items, n_factors = A.shape

    if rotation_type == "orthogonal":
        W = np.where(T != 0, 1.0, 0.0)

        R = np.eye(n_factors)

        for _ in range(max_iter):
            B = A @ R
            G = A.T @ (W * (B - T))

            U, S, Vt = linalg.svd(R - 0.1 * (A.T @ A) @ linalg.solve(A.T @ A, G))
            R_new = U @ Vt

            if np.max(np.abs(R_new - R)) < tol:
                break

            R = R_new

        return A @ R, R

    else:
        W = np.where(T != 0, 1.0, 0.0)

        R = np.eye(n_factors)

        for _ in range(max_iter):
            B = A @ R
            G = A.T @ (W * (B - T))

            R_new = R - 0.1 * G

            if np.max(np.abs(R_new - R)) < tol:
                break

            R = R_new

        return A @ R, R


def mirt_linking_summary(result: ProcrustesResult, model_old: "BaseItemModel") -> str:
    """Generate summary of MIRT linking results.

    Parameters
    ----------
    result : ProcrustesResult
        Procrustes rotation result.
    model_old : BaseItemModel
        Reference model.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MIRT Linking Summary (Procrustes Rotation)")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Transformation Parameters")
    lines.append("-" * 30)
    lines.append(f"Scaling factor: {result.scaling:.4f}")
    lines.append(f"RMSE: {result.rmse:.4f}")
    lines.append("")

    lines.append("Rotation Matrix R:")
    for row in result.rotation_matrix:
        lines.append("  " + "  ".join(f"{x:8.4f}" for x in row))
    lines.append("")

    lines.append("Translation Vector t:")
    lines.append("  " + "  ".join(f"{x:8.4f}" for x in result.translation))
    lines.append("")

    disc_old = np.asarray(model_old.discrimination)
    if disc_old.ndim == 1:
        disc_old = disc_old.reshape(-1, 1)

    phi = factor_congruence_coefficient(disc_old, result.transformed_loadings)
    lines.append("Factor Congruence Coefficients:")
    for i, row in enumerate(phi):
        lines.append(f"  Factor {i + 1}: " + "  ".join(f"{x:6.3f}" for x in row))
    lines.append("")

    lines.append("Transformation Equation:")
    lines.append("  A_transformed = scaling * A_new @ R + t")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
