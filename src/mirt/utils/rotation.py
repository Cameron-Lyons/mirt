"""Factor rotation methods for exploratory MIRT.

This module provides orthogonal and oblique rotation methods for
interpreting multidimensional IRT factor loadings. Every criterion except
promax is optimized by gradient projection, as in R's GPArotation package,
which R mirt's ``summary(rotate=...)`` calls.

References
----------
- Bernaards, C.A., & Jennrich, R.I. (2005). Gradient projection algorithms
  and software for arbitrary rotation criteria in factor analysis.
  Educational and Psychological Measurement, 65, 676-696.
- Browne, M.W. (2001). An overview of analytic rotation in exploratory
  factor analysis. Multivariate Behavioral Research, 36, 111-150.
- Jennrich, R.I. (2001). A simple general procedure for orthogonal
  rotation. Psychometrika, 66, 289-306.
- Jennrich, R.I. (2002). A simple general method for oblique rotation.
  Psychometrika, 67, 7-19.
- Jennrich, R.I., & Bentler, P.M. (2011). Exploratory bi-factor analysis.
  Psychometrika, 76, 537-549.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


RotationMethod = Literal[
    "varimax",
    "quartimax",
    "equamax",
    "geominT",
    "bifactorT",
    "targetT",
    "pstT",
    "oblimin",
    "quartimin",
    "promax",
    "geomin",
    "geominQ",
    "bifactorQ",
    "targetQ",
    "pstQ",
    "none",
]
RotationObjective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]
# Method names are matched case-insensitively.
_ORTHOGONAL_ROTATIONS = frozenset(
    {"varimax", "quartimax", "equamax", "geomint", "bifactort", "targett", "pstt"}
)
_OBLIQUE_ROTATIONS = frozenset(
    {
        "oblimin",
        "quartimin",
        "promax",
        "geomin",
        "geominq",
        "bifactorq",
        "targetq",
        "pstq",
    }
)
_TARGET_ROTATIONS = frozenset({"targett", "targetq", "pstt", "pstq"})
_VALID_ROTATIONS = _ORTHOGONAL_ROTATIONS | _OBLIQUE_ROTATIONS | {"none"}
_MAX_ROTATION_CONDITION = 1.0 / np.sqrt(np.finfo(np.float64).eps)
# Step halvings tried by each gradient-projection line search.
_LINE_SEARCH_STEPS = 12
# A random start replaces the incumbent only when it lowers the criterion by
# more than this relative amount, so equivalent minima keep the first start.
_RANDOM_START_IMPROVEMENT = 1e-8
# Logistic slopes are divided by D = 1.702 before standardizing, as in R mirt.
_NORMAL_OGIVE_SCALING = 1.702


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


def _validated_target(
    method: str,
    shape: tuple[int, int],
    target: ArrayLike | None,
    target_weights: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the target matrix, with zeros where unspecified, and its weights."""
    if method not in _TARGET_ROTATIONS:
        if target is not None or target_weights is not None:
            raise MirtValidationError(
                "target and target_weights apply only to the targetT, targetQ, "
                "pstT and pstQ rotations",
                parameter="target",
            )
        return None
    if target is None:
        raise MirtValidationError(
            f"the {method} rotation requires a target matrix",
            parameter="target",
        )
    try:
        target_matrix = np.array(target, dtype=np.float64)
    except (TypeError, ValueError):
        raise MirtValidationError(
            "target must be a numeric matrix", parameter="target"
        ) from None
    if target_matrix.shape != shape:
        raise MirtValidationError(
            f"target must have shape {shape}, got {target_matrix.shape}",
            parameter="target",
            value=target_matrix.shape,
        )
    if np.any(np.isinf(target_matrix)):
        raise MirtValidationError(
            "target must contain finite values, or NaN for unspecified entries",
            parameter="target",
        )
    specified = ~np.isnan(target_matrix)

    if target_weights is None:
        if method in {"pstt", "pstq"}:
            raise MirtValidationError(
                f"the {method} rotation requires target_weights",
                parameter="target_weights",
            )
        weights = specified.astype(np.float64)
    else:
        try:
            weights = np.array(target_weights, dtype=np.float64)
        except (TypeError, ValueError):
            raise MirtValidationError(
                "target_weights must be a numeric matrix",
                parameter="target_weights",
            ) from None
        if weights.shape != shape:
            raise MirtValidationError(
                f"target_weights must have shape {shape}, got {weights.shape}",
                parameter="target_weights",
                value=weights.shape,
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise MirtValidationError(
                "target_weights must be finite and non-negative",
                parameter="target_weights",
            )
        weights = np.where(specified, weights, 0.0)
    if not np.any(weights > 0.0):
        raise MirtValidationError(
            "target must specify at least one entry with a positive weight",
            parameter="target",
        )
    return np.where(specified, target_matrix, 0.0), weights


def _starting_rotations(
    n_factors: int,
    random_starts: int,
    seed: int | None,
) -> list[NDArray[np.float64]]:
    """Return the identity followed by uniformly random orthogonal starts."""
    starts = [np.eye(n_factors)]
    if random_starts:
        rng = np.random.default_rng(seed)
        for _ in range(random_starts):
            q, r = np.linalg.qr(rng.standard_normal((n_factors, n_factors)))
            signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
            starts.append(q * signs[None, :])
    return starts


def rotate_loadings(
    loadings: NDArray[np.float64],
    method: RotationMethod = "varimax",
    gamma: float | None = None,
    kappa: float = 4.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: bool | None = None,
    geomin_epsilon: float = 0.01,
    *,
    target: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    random_starts: int = 0,
    seed: int | None = None,
    reflect: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Rotate factor loadings for interpretability.

    Parameters
    ----------
    loadings : ndarray
        Unrotated loading matrix, shape (n_items, n_factors)
    method : str
        Rotation method, matched case-insensitively. The names follow R
        mirt's ``rotate`` options, with ``T`` for orthogonal and ``Q`` for
        oblique versions of a criterion. Orthogonal methods:

        - 'varimax': Maximize the variance of squared loadings within factors
        - 'quartimax': Simplify rows (items)
        - 'equamax': Orthomax compromise with weight ``n_factors / 2``, GPArotation's
          ``cfT(kappa = n_factors / (2 * n_items))``
        - 'geominT': Orthogonal geomin
        - 'bifactorT': Bifactor rotation (Jennrich & Bentler, 2011); the
          first column is the general factor
        - 'targetT', 'pstT': Rotation toward a (partially specified) target

        Oblique methods:

        - 'oblimin': Oblimin family with weight ``gamma``
        - 'quartimin': Oblimin with ``gamma = 0``
        - 'promax': Power target from the varimax solution, as ``stats::promax``
        - 'geomin' or 'geominQ': Oblique geomin
        - 'bifactorQ': Oblique bifactor rotation (biquartimin on the group
          factors)
        - 'targetQ', 'pstQ': Oblique rotation toward a (partially specified)
          target

        and 'none' returns the loadings unchanged.
    gamma : float, optional
        Weight of the oblimin criterion: 0 (default) is direct quartimin,
        0.5 biquartimin and 1 covarimin. Other methods ignore it.
    kappa : float
        Power parameter for promax rotation. Default 4.
    max_iter : int
        Maximum gradient-projection iterations from each start.
    tol : float
        Convergence tolerance on the norm of the projected gradient
        (GPArotation's ``eps``, whose default is 1e-5).
    normalize : bool, optional
        Kaiser normalization: rotate the loadings with rows scaled to unit
        length, then restore the row lengths. ``None`` (default) follows R
        mirt, which normalizes only the varimax step of promax
        (``stats::promax``) and calls GPArotation with ``normalize = FALSE``
        for every other method. Promax always forms its power target from
        the de-normalized varimax loadings.
    geomin_epsilon : float
        Positive stabilization constant ``delta`` of the geomin criterion.
        Default 0.01.
    target : array_like of shape (n_items, n_factors), optional
        Target loadings of the target and partially specified target
        rotations, which minimize ``sum((W * (L - target)) ** 2)``. For
        'targetT' and 'targetQ', ``NaN`` marks unspecified entries
        (R's ``NA``). With Kaiser normalization the target applies to the
        normalized loadings, as in GPArotation.
    target_weights : array_like of shape (n_items, n_factors), optional
        Non-negative weights ``W`` of the target entries, R's ``W``:
        1 for specified and 0 for unspecified entries. Required for 'pstT'
        and 'pstQ'; for 'targetT' and 'targetQ' the weights default to one
        on every specified entry.
    random_starts : int
        Number of uniformly random orthogonal starting rotations tried in
        addition to the identity, as GPArotation's ``randomStarts``. The
        solution with the lowest criterion is returned; among equivalent
        minima the earliest start is kept. Default 0.
    seed : int, optional
        Seed of the random starts.
    reflect : bool
        Reflect each rotated column so that its largest absolute loading is
        positive, as R mirt does. Default True.

    Returns
    -------
    rotated_loadings : ndarray
        Rotated loading matrix, shape (n_items, n_factors)
    rotation_matrix : ndarray
        Rotation matrix T such that rotated = loadings @ T
    factor_correlation : ndarray or None
        Factor correlation matrix ``inv(T) @ inv(T).T`` for oblique
        rotations, None for orthogonal ones

    Notes
    -----
    Every method except promax minimizes its criterion by gradient
    projection (Jennrich, 2001, 2002) with a monotone Armijo line search
    from the identity and each random start, so a step never worsens the
    criterion. GPArotation's default Barzilai-Borwein step can stop at a
    different point within its tolerance. R mirt's ``summary()`` rotates
    the standardized loadings with ``rotate = "oblimin"`` by default; see
    :func:`get_rotated_loadings`.
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
    if normalize is not None and not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be None or a boolean")
    if not isinstance(reflect, (bool, np.bool_)):
        raise MirtValidationError("reflect must be a boolean", parameter="reflect")
    if (
        isinstance(random_starts, (bool, np.bool_))
        or not isinstance(random_starts, (int, np.integer))
        or random_starts < 0
    ):
        raise MirtValidationError(
            "random_starts must be a non-negative integer",
            parameter="random_starts",
            value=random_starts,
        )
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
    ):
        raise MirtValidationError(
            "seed must be an integer or None", parameter="seed", value=seed
        )

    if gamma is not None and not _is_finite_real(gamma):
        raise ValueError("gamma must be a finite value")
    if normalized_method == "promax" and (not _is_finite_real(kappa) or kappa <= 1):
        raise ValueError("kappa must be a finite value greater than 1")
    is_geomin = normalized_method in {"geomin", "geominq", "geomint"}
    if is_geomin and (not _is_finite_real(geomin_epsilon) or geomin_epsilon <= 0):
        raise ValueError("geomin_epsilon must be a finite positive value")
    target_spec = _validated_target(
        normalized_method, (n_items, n_factors), target, target_weights
    )

    if n_factors == 1:
        return loadings.copy(), np.eye(1), None

    if normalized_method == "none":
        return loadings.copy(), np.eye(n_factors), None

    if normalize is None:
        normalize = normalized_method == "promax"
    if normalize:
        normalized, row_scales, scaled_norms = _kaiser_normalize(loadings)
    else:
        normalized = loadings

    starts = _starting_rotations(n_factors, int(random_starts), seed)
    factor_corr: NDArray[np.float64] | None
    if normalized_method == "promax":
        rotated, T, factor_corr = _promax(
            loadings, normalized, float(kappa), max_iter, tol, starts
        )
    else:
        objective = _rotation_objective(
            normalized_method,
            0.0 if gamma is None else float(gamma),
            float(geomin_epsilon) if is_geomin else 0.0,
            target_spec,
        )
        oblique = normalized_method in _OBLIQUE_ROTATIONS
        T = _best_gradient_projection(
            normalized, objective, oblique, starts, max_iter, tol
        )
        if oblique:
            rotated, T, factor_corr = _standardize_oblique_solution(normalized, T)
        else:
            rotated = normalized @ T
            factor_corr = None

        if normalize:
            with np.errstate(over="ignore", invalid="ignore"):
                rotated = (rotated * scaled_norms) * row_scales
            if not np.all(np.isfinite(rotated)):
                raise ValueError(
                    "rotated loadings exceed the finite floating-point range"
                )

    if reflect:
        rotated, T, factor_corr = _reflect_columns(rotated, T, factor_corr)
    return rotated, T, factor_corr


def _rotation_objective(
    method: str,
    gamma: float,
    geomin_epsilon: float,
    target_spec: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> RotationObjective:
    """Return the criterion, to be minimized, of a gradient-projection method."""
    if method in {"varimax", "quartimax", "equamax"}:
        weight = {"varimax": 1.0, "quartimax": 0.0}.get(method)

        def orthomax(
            loadings: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64]]:
            orthomax_weight = loadings.shape[1] / 2 if weight is None else weight
            return _orthomax_objective(loadings, orthomax_weight)

        return orthomax
    if method in {"oblimin", "quartimin"}:
        oblimin_weight = gamma if method == "oblimin" else 0.0
        return lambda loadings: _oblimin_objective(loadings, oblimin_weight)
    if method in {"geomin", "geominq", "geomint"}:
        return lambda loadings: _geomin_objective(loadings, geomin_epsilon)
    if method in {"bifactort", "bifactorq"}:
        return _bifactor_objective
    if target_spec is not None:
        target_matrix, weights = target_spec
        return lambda loadings: _target_objective(loadings, target_matrix, weights)
    raise AssertionError("unreachable rotation method")  # pragma: no cover


def _orthomax_objective(
    loadings: NDArray[np.float64],
    gamma: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return the negated orthomax criterion and its loading gradient.

    The orthomax criterion ``sum(L**4) - gamma / p * sum(colsum(L**2) ** 2)``
    is varimax for ``gamma = 1``, quartimax for ``gamma = 0`` and equamax
    for ``gamma = k / 2``; it is negated and divided by four as in
    GPArotation's ``vgQ.varimax`` and ``vgQ.quartimax``.
    """
    squared = loadings * loadings
    centered = squared - (gamma / loadings.shape[0]) * squared.sum(
        axis=0, keepdims=True
    )
    return -float(np.sum(squared * centered)) / 4.0, -loadings * centered


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


def _bifactor_objective(
    loadings: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return the bifactor criterion and loading gradient.

    The criterion of Jennrich and Bentler (2011), GPArotation's
    ``vgQ.bifactor``, is the quartimin criterion of the group factors
    ``L[:, 1:]``; the general factor in the first column is unpenalized.
    """
    group = loadings[:, 1:]
    squared = group * group
    cross_products = squared.sum(axis=1, keepdims=True) - squared
    gradient = np.zeros_like(loadings)
    gradient[:, 1:] = 4.0 * group * cross_products
    return float(np.sum(squared * cross_products)), gradient


def _target_objective(
    loadings: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return the weighted least-squares distance to a target and its gradient."""
    residual = weights * (loadings - target)
    return float(np.sum(residual * residual)), 2.0 * weights * residual


def _best_gradient_projection(
    A: NDArray[np.float64],
    objective: RotationObjective,
    oblique: bool,
    starts: list[NDArray[np.float64]],
    max_iter: int,
    tol: float,
) -> NDArray[np.float64]:
    """Return the rotation of the start whose solution has the lowest criterion."""
    best: tuple[NDArray[np.float64], float] | None = None
    for start in starts:
        rotation, criterion = _gradient_projection(
            A, objective, oblique, start, max_iter, tol
        )
        if best is None or criterion < best[1] - _RANDOM_START_IMPROVEMENT * max(
            1.0, abs(best[1])
        ):
            best = (rotation, criterion)
    assert best is not None
    return best[0]


def _gradient_projection(
    A: NDArray[np.float64],
    objective: RotationObjective,
    oblique: bool,
    start: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], float]:
    """Minimize a rotation criterion by gradient projection.

    The orthogonal algorithm (GPArotation's ``GPForth``) moves the rotation
    ``T`` along the criterion gradient projected onto the tangent space of
    the orthogonal matrices and returns to them through the polar factor of
    an SVD; the rotated loadings are ``A @ T``. The oblique algorithm
    (``GPFoblq``) keeps unit-length columns of ``T`` and rotates to
    ``A @ inv(T).T``. Each iteration doubles the step and halves it until the
    Armijo condition holds, accepting a smaller decrease when it never does
    and stopping when no step lowers the criterion.

    Returns
    -------
    rotation : ndarray
        Matrix ``R`` whose rotated loadings are ``A @ R``.
    criterion : float
        Criterion value at the returned rotation.
    """
    transform = np.array(start, dtype=np.float64, copy=True)
    inverse_transform = np.linalg.inv(transform) if oblique else transform.T
    rotated = A @ (inverse_transform.T if oblique else transform)
    criterion, loading_gradient = objective(rotated)
    gradient = _transform_gradient(
        A, rotated, loading_gradient, inverse_transform, oblique
    )
    step_size = 1.0

    for _ in range(max_iter):
        if oblique:
            column_inner_products = np.sum(transform * gradient, axis=0)
            projected_gradient = gradient - transform * column_inner_products
        else:
            inner = transform.T @ gradient
            projected_gradient = gradient - transform @ ((inner + inner.T) / 2.0)
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

        for _ in range(_LINE_SEARCH_STEPS):
            candidate_transform = transform - step_size * projected_gradient
            if oblique:
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
            else:
                left, _, right_transpose = np.linalg.svd(candidate_transform)
                candidate_transform = left @ right_transpose
                candidate_inverse = candidate_transform.T
                candidate_loadings = A @ candidate_transform

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
            transform,
            inverse_transform,
            rotated,
            criterion,
            loading_gradient,
        ) = accepted
        gradient = _transform_gradient(
            A, rotated, loading_gradient, inverse_transform, oblique
        )

    return (inverse_transform.T if oblique else transform), criterion


def _transform_gradient(
    A: NDArray[np.float64],
    rotated: NDArray[np.float64],
    loading_gradient: NDArray[np.float64],
    inverse_transform: NDArray[np.float64],
    oblique: bool,
) -> NDArray[np.float64]:
    """Return the criterion gradient with respect to the rotation ``T``."""
    if oblique:
        return -(rotated.T @ loading_gradient @ inverse_transform).T
    return A.T @ loading_gradient


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


def _reflect_columns(
    rotated: NDArray[np.float64],
    rotation: NDArray[np.float64],
    factor_correlation: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Flip columns so that each one's largest absolute loading is positive."""
    largest = np.argmax(np.abs(rotated), axis=0)
    signs = np.where(rotated[largest, np.arange(rotated.shape[1])] < 0.0, -1.0, 1.0)
    if factor_correlation is not None:
        factor_correlation = factor_correlation * np.outer(signs, signs)
    return rotated * signs, rotation * signs, factor_correlation


def _promax(
    A: NDArray[np.float64],
    varimax_input: NDArray[np.float64],
    kappa: float,
    max_iter: int,
    tol: float,
    starts: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Promax rotation (oblique).

    As ``stats::promax`` does, rotate ``varimax_input`` (the loadings, or
    their Kaiser-normalized rows) to varimax, regress the power target
    ``V * |V| ** (kappa - 1)`` of the de-normalized varimax loadings ``V`` on
    ``V``, and rescale the factors to unit variance. When the loading
    matrix is rank deficient, retain the identified varimax solution and an
    identity factor correlation instead of constructing a singular transform.
    """
    T_varimax = _best_gradient_projection(
        varimax_input,
        lambda loadings: _orthomax_objective(loadings, 1.0),
        False,
        starts,
        max_iter,
        tol,
    )
    varimax_rotated = A @ T_varimax
    identity = np.eye(A.shape[1])

    # The regression is invariant to a common scale, which keeps the power
    # target in the floating-point range.
    scale = float(np.max(np.abs(varimax_rotated)))
    if not np.isfinite(scale) or scale <= 0.0:
        return varimax_rotated, T_varimax, identity
    scaled = varimax_rotated / scale
    target = np.sign(scaled) * np.abs(scaled) ** kappa
    try:
        oblique_transform, _, rank, _ = np.linalg.lstsq(scaled, target, rcond=None)
    except np.linalg.LinAlgError:
        oblique_transform = np.linalg.pinv(scaled) @ target
        rank = np.linalg.matrix_rank(scaled)

    if rank < A.shape[1]:
        return varimax_rotated, T_varimax, identity

    rotation = T_varimax @ oblique_transform
    try:
        return _standardize_oblique_solution(A, rotation)
    except ValueError:
        return varimax_rotated, T_varimax, identity


def _resolve_model(
    model: BaseItemModel | FitResult,
) -> tuple[BaseItemModel, FitResult | None]:
    """Return the item model and, for a ``FitResult``, the result itself."""
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        return model.model, model
    return model, None


def _rotatable_parameter(model: BaseItemModel) -> str | None:
    """Return the stored slopes that a factor rotation transforms, if any.

    These are the parameter named by the model's ``_rotatable_slopes`` when
    it stores factors on a last axis (unidimensional slopes are vectors), or
    for other models a two-dimensional ``loadings`` or ``slopes`` parameter.
    """
    params = model.parameters
    name = getattr(model, "_rotatable_slopes", None)
    n_factors = getattr(model, "n_factors", 0)
    if name is not None:
        values = params.get(name)
        if values is None or np.ndim(values) < 2 or np.shape(values)[-1] != n_factors:
            return None
        return name
    for name in ("loadings", "slopes"):
        if name in params and np.ndim(params[name]) == 2:
            return name
    return None


def _item_slope_matrix(
    model: BaseItemModel,
    name: str,
    factor_transform: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return item-by-factor slopes and R mirt's per-item standardizing scales.

    R mirt standardizes the loadings of a slope vector ``a`` as
    ``a* / sqrt(1 + a*' a*)`` with ``a* = c * a / 1.702``. The scale ``c`` is
    one except for generalized partial credit and nominal items with ``K > 2``
    categories, for which it is ``sqrt((K - 1) * 1.702)``, times the largest
    category score divided by ``K - 1`` for nominal items. A nominal item's
    slope vector is its highest-category slope divided by ``K - 1``, the
    slope per score unit under R's default scoring. ``factor_transform`` maps
    the stored factors to uncorrelated unit-variance ones.
    """
    from mirt.models.polytomous import GeneralizedPartialCredit, NominalResponseModel

    values = np.asarray(model.parameters[name], dtype=np.float64) @ factor_transform
    scales = np.ones(values.shape[0])
    if isinstance(model, NominalResponseModel) and values.ndim == 3:
        counts = np.asarray(model.n_categories, dtype=np.intp)
        rows = np.arange(values.shape[0])
        steps = np.maximum(counts - 1, 1).astype(np.float64)
        highest = values[rows, counts - 1]
        slopes = highest / steps[:, None]
        squared_norm = np.sum(highest * highest, axis=1)
        projections = np.einsum("icf,if->ic", values, highest)
        category_scores = np.divide(
            np.max(projections, axis=1),
            squared_norm,
            out=np.ones(values.shape[0]),
            where=squared_norm > 0.0,
        )
        scales = np.where(
            counts > 2, category_scores * np.sqrt(steps * _NORMAL_OGIVE_SCALING), 1.0
        )
        return slopes, scales
    if isinstance(model, GeneralizedPartialCredit) and values.ndim == 2:
        counts = np.asarray(model.n_categories, dtype=np.float64)
        scales = np.where(
            counts > 2, np.sqrt((counts - 1.0) * _NORMAL_OGIVE_SCALING), 1.0
        )
    if values.ndim != 2:
        raise MirtValidationError(
            f"{model.model_name} slopes do not form an item-by-factor loading matrix",
            parameter=name,
        )
    return values, scales


def _factor_transform(result: FitResult | None, n_factors: int) -> NDArray[np.float64]:
    """Return the Cholesky factor mapping uncorrelated factors to the fitted ones."""
    covariance = None if result is None else result.latent_covariance
    if covariance is None:
        return np.eye(n_factors)
    return np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))


def _model_rotation(
    model: BaseItemModel,
    result: FitResult | None,
    method: str,
    rotation_kwargs: Mapping[str, Any],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64],
]:
    """Rotate a model's standardized loadings as R mirt's ``summary()`` does.

    Returns
    -------
    rotated_loadings : ndarray
        Rotated standardized loadings, or the rotated matrix itself for a
        ``loadings`` parameter of unknown scale.
    rotated_slopes : ndarray
        Item slopes on the rotated factors.
    factor_correlation : ndarray or None
        Correlation of the rotated factors for an oblique rotation.
    rotation : ndarray
        Rotation ``T`` of the fitted factors: the model's slopes become
        ``slopes @ T``.
    """
    from mirt.models.multidimensional import standardized_loadings

    name = _rotatable_parameter(model)
    if name is not None:
        transform = _factor_transform(result, np.shape(model.parameters[name])[-1])
        if name == "loadings":
            slopes = np.asarray(model.parameters[name], dtype=np.float64) @ transform
            criterion_input = slopes
        else:
            slopes, scales = _item_slope_matrix(model, name, transform)
            criterion_input = standardized_loadings(slopes * scales[:, None])[0]
    elif callable(getattr(model, "get_loading_matrix", None)):
        slopes = np.asarray(model.get_loading_matrix(), dtype=np.float64)
        transform = _factor_transform(result, slopes.shape[1])
        slopes = slopes @ transform
        criterion_input = standardized_loadings(slopes)[0]
    else:
        raise ValueError("Model does not have loadings to rotate")

    rotated, rotation, factor_corr = rotate_loadings(
        criterion_input, method=method, **rotation_kwargs
    )
    return rotated, slopes @ rotation, factor_corr, transform @ rotation


def _validated_rotation_matrix(
    rotation_matrix: ArrayLike, n_factors: int
) -> NDArray[np.float64]:
    """Return a finite, well-conditioned square rotation matrix."""
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    expected_shape = (n_factors, n_factors)
    if rotation.shape != expected_shape:
        raise ValueError(
            f"rotation_matrix must have shape {expected_shape}, got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_matrix must contain only finite values")
    condition_number = float(np.linalg.cond(rotation))
    if not np.isfinite(condition_number) or condition_number >= _MAX_ROTATION_CONDITION:
        raise ValueError("rotation_matrix must be well-conditioned and nonsingular")
    return rotation


def _validated_factor_correlation(
    factor_correlation: ArrayLike,
    implied_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a factor correlation that matches the rotated latent covariance."""
    expected_shape = implied_covariance.shape
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
    if not np.allclose(correlation, implied_covariance, atol=1e-7, rtol=1e-7):
        raise ValueError("factor_correlation is inconsistent with rotation_matrix")
    return correlation


def _rotated_parameters(
    model: BaseItemModel,
    name: str,
    rotation: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Return the parameters that express the model on the rotated factors.

    The slopes ``a`` become ``a @ T``. Families whose multidimensional logits
    are ``a @ theta - sum(a) * b`` list their locations ``b`` in
    ``_slope_sum_locations``; they are rescaled to keep each item's
    intercepts ``sum(a) * b`` unchanged.
    """
    params = model.parameters
    slopes = np.asarray(params[name], dtype=np.float64)
    rotated = slopes @ rotation
    updates = {name: rotated}
    locations = tuple(getattr(model, "_slope_sum_locations", ()))
    if not locations or slopes.ndim != 2:
        return updates

    old_sums = np.sum(slopes, axis=1)
    new_sums = np.sum(rotated, axis=1)
    threshold = np.sqrt(np.finfo(np.float64).eps) * np.maximum(
        1.0, np.sum(np.abs(rotated), axis=1)
    )
    near_zero = np.abs(new_sums) <= threshold
    ratio = np.divide(old_sums, new_sums, out=np.zeros_like(old_sums), where=~near_zero)
    for location in locations:
        values = np.asarray(params[location], dtype=np.float64)
        shape = (-1,) + (1,) * (values.ndim - 1)
        intercepts = values * old_sums.reshape(shape)
        changed = np.any(
            np.abs(intercepts.reshape(values.shape[0], -1)) > threshold[:, None],
            axis=1,
        )
        if np.any(near_zero & changed):
            items = np.flatnonzero(near_zero & changed)
            raise MirtValidationError(
                f"the rotated slopes of item(s) {items.tolist()} sum to zero, so "
                f"{model.model_name}'s 'a @ theta - sum(a) * b' parameterization "
                "cannot keep their intercepts; use the slope-intercept "
                "MultidimensionalModel",
                parameter="rotation_matrix",
            )
        updates[location] = values * ratio.reshape(shape)
    return updates


def _record_rotated_population(
    result: FitResult,
    updated: Mapping[str, NDArray[np.float64]],
    inverse_rotation: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> None:
    """Re-express a fit's latent population and drop invalidated uncertainty."""
    identity = np.eye(covariance.shape[0])
    if result.latent_covariance is None and np.allclose(
        covariance, identity, rtol=0.0, atol=1e-12
    ):
        result.latent_covariance = None
    else:
        result.latent_covariance = covariance
    if result.latent_mean is not None:
        result.latent_mean = inverse_rotation @ result.latent_mean

    # Rotated parameters have no estimated sampling uncertainty, as in R.
    for name in updated:
        if name in result.standard_errors:
            result.standard_errors[name] = np.full(
                updated[name].shape, np.nan, dtype=np.float64
            )
    if result.vcov is not None and result.vcov_labels is not None:
        affected = [
            index
            for index, label in enumerate(result.vcov_labels)
            if label.split("[", 1)[0] in updated
        ]
        if affected:
            vcov = result.vcov.copy()
            vcov[affected, :] = np.nan
            vcov[:, affected] = np.nan
            result.vcov = vcov


def apply_rotation_to_model(
    model: BaseItemModel | FitResult,
    rotation_matrix: NDArray[np.float64] | str,
    factor_correlation: NDArray[np.float64] | None = None,
    **kwargs: Any,
) -> None:
    """Re-express a fitted exploratory MIRT model on rotated factors in place.

    The slopes ``a`` become ``a @ T``, and the factors ``theta`` become
    ``inv(T) @ theta`` with covariance ``inv(T) @ Sigma @ inv(T).T``, where
    ``Sigma`` is the fit's latent covariance (the identity by default). The
    rotated model therefore has the same marginal likelihood, response
    probabilities and fit statistics. Models parameterized as
    ``a @ theta - sum(a) * b`` (2PL, GRM, GPCM) also rescale ``b`` so that
    their intercepts are unchanged.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        A fitted exploratory model whose slopes are all free, such as the
        2PL, GRM, GPCM, NRM or ``MIRT`` model of an exploratory
        ``fit_mirt(..., n_factors=2)`` fit, or its ``FitResult``. An oblique
        rotation needs the ``FitResult``: its ``latent_covariance`` becomes
        the factor correlation and ``latent_mean`` is rotated, so scoring,
        simulation and diagnostics use the rotated population. The standard
        errors and ``vcov`` entries of the changed parameters become
        ``NaN``. Structured bifactor and confirmatory models cannot be
        updated in place.
    rotation_matrix : ndarray or str
        Rotation matrix ``T`` from :func:`rotate_loadings`, or the name of a
        rotation method, which rotates the standardized loadings as
        :func:`get_rotated_loadings` does.
    factor_correlation : ndarray, optional
        Factor correlation matrix for oblique rotations, checked against the
        rotated latent covariance. Omit it with a method name.
    **kwargs
        Arguments for :func:`rotate_loadings` when ``rotation_matrix`` is a
        method name.

    Raises
    ------
    MirtValidationError
        If an oblique rotation is applied to a bare model, if rotation
        options accompany a rotation matrix, or if a rotated item's slopes
        sum to zero in a ``sum(a) * b`` parameterization.
    """
    item_model, result = _resolve_model(model)
    if getattr(item_model, "model_type", None) == "confirmatory":
        raise ValueError("rotation can only be applied to exploratory models")

    parameter_name = _rotatable_parameter(item_model)
    if parameter_name is None:
        if "general_loadings" in item_model.parameters:
            raise ValueError(
                "structured bifactor loadings cannot be updated by a dense rotation"
            )
        raise ValueError("Model does not have freely rotatable loadings")
    if getattr(
        item_model, "_rotatable_slopes", None
    ) is not None and not item_model._rotational_indeterminacy(
        item_model.free_parameter_masks
    ):
        raise ValueError(
            "Model does not have freely rotatable loadings: its slopes are "
            "fixed or restricted"
        )

    n_factors = int(np.shape(item_model.parameters[parameter_name])[-1])
    if isinstance(rotation_matrix, str):
        if factor_correlation is not None:
            raise MirtValidationError(
                "factor_correlation follows from the rotation method; omit it",
                parameter="factor_correlation",
            )
        _, _, factor_correlation, rotation = _model_rotation(
            item_model, result, rotation_matrix, kwargs
        )
    else:
        if kwargs:
            raise MirtValidationError(
                f"rotation options {sorted(kwargs)} require a rotation method name",
                parameter="rotation_matrix",
            )
        rotation = _validated_rotation_matrix(rotation_matrix, n_factors)

    inverse_rotation = np.linalg.inv(rotation)
    covariance = (
        np.eye(n_factors)
        if result is None or result.latent_covariance is None
        else np.asarray(result.latent_covariance, dtype=np.float64)
    )
    rotated_covariance = inverse_rotation @ covariance @ inverse_rotation.T
    rotated_covariance = (rotated_covariance + rotated_covariance.T) / 2

    correlation = (
        None
        if factor_correlation is None
        else _validated_factor_correlation(factor_correlation, rotated_covariance)
    )
    if result is None and not np.allclose(
        rotated_covariance, np.eye(n_factors), atol=1e-8, rtol=0.0
    ):
        raise MirtValidationError(
            "an oblique rotation correlates the factors, which a bare model "
            "cannot record; pass the FitResult so that its latent_covariance "
            "holds the factor correlation",
            parameter="rotation_matrix",
        )

    updates = _rotated_parameters(item_model, parameter_name, rotation)
    item_model.set_parameters(**updates)
    if result is not None:
        _record_rotated_population(
            result, updates, inverse_rotation, rotated_covariance
        )
    item_model._rotation_matrix = rotation.copy()
    item_model._factor_correlation = None if correlation is None else correlation.copy()


def get_rotated_loadings(
    model: BaseItemModel | FitResult,
    method: str = "oblimin",
    *,
    standardized: bool = False,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Get rotated loadings from a fitted MIRT model.

    As R mirt's ``summary(rotate=...)`` does, the rotation is computed on
    the standardized loadings ``a* / sqrt(1 + a*' a*)`` with
    ``a* = a / 1.702``, without Kaiser normalization by default, and each
    column is reflected so that its largest loading is positive.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        A fitted model exposing a loading matrix, such as the 2PL, GRM,
        GPCM, NRM or ``MIRT`` model of an exploratory
        ``fit_mirt(..., n_factors=2)`` fit, or its ``FitResult``, whose
        ``latent_covariance`` then defines the unrotated factors. Structured
        bifactor models are supported for reading and rotation without
        in-place mutation.
    method : str
        Rotation method; see :func:`rotate_loadings`. Default 'oblimin', R
        mirt's ``summary()`` default.
    standardized : bool
        Return the rotated standardized loadings (R's ``summary()``) rather
        than the item slopes on the rotated factors (R's
        ``coef(rotate=...)``). Default False.
    **kwargs
        Additional arguments for rotate_loadings()

    Returns
    -------
    rotated_loadings : ndarray
        Rotated loading matrix
    factor_correlation : ndarray or None
        Factor correlation matrix for oblique rotations

    Notes
    -----
    Generalized partial credit and nominal items are standardized with R
    mirt's scale ``sqrt((K - 1) * 1.702)``; a nominal item's slopes are its
    highest-category slopes per score unit. Use :func:`apply_rotation_to_model`
    to re-express the fit itself on the rotated factors.
    """
    item_model, result = _resolve_model(model)
    rotated, slopes, factor_corr, _ = _model_rotation(
        item_model, result, method, kwargs
    )
    return (rotated if standardized else slopes), factor_corr


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
