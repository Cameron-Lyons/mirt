"""Multidimensional IRT linking and target rotation.

The linking functions use row-vector item slopes.  A source calibration is
placed on the target metric with

``a_linked = scaling * a_source @ rotation``

and, for every linear item intercept,

``d_linked = d_source - (a_source @ rotation) @ translation``.

Graded response and generalized partial credit models have one intercept
``-sum(a) * b_k`` per category threshold or step, and each one transforms
this way. The corresponding latent-coordinate transformation is implemented
by :func:`transform_mirt_theta`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


FloatArray = NDArray[np.float64]
ParameterStyle = Literal[
    "slopes_intercepts", "discrimination_difficulty", "discrimination_locations"
]


@dataclass
class ProcrustesResult:
    """Result of a multidimensional linking transformation.

    Parameters
    ----------
    rotation_matrix
        Orthogonal or nonsingular oblique transformation matrix.
    translation
        Latent-coordinate location shift.
    scaling
        Uniform dilation applied to the rotated item slopes.
    rmse
        Root mean squared anchor-slope residual.
    transformed_loadings
        All source slopes transformed to the target metric.
    transformed_intercepts
        All source linear intercepts transformed to the target metric. GRM
        and GPCM models have one column per category intercept
        ``-sum(a) * b_k``; columns beyond an item's categories are zero.
    intercept_rmse
        Root mean squared anchor-intercept residual over every category
        intercept, when available.
    anchor_items_old, anchor_items_new
        Anchor mappings used to estimate the transformation.
    """

    rotation_matrix: FloatArray
    translation: FloatArray
    scaling: float
    rmse: float
    transformed_loadings: FloatArray
    transformed_intercepts: FloatArray | None = None
    intercept_rmse: float | None = None
    anchor_items_old: tuple[int, ...] = ()
    anchor_items_new: tuple[int, ...] = ()


@dataclass(frozen=True)
class _LinearParameters:
    """Slopes and intercepts of a compensatory model.

    ``intercepts`` has one value per item, or for GRM and GPCM models one
    column per category location stored as ``location_name``; ``active``
    marks the columns within each item's categories.
    """

    slopes: FloatArray
    intercepts: FloatArray
    style: ParameterStyle
    location_name: str | None = None
    active: NDArray[np.bool_] | None = None

    def intercept_columns(self) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Intercepts as a matrix with one row per item, and its active mask."""
        if self.intercepts.ndim == 1:
            return self.intercepts[:, None], np.ones(
                (self.intercepts.size, 1), dtype=np.bool_
            )
        assert self.active is not None
        return self.intercepts, self.active

    def translated_intercepts(
        self, rotated_slopes: FloatArray, shift: FloatArray
    ) -> FloatArray:
        """Every active intercept ``d - (a @ R) @ t``; padding stays zero."""
        offset = rotated_slopes @ shift
        if self.intercepts.ndim == 1:
            return self.intercepts - offset
        assert self.active is not None
        return np.where(self.active, self.intercepts - offset[:, None], 0.0)


def _as_finite_matrix(value: object, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_matrix_pair(
    A_target: object,
    A_source: object,
    *,
    require_target_rank: bool = True,
) -> tuple[FloatArray, FloatArray]:
    target = _as_finite_matrix(A_target, "A_target")
    source = _as_finite_matrix(A_source, "A_source")
    if target.shape != source.shape:
        raise ValueError(
            "A_target and A_source must have the same shape; "
            f"got {target.shape} and {source.shape}"
        )

    n_rows, n_factors = target.shape
    if n_rows < n_factors:
        raise ValueError(
            f"at least {n_factors} rows are required to identify {n_factors} factors"
        )
    if np.linalg.matrix_rank(source) < n_factors:
        raise ValueError("A_source must have full column rank")
    if require_target_rank and np.linalg.matrix_rank(target) < n_factors:
        raise ValueError("A_target must have full column rank")
    return target, source


def _validate_solver_controls(max_iter: int, tol: float) -> None:
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("max_iter must be a positive integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")


def _extract_linear_parameters(model: BaseItemModel) -> _LinearParameters:
    """Extract the slope/intercept form used by compensatory MIRT models.

    GRM and GPCM models list their category locations ``b_k`` in
    ``_slope_sum_locations``; their intercepts are ``-sum(a) * b_k``.
    """
    try:
        params = model.parameters
    except (AttributeError, TypeError) as exc:
        raise ValueError("model must expose a parameters mapping") from exc

    category_locations = [
        name
        for name in getattr(model, "_slope_sum_locations", ())
        if name in params and np.ndim(params[name]) == 2
    ]
    if "slopes" in params and "intercepts" in params:
        slopes = _as_finite_matrix(params["slopes"], "model slopes")
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        style: ParameterStyle = "slopes_intercepts"
    elif "discrimination" in params and len(category_locations) == 1:
        return _extract_category_intercepts(model, params, category_locations[0])
    elif "discrimination" in params and "difficulty" in params:
        slopes = _as_finite_matrix(params["discrimination"], "model discrimination")
        difficulty = np.asarray(params["difficulty"], dtype=np.float64)
        if difficulty.ndim != 1 or difficulty.shape[0] != slopes.shape[0]:
            raise ValueError(
                "multidimensional discrimination/difficulty models must use one "
                "scalar difficulty per item"
            )
        if not np.all(np.isfinite(difficulty)):
            raise ValueError("model difficulty must contain only finite values")
        intercepts = -np.sum(slopes, axis=1) * difficulty
        style = "discrimination_difficulty"
    else:
        raise ValueError(
            "model must use either slopes/intercepts, multidimensional "
            "discrimination with scalar difficulty, or multidimensional "
            "discrimination with GRM thresholds or GPCM steps"
        )

    if intercepts.ndim != 1 or intercepts.shape[0] != slopes.shape[0]:
        raise ValueError("model intercepts must contain one value per item")
    if not np.all(np.isfinite(intercepts)):
        raise ValueError("model intercepts must contain only finite values")
    _validate_slope_dimensions(model, slopes)
    return _LinearParameters(slopes, intercepts, style)


def _validate_slope_dimensions(model: BaseItemModel, slopes: FloatArray) -> None:
    if slopes.shape[1] < 2:
        raise ValueError("multidimensional linking requires at least two factors")
    n_items = getattr(model, "n_items", slopes.shape[0])
    n_factors = getattr(model, "n_factors", slopes.shape[1])
    if slopes.shape != (n_items, n_factors):
        raise ValueError(
            "model slope shape does not match its declared item/factor dimensions"
        )


def _extract_category_intercepts(
    model: BaseItemModel,
    params: dict[str, FloatArray],
    location_name: str,
) -> _LinearParameters:
    """Extract one intercept ``-sum(a) * b_k`` per GRM threshold or GPCM step."""
    if np.ndim(params["discrimination"]) != 2:
        raise ValueError("multidimensional linking requires at least two factors")
    slopes = _as_finite_matrix(params["discrimination"], "model discrimination")
    locations = np.asarray(params[location_name], dtype=np.float64)
    if locations.shape[0] != slopes.shape[0]:
        raise ValueError(f"model {location_name} must contain one row per item")
    category_counts = np.asarray(model.n_categories, dtype=np.intp)
    if category_counts.shape != (slopes.shape[0],) or np.any(
        category_counts - 1 > locations.shape[1]
    ):
        raise ValueError(f"model {location_name} do not cover every item's categories")
    active = np.arange(locations.shape[1])[None, :] < (category_counts[:, None] - 1)
    if not np.all(np.isfinite(locations[active])):
        raise ValueError(f"model {location_name} must contain only finite values")
    intercepts = np.where(active, -np.sum(slopes, axis=1)[:, None] * locations, 0.0)
    _validate_slope_dimensions(model, slopes)
    return _LinearParameters(
        slopes, intercepts, "discrimination_locations", location_name, active
    )


def _validate_anchor_indices(
    indices: object,
    *,
    name: str,
    n_items: int,
    n_factors: int,
) -> NDArray[np.intp]:
    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if raw.size == 0:
        raise ValueError("at least one anchor item is required")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer indices")

    result = raw.astype(np.intp, copy=False)
    if result.size < n_factors:
        raise ValueError(
            f"at least {n_factors} anchor items are required for {n_factors} factors"
        )
    if np.any(result < 0) or np.any(result >= n_items):
        raise IndexError(f"{name} contains an item index outside [0, {n_items})")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must not contain duplicate item indices")
    return result


def _difficulty_from_intercepts(
    slopes: FloatArray, intercepts: FloatArray
) -> FloatArray:
    """Locations ``-d / sum(a)`` of one intercept, or a row of them, per item."""
    slope_sums = np.sum(slopes, axis=1)
    threshold = np.finfo(np.float64).eps * np.maximum(
        1.0, np.linalg.norm(slopes, axis=1)
    )
    near_zero = np.abs(slope_sums) <= threshold
    row_shape = (-1,) + (1,) * (intercepts.ndim - 1)
    nonzero = np.abs(intercepts) > threshold.reshape(row_shape)
    if intercepts.ndim > 1:
        nonzero = np.any(nonzero, axis=tuple(range(1, intercepts.ndim)))
    if np.any(near_zero & nonzero):
        raise ValueError(
            "the transformed intercept cannot be represented by the model's "
            "scalar-difficulty parameterization because an item has zero summed slope"
        )

    difficulty = np.zeros_like(intercepts)
    np.divide(
        -intercepts,
        slope_sums.reshape(row_shape),
        out=difficulty,
        where=np.broadcast_to(~near_zero.reshape(row_shape), intercepts.shape),
    )
    return difficulty


def _set_linear_parameters(
    model: BaseItemModel,
    extracted: _LinearParameters,
    slopes: FloatArray,
    intercepts: FloatArray,
) -> None:
    if extracted.style == "slopes_intercepts":
        pattern = getattr(model, "loading_pattern", None)
        if pattern is not None:
            pattern_array = np.asarray(pattern)
            forbidden = (pattern_array == 0) & ~np.isclose(slopes, 0.0)
            if np.any(forbidden):
                raise ValueError(
                    "the requested rotation is incompatible with the model's fixed "
                    "loading pattern"
                )
        model.set_parameters(slopes=slopes, intercepts=intercepts)
    elif extracted.style == "discrimination_locations":
        assert extracted.location_name is not None and extracted.active is not None
        stored = np.asarray(model.parameters[extracted.location_name])
        locations = np.where(
            extracted.active, _difficulty_from_intercepts(slopes, intercepts), stored
        )
        model.set_parameters(
            discrimination=slopes, **{extracted.location_name: locations}
        )
    else:
        difficulty = _difficulty_from_intercepts(slopes, intercepts)
        model.set_parameters(discrimination=slopes, difficulty=difficulty)


def _anchor_intercepts(
    target: _LinearParameters,
    source: _LinearParameters,
    old_indices: NDArray[np.intp],
    new_indices: NDArray[np.intp],
    source_intercepts: FloatArray | None = None,
) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
    """Pair every anchor intercept of the two forms.

    Returns each anchor's number of intercepts and the target and source
    intercepts, stacked anchor by anchor. ``source_intercepts`` replaces the
    source intercepts, for example with transformed values.
    """
    target_values, target_active = target.intercept_columns()
    source_values, source_active = source.intercept_columns()
    if source_intercepts is not None:
        source_values = source_intercepts.reshape(source_values.shape)
    counts = np.count_nonzero(target_active[old_indices], axis=1)
    if not np.array_equal(counts, np.count_nonzero(source_active[new_indices], axis=1)):
        raise ValueError(
            "corresponding anchor items must have the same number of categories"
        )
    # Active intercepts lead each row, so equal counts align the categories.
    return (
        counts,
        target_values[old_indices][target_active[old_indices]],
        source_values[new_indices][source_active[new_indices]],
    )


def link_mirt(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    rotation: Literal["orthogonal", "oblique"] = "orthogonal",
    scaling: bool = True,
    translation: bool = True,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ProcrustesResult:
    """Link a source MIRT calibration to a target calibration.

    ``model_old`` defines the target metric and ``model_new`` defines the source
    metric.  Anchor slopes identify the rotation and dilation.  Anchor linear
    intercepts identify the latent-coordinate translation by least squares.

    Parameters
    ----------
    model_old : BaseItemModel
        Target calibration. Models may use native ``slopes``/``intercepts``
        parameters, the multidimensional 2PL ``discrimination``/scalar
        ``difficulty`` convention, or multidimensional GRM thresholds or
        GPCM steps, whose category intercepts ``-sum(a) * b_k`` all
        identify the translation.
    model_new : BaseItemModel
        Source calibration, with the same number of factors.
    anchor_items_old, anchor_items_new : list[int]
        Corresponding anchor items. Polytomous anchors must have the same
        number of categories on both forms.
    rotation : {"orthogonal", "oblique"}
        ``"orthogonal"`` fits the orthogonal Procrustes rotation (with
        reflections). ``"oblique"`` fits a general nonsingular
        transformation by least squares, the oblique Procrustes linking of
        Li and Lissitz (2000), which recovers any exact linear metric
        difference.
    scaling : bool
        Whether to estimate a uniform dilation of the rotated slopes.
    translation : bool
        Whether to estimate the latent-coordinate translation.
    gamma : float
        Oblique weight in ``[0, 1]``. The default 1 is unrestricted least
        squares; smaller values ridge the oblique transformation toward the
        orthogonal solution, which ``gamma=0`` returns. The ridge can
        stabilize nearly collinear anchors but biases a genuine oblique
        difference. Ignored for orthogonal rotation.
    max_iter, tol : int, float
        Validated and retained for call compatibility; the oblique solution
        is computed directly.

    Returns
    -------
    ProcrustesResult
        Transformation and transformed source parameters.
    """
    if rotation not in {"orthogonal", "oblique"}:
        raise ValueError("rotation must be 'orthogonal' or 'oblique'")
    if not isinstance(scaling, (bool, np.bool_)):
        raise ValueError("scaling must be boolean")
    if not isinstance(translation, (bool, np.bool_)):
        raise ValueError("translation must be boolean")

    target_params = _extract_linear_parameters(model_old)
    source_params = _extract_linear_parameters(model_new)
    if target_params.slopes.shape[1] != source_params.slopes.shape[1]:
        raise ValueError(
            "target and source models must have the same number of factors"
        )

    n_factors = target_params.slopes.shape[1]
    old_indices = _validate_anchor_indices(
        anchor_items_old,
        name="anchor_items_old",
        n_items=target_params.slopes.shape[0],
        n_factors=n_factors,
    )
    new_indices = _validate_anchor_indices(
        anchor_items_new,
        name="anchor_items_new",
        n_items=source_params.slopes.shape[0],
        n_factors=n_factors,
    )
    if old_indices.size != new_indices.size:
        raise ValueError("anchor lists must have the same length")

    A_target = target_params.slopes[old_indices]
    A_source = source_params.slopes[new_indices]
    if rotation == "orthogonal":
        R, scale, rmse = orthogonal_procrustes_rotation(
            A_target, A_source, scaling=bool(scaling)
        )
    else:
        R, scale, rmse = oblique_procrustes_rotation(
            A_target,
            A_source,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            scaling=bool(scaling),
        )

    counts, target_intercepts, source_intercepts = _anchor_intercepts(
        target_params, source_params, old_indices, new_indices
    )
    if translation:
        # Every category intercept of an anchor shares its rotated slopes.
        shift, _, _, _ = linalg.lstsq(
            np.repeat(A_source @ R, counts, axis=0),
            source_intercepts - target_intercepts,
        )
    else:
        shift = np.zeros(n_factors, dtype=np.float64)

    transformed_slopes = scale * source_params.slopes @ R
    transformed_intercepts = source_params.translated_intercepts(
        source_params.slopes @ R, shift
    )
    _, _, linked_intercepts = _anchor_intercepts(
        target_params, source_params, old_indices, new_indices, transformed_intercepts
    )
    intercept_rmse = float(
        np.sqrt(np.mean((target_intercepts - linked_intercepts) ** 2))
    )

    return ProcrustesResult(
        rotation_matrix=R,
        translation=np.asarray(shift, dtype=np.float64),
        scaling=scale,
        rmse=rmse,
        transformed_loadings=transformed_slopes,
        transformed_intercepts=transformed_intercepts,
        intercept_rmse=intercept_rmse,
        anchor_items_old=tuple(int(index) for index in old_indices),
        anchor_items_new=tuple(int(index) for index in new_indices),
    )


def orthogonal_procrustes_rotation(
    A_target: FloatArray,
    A_source: FloatArray,
    scaling: bool = True,
) -> tuple[FloatArray, float, float]:
    """Fit an orthogonal rotation, optionally with uniform dilation.

    Reflections are permitted because factor signs are unidentified.  This is
    the standard orthogonal Procrustes solution rather than the proper-rotation
    variant used for physical coordinate systems.
    """
    if not isinstance(scaling, (bool, np.bool_)):
        raise ValueError("scaling must be boolean")
    target, source = _validate_matrix_pair(A_target, A_source)

    U, singular_values, Vt = linalg.svd(
        source.T @ target, full_matrices=False, check_finite=False
    )
    rotation = U @ Vt
    if scaling:
        denominator = float(np.sum(source * source))
        scale = float(np.sum(singular_values) / denominator)
    else:
        scale = 1.0

    residual = target - scale * source @ rotation
    rmse = float(np.sqrt(np.mean(residual**2)))
    return rotation, scale, rmse


def oblique_procrustes_rotation(
    A_target: FloatArray,
    A_source: FloatArray,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    scaling: bool = True,
) -> tuple[FloatArray, float, float]:
    """Fit an oblique Procrustes linking transformation.

    The default ``gamma=1`` returns the unrestricted least-squares
    transformation of ``A_source`` toward ``A_target``, which recovers any
    exact nonsingular metric difference.  ``gamma=0`` returns the orthogonal
    similarity solution, and intermediate values ridge the general
    transformation toward it, an opt-in regularization that biases genuine
    oblique differences.  ``max_iter`` and ``tol`` are validated and retained
    for call compatibility; the solution is computed directly without
    iteration.
    """
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not isinstance(scaling, (bool, np.bool_)):
        raise ValueError("scaling must be boolean")
    _validate_solver_controls(max_iter, tol)
    target, source = _validate_matrix_pair(A_target, A_source)

    orthogonal, orthogonal_scale, orthogonal_rmse = orthogonal_procrustes_rotation(
        target, source, scaling=True
    )
    if gamma <= np.finfo(np.float64).eps:
        if scaling:
            return orthogonal, orthogonal_scale, orthogonal_rmse
        residual = target - source @ orthogonal
        return orthogonal, 1.0, float(np.sqrt(np.mean(residual**2)))

    ata = source.T @ source
    atb = source.T @ target
    orthogonal_transform = orthogonal_scale * orthogonal
    if gamma == 1.0:
        # Normal equations square the condition number and can lose an
        # identifiable factor when anchor slopes are nearly collinear.
        transform, _, _, _ = linalg.lstsq(source, target, check_finite=False)
    else:
        matrix_scale = max(float(np.trace(ata) / ata.shape[0]), np.finfo(float).tiny)
        ridge = ((1.0 - gamma) / gamma) * matrix_scale
        transform = linalg.solve(
            ata + ridge * np.eye(ata.shape[0]),
            atb + ridge * orthogonal_transform,
            assume_a="pos",
            check_finite=False,
        )

    if np.linalg.matrix_rank(transform) < transform.shape[0]:
        raise ValueError("the fitted oblique transformation is singular")

    if scaling:
        scale = float(linalg.norm(transform, ord="fro") / np.sqrt(transform.shape[0]))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("the fitted oblique scaling is not positive and finite")
        rotation = transform / scale
    else:
        scale = 1.0
        rotation = transform

    residual = target - scale * source @ rotation
    rmse = float(np.sqrt(np.mean(residual**2)))
    return rotation, scale, rmse


def transform_mirt_parameters(
    model: BaseItemModel,
    R: FloatArray,
    t: FloatArray | None = None,
    s: float = 1.0,
    in_place: bool = False,
) -> BaseItemModel:
    """Transform compensatory MIRT item parameters to a linked metric.

    Slopes become ``s * a @ R`` and every intercept, including each GRM or
    GPCM category intercept ``-sum(a) * b_k``, becomes ``d - (a @ R) @ t``.
    """
    extracted = _extract_linear_parameters(model)
    rotation = _as_finite_matrix(R, "R")
    n_factors = extracted.slopes.shape[1]
    if rotation.shape != (n_factors, n_factors):
        raise ValueError(
            f"R must have shape {(n_factors, n_factors)}, got {rotation.shape}"
        )
    if np.linalg.matrix_rank(rotation) < n_factors:
        raise ValueError("R must be nonsingular")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    if not isinstance(in_place, (bool, np.bool_)):
        raise ValueError("in_place must be boolean")

    if t is None:
        shift = np.zeros(n_factors, dtype=np.float64)
    else:
        shift = np.asarray(t, dtype=np.float64)
        if shift.shape != (n_factors,):
            raise ValueError(f"t must have shape {(n_factors,)}, got {shift.shape}")
        if not np.all(np.isfinite(shift)):
            raise ValueError("t must contain only finite values")

    transformed_slopes = s * extracted.slopes @ rotation
    transformed_intercepts = extracted.translated_intercepts(
        extracted.slopes @ rotation, shift
    )

    output = model if in_place else deepcopy(model)
    output_params = _extract_linear_parameters(output)
    _set_linear_parameters(
        output, output_params, transformed_slopes, transformed_intercepts
    )
    return output


def transform_mirt_theta(
    theta: FloatArray,
    R: FloatArray,
    t: FloatArray | None = None,
    s: float = 1.0,
) -> FloatArray:
    """Transform source latent coordinates to the linked target metric.

    The transformation is ``(theta @ inv(R).T + t) / s`` and preserves the
    linear predictor when paired with :func:`transform_mirt_parameters`.
    A one-dimensional input is treated as one multidimensional point and the
    original dimensionality is preserved.
    """
    rotation = _as_finite_matrix(R, "R")
    if rotation.shape[0] != rotation.shape[1]:
        raise ValueError("R must be square")
    n_factors = rotation.shape[0]
    if np.linalg.matrix_rank(rotation) < n_factors:
        raise ValueError("R must be nonsingular")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")

    values = np.asarray(theta, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        if values.shape != (n_factors,):
            raise ValueError(
                f"theta must have shape {(n_factors,)} or (n, {n_factors})"
            )
        values_2d = values[None, :]
    elif values.ndim == 2 and values.shape[1] == n_factors:
        values_2d = values
    else:
        raise ValueError(f"theta must have shape {(n_factors,)} or (n, {n_factors})")
    if not np.all(np.isfinite(values_2d)):
        raise ValueError("theta must contain only finite values")

    if t is None:
        shift = np.zeros(n_factors, dtype=np.float64)
    else:
        shift = np.asarray(t, dtype=np.float64)
        if shift.shape != (n_factors,):
            raise ValueError(f"t must have shape {(n_factors,)}, got {shift.shape}")
        if not np.all(np.isfinite(shift)):
            raise ValueError("t must contain only finite values")

    transformed = (linalg.solve(rotation, values_2d.T).T + shift) / s
    return transformed[0] if was_vector else transformed


def factor_congruence_coefficient(
    A1: FloatArray,
    A2: FloatArray,
) -> FloatArray:
    """Compute all pairwise Tucker congruence coefficients.

    The two matrices must describe the same items but may contain different
    numbers of factors.  The result has shape ``(A1 factors, A2 factors)``.
    """
    first = _as_finite_matrix(A1, "A1")
    second = _as_finite_matrix(A2, "A2")
    if first.shape[0] != second.shape[0]:
        raise ValueError("A1 and A2 must contain the same number of rows")

    first_norms = np.linalg.norm(first, axis=0)
    second_norms = np.linalg.norm(second, axis=0)
    if np.any(first_norms == 0) or np.any(second_norms == 0):
        raise ValueError("factor congruence is undefined for a zero loading column")
    return (first.T @ second) / np.outer(first_norms, second_norms)


def match_factors(
    A_target: FloatArray,
    A_source: FloatArray,
) -> tuple[FloatArray, list[int]]:
    """Optimally permute and sign-align source factors to target factors."""
    target = _as_finite_matrix(A_target, "A_target")
    source = _as_finite_matrix(A_source, "A_source")
    if target.shape[0] != source.shape[0]:
        raise ValueError("A_target and A_source must contain the same number of rows")
    if target.shape[1] != source.shape[1]:
        raise ValueError("factor matching requires the same number of factors")

    congruence = factor_congruence_coefficient(target, source)
    row_indices, column_indices = linear_sum_assignment(-np.abs(congruence))
    order = np.empty(target.shape[1], dtype=np.intp)
    order[row_indices] = column_indices

    matched = source[:, order].copy()
    signs = np.where(congruence[np.arange(target.shape[1]), order] < 0.0, -1.0, 1.0)
    matched *= signs
    return matched, [int(index) for index in order]


def compute_mirt_linking_fit(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    procrustes_result: ProcrustesResult,
) -> dict[str, float]:
    """Compute anchor-slope and intercept fit statistics for a link."""
    target_params = _extract_linear_parameters(model_old)
    source_params = _extract_linear_parameters(model_new)
    if target_params.slopes.shape[1] != source_params.slopes.shape[1]:
        raise ValueError(
            "target and source models must have the same number of factors"
        )
    n_factors = target_params.slopes.shape[1]
    old_indices = _validate_anchor_indices(
        anchor_items_old,
        name="anchor_items_old",
        n_items=target_params.slopes.shape[0],
        n_factors=n_factors,
    )
    new_indices = _validate_anchor_indices(
        anchor_items_new,
        name="anchor_items_new",
        n_items=source_params.slopes.shape[0],
        n_factors=n_factors,
    )
    if old_indices.size != new_indices.size:
        raise ValueError("anchor lists must have the same length")

    transformed = _as_finite_matrix(
        procrustes_result.transformed_loadings, "transformed_loadings"
    )
    if transformed.shape != source_params.slopes.shape:
        raise ValueError(
            "transformed_loadings shape must match the source model slope shape"
        )

    target_anchor = target_params.slopes[old_indices]
    transformed_anchor = transformed[new_indices]
    residual = target_anchor - transformed_anchor
    rmse = float(np.sqrt(np.mean(residual**2)))
    congruence = factor_congruence_coefficient(target_anchor, transformed_anchor)
    mean_congruence = float(np.mean(np.diag(congruence)))

    centered = target_anchor - np.mean(target_anchor, axis=0)
    ss_total = float(np.sum(centered * centered))
    ss_residual = float(np.sum(residual * residual))
    if ss_total <= np.finfo(np.float64).eps:
        r_squared = 1.0 if ss_residual <= np.finfo(np.float64).eps else 0.0
    else:
        r_squared = 1.0 - ss_residual / ss_total

    result = {
        "rmse": rmse,
        "mean_congruence": mean_congruence,
        "r_squared": float(r_squared),
        "scaling": float(procrustes_result.scaling),
    }
    if procrustes_result.transformed_intercepts is not None:
        transformed_intercepts = np.asarray(
            procrustes_result.transformed_intercepts, dtype=np.float64
        )
        if transformed_intercepts.shape != source_params.intercepts.shape:
            raise ValueError(
                "transformed_intercepts shape must match the source model intercept shape"
            )
        if not np.all(np.isfinite(transformed_intercepts)):
            raise ValueError("transformed_intercepts must contain only finite values")
        _, target_intercepts, linked_intercepts = _anchor_intercepts(
            target_params,
            source_params,
            old_indices,
            new_indices,
            transformed_intercepts,
        )
        intercept_residual = target_intercepts - linked_intercepts
        result["intercept_rmse"] = float(
            np.sqrt(np.mean(intercept_residual * intercept_residual))
        )
    return result


def target_rotation(
    A: FloatArray,
    T: FloatArray,
    rotation_type: Literal["orthogonal", "oblique"] = "orthogonal",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[FloatArray, FloatArray]:
    """Rotate a loading matrix toward a fully specified target matrix.

    Both rotations minimize ``sum((A @ rotation - T) ** 2)``; zero target
    entries are target zeros.

    Parameters
    ----------
    A : ndarray of shape (n_items, n_factors)
        Loadings to rotate, with full column rank.
    T : ndarray of shape (n_items, n_factors)
        Target loadings.
    rotation_type : {"orthogonal", "oblique"}
        ``"orthogonal"`` returns the exact orthogonal Procrustes rotation,
        GPArotation's ``targetT``. ``"oblique"`` is the oblique target
        rotation of Browne (2001), GPArotation's ``targetQ``: rotated
        factors keep unit variance, so the implied factor correlation
        matrix ``inv(rotation) @ inv(rotation).T`` has a unit diagonal. It is
        minimized by gradient projection from the identity, as in
        GPArotation. An unrestricted least-squares transformation, which
        also rescales the factors, is
        :func:`oblique_procrustes_rotation` with ``scaling=False``.
    max_iter : int
        Maximum gradient-projection iterations of the oblique rotation.
    tol : float
        Convergence tolerance on the norm of the projected gradient of the
        oblique rotation (GPArotation's ``eps``).

    Returns
    -------
    rotated : ndarray of shape (n_items, n_factors)
        ``A @ rotation``.
    rotation : ndarray of shape (n_factors, n_factors)
        Rotation matrix.
    """
    if rotation_type not in {"orthogonal", "oblique"}:
        raise ValueError("rotation_type must be 'orthogonal' or 'oblique'")
    _validate_solver_controls(max_iter, tol)
    loadings = _as_finite_matrix(A, "A")
    target = _as_finite_matrix(T, "T")
    if loadings.shape != target.shape:
        raise ValueError(
            f"A and T must have the same shape; got {loadings.shape} and {target.shape}"
        )
    n_factors = loadings.shape[1]
    if loadings.shape[0] < n_factors:
        raise ValueError(
            f"at least {n_factors} rows are required to identify {n_factors} factors"
        )
    if np.linalg.matrix_rank(loadings) < n_factors:
        raise ValueError("A must have full column rank")

    if rotation_type == "orthogonal":
        U, _, Vt = linalg.svd(
            loadings.T @ target, full_matrices=False, check_finite=False
        )
        rotation = U @ Vt
    else:
        from mirt.utils.rotation import rotate_loadings

        try:
            _, rotation, _ = rotate_loadings(
                loadings,
                method="targetQ",
                target=target,
                max_iter=int(max_iter),
                tol=float(tol),
                reflect=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"the fitted oblique rotation is singular or ill-conditioned: {exc}"
            ) from exc
    return loadings @ rotation, rotation


def mirt_linking_summary(result: ProcrustesResult, model_old: BaseItemModel) -> str:
    """Generate a compact text summary of MIRT linking results."""
    lines = [
        "=" * 60,
        "MIRT Linking Summary (Procrustes Rotation)",
        "=" * 60,
        "",
        "Transformation Parameters",
        "-" * 30,
        f"Scaling factor: {result.scaling:.4f}",
        f"Slope RMSE: {result.rmse:.4f}",
    ]
    if result.intercept_rmse is not None:
        lines.append(f"Intercept RMSE: {result.intercept_rmse:.4f}")
    lines.extend(["", "Rotation Matrix R:"])
    for row in result.rotation_matrix:
        lines.append("  " + "  ".join(f"{value:8.4f}" for value in row))
    lines.extend(
        [
            "",
            "Translation Vector t:",
            "  " + "  ".join(f"{value:8.4f}" for value in result.translation),
            "",
        ]
    )

    target_params = _extract_linear_parameters(model_old)
    transformed = _as_finite_matrix(result.transformed_loadings, "transformed_loadings")
    if result.anchor_items_old and result.anchor_items_new:
        target_for_congruence = target_params.slopes[list(result.anchor_items_old)]
        transformed_for_congruence = transformed[list(result.anchor_items_new)]
    elif target_params.slopes.shape[0] == transformed.shape[0]:
        target_for_congruence = target_params.slopes
        transformed_for_congruence = transformed
    else:
        target_for_congruence = None
        transformed_for_congruence = None

    lines.append("Factor Congruence Coefficients:")
    if target_for_congruence is None or transformed_for_congruence is None:
        lines.append("  unavailable (anchor mappings were not retained)")
    else:
        congruence = factor_congruence_coefficient(
            target_for_congruence, transformed_for_congruence
        )
        for index, row in enumerate(congruence):
            formatted = "  ".join(f"{value:6.3f}" for value in row)
            lines.append(f"  Factor {index + 1}: {formatted}")

    lines.extend(
        [
            "",
            "Transformation Equations:",
            "  a_linked = scaling * a_new @ R",
            "  d_linked = d_new - (a_new @ R) @ t",
            "  theta_linked = (theta_new @ inv(R).T + t) / scaling",
            "",
            "=" * 60,
        ]
    )
    return "\n".join(lines)
