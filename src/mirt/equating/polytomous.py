"""Linking methods for unidimensional polytomous IRT models.

The public linkers estimate an affine transformation that places a new
calibration on an old/reference scale::

    theta_old = A * theta_new + B

All implementations share the same transformation direction, validation,
stable category-probability kernels, and positive-scale optimization.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats
from scipy.special import expit, softmax

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.equating.linking import (
    AnchorDiagnostics,
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
)

_ORDERED_METHODS = frozenset({"mean_sigma", "mean_mean", "stocking_lord", "haebara"})
_NRM_METHODS = frozenset({"stocking_lord", "haebara"})


@dataclass
class PolytomousLinkingResult(LinkingResult):
    """Extended linking result for polytomous models.

    Attributes
    ----------
    category_fit : dict[int, float] | None
        Optional category-level fit statistics.
    """

    category_fit: dict[int, float] | None = None


def link_grm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link a new Graded Response Model to an old/reference calibration.

    The returned constants place ``model_new`` on the ``model_old`` scale:
    ``a_old = a_new / A`` and ``threshold_old = A * threshold_new + B``.
    Corresponding anchor items must have the same number of categories.
    """
    if method not in _ORDERED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    old_items, new_items, theta_grid, normalized_weights = _validate_link_inputs(
        model_old,
        model_new,
        anchors_old,
        anchors_new,
        theta_range,
        n_theta,
        weights,
    )
    disc_old = _extract_discrimination(model_old, old_items, "old")
    disc_new = _extract_discrimination(model_new, new_items, "new")
    thresholds_old = _extract_thresholds(model_old, old_items, "old")
    thresholds_new = _extract_thresholds(model_new, new_items, "new")

    A, B, convergence_info = _link_ordered_parameters(
        disc_old,
        thresholds_old,
        disc_new,
        thresholds_new,
        theta_grid,
        normalized_weights,
        method,
        "grm",
    )
    A, B = _validate_linking_constants(A, B, method)

    fit_statistics = None
    anchor_diagnostics = None
    if compute_diagnostics:
        fit_statistics = _compute_grm_fit(
            disc_old,
            thresholds_old,
            disc_new,
            thresholds_new,
            A,
            B,
            theta_grid,
            normalized_weights,
        )
        anchor_diagnostics = _compute_grm_diagnostics(
            disc_old,
            thresholds_old,
            disc_new,
            thresholds_new,
            A,
            B,
            old_items,
            theta_grid,
        )

    return LinkingResult(
        constants=LinkingConstants(A=A, B=B, method=method),
        anchor_items=old_items,
        fit_statistics=fit_statistics,
        anchor_diagnostics=anchor_diagnostics,
        convergence_info=convergence_info,
    )


def link_gpcm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link a new Generalized Partial Credit Model to a reference scale.

    The returned constants place ``model_new`` on the ``model_old`` scale:
    ``a_old = a_new / A`` and ``step_old = A * step_new + B``.
    Corresponding anchor items must have the same number of categories.
    """
    if method not in _ORDERED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    old_items, new_items, theta_grid, normalized_weights = _validate_link_inputs(
        model_old,
        model_new,
        anchors_old,
        anchors_new,
        theta_range,
        n_theta,
        weights,
    )
    disc_old = _extract_discrimination(model_old, old_items, "old")
    disc_new = _extract_discrimination(model_new, new_items, "new")
    steps_old = _extract_steps(model_old, old_items, "old")
    steps_new = _extract_steps(model_new, new_items, "new")

    A, B, convergence_info = _link_ordered_parameters(
        disc_old,
        steps_old,
        disc_new,
        steps_new,
        theta_grid,
        normalized_weights,
        method,
        "gpcm",
    )
    A, B = _validate_linking_constants(A, B, method)

    fit_statistics = None
    anchor_diagnostics = None
    if compute_diagnostics:
        fit_statistics = _compute_gpcm_fit(
            disc_old,
            steps_old,
            disc_new,
            steps_new,
            A,
            B,
            theta_grid,
            normalized_weights,
        )
        anchor_diagnostics = _compute_gpcm_diagnostics(
            disc_old,
            steps_old,
            disc_new,
            steps_new,
            A,
            B,
            old_items,
            theta_grid,
        )

    return LinkingResult(
        constants=LinkingConstants(A=A, B=B, method=method),
        anchor_items=old_items,
        fit_statistics=fit_statistics,
        anchor_diagnostics=anchor_diagnostics,
        convergence_info=convergence_info,
    )


def link_nrm(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
    method: Literal["stocking_lord", "haebara"] = "stocking_lord",
    compute_diagnostics: bool = True,
) -> LinkingResult:
    """Link a new Nominal Response Model to an old/reference calibration.

    NRM category logits transform as ``slope_old = slope_new / A`` and
    ``intercept_old = intercept_new - slope_new * B / A``. Both expected-score
    (Stocking-Lord) and category-curve (Haebara) objectives are available.
    """
    if method not in _NRM_METHODS:
        raise ValueError(f"Unknown method: {method}")
    old_items, new_items, theta_grid, normalized_weights = _validate_link_inputs(
        model_old,
        model_new,
        anchors_old,
        anchors_new,
        theta_range,
        n_theta,
        weights,
    )
    slopes_old, intercepts_old = _extract_nrm_parameters(model_old, old_items, "old")
    slopes_new, intercepts_new = _extract_nrm_parameters(model_new, new_items, "new")

    A, B, convergence_info = _link_nrm_parameters(
        slopes_old,
        intercepts_old,
        slopes_new,
        intercepts_new,
        theta_grid,
        normalized_weights,
        method,
    )
    A, B = _validate_linking_constants(A, B, method)

    fit_statistics = None
    anchor_diagnostics = None
    if compute_diagnostics:
        fit_statistics = _compute_nrm_fit(
            slopes_old,
            intercepts_old,
            slopes_new,
            intercepts_new,
            A,
            B,
            theta_grid,
            normalized_weights,
        )
        anchor_diagnostics = _compute_nrm_diagnostics(
            slopes_old,
            intercepts_old,
            slopes_new,
            intercepts_new,
            A,
            B,
            old_items,
            theta_grid,
        )

    return LinkingResult(
        constants=LinkingConstants(A=A, B=B, method=method),
        anchor_items=old_items,
        fit_statistics=fit_statistics,
        anchor_diagnostics=anchor_diagnostics,
        convergence_info=convergence_info,
    )


def transform_polytomous_parameters(
    model: "BaseItemModel",
    A: float,
    B: float,
    model_type: Literal["grm", "gpcm", "nrm"] = "grm",
    in_place: bool = False,
) -> "BaseItemModel":
    """Place polytomous item parameters on ``A * theta + B``.

    Discriminations and NRM slopes divide by ``A``. GRM thresholds and GPCM
    steps become ``A * location + B``. NRM intercepts become
    ``intercept - slope * B / A``.
    """
    scale, shift = _validate_transform_constants(A, B)
    if model_type not in {"grm", "gpcm", "nrm"}:
        raise ValueError(f"Unknown polytomous model type: {model_type}")
    if model.n_factors != 1:
        raise ValueError("Polytomous transformation requires a unidimensional model")

    transformed = model if in_place else model.copy()
    parameters = transformed.parameters

    if model_type == "grm":
        discrimination = _require_parameter(parameters, ("discrimination",), "GRM")
        location_name, thresholds = _require_named_parameter(
            parameters, ("thresholds", "difficulty"), "GRM"
        )
        transformed.set_parameters(
            discrimination=discrimination / scale,
            **{
                location_name: _transform_ordered_storage(
                    thresholds, transformed.n_categories, scale, shift
                )
            },
        )
    elif model_type == "gpcm":
        discrimination = _require_parameter(parameters, ("discrimination",), "GPCM")
        location_name, steps = _require_named_parameter(
            parameters, ("steps", "step_parameters"), "GPCM"
        )
        transformed.set_parameters(
            discrimination=discrimination / scale,
            **{
                location_name: _transform_ordered_storage(
                    steps, transformed.n_categories, scale, shift
                )
            },
        )
    else:
        slopes = _require_parameter(parameters, ("slopes",), "NRM")
        intercepts = _require_parameter(parameters, ("intercepts",), "NRM")
        new_slopes = slopes.copy()
        new_intercepts = intercepts.copy()
        for item, n_categories in enumerate(transformed.n_categories):
            new_slopes[item, :n_categories] = slopes[item, :n_categories] / scale
            new_intercepts[item, :n_categories] = (
                intercepts[item, :n_categories]
                - slopes[item, :n_categories] * shift / scale
            )
        transformed.set_parameters(
            slopes=new_slopes,
            intercepts=new_intercepts,
        )

    return transformed


def _transform_ordered_storage(
    values: NDArray[np.float64],
    category_counts: list[int],
    A: float,
    B: float,
) -> NDArray[np.float64]:
    """Transform active category locations while preserving padding."""
    if values.ndim == 1 and values.shape == (len(category_counts),):
        if any(n_categories != 2 for n_categories in category_counts):
            raise ValueError("Ordered category parameters have an invalid shape")
        return A * values + B
    if values.ndim != 2 or values.shape[0] != len(category_counts):
        raise ValueError("Ordered category parameters have an invalid shape")
    transformed = values.copy()
    for item, n_categories in enumerate(category_counts):
        count = n_categories - 1
        transformed[item, :count] = A * values[item, :count] + B
    return transformed


def _validate_link_inputs(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchors_old: list[int],
    anchors_new: list[int],
    theta_range: tuple[float, float],
    n_theta: int,
    weights: NDArray[np.float64] | None,
) -> tuple[list[int], list[int], NDArray[np.float64], NDArray[np.float64]]:
    """Validate paired anchors and construct a normalized integration grid."""
    if model_old.n_factors != 1 or model_new.n_factors != 1:
        raise ValueError("Polytomous linking requires unidimensional models")
    if not model_old.is_polytomous or not model_new.is_polytomous:
        raise ValueError("Polytomous linking requires polytomous item models")
    if len(anchors_old) != len(anchors_new):
        raise ValueError("Anchor lists must have same length")
    if len(anchors_old) < 2:
        raise ValueError("At least 2 anchor items are required")

    normalized: list[list[int]] = []
    for label, anchors, n_items in (
        ("old", anchors_old, model_old.n_items),
        ("new", anchors_new, model_new.n_items),
    ):
        current: list[int] = []
        for anchor in anchors:
            if isinstance(anchor, (bool, np.bool_)) or not isinstance(
                anchor, (int, np.integer)
            ):
                raise ValueError(
                    f"Anchor indices for the {label} model must be integers"
                )
            index = int(anchor)
            if index < 0 or index >= n_items:
                raise ValueError(
                    f"Anchor index {index} out of range for the {label} model "
                    f"with {n_items} items"
                )
            current.append(index)
        if len(set(current)) != len(current):
            raise ValueError(f"Anchor indices for the {label} model must be unique")
        normalized.append(current)

    old_items, new_items = normalized
    old_categories = model_old.n_categories
    new_categories = model_new.n_categories
    for old_item, new_item in zip(old_items, new_items, strict=True):
        if old_categories[old_item] != new_categories[new_item]:
            raise ValueError(
                "Corresponding anchor items must have the same number of categories"
            )

    theta_grid, normalized_weights = _validate_curve_grid(theta_range, n_theta, weights)
    return old_items, new_items, theta_grid, normalized_weights


def _validate_curve_grid(
    theta_range: tuple[float, float],
    n_theta: int,
    weights: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate integration settings and normalize weights."""
    if not isinstance(theta_range, (tuple, list)) or len(theta_range) != 2:
        raise ValueError("theta_range must contain exactly two values")
    lower, upper = float(theta_range[0]), float(theta_range[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite, increasing values")
    if isinstance(n_theta, (bool, np.bool_)) or not isinstance(
        n_theta, (int, np.integer)
    ):
        raise ValueError("n_theta must be an integer")
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")

    theta_grid = np.linspace(lower, upper, int(n_theta))
    if weights is None:
        normalized_weights = stats.norm.pdf(theta_grid)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (n_theta,):
            raise ValueError(f"weights must have shape ({n_theta},)")
        if not np.all(np.isfinite(normalized_weights)):
            raise ValueError("weights must be finite")
        if np.any(normalized_weights < 0.0):
            raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(normalized_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("weights must have a positive sum")
    return theta_grid, normalized_weights / weight_sum


def _validate_transform_constants(A: float, B: float) -> tuple[float, float]:
    """Require a finite, orientation-preserving transformation."""
    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A must be finite and positive")
    if not np.isfinite(shift):
        raise ValueError("B must be finite")
    return scale, shift


def _validate_linking_constants(A: float, B: float, method: str) -> tuple[float, float]:
    """Require finite constants from a linking method."""
    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"{method} linking did not produce a positive finite slope")
    if not np.isfinite(shift):
        raise RuntimeError(f"{method} linking did not produce a finite intercept")
    return scale, shift


def _require_parameter(
    parameters: dict[str, NDArray[np.float64]], names: tuple[str, ...], label: str
) -> NDArray[np.float64]:
    """Return the first named parameter or fail instead of inventing defaults."""
    _, values = _require_named_parameter(parameters, names, label)
    return values


def _require_named_parameter(
    parameters: dict[str, NDArray[np.float64]], names: tuple[str, ...], label: str
) -> tuple[str, NDArray[np.float64]]:
    """Return a parameter's storage name and validated finite values."""
    for name in names:
        if name in parameters:
            values = np.asarray(parameters[name], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label} parameters must be finite")
            return name, values
    expected = " or ".join(names)
    raise ValueError(f"{label} model is missing required {expected} parameters")


def _extract_discrimination(
    model: "BaseItemModel", items: list[int], label: str
) -> NDArray[np.float64]:
    """Extract finite positive scalar discriminations."""
    try:
        discrimination = np.asarray(model.discrimination, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(f"The {label} model has no discrimination parameters") from exc
    if discrimination.ndim == 2 and discrimination.shape == (model.n_items, 1):
        discrimination = discrimination[:, 0]
    if discrimination.shape != (model.n_items,):
        raise ValueError(
            f"Discrimination parameters for the {label} model must contain one value per item"
        )
    selected = discrimination[items]
    if not np.all(np.isfinite(selected)):
        raise ValueError(
            f"Discrimination parameters for the {label} model must be finite"
        )
    if np.any(selected <= 0.0):
        raise ValueError(
            f"Discrimination parameters for the {label} model must be positive"
        )
    return selected


def _extract_thresholds(
    model: "BaseItemModel", items: list[int], label: str = "model"
) -> list[NDArray[np.float64]]:
    """Extract actual GRM thresholds, excluding padded storage columns."""
    return _extract_ordered_parameters(
        model,
        items,
        ("thresholds", "difficulty"),
        label,
        "threshold",
        require_increasing=True,
    )


def _extract_steps(
    model: "BaseItemModel", items: list[int], label: str = "model"
) -> list[NDArray[np.float64]]:
    """Extract actual GPCM steps, excluding padded storage columns."""
    return _extract_ordered_parameters(
        model,
        items,
        ("steps", "step_parameters"),
        label,
        "step",
        require_increasing=False,
    )


def _extract_ordered_parameters(
    model: "BaseItemModel",
    items: list[int],
    names: tuple[str, ...],
    label: str,
    parameter_label: str,
    require_increasing: bool,
) -> list[NDArray[np.float64]]:
    """Extract item-specific category locations from dense model storage."""
    values = _require_parameter(model.parameters, names, label.upper())
    category_counts = model.n_categories
    if values.ndim == 1:
        if values.shape != (model.n_items,) or any(
            category_counts[item] != 2 for item in items
        ):
            raise ValueError(
                f"{parameter_label.title()} parameters for the {label} model have an invalid shape"
            )
        result = [np.array([values[item]], dtype=np.float64) for item in items]
    elif values.ndim == 2 and values.shape[0] == model.n_items:
        result = []
        for item in items:
            count = category_counts[item] - 1
            if values.shape[1] < count:
                raise ValueError(
                    f"{parameter_label.title()} parameters for the {label} model "
                    f"do not cover item {item}'s categories"
                )
            result.append(np.asarray(values[item, :count], dtype=np.float64).copy())
    else:
        raise ValueError(
            f"{parameter_label.title()} parameters for the {label} model have an invalid shape"
        )

    if not all(np.all(np.isfinite(values_item)) for values_item in result):
        raise ValueError(
            f"{parameter_label.title()} parameters for the {label} model must be finite"
        )
    if require_increasing and any(
        np.any(np.diff(values_item) <= 0.0) for values_item in result
    ):
        raise ValueError(
            f"Threshold parameters for the {label} model must be strictly increasing"
        )
    return result


def _extract_nrm_parameters(
    model: "BaseItemModel", items: list[int], label: str
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Extract actual NRM categories and reject unidentifiable scale data."""
    slopes = _require_parameter(model.parameters, ("slopes",), "NRM")
    intercepts = _require_parameter(model.parameters, ("intercepts",), "NRM")
    if (
        slopes.ndim != 2
        or intercepts.ndim != 2
        or slopes.shape[0] != model.n_items
        or intercepts.shape[0] != model.n_items
    ):
        raise ValueError(f"NRM parameters for the {label} model have an invalid shape")

    result_slopes: list[NDArray[np.float64]] = []
    result_intercepts: list[NDArray[np.float64]] = []
    for item in items:
        count = model.n_categories[item]
        if slopes.shape[1] < count or intercepts.shape[1] < count:
            raise ValueError(f"NRM parameters for the {label} model are incomplete")
        result_slopes.append(slopes[item, :count].copy())
        result_intercepts.append(intercepts[item, :count].copy())

    centered = np.concatenate([values - np.mean(values) for values in result_slopes])
    if np.linalg.norm(centered) < 1e-10:
        raise ValueError(f"NRM slopes for the {label} model do not identify a scale")
    return result_slopes, result_intercepts


def _mean_sigma_polytomous(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Match the location mean and standard deviation."""
    scale_old = float(np.std(diff_old, ddof=1))
    scale_new = float(np.std(diff_new, ddof=1))
    if scale_old < 1e-10 or scale_new < 1e-10:
        A = float(np.mean(disc_new) / np.mean(disc_old))
    else:
        A = scale_old / scale_new
    B = float(np.mean(diff_old) - A * np.mean(diff_new))
    return A, B, {"method": "mean_sigma"}


def _mean_mean_polytomous(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Match mean discrimination and mean location."""
    A = float(np.mean(disc_new) / np.mean(disc_old))
    B = float(np.mean(diff_old) - A * np.mean(diff_new))
    return A, B, {"method": "mean_mean"}


def _grm_category_probs(
    theta: NDArray[np.float64],
    disc: float,
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute stable GRM category probabilities for one item."""
    result = _grm_category_probs_batch(
        np.asarray(theta, dtype=np.float64),
        np.array([disc], dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64)[None, :],
    )
    return result[:, 0, :]


def _grm_category_probs_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute GRM curves for equally sized items in one batch."""
    cumulative = expit(
        discrimination[None, :, None] * (theta[:, None, None] - thresholds[None, :, :])
    )
    n_categories = thresholds.shape[1] + 1
    probabilities = np.empty(
        (theta.size, discrimination.size, n_categories), dtype=np.float64
    )
    probabilities[:, :, 0] = 1.0 - cumulative[:, :, 0]
    if n_categories > 2:
        probabilities[:, :, 1:-1] = cumulative[:, :, :-1] - cumulative[:, :, 1:]
    probabilities[:, :, -1] = cumulative[:, :, -1]
    return probabilities


def _gpcm_category_probs(
    theta: NDArray[np.float64],
    disc: float,
    steps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute stable GPCM category probabilities for one item."""
    result = _gpcm_category_probs_batch(
        np.asarray(theta, dtype=np.float64),
        np.array([disc], dtype=np.float64),
        np.asarray(steps, dtype=np.float64)[None, :],
    )
    return result[:, 0, :]


def _gpcm_category_probs_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    steps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute GPCM curves for equally sized items in one stable batch."""
    cumulative_steps = np.concatenate(
        (np.zeros((steps.shape[0], 1)), np.cumsum(steps, axis=1)), axis=1
    )
    categories = np.arange(steps.shape[1] + 1, dtype=np.float64)
    logits = discrimination[None, :, None] * (
        theta[:, None, None] * categories[None, None, :] - cumulative_steps[None, :, :]
    )
    return softmax(logits, axis=2)


def _nrm_category_probs(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute stable NRM category probabilities for one item."""
    logits = theta[:, None] * slopes[None, :] + intercepts[None, :]
    return softmax(logits, axis=1)


def _nrm_category_probs_batch(
    theta: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute NRM curves for equally sized items in one stable batch."""
    logits = theta[:, None, None] * slopes[None, :, :] + intercepts[None, :, :]
    return softmax(logits, axis=2)


def _group_indices(
    parameters: list[NDArray[np.float64]], max_batch_size: int = 2048
) -> list[NDArray[np.int_]]:
    """Group item positions by width in memory-bounded evaluation batches."""
    groups: dict[int, list[int]] = {}
    for index, values in enumerate(parameters):
        groups.setdefault(len(values), []).append(index)
    return [
        np.asarray(indices[start : start + max_batch_size], dtype=np.int64)
        for indices in groups.values()
        for start in range(0, len(indices), max_batch_size)
    ]


def _ordered_probability_batch(
    model_type: Literal["grm", "gpcm"],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    locations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dispatch an ordered-model batch without duplicating objectives."""
    if model_type == "grm":
        return _grm_category_probs_batch(theta, discrimination, locations)
    return _gpcm_category_probs_batch(theta, discrimination, locations)


def _link_ordered_parameters(
    disc_old: NDArray[np.float64],
    locations_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    locations_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    method: str,
    model_type: Literal["grm", "gpcm"],
) -> tuple[float, float, dict]:
    """Dispatch moment or curve linking for an ordered model."""
    centers_old = np.array([np.mean(values) for values in locations_old])
    centers_new = np.array([np.mean(values) for values in locations_new])
    if method == "mean_sigma":
        return _mean_sigma_polytomous(disc_old, centers_old, disc_new, centers_new)
    if method == "mean_mean":
        return _mean_mean_polytomous(disc_old, centers_old, disc_new, centers_new)
    return _curve_link_ordered(
        disc_old,
        locations_old,
        disc_new,
        locations_new,
        theta_grid,
        weights,
        method,
        model_type,
    )


def _curve_link_ordered(
    disc_old: NDArray[np.float64],
    locations_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    locations_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    method: str,
    model_type: Literal["grm", "gpcm"],
) -> tuple[float, float, dict]:
    """Fit a positive affine transformation with vectorized curve batches."""
    groups = []
    tcc_old = np.zeros(theta_grid.size)
    for indices in _group_indices(locations_old):
        old_locations = np.stack([locations_old[index] for index in indices])
        new_locations = np.stack([locations_new[index] for index in indices])
        curves_old = _ordered_probability_batch(
            model_type, theta_grid, disc_old[indices], old_locations
        )
        scores = np.arange(curves_old.shape[2], dtype=np.float64)
        tcc_old += np.sum(curves_old * scores[None, None, :], axis=(1, 2))
        groups.append((indices, new_locations, curves_old, scores))

    centers_old = np.array([np.mean(values) for values in locations_old])
    centers_new = np.array([np.mean(values) for values in locations_new])
    initial_A, initial_B, _ = _mean_sigma_polytomous(
        disc_old, centers_old, disc_new, centers_new
    )

    def criterion(parameters: NDArray[np.float64]) -> float:
        log_A, B = float(parameters[0]), float(parameters[1])
        if not np.isfinite(log_A) or not np.isfinite(B) or abs(log_A) > 20.0:
            return float("inf")
        A = float(np.exp(log_A))
        if method == "stocking_lord":
            tcc_new = np.zeros(theta_grid.size)
            for indices, new_locations, _, scores in groups:
                curves_new = _ordered_probability_batch(
                    model_type,
                    theta_grid,
                    disc_new[indices] / A,
                    A * new_locations + B,
                )
                tcc_new += np.sum(curves_new * scores[None, None, :], axis=(1, 2))
            return float(np.sum(weights * (tcc_old - tcc_new) ** 2))

        total = 0.0
        for indices, new_locations, curves_old, _ in groups:
            curves_new = _ordered_probability_batch(
                model_type,
                theta_grid,
                disc_new[indices] / A,
                A * new_locations + B,
            )
            total += float(
                np.sum(weights[:, None, None] * (curves_old - curves_new) ** 2)
            )
        return total

    return _minimize_positive_scale(criterion, initial_A, initial_B, method)


def _stocking_lord_grm(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord expected-score matching for GRM."""
    return _curve_link_ordered(
        disc_old,
        thresholds_old,
        disc_new,
        thresholds_new,
        theta_grid,
        weights,
        "stocking_lord",
        "grm",
    )


def _haebara_grm(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Haebara category-curve matching for GRM."""
    return _curve_link_ordered(
        disc_old,
        thresholds_old,
        disc_new,
        thresholds_new,
        theta_grid,
        weights,
        "haebara",
        "grm",
    )


def _stocking_lord_gpcm(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord expected-score matching for GPCM."""
    return _curve_link_ordered(
        disc_old,
        steps_old,
        disc_new,
        steps_new,
        theta_grid,
        weights,
        "stocking_lord",
        "gpcm",
    )


def _haebara_gpcm(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Haebara category-curve matching for GPCM."""
    return _curve_link_ordered(
        disc_old,
        steps_old,
        disc_new,
        steps_new,
        theta_grid,
        weights,
        "haebara",
        "gpcm",
    )


def _minimize_positive_scale(
    criterion: Callable[[NDArray[np.float64]], float],
    initial_A: float,
    initial_B: float,
    method: str,
) -> tuple[float, float, dict]:
    """Minimize a curve objective in log-scale coordinates."""
    initial_A, initial_B = _validate_linking_constants(
        initial_A, initial_B, f"{method} initialization"
    )
    initial = np.array([np.log(initial_A), initial_B], dtype=np.float64)
    initial_value = float(criterion(initial))
    if not np.isfinite(initial_value):
        raise RuntimeError(f"{method} linking objective is not finite")
    if initial_value <= 1e-14:
        return (
            initial_A,
            initial_B,
            {
                "method": method,
                "success": True,
                "fun": initial_value,
                "nit": 0,
            },
        )

    result = optimize.minimize(
        criterion,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"{method} linking failed to converge: {result.message}")
    A, B = float(np.exp(result.x[0])), float(result.x[1])
    A, B = _validate_linking_constants(A, B, method)
    return (
        A,
        B,
        {
            "method": method,
            "success": True,
            "fun": float(result.fun),
            "nit": int(result.nit),
        },
    )


def _nrm_initial_constants(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
) -> tuple[float, float]:
    """Estimate NRM constants from category-centered parameters."""
    centered_slopes_old = np.concatenate(
        [values - np.mean(values) for values in slopes_old]
    )
    centered_slopes_new = np.concatenate(
        [values - np.mean(values) for values in slopes_new]
    )
    centered_intercepts_old = np.concatenate(
        [values - np.mean(values) for values in intercepts_old]
    )
    centered_intercepts_new = np.concatenate(
        [values - np.mean(values) for values in intercepts_new]
    )
    old_norm = float(np.dot(centered_slopes_old, centered_slopes_old))
    new_norm = float(np.dot(centered_slopes_new, centered_slopes_new))
    if old_norm < 1e-20 or new_norm < 1e-20:
        raise ValueError("NRM slopes do not identify a linking scale")
    A = float(np.sqrt(new_norm / old_norm))
    intercept_difference = centered_intercepts_old - centered_intercepts_new
    B = float(
        -A
        * np.dot(centered_slopes_new, intercept_difference)
        / np.dot(centered_slopes_new, centered_slopes_new)
    )
    return A, B


def _link_nrm_parameters(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    method: str,
) -> tuple[float, float, dict]:
    """Fit an NRM transformation with stable grouped curve batches."""
    groups = []
    tcc_old = np.zeros(theta_grid.size)
    for indices in _group_indices(slopes_old):
        old_slopes = np.stack([slopes_old[index] for index in indices])
        old_intercepts = np.stack([intercepts_old[index] for index in indices])
        new_slopes = np.stack([slopes_new[index] for index in indices])
        new_intercepts = np.stack([intercepts_new[index] for index in indices])
        curves_old = _nrm_category_probs_batch(theta_grid, old_slopes, old_intercepts)
        scores = np.arange(curves_old.shape[2], dtype=np.float64)
        tcc_old += np.sum(curves_old * scores[None, None, :], axis=(1, 2))
        groups.append((new_slopes, new_intercepts, curves_old, scores))

    initial_A, initial_B = _nrm_initial_constants(
        slopes_old, intercepts_old, slopes_new, intercepts_new
    )

    def criterion(parameters: NDArray[np.float64]) -> float:
        log_A, B = float(parameters[0]), float(parameters[1])
        if not np.isfinite(log_A) or not np.isfinite(B) or abs(log_A) > 20.0:
            return float("inf")
        A = float(np.exp(log_A))
        if method == "stocking_lord":
            tcc_new = np.zeros(theta_grid.size)
            for new_slopes, new_intercepts, _, scores in groups:
                curves_new = _nrm_category_probs_batch(
                    theta_grid,
                    new_slopes / A,
                    new_intercepts - new_slopes * B / A,
                )
                tcc_new += np.sum(curves_new * scores[None, None, :], axis=(1, 2))
            return float(np.sum(weights * (tcc_old - tcc_new) ** 2))

        total = 0.0
        for new_slopes, new_intercepts, curves_old, _ in groups:
            curves_new = _nrm_category_probs_batch(
                theta_grid,
                new_slopes / A,
                new_intercepts - new_slopes * B / A,
            )
            total += float(
                np.sum(weights[:, None, None] * (curves_old - curves_new) ** 2)
            )
        return total

    return _minimize_positive_scale(criterion, initial_A, initial_B, method)


def _stocking_lord_nrm(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float, dict]:
    """Stocking-Lord expected-score matching for NRM."""
    return _link_nrm_parameters(
        slopes_old,
        intercepts_old,
        slopes_new,
        intercepts_new,
        theta_grid,
        weights,
        "stocking_lord",
    )


def _ordered_curve_summaries(
    disc_old: NDArray[np.float64],
    locations_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    locations_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    model_type: Literal["grm", "gpcm"],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected-score curves and per-item category-curve areas."""
    tcc_old = np.zeros(theta_grid.size)
    tcc_new = np.zeros(theta_grid.size)
    areas = np.zeros(len(disc_old))
    for indices in _group_indices(locations_old):
        old_locations = np.stack([locations_old[index] for index in indices])
        new_locations = np.stack([locations_new[index] for index in indices])
        curves_old = _ordered_probability_batch(
            model_type, theta_grid, disc_old[indices], old_locations
        )
        curves_new = _ordered_probability_batch(
            model_type,
            theta_grid,
            disc_new[indices] / A,
            A * new_locations + B,
        )
        scores = np.arange(curves_old.shape[2], dtype=np.float64)
        tcc_old += np.sum(curves_old * scores[None, None, :], axis=(1, 2))
        tcc_new += np.sum(curves_new * scores[None, None, :], axis=(1, 2))
        areas[indices] = np.trapezoid(
            np.sum(np.abs(curves_old - curves_new), axis=2),
            theta_grid,
            axis=0,
        )
    return tcc_old, tcc_new, areas


def _compute_ordered_fit(
    disc_old: NDArray[np.float64],
    locations_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    locations_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    model_type: Literal["grm", "gpcm"],
) -> LinkingFitStatistics:
    """Compute parameter and expected-score fit for an ordered model."""
    diff_a = disc_old - disc_new / A
    diff_locations = np.concatenate(
        [
            old_values - (A * new_values + B)
            for old_values, new_values in zip(locations_old, locations_new, strict=True)
        ]
    )
    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    rmse_b = float(np.sqrt(np.mean(diff_locations**2)))
    mad_a = float(np.mean(np.abs(diff_a)))
    mad_b = float(np.mean(np.abs(diff_locations)))
    tcc_old, tcc_new, _ = _ordered_curve_summaries(
        disc_old,
        locations_old,
        disc_new,
        locations_new,
        A,
        B,
        theta_grid,
        model_type,
    )
    tcc_rmse = float(np.sqrt(np.sum(weights * (tcc_old - tcc_new) ** 2)))
    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=mad_a,
        mad_b=mad_b,
        weighted_rmse=float(np.hypot(rmse_a, rmse_b)),
        tcc_rmse=tcc_rmse,
    )


def _compute_grm_fit(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute fit statistics for GRM linking."""
    return _compute_ordered_fit(
        disc_old,
        thresholds_old,
        disc_new,
        thresholds_new,
        A,
        B,
        theta_grid,
        weights,
        "grm",
    )


def _compute_gpcm_fit(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute fit statistics for GPCM linking."""
    return _compute_ordered_fit(
        disc_old,
        steps_old,
        disc_new,
        steps_new,
        A,
        B,
        theta_grid,
        weights,
        "gpcm",
    )


def _robust_z_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute MAD-based scores while retaining isolated tied outliers."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    if mad >= 1e-10:
        return (values - median) / mad
    matches = np.isclose(values, median, rtol=1e-8, atol=1e-10)
    scores = np.zeros_like(values)
    differences = values[~matches] - median
    scores[~matches] = np.copysign(np.inf, differences)
    return scores


def _compute_ordered_diagnostics(
    disc_old: NDArray[np.float64],
    locations_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    locations_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
    model_type: Literal["grm", "gpcm"],
) -> AnchorDiagnostics:
    """Compute per-anchor parameter and full-category curve diagnostics."""
    signed_diff_a = disc_old - disc_new / A
    signed_diff_b = np.array(
        [
            np.mean(old_values - (A * new_values + B))
            for old_values, new_values in zip(locations_old, locations_new, strict=True)
        ]
    )
    _, _, area_diff = _ordered_curve_summaries(
        disc_old,
        locations_old,
        disc_new,
        locations_new,
        A,
        B,
        theta_grid,
        model_type,
    )
    combined = np.sqrt(signed_diff_a**2 + signed_diff_b**2 + area_diff**2)
    robust_z = _robust_z_scores(combined)
    return AnchorDiagnostics(
        item_indices=anchor_indices,
        signed_diff_a=signed_diff_a,
        signed_diff_b=signed_diff_b,
        area_diff=area_diff,
        robust_z=robust_z,
        flagged=np.abs(robust_z) > 2.5,
    )


def _compute_grm_diagnostics(
    disc_old: NDArray[np.float64],
    thresholds_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    thresholds_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
) -> AnchorDiagnostics:
    """Compute anchor diagnostics for GRM linking."""
    return _compute_ordered_diagnostics(
        disc_old,
        thresholds_old,
        disc_new,
        thresholds_new,
        A,
        B,
        anchor_indices,
        theta_grid,
        "grm",
    )


def _compute_gpcm_diagnostics(
    disc_old: NDArray[np.float64],
    steps_old: list[NDArray[np.float64]],
    disc_new: NDArray[np.float64],
    steps_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
) -> AnchorDiagnostics:
    """Compute anchor diagnostics for GPCM linking."""
    return _compute_ordered_diagnostics(
        disc_old,
        steps_old,
        disc_new,
        steps_new,
        A,
        B,
        anchor_indices,
        theta_grid,
        "gpcm",
    )


def _nrm_curve_summaries(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute NRM expected-score curves and item areas."""
    tcc_old = np.zeros(theta_grid.size)
    tcc_new = np.zeros(theta_grid.size)
    areas = np.zeros(len(slopes_old))
    for indices in _group_indices(slopes_old):
        old_slopes = np.stack([slopes_old[index] for index in indices])
        old_intercepts = np.stack([intercepts_old[index] for index in indices])
        new_slopes = np.stack([slopes_new[index] for index in indices])
        new_intercepts = np.stack([intercepts_new[index] for index in indices])
        curves_old = _nrm_category_probs_batch(theta_grid, old_slopes, old_intercepts)
        curves_new = _nrm_category_probs_batch(
            theta_grid,
            new_slopes / A,
            new_intercepts - new_slopes * B / A,
        )
        scores = np.arange(curves_old.shape[2], dtype=np.float64)
        tcc_old += np.sum(curves_old * scores[None, None, :], axis=(1, 2))
        tcc_new += np.sum(curves_new * scores[None, None, :], axis=(1, 2))
        areas[indices] = np.trapezoid(
            np.sum(np.abs(curves_old - curves_new), axis=2),
            theta_grid,
            axis=0,
        )
    return tcc_old, tcc_new, areas


def _centered_flattened(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Flatten category parameters after removing irrelevant item constants."""
    return np.concatenate([item - np.mean(item) for item in values])


def _compute_nrm_fit(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> LinkingFitStatistics:
    """Compute identification-invariant NRM fit statistics."""
    slopes_transformed = [values / A for values in slopes_new]
    intercepts_transformed = [
        intercept - slope * B / A
        for slope, intercept in zip(slopes_new, intercepts_new, strict=True)
    ]
    diff_a = _centered_flattened(slopes_old) - _centered_flattened(slopes_transformed)
    diff_b = _centered_flattened(intercepts_old) - _centered_flattened(
        intercepts_transformed
    )
    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))
    tcc_old, tcc_new, _ = _nrm_curve_summaries(
        slopes_old,
        intercepts_old,
        slopes_new,
        intercepts_new,
        A,
        B,
        theta_grid,
    )
    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=float(np.mean(np.abs(diff_a))),
        mad_b=float(np.mean(np.abs(diff_b))),
        weighted_rmse=float(np.hypot(rmse_a, rmse_b)),
        tcc_rmse=float(np.sqrt(np.sum(weights * (tcc_old - tcc_new) ** 2))),
    )


def _compute_nrm_diagnostics(
    slopes_old: list[NDArray[np.float64]],
    intercepts_old: list[NDArray[np.float64]],
    slopes_new: list[NDArray[np.float64]],
    intercepts_new: list[NDArray[np.float64]],
    A: float,
    B: float,
    anchor_indices: list[int],
    theta_grid: NDArray[np.float64],
) -> AnchorDiagnostics:
    """Compute category-contrast and curve diagnostics for NRM anchors."""
    slopes_transformed = [values / A for values in slopes_new]
    intercepts_transformed = [
        intercept - slope * B / A
        for slope, intercept in zip(slopes_new, intercepts_new, strict=True)
    ]
    signed_diff_a = np.array(
        [
            np.mean((old - old[0]) - (new - new[0]))
            for old, new in zip(slopes_old, slopes_transformed, strict=True)
        ]
    )
    signed_diff_b = np.array(
        [
            np.mean((old - old[0]) - (new - new[0]))
            for old, new in zip(intercepts_old, intercepts_transformed, strict=True)
        ]
    )
    _, _, area_diff = _nrm_curve_summaries(
        slopes_old,
        intercepts_old,
        slopes_new,
        intercepts_new,
        A,
        B,
        theta_grid,
    )
    combined = np.sqrt(signed_diff_a**2 + signed_diff_b**2 + area_diff**2)
    robust_z = _robust_z_scores(combined)
    return AnchorDiagnostics(
        item_indices=anchor_indices,
        signed_diff_a=signed_diff_a,
        signed_diff_b=signed_diff_b,
        area_diff=area_diff,
        robust_z=robust_z,
        flagged=np.abs(robust_z) > 2.5,
    )
