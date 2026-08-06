"""Diagnostics and uncertainty estimates for dichotomous IRT linking."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats
from scipy.special import expit

from mirt.equating.linking import (
    LinkingFitStatistics,
    LinkingResult,
    _bisector_link,
    _orthogonal_link,
    link,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_SUPPORTED_METHODS = frozenset(
    {
        "mean_sigma",
        "mean_mean",
        "stocking_lord",
        "haebara",
        "tcc",
        "bisector",
        "orthogonal",
    }
)
_LOGISTIC_MODEL_NAMES = frozenset({"1PL", "2PL", "3PL", "4PL", "5PL"})

RefitCallback = Callable[["BaseItemModel", NDArray[np.float64]], "BaseItemModel"]


def bootstrap_linking_se(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    responses_old: NDArray[np.float64] | None,
    responses_new: NDArray[np.float64] | None,
    anchors_old: list[int],
    anchors_new: list[int],
    method: str = "stocking_lord",
    n_bootstrap: int = 200,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    seed: int | np.random.Generator | None = None,
    refit: RefitCallback | None = None,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute bootstrap standard errors for linking constants.

    With both response matrices omitted, this performs a paired anchor-item
    bootstrap. With both response matrices supplied, persons are resampled
    independently within each form and ``refit`` recalibrates each sampled
    form before linking. Recalibration is explicit because fit settings are
    study-specific and cannot be inferred safely from fitted model objects.

    Parameters
    ----------
    model_old, model_new : BaseItemModel
        Old/reference and new calibrations.
    responses_old, responses_new : ndarray or None
        Response matrices. Supply both or neither.
    anchors_old, anchors_new : list[int]
        Corresponding anchor indices.
    method : str
        Any public dichotomous linking method.
    n_bootstrap : int
        Number of replicates; must be at least two.
    theta_range, n_theta
        Integration grid for curve methods.
    seed : int, numpy.random.Generator, or None
        Random source for reproducible resampling.
    refit : callable or None
        ``refit(model, sampled_responses) -> fitted_model``. Required when
        response matrices are supplied.

    Returns
    -------
    tuple
        Standard errors for A and B followed by the replicate samples.
    """
    _validate_method(method)
    _validate_bootstrap_count(n_bootstrap)
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    response_matrices = _validate_bootstrap_responses(
        model_old, model_new, responses_old, responses_new, refit
    )
    rng = np.random.default_rng(seed)

    parameters_old = _extract_link_parameters(model_old, anchors_old, "old")
    parameters_new = _extract_link_parameters(model_new, anchors_new, "new")
    n_anchors = len(anchors_old)
    A_samples = np.empty(n_bootstrap, dtype=np.float64)
    B_samples = np.empty(n_bootstrap, dtype=np.float64)

    for replicate in range(n_bootstrap):
        if response_matrices is not None:
            assert refit is not None
            old_responses, new_responses = response_matrices
            old_rows = rng.integers(0, old_responses.shape[0], old_responses.shape[0])
            new_rows = rng.integers(0, new_responses.shape[0], new_responses.shape[0])
            fitted_old = refit(model_old.copy(), old_responses[old_rows].copy())
            fitted_new = refit(model_new.copy(), new_responses[new_rows].copy())
            _validate_anchor_pairs(fitted_old, fitted_new, anchors_old, anchors_new)
            replicate_old = _extract_link_parameters(
                fitted_old, anchors_old, "refitted old"
            )
            replicate_new = _extract_link_parameters(
                fitted_new, anchors_new, "refitted new"
            )
        else:
            sampled = rng.integers(0, n_anchors, n_anchors)
            replicate_old = tuple(values[sampled] for values in parameters_old)
            replicate_new = tuple(values[sampled] for values in parameters_new)

        try:
            A, B = _estimate_constants(
                replicate_old,
                replicate_new,
                method,
                theta_grid,
                weights,
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            raise RuntimeError(
                f"Bootstrap replicate {replicate + 1} failed: {exc}"
            ) from exc
        A_samples[replicate] = A
        B_samples[replicate] = B

    return (
        float(np.std(A_samples, ddof=1)),
        float(np.std(B_samples, ddof=1)),
        A_samples,
        B_samples,
    )


def delta_method_se(
    linking_result: LinkingResult,
    vcov_old: NDArray[np.float64],
    vcov_new: NDArray[np.float64],
    anchors_old: list[int],
    anchors_new: list[int],
    model_old: BaseItemModel | None = None,
    model_new: BaseItemModel | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
) -> tuple[float, float]:
    """Propagate both forms' parameter covariance to linking constants.

    The covariance matrices follow the package's flattened parameter order:
    all discrimination parameters, followed by all difficulty parameters,
    followed by any additional model parameters. Numerical differentiation
    supports every public dichotomous linking method and includes covariance
    among every link-relevant parameter estimate within each form.

    ``model_old`` and ``model_new`` are required because a Jacobian cannot be
    recovered from A and B alone.
    """
    if model_old is None or model_new is None:
        raise ValueError(
            "model_old and model_new are required for delta-method propagation"
        )
    method = linking_result.constants.method
    _validate_method(method)
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    parameters_old = _extract_link_parameters(model_old, anchors_old, "old")
    parameters_new = _extract_link_parameters(model_new, anchors_new, "new")
    covariance_old = _validate_covariance(vcov_old, model_old.n_parameters, "old")
    covariance_new = _validate_covariance(vcov_new, model_new.n_parameters, "new")

    components_old = _delta_components("old", model_old, anchors_old, parameters_old)
    components_new = _delta_components("new", model_new, anchors_new, parameters_new)
    components = components_old + components_new
    parameter_vector = np.concatenate([component[2] for component in components])

    def constants_from_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
        varied_old = list(parameters_old)
        varied_new = list(parameters_new)
        offset = 0
        for form, component_index, component_values, _ in components:
            size = component_values.size
            varied = varied_old if form == "old" else varied_new
            varied[component_index] = values[offset : offset + size]
            offset += size
        A, B = _estimate_constants(
            tuple(varied_old),
            tuple(varied_new),
            method,
            theta_grid,
            weights,
        )
        return np.array([A, B], dtype=np.float64)

    jacobian = np.empty((2, parameter_vector.size), dtype=np.float64)
    discrimination_positions: set[int] = set()
    offset = 0
    for _, component_index, component_values, _ in components:
        if component_index == 0:
            discrimination_positions.update(
                range(offset, offset + component_values.size)
            )
        offset += component_values.size
    for index, value in enumerate(parameter_vector):
        step = 1e-5 * max(1.0, abs(float(value)))
        if index in discrimination_positions:
            step = min(step, float(value) * 0.25)
        upper = parameter_vector.copy()
        lower = parameter_vector.copy()
        upper[index] += step
        lower[index] -= step
        jacobian[:, index] = (
            constants_from_vector(upper) - constants_from_vector(lower)
        ) / (2.0 * step)

    old_indices = np.concatenate([component[3] for component in components_old]).astype(
        np.int64, copy=False
    )
    new_indices = np.concatenate([component[3] for component in components_new]).astype(
        np.int64, copy=False
    )
    selected_old = covariance_old[np.ix_(old_indices, old_indices)]
    selected_new = covariance_new[np.ix_(new_indices, new_indices)]
    n_old = old_indices.size
    n_new = new_indices.size
    combined_covariance = np.zeros((n_old + n_new, n_old + n_new), dtype=np.float64)
    combined_covariance[:n_old, :n_old] = selected_old
    combined_covariance[n_old:, n_old:] = selected_new
    propagated = jacobian @ combined_covariance @ jacobian.T
    variances = np.diag(propagated)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(propagated))))
    if np.any(variances < -tolerance):
        raise ValueError("Propagated covariance produced a negative variance")
    return float(np.sqrt(max(variances[0], 0.0))), float(
        np.sqrt(max(variances[1], 0.0))
    )


def compute_linking_fit(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchors_old: list[int],
    anchors_new: list[int],
    A: float,
    B: float,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    weights: NDArray[np.float64] | None = None,
) -> LinkingFitStatistics:
    """Compute parameter and full-curve fit for a linking solution."""
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    scale, shift = _validate_constants(A, B)
    theta_grid, normalized_weights = _validate_curve_grid(theta_range, n_theta, weights)
    parameters_old = _extract_link_parameters(model_old, anchors_old, "old")
    parameters_new = _extract_link_parameters(model_new, anchors_new, "new")
    disc_old, diff_old = parameters_old[:2]
    disc_new, diff_new = parameters_new[:2]

    disc_new_transformed = disc_new / scale
    diff_new_transformed = scale * diff_new + shift
    diff_a = disc_old - disc_new_transformed
    diff_b = diff_old - diff_new_transformed
    rmse_a = float(np.sqrt(np.mean(diff_a**2)))
    rmse_b = float(np.sqrt(np.mean(diff_b**2)))

    curves_old = _model_curves(
        model_old, theta_grid, anchors_old, parameters_old, "old"
    )
    theta_new = (theta_grid - shift) / scale
    curves_new = _model_curves(model_new, theta_new, anchors_new, parameters_new, "new")
    tcc_old = curves_old.sum(axis=1)
    tcc_new = curves_new.sum(axis=1)
    tcc_rmse = float(np.sqrt(np.sum(normalized_weights * (tcc_old - tcc_new) ** 2)))
    return LinkingFitStatistics(
        rmse_a=rmse_a,
        rmse_b=rmse_b,
        mad_a=float(np.mean(np.abs(diff_a))),
        mad_b=float(np.mean(np.abs(diff_b))),
        weighted_rmse=float(np.hypot(rmse_a, rmse_b)),
        tcc_rmse=tcc_rmse,
    )


def linking_summary(
    result: LinkingResult,
    model_old: BaseItemModel,
    model_new: BaseItemModel,
) -> str:
    """Generate a formatted, directionally explicit linking summary."""
    lines = ["=" * 60, "IRT Linking Summary", "=" * 60, ""]
    lines.extend(
        [
            f"Reference model: {model_old.model_name} ({model_old.n_items} items)",
            f"New model:       {model_new.model_name} ({model_new.n_items} items)",
            "",
            "Transformation Constants",
            "-" * 30,
            f"Method: {result.constants.method}",
            f"A (slope):     {result.constants.A:8.4f}",
            f"B (intercept): {result.constants.B:8.4f}",
        ]
    )
    if result.constants.A_se is not None:
        lines.append(f"SE(A):         {result.constants.A_se:8.4f}")
    if result.constants.B_se is not None:
        lines.append(f"SE(B):         {result.constants.B_se:8.4f}")

    lines.extend(
        [
            "",
            "Anchor Items",
            "-" * 30,
            f"Number of anchors: {len(result.anchor_items)}",
            f"Reference indices: {result.anchor_items}",
        ]
    )
    if result.fit_statistics is not None:
        fit = result.fit_statistics
        lines.extend(
            [
                "",
                "Fit Statistics",
                "-" * 30,
                f"RMSE (discrimination): {fit.rmse_a:.4f}",
                f"RMSE (difficulty):     {fit.rmse_b:.4f}",
                f"MAD (discrimination):  {fit.mad_a:.4f}",
                f"MAD (difficulty):      {fit.mad_b:.4f}",
                f"Weighted RMSE:          {fit.weighted_rmse:.4f}",
                f"TCC RMSE:               {fit.tcc_rmse:.4f}",
            ]
        )

    if result.anchor_diagnostics is not None:
        diagnostics = result.anchor_diagnostics
        n_flagged = int(np.sum(diagnostics.flagged))
        lines.extend(
            [
                "",
                "Anchor Diagnostics",
                "-" * 30,
                f"Items flagged for drift: {n_flagged}",
            ]
        )
        for position in np.flatnonzero(diagnostics.flagged):
            lines.append(
                f"  Item {diagnostics.item_indices[position]}: "
                f"z = {diagnostics.robust_z[position]:.2f}, "
                f"area = {diagnostics.area_diff[position]:.3f}"
            )

    if result.convergence_info is not None:
        lines.extend(["", "Convergence Information", "-" * 30])
        lines.extend(
            f"{key}: {value}" for key, value in result.convergence_info.items()
        )

    A, B = result.constants.A, result.constants.B
    lines.extend(
        [
            "",
            "Transformation Equations",
            "-" * 30,
            f"theta_old = {A:.4f} * theta_new + {B:.4f}",
            f"a_new_on_old = a_new / {A:.4f}",
            f"b_new_on_old = {A:.4f} * b_new + {B:.4f}",
            "",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def compare_linking_methods(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchors_old: list[int],
    anchors_new: list[int],
    methods: list[str] | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
) -> dict[str, dict]:
    """Compare linking constants, fit, and drift flags across methods."""
    selected_methods = methods or [
        "mean_sigma",
        "mean_mean",
        "stocking_lord",
        "haebara",
        "tcc",
        "bisector",
        "orthogonal",
    ]
    results: dict[str, dict] = {}
    for method in selected_methods:
        try:
            result = link(
                model_old,
                model_new,
                anchors_old,
                anchors_new,
                method=method,
                theta_range=theta_range,
                n_theta=n_theta,
                compute_diagnostics=True,
            )
            fit = compute_linking_fit(
                model_old,
                model_new,
                anchors_old,
                anchors_new,
                result.constants.A,
                result.constants.B,
                theta_range,
                n_theta,
            )
            diagnostics = result.anchor_diagnostics
            results[method] = {
                "A": result.constants.A,
                "B": result.constants.B,
                "rmse_a": fit.rmse_a if fit else None,
                "rmse_b": fit.rmse_b if fit else None,
                "tcc_rmse": fit.tcc_rmse if fit else None,
                "n_flagged": int(np.sum(diagnostics.flagged)) if diagnostics else 0,
            }
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            results[method] = {"error": str(exc)}
    return results


def parameter_recovery_summary(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchors_old: list[int],
    anchors_new: list[int],
    A: float,
    B: float,
) -> str:
    """Generate a validated paired-anchor parameter recovery table."""
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    scale, shift = _validate_constants(A, B)
    disc_old, diff_old, _, _, _ = _extract_link_parameters(
        model_old, anchors_old, "old"
    )
    disc_new, diff_new, _, _, _ = _extract_link_parameters(
        model_new, anchors_new, "new"
    )
    disc_new_transformed = disc_new / scale
    diff_new_transformed = scale * diff_new + shift

    lines = [
        "Parameter Recovery After Transformation",
        "=" * 80,
        f"{'Old':>6} {'New':>6} {'a_old':>8} {'a_trans':>8} "
        f"{'diff_a':>8} {'b_old':>8} {'b_trans':>8} {'diff_b':>8}",
        "-" * 80,
    ]
    for position, (old_item, new_item) in enumerate(
        zip(anchors_old, anchors_new, strict=True)
    ):
        lines.append(
            f"{old_item:>6} {new_item:>6} {disc_old[position]:>8.3f} "
            f"{disc_new_transformed[position]:>8.3f} "
            f"{disc_old[position] - disc_new_transformed[position]:>8.3f} "
            f"{diff_old[position]:>8.3f} {diff_new_transformed[position]:>8.3f} "
            f"{diff_old[position] - diff_new_transformed[position]:>8.3f}"
        )

    rmse_a = float(np.sqrt(np.mean((disc_old - disc_new_transformed) ** 2)))
    rmse_b = float(np.sqrt(np.mean((diff_old - diff_new_transformed) ** 2)))
    corr_a = _safe_correlation(disc_old, disc_new_transformed)
    corr_b = _safe_correlation(diff_old, diff_new_transformed)
    lines.extend(
        [
            "-" * 80,
            f"RMSE(a): {rmse_a:.4f}    RMSE(b): {rmse_b:.4f}",
            f"Corr(a): {_format_correlation(corr_a)}    "
            f"Corr(b): {_format_correlation(corr_b)}",
            "=" * 80,
        ]
    )
    return "\n".join(lines)


def _validate_method(method: str) -> None:
    """Reject unsupported methods instead of silently changing estimators."""
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unknown linking method: {method}")


def _validate_bootstrap_count(n_bootstrap: int) -> None:
    """Require enough integer replicates for a sample standard deviation."""
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise ValueError("n_bootstrap must be an integer")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")


def _validate_bootstrap_responses(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    responses_old: NDArray[np.float64] | None,
    responses_new: NDArray[np.float64] | None,
    refit: RefitCallback | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Validate response-bootstrap mode and its explicit recalibration hook."""
    if (responses_old is None) != (responses_new is None):
        raise ValueError("responses_old and responses_new must be supplied together")
    if responses_old is None:
        if refit is not None:
            raise ValueError("refit requires response matrices")
        return None
    if refit is None:
        raise ValueError("refit is required when response matrices are supplied")
    matrices: list[NDArray[np.float64]] = []
    for label, responses, model in (
        ("old", responses_old, model_old),
        ("new", responses_new, model_new),
    ):
        values = np.asarray(responses)
        if values.ndim != 2 or values.shape[1] != model.n_items:
            raise ValueError(
                f"responses_{label} must have shape (n_persons, {model.n_items})"
            )
        if values.shape[0] < 2:
            raise ValueError(f"responses_{label} must contain at least two persons")
        matrices.append(values)
    return matrices[0], matrices[1]


def _validate_anchor_pairs(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchors_old: list[int],
    anchors_new: list[int],
) -> tuple[list[int], list[int]]:
    """Validate and normalize corresponding anchor indices."""
    if model_old.n_factors != 1 or model_new.n_factors != 1:
        raise ValueError("Linking diagnostics require unidimensional models")
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
    return normalized[0], normalized[1]


def _extract_link_parameters(
    model: BaseItemModel, anchors: list[int], label: str
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extract validated dichotomous parameters for selected anchors."""
    try:
        discrimination = np.asarray(model.discrimination, dtype=np.float64)
        difficulty = np.asarray(model.difficulty, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError(
            f"The {label} model must expose discrimination and difficulty parameters"
        ) from exc
    if discrimination.ndim == 2 and discrimination.shape == (model.n_items, 1):
        discrimination = discrimination[:, 0]
    if discrimination.shape != (model.n_items,) or difficulty.shape != (model.n_items,):
        raise ValueError(
            f"The {label} model must contain one discrimination and difficulty per item"
        )
    lower = np.asarray(
        getattr(model, "guessing", np.zeros(model.n_items)), dtype=np.float64
    )
    upper = np.asarray(
        getattr(model, "upper", np.ones(model.n_items)), dtype=np.float64
    )
    asymmetry = np.asarray(
        getattr(model, "asymmetry", np.ones(model.n_items)), dtype=np.float64
    )
    if (
        lower.shape != (model.n_items,)
        or upper.shape != (model.n_items,)
        or asymmetry.shape != (model.n_items,)
    ):
        raise ValueError(
            f"Curve parameters for the {label} model have an invalid shape"
        )

    selected = (
        discrimination[anchors],
        difficulty[anchors],
        lower[anchors],
        upper[anchors],
        asymmetry[anchors],
    )
    if not all(np.all(np.isfinite(values)) for values in selected):
        raise ValueError(f"Item parameters for the {label} model must be finite")
    if np.any(selected[0] <= 0.0):
        raise ValueError(f"Discriminations for the {label} model must be positive")
    if np.any(selected[2] < 0.0) or np.any(selected[3] > 1.0):
        raise ValueError(f"Asymptotes for the {label} model must lie in [0, 1]")
    if np.any(selected[2] >= selected[3]):
        raise ValueError(
            f"Lower asymptotes for the {label} model must be below upper asymptotes"
        )
    if np.any(selected[4] <= 0.0):
        raise ValueError(f"Asymmetries for the {label} model must be positive")
    return selected


def _validate_curve_grid(
    theta_range: tuple[float, float],
    n_theta: int,
    weights: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate an integration grid and normalize its weights."""
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


def _validate_constants(A: float, B: float) -> tuple[float, float]:
    """Require finite orientation-preserving linking constants."""
    scale, shift = float(A), float(B)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A must be finite and positive")
    if not np.isfinite(shift):
        raise ValueError("B must be finite")
    return scale, shift


def _validate_covariance(
    covariance: NDArray[np.float64], minimum_size: int, label: str
) -> NDArray[np.float64]:
    """Validate covariance size, symmetry, finiteness, and definiteness."""
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"vcov_{label} must be a square matrix")
    if values.shape[0] < minimum_size:
        raise ValueError(f"vcov_{label} must cover all fitted model parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"vcov_{label} must be finite")
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"vcov_{label} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"vcov_{label} must be positive semidefinite")
    return values


def _delta_components(
    form: str,
    model: BaseItemModel,
    anchors: list[int],
    extracted: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> list[tuple[str, int, NDArray[np.float64], NDArray[np.int64]]]:
    """Map link-relevant arrays to their flattened covariance positions."""
    parameter_offsets: dict[str, int] = {}
    offset = 0
    for name, values in model.parameters.items():
        parameter_offsets[name] = offset
        offset += values.size

    components: list[tuple[str, int, NDArray[np.float64], NDArray[np.int64]]] = []
    for name, component_index in (
        ("discrimination", 0),
        ("difficulty", 1),
        ("guessing", 2),
        ("upper", 3),
        ("asymmetry", 4),
    ):
        if name not in parameter_offsets:
            continue
        indices = parameter_offsets[name] + np.asarray(anchors, dtype=np.int64)
        components.append((form, component_index, extracted[component_index], indices))
    return components


def _icc_matrix(
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    theta: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate stable dichotomous item curves in one batch."""
    logistic = expit(discrimination[None, :] * (theta[:, None] - difficulty[None, :]))
    shaped = logistic if np.all(asymmetry == 1.0) else logistic ** asymmetry[None, :]
    return lower[None, :] + (upper - lower)[None, :] * shaped


def _model_curves(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    anchors: list[int],
    parameters: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    label: str,
) -> NDArray[np.float64]:
    """Evaluate the model's native response function for selected items."""
    if model.model_name in _LOGISTIC_MODEL_NAMES:
        curves = _icc_matrix(*parameters[:2], theta, *parameters[2:])
        expected_shape = (theta.size, len(anchors))
    else:
        all_curves = np.asarray(model.probability(theta), dtype=np.float64)
        expected_shape = (theta.size, model.n_items)
        if all_curves.shape != expected_shape:
            raise ValueError(
                f"The {label} model returned curves with shape {all_curves.shape}; "
                f"expected {expected_shape}"
            )
        curves = all_curves[:, anchors]
        expected_shape = (theta.size, len(anchors))
    if curves.shape != expected_shape:
        raise ValueError(
            f"The {label} model returned curves with shape {curves.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(curves)):
        raise ValueError(f"Response curves for the {label} model must be finite")
    if np.any(curves < 0.0) or np.any(curves > 1.0):
        raise ValueError(f"Response curves for the {label} model must lie in [0, 1]")
    return curves


def _estimate_constants(
    parameters_old: tuple[NDArray[np.float64], ...],
    parameters_new: tuple[NDArray[np.float64], ...],
    method: str,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate constants from aligned parameter arrays."""
    _validate_method(method)
    disc_old, diff_old, lower_old, upper_old, asymmetry_old = parameters_old
    disc_new, diff_new, lower_new, upper_new, asymmetry_new = parameters_new
    if method == "mean_sigma":
        sd_old = float(np.std(diff_old, ddof=1))
        sd_new = float(np.std(diff_new, ddof=1))
        if sd_old < 1e-10 or sd_new < 1e-10:
            A = float(np.mean(disc_new) / np.mean(disc_old))
        else:
            A = sd_old / sd_new
        B = float(np.mean(diff_old) - A * np.mean(diff_new))
    elif method == "mean_mean":
        A = float(np.mean(disc_new) / np.mean(disc_old))
        B = float(np.mean(diff_old) - A * np.mean(diff_new))
    elif method in {"stocking_lord", "haebara", "tcc"}:
        A, B = _curve_link(
            disc_old,
            diff_old,
            lower_old,
            upper_old,
            asymmetry_old,
            disc_new,
            diff_new,
            lower_new,
            upper_new,
            asymmetry_new,
            theta_grid,
            weights,
            method,
        )
    elif method == "bisector":
        A, B, _ = _bisector_link(disc_old, diff_old, disc_new, diff_new)
    else:
        A, B, _ = _orthogonal_link(disc_old, diff_old, disc_new, diff_new)
    return _validate_constants(A, B)


def _curve_link(
    disc_old: NDArray[np.float64],
    diff_old: NDArray[np.float64],
    lower_old: NDArray[np.float64],
    upper_old: NDArray[np.float64],
    asymmetry_old: NDArray[np.float64],
    disc_new: NDArray[np.float64],
    diff_new: NDArray[np.float64],
    lower_new: NDArray[np.float64],
    upper_new: NDArray[np.float64],
    asymmetry_new: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    method: str,
) -> tuple[float, float]:
    """Fit a positive-scale curve-matching transformation."""
    curves_old = _icc_matrix(
        disc_old,
        diff_old,
        theta_grid,
        lower_old,
        upper_old,
        asymmetry_old,
    )
    sd_old = float(np.std(diff_old, ddof=1))
    sd_new = float(np.std(diff_new, ddof=1))
    if sd_old < 1e-10 or sd_new < 1e-10:
        initial_A = float(np.mean(disc_new) / np.mean(disc_old))
    else:
        initial_A = sd_old / sd_new
    initial_B = float(np.mean(diff_old) - initial_A * np.mean(diff_new))
    initial_A, initial_B = _validate_constants(initial_A, initial_B)

    def criterion(values: NDArray[np.float64]) -> float:
        log_A, B = float(values[0]), float(values[1])
        if not np.isfinite(log_A) or not np.isfinite(B) or abs(log_A) > 20.0:
            return float("inf")
        A = float(np.exp(log_A))
        curves_new = _icc_matrix(
            disc_new / A,
            A * diff_new + B,
            theta_grid,
            lower_new,
            upper_new,
            asymmetry_new,
        )
        if method in {"stocking_lord", "tcc"}:
            difference = curves_old.sum(axis=1) - curves_new.sum(axis=1)
            return float(np.sum(weights * difference**2))
        return float(np.sum(weights[:, None] * (curves_old - curves_new) ** 2))

    initial = np.array([np.log(initial_A), initial_B])
    initial_value = criterion(initial)
    if initial_value <= 1e-14:
        return initial_A, initial_B
    result = optimize.minimize(
        criterion,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"{method} linking failed to converge: {result.message}")
    return _validate_constants(float(np.exp(result.x[0])), float(result.x[1]))


def _safe_correlation(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float | None:
    """Compute correlation without warnings for short or constant vectors."""
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _format_correlation(value: float | None) -> str:
    """Format an undefined correlation explicitly."""
    return "n/a" if value is None else f"{value:.4f}"
