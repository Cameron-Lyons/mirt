"""Diagnostics and uncertainty estimates for dichotomous IRT linking."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.equating.linking import (
    _CLOSED_FORM_LINKING_METHODS,
    _CURVE_LINKING_METHODS,
    LinkingFitStatistics,
    LinkingResult,
    LinkParameters,
    _bisector_link,
    _closed_form_bootstrap_samples,
    _compute_fit_statistics,
    _CurveObjective,
    _extract_link_parameters,
    _fit_curve_link,
    _link_form,
    _LinkForm,
    _normalize_anchor_indices,
    _orthogonal_link,
    _validate_curve_grid,
    _validate_transform_constants,
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
# Covariance names of the link-relevant parameters, in LinkParameters order.
_DELTA_PARAMETER_NAMES = (
    "discrimination",
    "difficulty",
    "guessing",
    "upper",
    "asymmetry",
)
_DELTA_STEP = 1e-5

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
    n_jobs: int = 1,
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
    n_jobs : int
        Positive number of worker threads for curve-based or response-refit
        replicates. Closed-form anchor bootstraps are already vectorized.
        Default 1.

    Returns
    -------
    tuple
        Standard errors for A and B followed by the replicate samples.
    """
    _validate_method(method)
    _validate_bootstrap_count(n_bootstrap)
    n_jobs = _validate_job_count(n_jobs)
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    response_matrices = _validate_bootstrap_responses(
        model_old, model_new, responses_old, responses_new, refit
    )
    rng = np.random.default_rng(seed)

    form_old = _link_form(model_old, anchors_old, "old")
    form_new = _link_form(model_new, anchors_new, "new")
    n_anchors = len(anchors_old)

    if response_matrices is None and method in _CLOSED_FORM_LINKING_METHODS:
        A_samples, B_samples = _closed_form_bootstrap_samples(
            form_old.discrimination,
            form_old.difficulty,
            form_new.discrimination,
            form_new.difficulty,
            method,
            n_bootstrap,
            rng,
            fallback_on_either_scale=True,
        )
        invalid = np.flatnonzero(
            (~np.isfinite(A_samples)) | (A_samples <= 0.0) | (~np.isfinite(B_samples))
        )
        if invalid.size:
            replicate = int(invalid[0])
            try:
                _validate_transform_constants(
                    A_samples[replicate], B_samples[replicate]
                )
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                raise RuntimeError(
                    f"Bootstrap replicate {replicate + 1} failed: {exc}"
                ) from exc
        return (
            float(np.std(A_samples, ddof=1)),
            float(np.std(B_samples, ddof=1)),
            A_samples,
            B_samples,
        )

    A_samples = np.empty(n_bootstrap, dtype=np.float64)
    B_samples = np.empty(n_bootstrap, dtype=np.float64)

    def draw_sample() -> NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.intp]]:
        if response_matrices is None:
            return rng.integers(0, n_anchors, n_anchors)
        old_responses, new_responses = response_matrices
        return (
            rng.integers(0, old_responses.shape[0], old_responses.shape[0]),
            rng.integers(0, new_responses.shape[0], new_responses.shape[0]),
        )

    def run_replicate(
        replicate: int,
        sample: NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.intp]],
    ) -> tuple[float, float]:
        return _linking_bootstrap_replicate(
            replicate,
            sample,
            model_old,
            model_new,
            response_matrices,
            anchors_old,
            anchors_new,
            form_old,
            form_new,
            method,
            theta_grid,
            weights,
            refit,
        )

    if n_jobs == 1:
        for replicate in range(n_bootstrap):
            A_samples[replicate], B_samples[replicate] = run_replicate(
                replicate, draw_sample()
            )
    else:
        # Keep only a small multiple of the worker count in flight. Response
        # bootstrap indices can be large, so eagerly submitting every replicate
        # would replace compute savings with an avoidable memory spike.
        max_workers = min(n_jobs, n_bootstrap)
        pending: deque[tuple[int, Future[tuple[float, float]]]] = deque()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for replicate in range(n_bootstrap):
                pending.append(
                    (
                        replicate,
                        executor.submit(run_replicate, replicate, draw_sample()),
                    )
                )
                if len(pending) >= 2 * max_workers:
                    index, future = pending.popleft()
                    A_samples[index], B_samples[index] = future.result()
            while pending:
                index, future = pending.popleft()
                A_samples[index], B_samples[index] = future.result()

    return (
        float(np.std(A_samples, ddof=1)),
        float(np.std(B_samples, ddof=1)),
        A_samples,
        B_samples,
    )


def _linking_bootstrap_replicate(
    replicate: int,
    sample: NDArray[np.intp] | tuple[NDArray[np.intp], NDArray[np.intp]],
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    response_matrices: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
    anchors_old: list[int],
    anchors_new: list[int],
    form_old: _LinkForm,
    form_new: _LinkForm,
    method: str,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    refit: RefitCallback | None,
) -> tuple[float, float]:
    """Evaluate one pre-sampled linking replicate."""
    if response_matrices is not None:
        assert refit is not None
        assert isinstance(sample, tuple)
        old_responses, new_responses = response_matrices
        old_rows, new_rows = sample
        fitted_old = refit(model_old.copy(), old_responses[old_rows].copy())
        fitted_new = refit(model_new.copy(), new_responses[new_rows].copy())
        _validate_anchor_pairs(fitted_old, fitted_new, anchors_old, anchors_new)
        replicate_old = _link_form(fitted_old, anchors_old, "refitted old")
        replicate_new = _link_form(fitted_new, anchors_new, "refitted new")
    else:
        assert isinstance(sample, np.ndarray)
        replicate_old = form_old.take(sample)
        replicate_new = form_new.take(sample)

    try:
        return _estimate_constants(
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
    followed by any additional model parameters. Every public dichotomous
    linking method is supported, including covariance among every
    link-relevant parameter estimate within each form.

    For Stocking-Lord, TCC, and Haebara links between 1PL-5PL forms, the
    constants are differentiated through the criterion's first-order
    condition (implicit-function theorem) using its closed-form gradient, so
    no re-optimization is needed. Moment methods and model-native curve
    families use central differences of re-estimated constants. Asymptotes
    estimated on their [0, 1] bound are differenced one-sidedly.

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
    form_old = _link_form(model_old, anchors_old, "old")
    form_new = _link_form(model_new, anchors_new, "new")
    covariance_old = _validate_covariance(vcov_old, model_old.n_parameters, "old")
    covariance_new = _validate_covariance(vcov_new, model_new.n_parameters, "new")

    components_old = _delta_components(
        "old", model_old, anchors_old, form_old.parameters
    )
    components_new = _delta_components(
        "new", model_new, anchors_new, form_new.parameters
    )
    components = components_old + components_new
    parameter_vector = np.concatenate([component[2] for component in components])

    def forms_from_vector(values: NDArray[np.float64]) -> tuple[_LinkForm, _LinkForm]:
        varied = {"old": list(form_old.parameters), "new": list(form_new.parameters)}
        offset = 0
        for form, component_index, component_values, _ in components:
            size = component_values.size
            varied[form][component_index] = values[offset : offset + size]
            offset += size
        return (
            _perturbed_form(form_old, model_old, anchors_old, varied["old"]),
            _perturbed_form(form_new, model_new, anchors_new, varied["new"]),
        )

    steps = _DELTA_STEP * np.maximum(1.0, np.abs(parameter_vector))
    lowest = np.full(parameter_vector.size, -np.inf)
    highest = np.full(parameter_vector.size, np.inf)
    offset = 0
    for _, component_index, component_values, _ in components:
        positions = slice(offset, offset + component_values.size)
        if component_index == 0:
            steps[positions] = np.minimum(steps[positions], 0.25 * component_values)
        elif component_index == 2:
            lowest[positions] = 0.0
        elif component_index == 3:
            highest[positions] = 1.0
        offset += component_values.size
    # Asymptote probes stay in [0, 1], which fitted models enforce, so an
    # estimate on its bound gets a one-sided difference.
    probes = (
        np.maximum(parameter_vector - steps, lowest),
        np.minimum(parameter_vector + steps, highest),
    )

    if method in _CURVE_LINKING_METHODS and not (
        form_old.is_native or form_new.is_native
    ):
        jacobian = _implicit_curve_jacobian(
            form_old,
            form_new,
            method,
            theta_grid,
            weights,
            forms_from_vector,
            parameter_vector,
            probes,
        )
    else:

        def constants_from_vector(
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            varied_old, varied_new = forms_from_vector(values)
            return np.array(
                _estimate_constants(varied_old, varied_new, method, theta_grid, weights)
            )

        jacobian = _difference_columns(constants_from_vector, parameter_vector, probes)

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


def _implicit_curve_jacobian(
    form_old: _LinkForm,
    form_new: _LinkForm,
    method: str,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    forms_from_vector: Callable[[NDArray[np.float64]], tuple[_LinkForm, _LinkForm]],
    parameter_vector: NDArray[np.float64],
    probes: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Differentiate fitted curve-link constants by the implicit-function theorem.

    At the minimizer ``x*`` of the criterion ``F(x; p)`` in ``x = (log A, B)``,
    ``dx*/dp = -H^{-1} d2F/dx dp``. Both second-derivative blocks are
    differences of the closed-form gradient, so no re-optimization is needed.
    """
    A, B, _ = _fit_curve_link(form_old, form_new, theta_grid, weights, method)
    solution = np.array([np.log(A), B], dtype=np.float64)

    def gradient(
        forms: tuple[_LinkForm, _LinkForm], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _CurveObjective(*forms, theta_grid, weights, method).gradient(params)

    hessian = np.empty((2, 2), dtype=np.float64)
    for column in range(2):
        offset = np.zeros(2, dtype=np.float64)
        offset[column] = _DELTA_STEP * max(1.0, abs(float(solution[column])))
        hessian[:, column] = (
            gradient((form_old, form_new), solution + offset)
            - gradient((form_old, form_new), solution - offset)
        ) / (2.0 * offset[column])
    hessian = 0.5 * (hessian + hessian.T)

    mixed = _difference_columns(
        lambda values: gradient(forms_from_vector(values), solution),
        parameter_vector,
        probes,
    )

    try:
        derivatives = -np.linalg.solve(hessian, mixed)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The linking criterion is singular at its minimum; "
            "delta-method derivatives are undefined"
        ) from exc
    derivatives[0] *= A
    return derivatives


def _difference_columns(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    center: NDArray[np.float64],
    probes: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Difference quotients of a 2-vector function, one column per parameter."""
    below, above = probes
    columns = np.empty((2, center.size), dtype=np.float64)
    for index in range(center.size):
        lower = center.copy()
        upper = center.copy()
        lower[index] = below[index]
        upper[index] = above[index]
        columns[:, index] = (function(upper) - function(lower)) / (
            above[index] - below[index]
        )
    return columns


def _perturbed_form(
    form: _LinkForm,
    model: BaseItemModel,
    anchors: list[int],
    parameters: list[NDArray[np.float64]],
) -> _LinkForm:
    """Rebuild one form's curves at perturbed link-relevant parameters."""
    if not form.is_native:
        return form.with_parameters(
            (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
        )
    varied = model.copy()
    updates: dict[str, NDArray[np.float64]] = {}
    for component_index, name in enumerate(_DELTA_PARAMETER_NAMES):
        if name not in model.parameters:
            continue
        values = np.array(model.parameters[name], dtype=np.float64, copy=True)
        values.reshape(values.shape[0], -1)[anchors, 0] = parameters[component_index]
        updates[name] = values
    varied.set_parameters(**updates)
    return _link_form(varied, anchors, form.label)


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
    scale, shift = _validate_transform_constants(A, B)
    theta_grid, normalized_weights = _validate_curve_grid(theta_range, n_theta, weights)
    return _compute_fit_statistics(
        _link_form(model_old, anchors_old, "old"),
        _link_form(model_new, anchors_new, "new"),
        scale,
        shift,
        theta_grid,
        normalized_weights,
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
    scale, shift = _validate_transform_constants(A, B)
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


def _validate_job_count(n_jobs: int) -> int:
    """Normalize a positive worker count without accepting booleans."""
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(
        n_jobs, (int, np.integer)
    ):
        raise ValueError("n_jobs must be a positive integer")
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return int(n_jobs)


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

    return _normalize_anchor_indices(
        anchors_old, anchors_new, model_old.n_items, model_new.n_items
    )


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
    extracted: LinkParameters,
) -> list[tuple[str, int, NDArray[np.float64], NDArray[np.int64]]]:
    """Map link-relevant arrays to their flattened covariance positions."""
    parameter_offsets: dict[str, int] = {}
    offset = 0
    for name, values in model.parameters.items():
        parameter_offsets[name] = offset
        offset += values.size

    components: list[tuple[str, int, NDArray[np.float64], NDArray[np.int64]]] = []
    for component_index, name in enumerate(_DELTA_PARAMETER_NAMES):
        if name not in parameter_offsets:
            continue
        indices = parameter_offsets[name] + np.asarray(anchors, dtype=np.int64)
        components.append((form, component_index, extracted[component_index], indices))
    return components


def _estimate_constants(
    form_old: _LinkForm,
    form_new: _LinkForm,
    method: str,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate constants from aligned anchor forms."""
    _validate_method(method)
    disc_old, diff_old = form_old.discrimination, form_old.difficulty
    disc_new, diff_new = form_new.discrimination, form_new.difficulty
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
    elif method in _CURVE_LINKING_METHODS:
        A, B, _ = _fit_curve_link(form_old, form_new, theta_grid, weights, method)
    elif method == "bisector":
        A, B, _ = _bisector_link(disc_old, diff_old, disc_new, diff_new)
    else:
        A, B, _ = _orthogonal_link(disc_old, diff_old, disc_new, diff_new)
    return _validate_transform_constants(A, B)


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
