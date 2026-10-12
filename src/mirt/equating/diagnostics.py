"""Diagnostics and uncertainty estimates for IRT linking."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.equating.linking import (
    _CLOSED_FORM_LINKING_METHODS,
    _CURVE_LINKING_METHODS,
    LinkingFitStatistics,
    LinkingResult,
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
from mirt.equating.polytomous import (
    _NRM_METHODS,
    _ORDERED_METHODS,
    _polytomous_constants,
    _polytomous_criterion_gradient,
    _polytomous_form,
    _polytomous_parameter_names,
    _PolytomousForm,
    _require_polytomous_family,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


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
# Probe domains: positive parameters keep steps below a quarter of their
# value; probabilities stay in [0, 1], below 1 where 1 is invalid.
_POSITIVE_PARAMETERS = frozenset({"discrimination", "asymmetry"})
_PROBABILITY_UPPER_BOUNDS = {
    "guessing": float(np.nextafter(1.0, 0.0)),
    "zero_inflation": float(np.nextafter(1.0, 0.0)),
    "upper": 1.0,
}

RefitCallback = Callable[["BaseItemModel", NDArray[np.float64]], "BaseItemModel"]
# Anchor curve sources: the core dichotomous kernel or a polytomous family's.
_AnchorForm = _LinkForm | _PolytomousForm


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
        Any public dichotomous linking method, or for GRM, GPCM/PCM, and NRM
        forms a method of :func:`~mirt.equating.link_grm`,
        :func:`~mirt.equating.link_gpcm`, or :func:`~mirt.equating.link_nrm`.
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
    anchors_old, anchors_new = _validate_linking_pair(
        model_old, model_new, anchors_old, anchors_new, method
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    response_matrices = _validate_bootstrap_responses(
        model_old, model_new, responses_old, responses_new, refit
    )
    rng = np.random.default_rng(seed)

    form_old = _anchor_form(model_old, anchors_old, "old")
    form_new = _anchor_form(model_new, anchors_new, "new")
    n_anchors = len(anchors_old)

    if (
        response_matrices is None
        and method in _CLOSED_FORM_LINKING_METHODS
        and isinstance(form_old, _LinkForm)
        and isinstance(form_new, _LinkForm)
    ):
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
    form_old: _AnchorForm,
    form_new: _AnchorForm,
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
        _validate_linking_pair(fitted_old, fitted_new, anchors_old, anchors_new, method)
        replicate_old = _anchor_form(fitted_old, anchors_old, "refitted old")
        replicate_new = _anchor_form(fitted_new, anchors_new, "refitted new")
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
    vcov_old: NDArray[np.float64] | FitResult,
    vcov_new: NDArray[np.float64] | FitResult,
    anchors_old: list[int],
    anchors_new: list[int],
    model_old: BaseItemModel | None = None,
    model_new: BaseItemModel | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
) -> tuple[float, float]:
    """Propagate both forms' parameter covariance to linking constants.

    Every public dichotomous linking method is supported, as are the
    methods of :func:`~mirt.equating.link_grm`,
    :func:`~mirt.equating.link_gpcm`, and :func:`~mirt.equating.link_nrm`
    for GRM, GPCM/PCM, and NRM forms, including covariance among every
    link-relevant parameter estimate within each form.

    Parameters
    ----------
    linking_result : LinkingResult
        Result whose ``constants.method`` selects the linking estimator.
    vcov_old, vcov_new : ndarray or FitResult
        Parameter covariance of each calibration. A ``FitResult`` is mapped
        through ``vcov_labels``. A result without ``vcov``, such as one
        fitted with ``se_method="complete_data"``, falls back to its squared
        standard errors with a warning; the standard errors of A and B then
        ignore covariances between parameter estimates. A matrix may cover
        the free parameters in ``FitResult.vcov`` order (one row per free
        coordinate, in ``model.parameters`` order with each array raveled
        row-major and fixed coordinates left out), or every
        stored coordinate in that order. Rows and columns that are ``NaN``
        throughout, such as those of estimates held at an optimizer bound,
        are treated as known exactly (zero variance), like fixed
        coordinates. Only the rows and columns of the anchors' parameters
        are read and validated.
    anchors_old, anchors_new : list[int]
        Corresponding anchor indices. Polytomous anchors must have the same
        number of categories on both forms.
    model_old, model_new : BaseItemModel or None
        Calibrated models. They default to the models of ``FitResult``
        covariance arguments and are otherwise required, because a Jacobian
        cannot be recovered from A and B alone.
    theta_range : tuple[float, float]
        Integration range of the curve criteria. Default ``(-4, 4)``.
    n_theta : int
        Number of integration points. Default 61.

    Returns
    -------
    tuple[float, float]
        Standard errors of A and B.

    Raises
    ------
    ValueError
        If the models are missing, a covariance matrix has neither accepted
        size, a ``FitResult`` has neither ``vcov`` nor standard errors, or
        the anchors' block of a covariance is not a symmetric, positive
        semidefinite matrix whose entries are finite apart from ``NaN`` rows
        and columns.

    Notes
    -----
    For Stocking-Lord, TCC, and Haebara links between built-in 1PL-5PL
    forms, and Stocking-Lord and Haebara links of GRM, GPCM/PCM, and NRM
    forms, the constants are differentiated through the criterion's
    first-order condition (implicit-function theorem) using its closed-form
    gradient, so no re-optimization is needed. Moment methods and
    model-native curve families use central differences of re-estimated
    constants. Native curve families propagate every per-item parameter of
    the anchors, such as zero-inflation probabilities, because their curves
    depend on all of them; polytomous anchors propagate their
    discriminations (NRM category slopes) and their active thresholds,
    steps, or category intercepts. Only coordinates with nonzero covariance
    are differenced. Probability parameters estimated on their [0, 1] bound
    are differenced one-sidedly.
    """
    from mirt.results.fit_result import FitResult

    if model_old is None and isinstance(vcov_old, FitResult):
        model_old = vcov_old.model
    if model_new is None and isinstance(vcov_new, FitResult):
        model_new = vcov_new.model
    if model_old is None or model_new is None:
        raise ValueError(
            "model_old and model_new are required for delta-method propagation"
        )
    method = linking_result.constants.method
    _validate_method(method)
    anchors_old, anchors_new = _validate_linking_pair(
        model_old, model_new, anchors_old, anchors_new, method
    )
    theta_grid, weights = _validate_curve_grid(theta_range, n_theta, None)
    old = _delta_side(model_old, anchors_old, "old")
    new = _delta_side(model_new, anchors_new, "new")
    covariance_old = _anchor_covariance(vcov_old, model_old, old.components, "old")
    covariance_new = _anchor_covariance(vcov_new, model_new, new.components, "new")

    n_old = covariance_old.shape[0]
    n_total = n_old + covariance_new.shape[0]
    combined_covariance = np.zeros((n_total, n_total), dtype=np.float64)
    combined_covariance[:n_old, :n_old] = covariance_old
    combined_covariance[n_old:, n_old:] = covariance_new
    # Coordinates without covariance cannot move the constants' variance.
    active = np.flatnonzero(np.any(combined_covariance != 0.0, axis=1))
    if active.size == 0:
        return 0.0, 0.0

    jacobian = _constants_jacobian(method, old, new, active, theta_grid, weights)
    propagated = jacobian @ combined_covariance[np.ix_(active, active)] @ jacobian.T
    variances = np.diag(propagated)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(propagated))))
    if np.any(variances < -tolerance):
        raise ValueError("Propagated covariance produced a negative variance")
    return float(np.sqrt(max(variances[0], 0.0))), float(
        np.sqrt(max(variances[1], 0.0))
    )


@dataclass(frozen=True, eq=False)
class _DeltaSide:
    """One linked form's anchors, curve source, and varied coordinates."""

    model: BaseItemModel
    anchors: list[int]
    form: _AnchorForm
    components: list[_DeltaComponent]


def _delta_side(model: BaseItemModel, anchors: list[int], label: str) -> _DeltaSide:
    """Build the delta-method view of validated anchors of one form."""
    form = _anchor_form(model, anchors, label)
    return _DeltaSide(model, anchors, form, _delta_components(model, anchors, form))


def _constants_jacobian(
    method: str,
    old: _DeltaSide,
    new: _DeltaSide,
    active: NDArray[np.intp],
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Derivatives of ``(A, B)`` with respect to the active coordinates.

    ``active`` indexes the old form's component coordinates followed by the
    new form's.
    """
    components = old.components + new.components
    n_old = sum(component.values.size for component in old.components)
    base_vector = np.concatenate([component.values for component in components])
    lowest, highest, steps = _probe_domain(components)

    def forms_with_coordinate(
        index: int, value: float
    ) -> tuple[_AnchorForm, _AnchorForm]:
        """Both forms with active coordinate ``index`` set to ``value``."""
        coordinate = int(active[index])
        if coordinate < n_old:
            return _perturbed_form(old, coordinate, value), new.form
        return old.form, _perturbed_form(new, coordinate - n_old, value)

    parameter_vector = base_vector[active]
    steps = np.minimum(
        steps[active], _DELTA_STEP * np.maximum(1.0, np.abs(parameter_vector))
    )
    probes = (
        np.maximum(parameter_vector - steps, lowest[active]),
        np.minimum(parameter_vector + steps, highest[active]),
    )

    forms = (old.form, new.form)
    if method in _CURVE_LINKING_METHODS:
        if isinstance(old.form, _PolytomousForm):
            A, B, _ = _polytomous_constants(*forms, method, theta_grid, weights)

            def gradient(
                pair: tuple[_AnchorForm, _AnchorForm], params: NDArray[np.float64]
            ) -> NDArray[np.float64]:
                return _polytomous_criterion_gradient(
                    *pair, theta_grid, weights, method, params
                )

            return _implicit_curve_jacobian(
                (A, B), gradient, forms, forms_with_coordinate, probes
            )
        if not (old.form.is_native or new.form.is_native):
            A, B, _ = _fit_curve_link(*forms, theta_grid, weights, method)

            def gradient(
                pair: tuple[_AnchorForm, _AnchorForm], params: NDArray[np.float64]
            ) -> NDArray[np.float64]:
                return _CurveObjective(*pair, theta_grid, weights, method).gradient(
                    params
                )

            return _implicit_curve_jacobian(
                (A, B), gradient, forms, forms_with_coordinate, probes
            )

    def constants_with_coordinate(index: int, value: float) -> NDArray[np.float64]:
        varied_old, varied_new = forms_with_coordinate(index, value)
        return np.array(
            _estimate_constants(varied_old, varied_new, method, theta_grid, weights)
        )

    return _difference_columns(constants_with_coordinate, probes)


def _implicit_curve_jacobian(
    constants: tuple[float, float],
    gradient: Callable[
        [tuple[_AnchorForm, _AnchorForm], NDArray[np.float64]], NDArray[np.float64]
    ],
    forms: tuple[_AnchorForm, _AnchorForm],
    forms_with_coordinate: Callable[[int, float], tuple[_AnchorForm, _AnchorForm]],
    probes: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Differentiate fitted curve-link constants by the implicit-function theorem.

    At the minimizer ``x*`` of the criterion ``F(x; p)`` in ``x = (log A, B)``,
    ``dx*/dp = -H^{-1} d2F/dx dp``. Both second-derivative blocks are
    differences of the closed-form ``gradient(forms, x)``, so no
    re-optimization is needed. ``constants`` are the fitted ``(A, B)``, and
    ``forms_with_coordinate(index, value)`` returns the forms with one
    parameter set to a probe value.
    """
    A, B = constants
    solution = np.array([np.log(A), B], dtype=np.float64)

    hessian = np.empty((2, 2), dtype=np.float64)
    for column in range(2):
        offset = np.zeros(2, dtype=np.float64)
        offset[column] = _DELTA_STEP * max(1.0, abs(float(solution[column])))
        hessian[:, column] = (
            gradient(forms, solution + offset) - gradient(forms, solution - offset)
        ) / (2.0 * offset[column])
    hessian = 0.5 * (hessian + hessian.T)

    mixed = _difference_columns(
        lambda index, value: gradient(forms_with_coordinate(index, value), solution),
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
    function: Callable[[int, float], NDArray[np.float64]],
    probes: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Difference quotients of a 2-vector function, one column per parameter.

    ``function(index, value)`` evaluates the function with only parameter
    ``index`` moved to ``value``, so each probe rebuilds a single coordinate.
    """
    below, above = probes
    columns = np.empty((2, below.size), dtype=np.float64)
    for index in range(below.size):
        columns[:, index] = (
            function(index, float(above[index])) - function(index, float(below[index]))
        ) / (above[index] - below[index])
    return columns


@dataclass(frozen=True, eq=False)
class _DeltaComponent:
    """Anchor coordinates of one stored parameter varied by the delta method.

    ``values`` holds the anchors' rows raveled row-major and ``indices``
    their flat indices in the stored parameter array.
    """

    name: str
    values: NDArray[np.float64]
    indices: NDArray[np.intp]


def _delta_components(
    model: BaseItemModel, anchors: list[int], form: _AnchorForm
) -> list[_DeltaComponent]:
    """Map the anchors' curve parameters to stored covariance positions.

    Closed-form forms depend on the five logistic parameters. Native forms
    depend on every per-item parameter, so all of them are included.
    Polytomous forms depend on each anchor's discrimination (NRM category
    slopes) and active thresholds, steps, or category intercepts.
    """
    parameters = model.parameters
    rows = np.asarray(anchors, dtype=np.intp)

    if isinstance(form, _PolytomousForm):
        polytomous = []
        for name, values in zip(
            _polytomous_parameter_names(model),
            (form.slopes, form.locations),
            strict=True,
        ):
            # Active categories lead each stored row.
            width = parameters[name].size // model.n_items
            indices = np.concatenate(
                [
                    row * width + np.arange(item.size)
                    for row, item in zip(rows, values, strict=True)
                ]
            )
            polytomous.append(_DeltaComponent(name, np.concatenate(values), indices))
        return polytomous

    if not form.is_native:
        extracted = dict(zip(_DELTA_PARAMETER_NAMES, form.parameters, strict=True))
        return [
            _DeltaComponent(name, extracted[name], rows)
            for name in _DELTA_PARAMETER_NAMES
            if name in parameters
        ]

    # Shared and model-level parameters are never per-item, even when their
    # length happens to equal the number of items.
    not_per_item: frozenset[str] = getattr(
        model, "_shared_parameters", frozenset()
    ) | getattr(model, "_global_parameters", frozenset())
    components = []
    for name, values in parameters.items():
        if values.ndim == 0 or values.shape[0] != model.n_items or name in not_per_item:
            continue
        width = values.size // model.n_items
        indices = (rows[:, None] * width + np.arange(width)).ravel()
        selected = values.reshape(model.n_items, width)[rows]
        components.append(
            _DeltaComponent(name, selected.ravel().astype(np.float64), indices)
        )
    return components


def _component_positions(
    components: list[_DeltaComponent],
    parameters: dict[str, NDArray[np.float64]],
    names: list[str],
) -> NDArray[np.intp]:
    """Component coordinates in the raveled concatenation of ``names``."""
    offsets: dict[str, int] = {}
    offset = 0
    for name in names:
        offsets[name] = offset
        offset += parameters[name].size
    if not components:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(
        [offsets[component.name] + component.indices for component in components]
    )


def _probe_domain(
    components: list[_DeltaComponent],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return lower and upper probe limits and step caps per coordinate."""
    lowest, highest, steps = [], [], []
    for component in components:
        size = component.values.size
        upper = _PROBABILITY_UPPER_BOUNDS.get(component.name)
        lowest.append(np.full(size, -np.inf if upper is None else 0.0))
        highest.append(np.full(size, np.inf if upper is None else upper))
        if component.name in _POSITIVE_PARAMETERS:
            steps.append(0.25 * component.values)
        else:
            steps.append(np.full(size, np.inf))
    return np.concatenate(lowest), np.concatenate(highest), np.concatenate(steps)


def _perturbed_form(side: _DeltaSide, coordinate: int, value: float) -> _AnchorForm:
    """Rebuild one form's curves with one component coordinate replaced.

    ``coordinate`` indexes the concatenated component values.
    """
    for part, component in enumerate(side.components):
        if coordinate < component.values.size:
            break
        coordinate -= component.values.size
    form = side.form
    if component.values[coordinate] == value:
        return form
    if isinstance(form, _PolytomousForm):
        return form.replaced(part, coordinate, value)
    values = component.values.copy()
    values[coordinate] = value
    if not form.is_native:
        parameters = list(form.parameters)
        parameters[_DELTA_PARAMETER_NAMES.index(component.name)] = values
        return form.with_parameters(
            (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
        )
    model, anchors = side.model, side.anchors
    array = np.array(model.parameters[component.name], dtype=np.float64, copy=True)
    array.reshape(model.n_items, -1)[anchors] = values.reshape(len(anchors), -1)
    varied = model.copy()
    varied.set_parameters(**{component.name: array})
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


def _validate_linking_pair(
    model_old: BaseItemModel,
    model_new: BaseItemModel,
    anchors_old: list[int],
    anchors_new: list[int],
    method: str,
) -> tuple[list[int], list[int]]:
    """Validate anchors, a shared response family, and its linking method.

    Dichotomous forms accept every public method; GRM, GPCM/PCM, and NRM
    forms the methods of their family's linker, with matching anchor
    category counts.
    """
    anchors_old, anchors_new = _validate_anchor_pairs(
        model_old, model_new, anchors_old, anchors_new
    )
    if model_old.is_polytomous != model_new.is_polytomous:
        raise ValueError("Linked forms must both be dichotomous or both polytomous")
    if not model_old.is_polytomous:
        return anchors_old, anchors_new
    family = _require_polytomous_family(model_old)
    if _require_polytomous_family(model_new) != family:
        raise ValueError(
            "Linked forms must share one response family; got "
            f"{model_old.model_name} and {model_new.model_name}"
        )
    supported = _NRM_METHODS if family == "nrm" else _ORDERED_METHODS
    if method not in supported:
        raise ValueError(
            f"Unknown linking method for {model_old.model_name} forms: {method}; "
            f"use one of {', '.join(sorted(supported))}"
        )
    old_categories = model_old.n_categories
    new_categories = model_new.n_categories
    if any(
        old_categories[old] != new_categories[new]
        for old, new in zip(anchors_old, anchors_new, strict=True)
    ):
        raise ValueError(
            "Corresponding anchor items must have the same number of categories"
        )
    return anchors_old, anchors_new


def _anchor_form(model: BaseItemModel, anchors: list[int], label: str) -> _AnchorForm:
    """Curve source of validated anchors: a polytomous family's or the core's."""
    if model.is_polytomous:
        return _polytomous_form(model, anchors, label)
    return _link_form(model, anchors, label)


def _anchor_covariance(
    covariance: NDArray[np.float64] | FitResult,
    model: BaseItemModel,
    components: list[_DeltaComponent],
    label: str,
    names: tuple[str, str] | None = None,
) -> NDArray[np.float64]:
    """Return the validated covariance of the components' anchor coordinates.

    Rows follow the concatenated component values. Messages call the
    arguments ``names``, by default ``("vcov_<label>", "model_<label>")``. ``FitResult`` covariances
    are mapped through their row labels; a result without ``vcov`` falls back
    to its squared standard errors, with a warning, as covariances between
    estimates are then unknown. Matrices with one row per free parameter are
    read through the model's free masks, so fixed coordinates get zero
    variance; matrices with one row per stored coordinate are read directly.
    Only the anchors' block is copied and validated.
    """
    from mirt.exceptions import MirtValidationError
    from mirt.results.fit_result import FitResult

    argument, model_argument = names or (f"vcov_{label}", f"model_{label}")
    parameters = model.parameters
    if isinstance(covariance, FitResult):
        varied = {component.name for component in components}
        names_varied = [name for name in parameters if name in varied]
        fitted = covariance.model.parameters
        if any(
            name not in fitted or np.shape(fitted[name]) != parameters[name].shape
            for name in names_varied
        ):
            raise ValueError(
                f"{argument} is a FitResult whose model parameters do not "
                f"match {model_argument}"
            )
        try:
            matrix, _, joint = covariance._parameter_covariance(names_varied)
        except MirtValidationError as exc:
            raise ValueError(f"{argument}: {exc.message}") from exc
        if not joint:
            warnings.warn(
                f"{argument} has no parameter covariance (vcov); its squared "
                "standard errors are used, ignoring covariances between estimates",
                UserWarning,
                stacklevel=3,
            )
        positions = _component_positions(components, parameters, names_varied)
        return _validate_covariance(matrix[np.ix_(positions, positions)], argument)

    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{argument} must be a square matrix")
    stored_names = list(parameters)
    stored_size = sum(array.size for array in parameters.values())
    positions = _component_positions(components, parameters, stored_names)
    masks = model.free_parameter_masks
    free = np.concatenate([np.ravel(masks[name]) for name in stored_names])
    # One row per free coordinate, as in FitResult.vcov. Tied coordinates,
    # such as a common 1PL slope, keep their own rows, so this can exceed
    # model.n_parameters.
    n_free = int(np.count_nonzero(free))
    if values.shape[0] == stored_size:
        return _validate_covariance(values[np.ix_(positions, positions)], argument)
    if values.shape[0] == n_free:
        # Free-layout rows of the anchors' coordinates; fixed ones have none.
        kept = free[positions]
        rows = (np.cumsum(free) - 1)[positions][kept]
        block = np.zeros((positions.size, positions.size), dtype=np.float64)
        block[np.ix_(kept, kept)] = values[np.ix_(rows, rows)]
        return _validate_covariance(block, argument)
    raise ValueError(
        f"{argument} has shape {values.shape}; expected ({n_free}, {n_free}) "
        f"over the free parameters, as in FitResult.vcov, or "
        f"({stored_size}, {stored_size}) over every stored parameter"
    )


def _validate_covariance(
    covariance: NDArray[np.float64], argument: str
) -> NDArray[np.float64]:
    """Validate symmetry, finiteness, and definiteness of a covariance block.

    Rows and columns whose variance is ``NaN`` belong to coordinates without
    an estimable variance and become zero. The block is modified in place.
    """
    values = covariance
    missing = np.isnan(np.diag(values))
    if np.any(missing):
        values[missing, :] = 0.0
        values[:, missing] = 0.0
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{argument} must be finite apart from NaN rows and columns of "
            "non-estimable coordinates"
        )
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{argument} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{argument} must be positive semidefinite")
    return values


def _estimate_constants(
    form_old: _AnchorForm,
    form_new: _AnchorForm,
    method: str,
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate constants from aligned anchor forms."""
    _validate_method(method)
    if isinstance(form_old, _PolytomousForm) or isinstance(form_new, _PolytomousForm):
        assert isinstance(form_old, _PolytomousForm)
        assert isinstance(form_new, _PolytomousForm)
        A, B, _ = _polytomous_constants(form_old, form_new, method, theta_grid, weights)
        return _validate_transform_constants(A, B)
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
