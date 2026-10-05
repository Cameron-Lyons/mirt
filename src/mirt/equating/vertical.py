"""Vertical scaling and grade-level linking for IRT models.

This module provides vertical scaling functionality for linking tests
across grade levels using common anchor item designs, with support
for monotonicity constraints and growth curve estimation.

Examples
--------
Basic vertical scaling with chain linking:

>>> from mirt.equating.vertical import vertical_scale, GradeData
>>> grade_data = [
...     GradeData("Grade 3", responses_g3, anchor_items_above=[0, 1, 2]),
...     GradeData("Grade 4", responses_g4, anchor_items_below=[10, 11, 12],
...               anchor_items_above=[0, 1, 2]),
...     GradeData("Grade 5", responses_g5, anchor_items_below=[10, 11, 12]),
... ]
>>> result = vertical_scale(grade_data)
>>> print(result.growth_curve)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mirt.equating.linking import LinkingResult
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.results import MultigroupFitResult
    from mirt.results.score_result import ScoreResult


_VERTICAL_METHODS = frozenset(
    {"chain", "concurrent", "fixed_anchor", "floating_anchor"}
)
_LINKING_METHODS = frozenset(
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


@dataclass
class GradeData:
    """Data for a single grade level in vertical scaling.

    Attributes
    ----------
    grade_label : str | int
        Label identifying the grade level.
    responses : NDArray[np.int_]
        Response matrix (n_persons x n_items) for this grade.
    anchor_items_below : list[int] | None
        Indices of items shared with the grade below.
    anchor_items_above : list[int] | None
        Indices of items shared with the grade above.
    """

    grade_label: str | int
    responses: NDArray[np.int_]
    anchor_items_below: list[int] | None = None
    anchor_items_above: list[int] | None = None


@dataclass
class VerticalScaleResult:
    """Result of vertical scaling procedure.

    Attributes
    ----------
    grade_transformations : dict[str | int, tuple[float, float]]
        Linear transformation constants (A, B) for each grade to the
        common vertical scale. Empty for anchor calibration, where refitted
        parameters cannot be represented by affine original-score maps.
    grade_means : dict[str | int, float]
        Grade population means for anchor calibration, or means of linked
        person estimates for chain and concurrent linking.
    grade_sds : dict[str | int, float]
        Grade population standard deviations for anchor calibration, or
        standard deviations of linked person estimates for linking.
    linking_results : list[LinkingResult]
        Detailed linking results for each adjacent grade pair.
    monotonicity_violations : list[tuple]
        Adjacent grade pairs with decreasing means. Linking retains pairs
        detected before correction; unconstrained anchor calibration reports
        violations in the fitted population means.
    growth_curve : NDArray[np.float64]
        Mean ability by grade level.
    method : str
        Vertical scaling method used.
    reference_grade : int
        Index of the grade that defines the common scale.
    calibrated_models : dict[str | int, BaseItemModel]
        Models in the original item order for each grade, fitted jointly on
        the common scale by fixed- or floating-anchor calibration.
    scores : dict[str | int, ScoreResult]
        EAP person scores under each fitted grade distribution.
    latent_distributions : dict[str | int, GroupLatentDistribution]
        Estimated grade population distributions; the reference has mean
        zero and variance one.
    calibration_result : MultigroupFitResult | None
        Joint calibration diagnostics and fitted global-item model.
    item_maps : dict[str | int, NDArray[np.int_]]
        Original grade item positions mapped to physical-item columns in
        the joint calibration model.
    free_parameter_masks : dict[str | int, dict[str, NDArray[np.bool_]]]
        Effective local item-parameter masks for joint calibration, excluding
        fixed anchors and structural padding. The same masks are attached to
        the calibrated models for constrained-fit parameter counts and
        diagnostics.
    """

    grade_transformations: dict[str | int, tuple[float, float]]
    grade_means: dict[str | int, float]
    grade_sds: dict[str | int, float]
    linking_results: list[LinkingResult]
    monotonicity_violations: list[tuple]
    growth_curve: NDArray[np.float64]
    method: str
    reference_grade: int = 0
    calibrated_models: dict[str | int, BaseItemModel] = field(default_factory=dict)
    scores: dict[str | int, ScoreResult] = field(default_factory=dict)
    latent_distributions: dict[str | int, GroupLatentDistribution] = field(
        default_factory=dict
    )
    calibration_result: MultigroupFitResult | None = None
    item_maps: dict[str | int, NDArray[np.int_]] = field(default_factory=dict)
    free_parameter_masks: dict[str | int, dict[str, NDArray[np.bool_]]] = field(
        default_factory=dict
    )


@dataclass
class VerticalScaleDiagnostics:
    """Diagnostics for vertical scale quality assessment.

    Attributes
    ----------
    grade_separation : NDArray[np.float64]
        Effect size (Cohen's d) between adjacent grades.
    growth_per_grade : NDArray[np.float64]
        Mean ability growth from each grade to the next.
    cumulative_growth : NDArray[np.float64]
        Cumulative growth from the reference grade.
    anchor_stability : dict[tuple, float]
        RMSE of anchor item parameters after transformation for each
        grade pair.
    """

    grade_separation: NDArray[np.float64]
    growth_per_grade: NDArray[np.float64]
    cumulative_growth: NDArray[np.float64]
    anchor_stability: dict[tuple, float]


@dataclass
class _GradeModelInfo:
    """Internal: Model and theta info for a grade."""

    model: BaseItemModel
    theta: NDArray[np.float64]
    label: str | int
    n_items: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_items = int(self.model.n_items)


def vertical_scale(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None = None,
    method: Literal["chain", "concurrent", "fixed_anchor", "floating_anchor"] = "chain",
    linking_method: str = "stocking_lord",
    reference_grade: int = 0,
    enforce_monotonicity: bool = True,
    n_quadpts: int = 31,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> VerticalScaleResult:
    """Create a vertical scale linking multiple grade levels.

    Vertical scaling places ability estimates from different grade-level
    tests onto a common developmental scale, enabling growth measurement
    across grades.

    Parameters
    ----------
    grade_data : list[GradeData]
        Data for each grade level, ordered from lowest to highest grade.
    models : list[BaseItemModel] | None
        Pre-fitted IRT models for each grade. If None, linking modes fit
        separate 2PL models, while anchor modes calibrate a joint 2PL model.
        Fixed-anchor calibration uses the reference model's anchor values;
        it fits that reference model first when models are omitted. Anchor
        calibration accepts one common built-in unidimensional model family:
        1PL, 2PL, 3PL, 4PL, GRM, GPCM, PCM, or NRM. Grade forms may differ in
        width and item order; shared polytomous items must have matching
        category counts.
    method : str
        Vertical scaling method:
        - "chain": Sequential pairwise linking (default)
        - "concurrent": Simultaneous curve matching across grade calibrations
        - "fixed_anchor": Joint response calibration with reference anchors fixed
        - "floating_anchor": Joint response calibration with shared anchors estimated
    linking_method : str
        Method for pairwise linking. Concurrent scaling supports
        ``"stocking_lord"``, ``"tcc"``, and ``"haebara"``. Anchor calibration
        estimates parameters from responses and does not use this linker.
    reference_grade : int
        Index of grade to use as reference (scale origin). Default is 0
        (lowest grade).
    enforce_monotonicity : bool
        Anchor calibration constrains population means to be nondecreasing
        during density estimation, preserving the identified reference mean.
        Chain and concurrent linking adjust grade locations after fitting to
        make person-score means strictly increasing.
    n_quadpts : int
        Quadrature points for fixed- and floating-anchor calibration.
    max_iter : int
        Maximum joint calibration EM iterations.
    tol : float
        Joint calibration log-likelihood convergence tolerance.

    Returns
    -------
    VerticalScaleResult
        Transformations, means, and growth curve for linking; calibrated
        models, population distributions, and person scores for anchor
        calibration. Anchor calibration has no affine grade transformations.

    Raises
    ------
    ValueError
        If fewer than 2 grades provided or anchor structure is invalid.

    Examples
    --------
    >>> grade_data = [
    ...     GradeData("G3", responses_g3, anchor_items_above=[0, 1, 2]),
    ...     GradeData("G4", responses_g4, anchor_items_below=[10, 11, 12],
    ...               anchor_items_above=[0, 1, 2]),
    ...     GradeData("G5", responses_g5, anchor_items_below=[10, 11, 12]),
    ... ]
    >>> result = vertical_scale(grade_data)
    """
    _validate_vertical_inputs(
        grade_data,
        models,
        method,
        linking_method,
        reference_grade,
    )
    reference_grade = int(reference_grade)

    if method in ("fixed_anchor", "floating_anchor"):
        from mirt.equating._vertical_calibration import calibrate_vertical_scale

        return calibrate_vertical_scale(
            grade_data,
            models,
            method,
            reference_grade,
            enforce_monotonicity=enforce_monotonicity,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
        )

    grade_models = _fit_grade_models(grade_data, models)

    if method == "chain":
        result = _chain_vertical_scale(
            grade_data,
            grade_models,
            linking_method,
            reference_grade,
        )
    elif method == "concurrent":
        result = _concurrent_vertical_scale(
            grade_data,
            grade_models,
            linking_method,
            reference_grade,
        )

    if enforce_monotonicity:
        result = _enforce_monotonicity(result, grade_data, reference_grade)

    return result


def compute_vertical_diagnostics(
    result: VerticalScaleResult,
    grade_data: list[GradeData],
) -> VerticalScaleDiagnostics:
    """Compute diagnostics for vertical scale quality.

    Parameters
    ----------
    result : VerticalScaleResult
        Output from vertical_scale().
    grade_data : list[GradeData]
        Original grade data.

    Returns
    -------
    VerticalScaleDiagnostics
        Diagnostic statistics for the vertical scale.
    """
    if not 0 <= result.reference_grade < len(grade_data):
        raise ValueError("result.reference_grade is out of range for grade_data")
    labels = [gd.grade_label for gd in grade_data]

    means = np.array([result.grade_means[label] for label in labels])
    sds = np.array([result.grade_sds[label] for label in labels])

    growth_per_grade = np.diff(means)
    cumulative_growth = means - means[result.reference_grade]

    pooled_sds = np.sqrt((sds[:-1] ** 2 + sds[1:] ** 2) / 2)
    grade_separation = np.divide(
        growth_per_grade,
        pooled_sds,
        out=np.zeros_like(growth_per_grade),
        where=pooled_sds > 1e-10,
    )

    anchor_stability: dict[tuple, float] = {}
    for i, link_result in enumerate(result.linking_results):
        pair_key = (labels[i], labels[i + 1])
        if link_result.fit_statistics is not None:
            anchor_stability[pair_key] = link_result.fit_statistics.weighted_rmse
        else:
            anchor_stability[pair_key] = float("nan")

    return VerticalScaleDiagnostics(
        grade_separation=grade_separation,
        growth_per_grade=growth_per_grade,
        cumulative_growth=cumulative_growth,
        anchor_stability=anchor_stability,
    )


def vertical_scale_summary(result: VerticalScaleResult) -> str:
    """Generate a text summary of vertical scaling results.

    Parameters
    ----------
    result : VerticalScaleResult
        Output from vertical_scale().

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = [
        "Vertical Scaling Summary",
        "=" * 40,
        f"Method: {result.method}",
        f"Number of grades: {len(result.grade_means)}",
        f"Reference grade: {list(result.grade_means)[result.reference_grade]}",
        "",
        "Grade Statistics:",
        "-" * 40,
        (
            f"{'Grade':<15} {'Mean':>10} {'SD':>10} {'A':>8} {'B':>8}"
            if result.grade_transformations
            else f"{'Grade':<15} {'Mean':>10} {'SD':>10}"
        ),
        "-" * 40,
    ]

    for label in result.grade_means:
        mean = result.grade_means[label]
        sd = result.grade_sds[label]
        line = f"{str(label):<15} {mean:>10.3f} {sd:>10.3f}"
        if result.grade_transformations:
            A, B = result.grade_transformations[label]
            line += f" {A:>8.3f} {B:>8.3f}"
        lines.append(line)

    if result.calibration_result is not None:
        lines.extend(
            [
                "",
                "Joint Anchor Calibration:",
                f"Physical items: {result.calibration_result.model.n_items}",
                f"Log-likelihood: {result.calibration_result.log_likelihood:.4f}",
                f"EM iterations: {result.calibration_result.n_iterations}",
            ]
        )

    lines.extend(
        [
            "",
            "Growth Curve:",
            "-" * 40,
        ]
    )

    labels = list(result.grade_means.keys())
    for i, (label, growth) in enumerate(zip(labels, result.growth_curve, strict=True)):
        lines.append(f"  {label}: {growth:.3f}")

    if result.monotonicity_violations:
        lines.extend(
            [
                "",
                "Monotonicity Violations:",
            ]
        )
        for v1, v2 in result.monotonicity_violations:
            lines.append(f"  {v1} -> {v2}")

    return "\n".join(lines)


def plot_vertical_scale(
    result: VerticalScaleResult,
    show_error_bands: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    """Plot vertical scale growth curve.

    Parameters
    ----------
    result : VerticalScaleResult
        Output from vertical_scale().
    show_error_bands : bool
        If True, show +/- 1 SD bands.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(result.grade_means.keys())
    means = np.array([result.grade_means[label] for label in labels])
    sds = np.array([result.grade_sds[label] for label in labels])

    x = np.arange(len(labels))

    ax.plot(x, means, "o-", linewidth=2, markersize=8, label="Mean ability")

    if show_error_bands:
        ax.fill_between(
            x,
            means - sds,
            means + sds,
            alpha=0.3,
            label="±1 SD",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_xlabel("Grade Level")
    ax.set_ylabel("Ability (θ)")
    ax.set_title("Vertical Scale Growth Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def _validate_vertical_inputs(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None,
    method: str,
    linking_method: str,
    reference_grade: int,
) -> None:
    """Validate scale configuration before fitting or scoring any models."""
    if method not in _VERTICAL_METHODS:
        raise ValueError(f"Unknown vertical scaling method: {method}")
    if linking_method not in _LINKING_METHODS:
        raise ValueError(f"Unknown linking method: {linking_method}")
    if method == "concurrent" and linking_method not in {
        "stocking_lord",
        "tcc",
        "haebara",
    }:
        raise ValueError("Concurrent vertical scaling requires a curve-matching linker")
    if len(grade_data) < 2:
        raise ValueError(
            f"Vertical scaling requires at least 2 grades, got {len(grade_data)}"
        )
    if isinstance(reference_grade, (bool, np.bool_)) or not isinstance(
        reference_grade, (int, np.integer)
    ):
        raise ValueError("reference_grade must be an integer index")
    if reference_grade < 0 or reference_grade >= len(grade_data):
        raise ValueError(
            f"reference_grade must be in [0, {len(grade_data)}), got {reference_grade}"
        )

    labels = [gd.grade_label for gd in grade_data]
    if any(
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (str, int, np.integer))
        for label in labels
    ):
        raise ValueError("Grade labels must be strings or integers")
    if len(set(labels)) != len(labels):
        raise ValueError("Grade labels must be unique")

    from mirt.utils.data import validate_responses

    for gd in grade_data:
        responses = validate_responses(gd.responses)
        if responses.shape[0] < 2:
            raise ValueError(
                f"Grade '{gd.grade_label}' must contain at least 2 response rows"
            )

    if models is not None:
        if len(models) != len(grade_data):
            raise ValueError(
                f"models must contain one model per grade: expected "
                f"{len(grade_data)}, got {len(models)}"
            )
        for gd, model in zip(grade_data, models, strict=True):
            n_response_items = np.asarray(gd.responses).shape[1]
            if model.n_items != n_response_items:
                raise ValueError(
                    f"Model for grade '{gd.grade_label}' has {model.n_items} items, "
                    f"but responses have {n_response_items}"
                )
            if model.n_factors != 1:
                raise ValueError("Vertical scaling requires unidimensional models")

    _validate_anchor_structure(grade_data)


def _validate_anchor_structure(grade_data: list[GradeData]) -> None:
    """Validate that adjacent grades have explicit, usable anchor mappings.

    When only one side of an adjacent pair is supplied, matching item indices
    are inferred for the other form. Supplying both sides supports anchors in
    different item positions.
    """
    for i in range(len(grade_data) - 1):
        lower = grade_data[i]
        upper = grade_data[i + 1]
        anchors_lower, anchors_upper = _resolve_anchor_pair(lower, upper)
        if len(anchors_lower) != len(anchors_upper):
            raise ValueError(
                f"Anchor item count mismatch between grades "
                f"'{lower.grade_label}' ({len(anchors_lower)}) and "
                f"'{upper.grade_label}' ({len(anchors_upper)})"
            )
        if len(anchors_lower) < 2:
            raise ValueError(
                f"At least 2 anchor items are required between grades "
                f"'{lower.grade_label}' and '{upper.grade_label}'"
            )

        _validate_anchor_indices(
            anchors_lower,
            np.asarray(lower.responses).shape[1],
            lower.grade_label,
        )
        _validate_anchor_indices(
            anchors_upper,
            np.asarray(upper.responses).shape[1],
            upper.grade_label,
        )


def _resolve_anchor_pair(
    lower: GradeData, upper: GradeData
) -> tuple[list[int], list[int]]:
    """Resolve corresponding anchor indices for one adjacent grade pair."""
    anchors_lower = lower.anchor_items_above
    anchors_upper = upper.anchor_items_below
    if anchors_lower is None and anchors_upper is None:
        raise ValueError(
            f"No anchor items connecting grade '{lower.grade_label}' "
            f"to grade '{upper.grade_label}'. Specify anchor_items_above "
            f"for the lower grade or anchor_items_below for the upper grade."
        )
    if anchors_lower is None:
        anchors_lower = anchors_upper
    if anchors_upper is None:
        anchors_upper = anchors_lower
    assert anchors_lower is not None and anchors_upper is not None
    return list(anchors_lower), list(anchors_upper)


def _validate_anchor_indices(
    anchors: list[int], n_items: int, label: str | int
) -> None:
    """Validate anchor index type, uniqueness, and form bounds."""
    normalized: list[int] = []
    for anchor in anchors:
        if isinstance(anchor, (bool, np.bool_)) or not isinstance(
            anchor, (int, np.integer)
        ):
            raise ValueError(f"Anchor indices for grade '{label}' must be integers")
        index = int(anchor)
        if index < 0 or index >= n_items:
            raise ValueError(
                f"Anchor index {index} out of range for grade '{label}' "
                f"with {n_items} items"
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Anchor indices for grade '{label}' must be unique")


def _fit_grade_models(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None,
) -> list[_GradeModelInfo]:
    """Fit IRT models for each grade or use provided models."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    grade_models = []

    for i, gd in enumerate(grade_data):
        if models is not None:
            model = models[i]
        else:
            result = fit_mirt(gd.responses, model="2PL", verbose=False)
            model = result.model

        score_result = fscores(model, gd.responses, method="EAP")
        theta = np.asarray(score_result.theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        expected_shape = (np.asarray(gd.responses).shape[0], 1)
        if theta.shape != expected_shape:
            raise ValueError(
                f"Scores for grade '{gd.grade_label}' have shape {theta.shape}; "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError(f"Scores for grade '{gd.grade_label}' must be finite")

        grade_models.append(
            _GradeModelInfo(
                model=model,
                theta=theta,
                label=gd.grade_label,
            )
        )

    return grade_models


def _chain_vertical_scale(
    grade_data: list[GradeData],
    grade_models: list[_GradeModelInfo],
    linking_method: str,
    reference_grade: int,
) -> VerticalScaleResult:
    """Perform chain vertical scaling via sequential pairwise linking."""
    from mirt.equating.chain import chain_link

    pairs = [
        _resolve_anchor_pair(lower, upper)
        for lower, upper in zip(grade_data[:-1], grade_data[1:], strict=True)
    ]
    chain = chain_link(
        [gm.model for gm in grade_models],
        pairs,
        method=linking_method,
        reference_index=reference_grade,
        compute_drift=False,
    )
    return _vertical_scale_result(
        grade_models,
        chain.cumulative_A,
        chain.cumulative_B,
        chain.pairwise_results,
        "chain",
        reference_grade,
    )


def _vertical_scale_result(
    grade_models: list[_GradeModelInfo],
    final_A: list[float],
    final_B: list[float],
    linking_results: list[LinkingResult],
    method: str,
    reference_grade: int,
) -> VerticalScaleResult:
    """Summarize scores under the estimated common-scale transformations."""
    grade_transformations = {}
    grade_means = {}
    grade_sds = {}

    for i, gm in enumerate(grade_models):
        A, B = final_A[i], final_B[i]
        grade_transformations[gm.label] = (A, B)

        theta_transformed = A * gm.theta + B
        grade_means[gm.label] = float(np.mean(theta_transformed))
        grade_sds[gm.label] = float(np.std(theta_transformed, ddof=1))

    growth_curve = np.array([grade_means[gm.label] for gm in grade_models])

    return VerticalScaleResult(
        grade_transformations=grade_transformations,
        grade_means=grade_means,
        grade_sds=grade_sds,
        linking_results=linking_results,
        monotonicity_violations=[],
        growth_curve=growth_curve,
        method=method,
        reference_grade=reference_grade,
    )


def _concurrent_vertical_scale(
    grade_data: list[GradeData],
    grade_models: list[_GradeModelInfo],
    linking_method: str,
    reference_grade: int,
) -> VerticalScaleResult:
    """Match all grade curves jointly on a common reference grid."""
    from mirt.equating.chain import concurrent_link
    from mirt.equating.linking import (
        LinkingConstants,
        LinkingResult,
        _compute_anchor_diagnostics,
        _compute_fit_statistics,
        _link_form,
        _validate_curve_grid,
    )

    pairs = [
        _resolve_anchor_pair(lower, upper)
        for lower, upper in zip(grade_data[:-1], grade_data[1:], strict=True)
    ]
    anchor_matrices = [
        [list(zip(anchors_lower, anchors_upper, strict=True))]
        for anchors_lower, anchors_upper in pairs
    ]
    transformations = concurrent_link(
        [gm.model for gm in grade_models],
        anchor_matrices,
        method=linking_method,
        max_iter=500,
        tol=1e-12,
        reference_index=reference_grade,
    )
    reference_A, reference_B = transformations[reference_grade]
    final_A = [A / reference_A for A, _ in transformations]
    final_B = [(B - reference_B) / reference_A for _, B in transformations]

    theta_grid, weights = _validate_curve_grid((-4.0, 4.0), 61, None)
    linking_results = []
    for index, (anchors_lower, anchors_upper) in enumerate(pairs):
        form_lower = _link_form(grade_models[index].model, anchors_lower, "lower")
        form_upper = _link_form(grade_models[index + 1].model, anchors_upper, "upper")
        # Recover the upper -> lower map implied by the joint common metric.
        A = final_A[index + 1] / final_A[index]
        B = (final_B[index + 1] - final_B[index]) / final_A[index]
        linking_results.append(
            LinkingResult(
                constants=LinkingConstants(A, B, method=linking_method),
                anchor_items=anchors_lower,
                anchor_diagnostics=_compute_anchor_diagnostics(
                    form_lower, form_upper, A, B, anchors_lower, theta_grid
                ),
                fit_statistics=_compute_fit_statistics(
                    form_lower, form_upper, A, B, theta_grid, weights
                ),
                convergence_info={
                    "method": linking_method,
                    "success": True,
                    "concurrent": True,
                },
            )
        )

    return _vertical_scale_result(
        grade_models, final_A, final_B, linking_results, "concurrent", reference_grade
    )


def _enforce_monotonicity(
    result: VerticalScaleResult,
    grade_data: list[GradeData],
    reference_grade: int,
) -> VerticalScaleResult:
    """Shift grade locations to ensure growth while preserving the reference."""
    labels = [gd.grade_label for gd in grade_data]
    means = np.array([result.grade_means[label] for label in labels])
    sds = np.array([result.grade_sds[label] for label in labels])

    violation_indices: set[int] = set()
    adjusted_means = means.copy()

    for i in range(reference_grade - 1, -1, -1):
        if adjusted_means[i] >= adjusted_means[i + 1]:
            violation_indices.add(i)
            adjusted_means[i] = adjusted_means[i + 1] - _minimum_growth(
                sds[i], sds[i + 1]
            )

    for i in range(reference_grade, len(means) - 1):
        if adjusted_means[i + 1] <= adjusted_means[i]:
            violation_indices.add(i)
            adjusted_means[i + 1] = adjusted_means[i] + _minimum_growth(
                sds[i], sds[i + 1]
            )

    if not violation_indices:
        return result

    new_transformations = {}
    new_means = {}

    for i, label in enumerate(labels):
        old_A, old_B = result.grade_transformations[label]
        old_mean = result.grade_means[label]
        mean_shift = float(adjusted_means[i] - old_mean)
        new_transformations[label] = (old_A, old_B + mean_shift)
        new_means[label] = float(adjusted_means[i])

    violations = [(labels[i], labels[i + 1]) for i in sorted(violation_indices)]

    return VerticalScaleResult(
        grade_transformations=new_transformations,
        grade_means=new_means,
        grade_sds=result.grade_sds,
        linking_results=result.linking_results,
        monotonicity_violations=violations,
        growth_curve=adjusted_means,
        method=result.method,
        reference_grade=reference_grade,
    )


def _minimum_growth(sd_lower: float, sd_upper: float) -> float:
    """Return a stable positive spacing for adjacent grade means."""
    pooled_sd = float(np.sqrt((sd_lower**2 + sd_upper**2) / 2))
    return max(0.1 * pooled_sd, 1e-6)
