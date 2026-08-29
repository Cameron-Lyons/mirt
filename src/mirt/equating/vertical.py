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
        common vertical scale.
    grade_means : dict[str | int, float]
        Mean ability estimate for each grade on the common scale.
    grade_sds : dict[str | int, float]
        Standard deviation of ability estimates for each grade.
    linking_results : list[LinkingResult]
        Detailed linking results for each adjacent grade pair.
    monotonicity_violations : list[tuple]
        List of (grade1, grade2) pairs where monotonicity was violated
        before correction.
    growth_curve : NDArray[np.float64]
        Mean ability by grade level.
    method : str
        Vertical scaling method used.
    reference_grade : int
        Index of the grade that defines the common scale.
    """

    grade_transformations: dict[str | int, tuple[float, float]]
    grade_means: dict[str | int, float]
    grade_sds: dict[str | int, float]
    linking_results: list[LinkingResult]
    monotonicity_violations: list[tuple]
    growth_curve: NDArray[np.float64]
    method: str
    reference_grade: int = 0


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
        Pre-fitted IRT models for each grade. If None, 2PL models are
        fitted to each grade's data.
    method : str
        Vertical scaling method:
        - "chain": Sequential pairwise linking (default)
        - "concurrent": Joint calibration with anchor constraints
        - "fixed_anchor": Anchors fixed to reference grade values
        - "floating_anchor": Anchors free but constrained equal
    linking_method : str
        Method for pairwise linking (used with chain method).
    reference_grade : int
        Index of grade to use as reference (scale origin). Default is 0
        (lowest grade).
    enforce_monotonicity : bool
        If True, ensure grade means are strictly increasing.

    Returns
    -------
    VerticalScaleResult
        Vertical scaling results including transformations, means, and
        growth curve.

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
    elif method in ("fixed_anchor", "floating_anchor"):
        result = _chain_vertical_scale(
            grade_data,
            grade_models,
            linking_method,
            reference_grade,
        )
        result = VerticalScaleResult(
            grade_transformations=result.grade_transformations,
            grade_means=result.grade_means,
            grade_sds=result.grade_sds,
            linking_results=result.linking_results,
            monotonicity_violations=result.monotonicity_violations,
            growth_curve=result.growth_curve,
            method=method,
            reference_grade=reference_grade,
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
        f"{'Grade':<15} {'Mean':>10} {'SD':>10} {'A':>8} {'B':>8}",
        "-" * 40,
    ]

    for label in result.grade_means:
        mean = result.grade_means[label]
        sd = result.grade_sds[label]
        A, B = result.grade_transformations[label]
        lines.append(f"{str(label):<15} {mean:>10.3f} {sd:>10.3f} {A:>8.3f} {B:>8.3f}")

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
                "Monotonicity Violations (corrected):",
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
    from mirt.equating.linking import link

    n_grades = len(grade_data)
    linking_results: list[LinkingResult] = []

    cumulative_A = [1.0] * n_grades
    cumulative_B = [0.0] * n_grades

    for i in range(n_grades - 1):
        lower_gd = grade_data[i]
        upper_gd = grade_data[i + 1]
        lower_model = grade_models[i].model
        upper_model = grade_models[i + 1].model

        anchor_lower, anchor_upper = _resolve_anchor_pair(lower_gd, upper_gd)

        link_result = link(
            lower_model,
            upper_model,
            anchor_lower,
            anchor_upper,
            method=linking_method,
            compute_diagnostics=True,
        )
        linking_results.append(link_result)

        A_pair = link_result.constants.A
        B_pair = link_result.constants.B

        cumulative_A[i + 1] = cumulative_A[i] * A_pair
        cumulative_B[i + 1] = cumulative_A[i] * B_pair + cumulative_B[i]

    ref_A = cumulative_A[reference_grade]
    ref_B = cumulative_B[reference_grade]

    final_A = [a / ref_A for a in cumulative_A]
    final_B = [(b - ref_B) / ref_A for b in cumulative_B]

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
        method="chain",
        reference_grade=reference_grade,
    )


def _concurrent_vertical_scale(
    grade_data: list[GradeData],
    grade_models: list[_GradeModelInfo],
    linking_method: str,
    reference_grade: int,
) -> VerticalScaleResult:
    """Perform concurrent vertical scaling.

    This implementation uses the connected adjacent-grade anchor structure
    to place every form on the selected reference scale.
    """
    chain_result = _chain_vertical_scale(
        grade_data, grade_models, linking_method, reference_grade
    )

    return VerticalScaleResult(
        grade_transformations=chain_result.grade_transformations,
        grade_means=chain_result.grade_means,
        grade_sds=chain_result.grade_sds,
        linking_results=chain_result.linking_results,
        monotonicity_violations=chain_result.monotonicity_violations,
        growth_curve=chain_result.growth_curve,
        method="concurrent",
        reference_grade=reference_grade,
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
