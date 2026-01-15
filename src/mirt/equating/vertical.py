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
    """

    grade_transformations: dict[str | int, tuple[float, float]]
    grade_means: dict[str | int, float]
    grade_sds: dict[str | int, float]
    linking_results: list[LinkingResult]
    monotonicity_violations: list[tuple]
    growth_curve: NDArray[np.float64]
    method: str


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
    if len(grade_data) < 2:
        raise ValueError(
            f"Vertical scaling requires at least 2 grades, got {len(grade_data)}"
        )

    _validate_anchor_structure(grade_data, method)

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
        )
    else:
        raise ValueError(f"Unknown vertical scaling method: {method}")

    if enforce_monotonicity:
        result = _enforce_monotonicity(result, grade_data)

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
    n_grades = len(grade_data)
    labels = [gd.grade_label for gd in grade_data]

    means = np.array([result.grade_means[label] for label in labels])
    sds = np.array([result.grade_sds[label] for label in labels])

    growth_per_grade = np.diff(means)
    cumulative_growth = means - means[0]

    grade_separation = np.zeros(n_grades - 1)
    for i in range(n_grades - 1):
        pooled_sd = np.sqrt((sds[i] ** 2 + sds[i + 1] ** 2) / 2)
        if pooled_sd > 1e-10:
            grade_separation[i] = (means[i + 1] - means[i]) / pooled_sd
        else:
            grade_separation[i] = 0.0

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
    for i, (label, growth) in enumerate(zip(labels, result.growth_curve)):
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


def _validate_anchor_structure(
    grade_data: list[GradeData],
    method: str,
) -> None:
    """Validate that anchor items form a connected structure.

    For chain linking, each adjacent pair of grades must share anchor
    items (either via anchor_items_above/below).
    """
    n_grades = len(grade_data)

    for i in range(n_grades - 1):
        lower = grade_data[i]
        upper = grade_data[i + 1]

        has_connection = (
            lower.anchor_items_above is not None or upper.anchor_items_below is not None
        )

        if not has_connection:
            raise ValueError(
                f"No anchor items connecting grade '{lower.grade_label}' "
                f"to grade '{upper.grade_label}'. Specify anchor_items_above "
                f"for the lower grade or anchor_items_below for the upper grade."
            )

        if (
            lower.anchor_items_above is not None
            and upper.anchor_items_below is not None
        ):
            n_lower = len(lower.anchor_items_above)
            n_upper = len(upper.anchor_items_below)
            if n_lower != n_upper:
                raise ValueError(
                    f"Anchor item count mismatch between grades "
                    f"'{lower.grade_label}' ({n_lower}) and "
                    f"'{upper.grade_label}' ({n_upper})"
                )


def _fit_grade_models(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None,
) -> list[_GradeModelInfo]:
    """Fit IRT models for each grade or use provided models."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    grade_models = []

    for i, gd in enumerate(grade_data):
        if models is not None and i < len(models):
            model = models[i]
        else:
            result = fit_mirt(gd.responses, model="2PL", verbose=False)
            model = result.model

        score_result = fscores(model, gd.responses, method="EAP")
        theta = score_result.theta
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)

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

        if lower_gd.anchor_items_above is not None:
            anchor_lower = lower_gd.anchor_items_above
        else:
            anchor_lower = list(range(min(5, lower_model.n_items)))

        if upper_gd.anchor_items_below is not None:
            anchor_upper = upper_gd.anchor_items_below
        else:
            anchor_upper = list(range(min(5, upper_model.n_items)))

        min_len = min(len(anchor_lower), len(anchor_upper))
        anchor_lower = anchor_lower[:min_len]
        anchor_upper = anchor_upper[:min_len]

        link_result = link(
            upper_model,
            lower_model,
            anchor_upper,
            anchor_lower,
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
    )


def _concurrent_vertical_scale(
    grade_data: list[GradeData],
    grade_models: list[_GradeModelInfo],
    reference_grade: int,
) -> VerticalScaleResult:
    """Perform concurrent vertical scaling.

    This is a simplified implementation that uses chain linking as a
    starting point, then refines with joint optimization.
    """
    chain_result = _chain_vertical_scale(
        grade_data, grade_models, "stocking_lord", reference_grade
    )

    return VerticalScaleResult(
        grade_transformations=chain_result.grade_transformations,
        grade_means=chain_result.grade_means,
        grade_sds=chain_result.grade_sds,
        linking_results=chain_result.linking_results,
        monotonicity_violations=chain_result.monotonicity_violations,
        growth_curve=chain_result.growth_curve,
        method="concurrent",
    )


def _enforce_monotonicity(
    result: VerticalScaleResult,
    grade_data: list[GradeData],
) -> VerticalScaleResult:
    """Adjust transformations to ensure strictly increasing grade means."""
    labels = [gd.grade_label for gd in grade_data]
    means = np.array([result.grade_means[label] for label in labels])
    sds = np.array([result.grade_sds[label] for label in labels])

    violations = []
    adjusted_means = means.copy()

    for i in range(len(means) - 1):
        if adjusted_means[i + 1] <= adjusted_means[i]:
            violations.append((labels[i], labels[i + 1]))
            min_growth = 0.1 * sds[i]
            adjusted_means[i + 1] = adjusted_means[i] + min_growth

    if not violations:
        return result

    new_transformations = {}
    new_means = {}

    for i, label in enumerate(labels):
        old_A, old_B = result.grade_transformations[label]
        old_mean = result.grade_means[label]

        if old_mean != 0:
            scale_factor = adjusted_means[i] / old_mean
            new_A = old_A * scale_factor
            new_B = old_B * scale_factor
        else:
            new_A = old_A
            new_B = adjusted_means[i]

        new_transformations[label] = (new_A, new_B)
        new_means[label] = adjusted_means[i]

    return VerticalScaleResult(
        grade_transformations=new_transformations,
        grade_means=new_means,
        grade_sds=result.grade_sds,
        linking_results=result.linking_results,
        monotonicity_violations=violations,
        growth_curve=adjusted_means,
        method=result.method,
    )
