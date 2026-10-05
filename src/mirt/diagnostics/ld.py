"""Local Dependence (LD) statistics for IRT models.

This module provides statistics for detecting violations of local independence,
a key assumption in IRT. When local independence is violated, item responses
are correlated beyond what is explained by the latent trait(s).

Statistics implemented:
- Yen's Q3 statistic (Yen, 1984)
- Chen & Thissen's LD χ² statistic (Chen & Thissen, 1997)
- G² (likelihood ratio) statistic
- Adjusted residual correlations

References:
    Chen, W. H., & Thissen, D. (1997). Local dependence indexes for item pairs
        using item response theory. Journal of Educational and Behavioral
        Statistics, 22(3), 265-289.
    Yen, W. M. (1984). Effects of local item dependence on the fit and equating
        performance of the three-parameter logistic model. Applied Psychological
        Measurement, 8(2), 125-145.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt._correlation import q3_correlations
from mirt._local_dependence import ld_pair_statistics
from mirt.constants import PROB_EPSILON
from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


@dataclass
class LDResult:
    """Results from local dependence analysis.

    Attributes
    ----------
    q3_matrix : NDArray
        Matrix of Yen's Q3 statistics for each item pair
    ld_chi2_matrix : NDArray
        Matrix of LD χ² statistics for each item pair
    g2_matrix : NDArray
        Matrix of G² (LR) statistics for each item pair
    adj_residual_corr : NDArray
        Matrix of adjusted residual correlations
    q3_flagged : list of tuple
        List of (item_i, item_j, q3_value) tuples flagged for LD
    chi2_flagged : list of tuple
        List of (item_i, item_j, chi2_value, selected_p_value) tuples flagged
        for LD. The final value reflects ``p_adjustment``.
    item_names : list of str or None
        Item names for labeling
    chi2_p_value_matrix : NDArray or None
        Raw chi-square p-values for every eligible item pair.
    chi2_adjusted_p_value_matrix : NDArray or None
        Multiplicity-adjusted chi-square p-values for every eligible pair.
    p_adjustment : str
        Multiple-testing adjustment used for chi-square pair selection.
    """

    q3_matrix: NDArray[np.float64]
    ld_chi2_matrix: NDArray[np.float64]
    g2_matrix: NDArray[np.float64]
    adj_residual_corr: NDArray[np.float64]
    q3_flagged: list[tuple[int, int, float]]
    chi2_flagged: list[tuple[int, int, float, float]]
    item_names: list[str] | None = None
    chi2_p_value_matrix: NDArray[np.float64] | None = None
    chi2_adjusted_p_value_matrix: NDArray[np.float64] | None = None
    p_adjustment: PValueAdjustment = "none"

    def summary(self) -> str:
        """Generate a formatted summary of LD results."""
        lines = [
            "Local Dependence Analysis Summary",
            "=" * 60,
            "",
        ]

        q3_upper = self.q3_matrix[np.triu_indices_from(self.q3_matrix, k=1)]
        lines.extend(
            [
                "Yen's Q3 Statistics:",
                f"  Mean Q3:     {np.mean(q3_upper):.4f}",
                f"  Max Q3:      {np.max(q3_upper):.4f}",
                f"  Min Q3:      {np.min(q3_upper):.4f}",
                f"  Pairs > 0.2: {np.sum(np.abs(q3_upper) > 0.2)}",
                "",
            ]
        )

        if self.q3_flagged:
            lines.append("Flagged item pairs (|Q3| > 0.2):")
            for i, j, q3 in sorted(self.q3_flagged, key=lambda x: -abs(x[2]))[:10]:
                if self.item_names:
                    lines.append(
                        f"  {self.item_names[i]} - {self.item_names[j]}: Q3 = {q3:.4f}"
                    )
                else:
                    lines.append(f"  Item {i + 1} - Item {j + 1}: Q3 = {q3:.4f}")
            lines.append("")

        chi2_upper = self.ld_chi2_matrix[np.triu_indices_from(self.ld_chi2_matrix, k=1)]
        p_value_label = (
            "p" if self.p_adjustment == "none" else f"{self.p_adjustment}-adjusted p"
        )
        lines.extend(
            [
                "LD Chi-Square Statistics:",
                f"  Mean χ²:        {np.nanmean(chi2_upper):.4f}",
                f"  Max χ²:         {np.nanmax(chi2_upper):.4f}",
                f"  Pairs {p_value_label} < 0.05: {len(self.chi2_flagged)}",
                "",
            ]
        )

        if self.chi2_flagged:
            lines.append(f"Flagged item pairs ({p_value_label} < 0.05):")
            for i, j, chi2, p in sorted(self.chi2_flagged, key=lambda x: x[3])[:10]:
                if self.item_names:
                    lines.append(
                        f"  {self.item_names[i]} - {self.item_names[j]}: χ² = {chi2:.2f}, p = {p:.4f}"
                    )
                else:
                    lines.append(
                        f"  Item {i + 1} - Item {j + 1}: χ² = {chi2:.2f}, p = {p:.4f}"
                    )
            lines.append("")

        lines.extend(
            [
                "Interpretation:",
                "  |Q3| > 0.2 suggests local dependence (Yen, 1984)",
                "  Significant LD χ² (p < 0.05) indicates model misfit",
                "  Consider: testlet models, bifactor models, or removing items",
            ]
        )

        return "\n".join(lines)


def compute_ld_statistics(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
    q3_threshold: float = 0.2,
    alpha: float = 0.05,
    p_adjust: PValueAdjustment = "none",
) -> LDResult:
    """Compute local dependence statistics for all item pairs.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : NDArray of shape (n_persons, n_items)
        Response matrix with integer responses
    theta : NDArray of shape (n_persons,) or (n_persons, n_factors), optional
        Ability estimates. If None, EAP estimates are computed.
    n_quadpts : int
        Number of quadrature points for computing expected values
    q3_threshold : float
        Threshold for flagging Q3 values (default 0.2)
    alpha : float
        Significance level for flagging LD χ² values
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, default="none"
        Multiple-testing adjustment across eligible item pairs.

    Returns
    -------
    LDResult
        Object containing all LD statistics and flagged pairs
    """
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    responses = np.asarray(responses)
    n_items = responses.shape[1]
    model, theta = _ability_matrix(model, responses, theta, n_quadpts=n_quadpts)

    residuals, positive_probabilities = _compute_residuals_and_positive_probabilities(
        model,
        responses,
        theta,
    )

    q3_matrix = _compute_q3(residuals, responses)
    adj_residual_corr = _adjust_q3(q3_matrix)

    ld_chi2_matrix, g2_matrix = _compute_ld_chi2_g2(
        model,
        responses,
        theta,
        n_quadpts,
        positive_probabilities=positive_probabilities,
    )

    rows, columns = np.triu_indices(n_items, k=1)
    q3_values = q3_matrix[rows, columns]
    q3_mask = np.abs(q3_values) > q3_threshold
    q3_flagged = [
        (int(i), int(j), float(value))
        for i, j, value in zip(
            rows[q3_mask], columns[q3_mask], q3_values[q3_mask], strict=True
        )
    ]

    chi2_values = ld_chi2_matrix[rows, columns]
    p_values, adjusted_p_values = _chi2_pair_p_values(chi2_values, p_adjust)
    chi2_mask = adjusted_p_values < alpha
    chi2_flagged = [
        (int(i), int(j), float(value), float(p_value))
        for i, j, value, p_value in zip(
            rows[chi2_mask],
            columns[chi2_mask],
            chi2_values[chi2_mask],
            adjusted_p_values[chi2_mask],
            strict=True,
        )
    ]

    chi2_p_value_matrix = _symmetric_pair_matrix(
        n_items,
        rows,
        columns,
        p_values,
    )
    chi2_adjusted_p_value_matrix = _symmetric_pair_matrix(
        n_items,
        rows,
        columns,
        adjusted_p_values,
    )

    item_names = model.item_names if hasattr(model, "item_names") else None

    return LDResult(
        q3_matrix=q3_matrix,
        ld_chi2_matrix=ld_chi2_matrix,
        g2_matrix=g2_matrix,
        adj_residual_corr=adj_residual_corr,
        q3_flagged=q3_flagged,
        chi2_flagged=chi2_flagged,
        item_names=item_names,
        chi2_p_value_matrix=chi2_p_value_matrix,
        chi2_adjusted_p_value_matrix=chi2_adjusted_p_value_matrix,
        p_adjustment=p_adjust,
    )


def compute_q3(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute Yen's Q3 statistics for all item pairs.

    Q3 is the correlation between residuals for pairs of items.
    Under local independence, Q3 should be approximately -1/(n_items - 1).

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates

    Returns
    -------
    NDArray
        Matrix of Q3 statistics
    """
    responses = np.asarray(responses)
    model, theta = _ability_matrix(model, responses, theta)

    residuals, _ = _compute_residuals_and_positive_probabilities(
        model, responses, theta
    )
    return _compute_q3(residuals, responses)


def compute_ld_chi2(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
    p_adjust: PValueAdjustment = "none",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Chen & Thissen's LD χ² statistics.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model, or the ``FitResult`` of a fit. Omitted abilities
        are then EAP scores under its estimated latent covariance.
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates
    n_quadpts : int
        Number of quadrature points
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, default="none"
        Multiple-testing adjustment across eligible item pairs.

    Returns
    -------
    chi2_matrix : NDArray
        Matrix of LD χ² statistics
    p_value_matrix : NDArray
        Matrix of raw or adjusted p-values, according to ``p_adjust``.
    """
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    responses = np.asarray(responses)
    n_items = responses.shape[1]
    model, theta = _ability_matrix(model, responses, theta, n_quadpts=n_quadpts)

    chi2_matrix, _ = _compute_ld_chi2_g2(model, responses, theta, n_quadpts)

    p_value_matrix = np.zeros_like(chi2_matrix)
    rows, columns = np.triu_indices(n_items, k=1)
    values = chi2_matrix[rows, columns]
    _, p_values = _chi2_pair_p_values(values, p_adjust)
    valid = ~np.isnan(p_values)
    valid_rows = rows[valid]
    valid_columns = columns[valid]
    p_value_matrix[valid_rows, valid_columns] = p_values[valid]
    p_value_matrix[valid_columns, valid_rows] = p_values[valid]

    return chi2_matrix, p_value_matrix


def _ability_matrix(
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None,
    **fscores_kwargs: int,
) -> tuple[BaseItemModel, NDArray[np.float64]]:
    """Return the item model and person abilities as an ``(n, n_factors)`` matrix.

    Missing abilities default to EAP scores under the latent population of a
    ``FitResult``, as :func:`mirt.fscores` computes them.
    """
    from mirt.results._common import resolve_item_model

    model: BaseItemModel = resolve_item_model(model_or_result)
    if theta is None:
        from mirt.scoring import fscores

        theta = fscores(
            model_or_result, responses, method="EAP", **fscores_kwargs
        ).theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and responses.shape[0] > 1:
        theta = theta.T
    return model, theta


def _compute_residuals_and_positive_probabilities(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute residuals and positive-response probabilities in one pass."""
    from mirt.diagnostics.residuals import (
        _all_item_expected_value_variance,
        _item_expected_value_variance,
    )

    n_persons, n_items = responses.shape
    batched = _all_item_expected_value_variance(model, theta, n_items)
    if batched is not None:
        probabilities, expected, variance = batched
        valid = responses >= 0
        residuals = np.where(
            valid,
            (responses - expected) / np.sqrt(variance + PROB_EPSILON),
            np.nan,
        )
        positive = (
            1.0 - probabilities[:, :, 0] if probabilities.ndim == 3 else probabilities
        )
        return residuals, np.asarray(positive, dtype=np.float64)

    residuals = np.full((n_persons, n_items), np.nan)
    positive = np.empty((n_persons, n_items), dtype=np.float64)
    for item_idx in range(n_items):
        probabilities, expected, variance = _item_expected_value_variance(
            model,
            theta,
            item_idx,
        )
        positive[:, item_idx] = (
            1.0 - probabilities[:, 0] if probabilities.ndim == 2 else probabilities
        )
        valid = responses[:, item_idx] >= 0
        residuals[valid, item_idx] = (
            responses[valid, item_idx] - expected[valid]
        ) / np.sqrt(variance[valid] + PROB_EPSILON)
    return residuals, positive


def _compute_q3(
    residuals: NDArray[np.float64],
    responses: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute Q3 (residual correlation) matrix."""
    return q3_correlations(residuals, responses >= 0)


def _adjust_q3(q3_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Adjust Q3 for its expected negative correlation under independence."""
    n_items = q3_matrix.shape[0]
    if n_items < 2:
        return q3_matrix.copy()
    return q3_matrix + 1.0 / (n_items - 1)


def _compute_ld_chi2_g2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    n_quadpts: int,
    *,
    positive_probabilities: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute LD χ² and G² statistics for item pairs.

    Uses the Chen & Thissen (1997) approach comparing observed and
    expected cross-classification frequencies.
    """
    del n_quadpts  # Retained for compatibility with the public call path.
    n_persons, n_items = responses.shape

    if positive_probabilities is None:
        positive_probabilities = np.empty((n_persons, n_items), dtype=np.float64)
        for item_idx in range(n_items):
            probabilities = np.asarray(model.probability(theta, item_idx))
            if probabilities.ndim == 2:
                probabilities = 1.0 - probabilities[:, 0]
            positive_probabilities[:, item_idx] = probabilities
    elif positive_probabilities.shape != (n_persons, n_items):
        raise ValueError("positive_probabilities must match the response matrix shape")

    chi2_matrix, g2_matrix = ld_pair_statistics(responses, positive_probabilities)
    assert g2_matrix is not None
    return chi2_matrix, g2_matrix


def flag_ld_pairs(
    ld_result: LDResult,
    q3_threshold: float = 0.2,
    chi2_alpha: float = 0.05,
    method: str = "q3",
    p_adjust: PValueAdjustment | None = None,
) -> list[tuple[int, int]]:
    """Get list of item pairs flagged for local dependence.

    Parameters
    ----------
    ld_result : LDResult
        Result from compute_ld_statistics
    q3_threshold : float
        Threshold for Q3 (absolute value)
    chi2_alpha : float
        Significance level for chi-square test
    method : str
        Method to use: "q3", "chi2", or "both"
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"} or None
        Multiple-testing adjustment for chi-square selection. If omitted, use
        the adjustment stored on ``ld_result``.

    Returns
    -------
    list of tuple
        List of (item_i, item_j) pairs flagged for LD

    Raises
    ------
    TypeError
        If ``method`` or an active threshold has the wrong type.
    ValueError
        If ``method`` is unsupported or an active threshold is out of range.
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method not in {"q3", "chi2", "both"}:
        raise ValueError("method must be 'q3', 'chi2', or 'both'")

    if method in {"q3", "both"}:
        q3_threshold = _validate_finite_pair_scalar(
            q3_threshold,
            name="q3_threshold",
            lower=0.0,
            lower_inclusive=True,
        )
    if method in {"chi2", "both"}:
        chi2_alpha = _validate_finite_pair_scalar(
            chi2_alpha,
            name="chi2_alpha",
            lower=0.0,
            upper=1.0,
        )
        if p_adjust is None:
            p_adjust = ld_result.p_adjustment
        p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")

    rows, columns = np.triu_indices_from(ld_result.q3_matrix, k=1)
    selected = np.zeros(rows.size, dtype=bool)

    if method in ("q3", "both"):
        q3_values = ld_result.q3_matrix[rows, columns]
        selected |= np.abs(q3_values) > q3_threshold

    if method in ("chi2", "both"):
        chi2_values = ld_result.ld_chi2_matrix[rows, columns]
        _, selected_p_values = _chi2_pair_p_values(chi2_values, p_adjust)
        selected |= selected_p_values < chi2_alpha

    return [
        (int(i), int(j)) for i, j in zip(rows[selected], columns[selected], strict=True)
    ]


def ld_summary_table(
    ld_result: LDResult,
    top_n: int = 20,
) -> str:
    """Create a formatted table of top LD pairs.

    Parameters
    ----------
    ld_result : LDResult
        Result from compute_ld_statistics
    top_n : int
        Number of top pairs to display

    Returns
    -------
    str
        Formatted table

    Raises
    ------
    TypeError
        If ``top_n`` is not an integer.
    ValueError
        If ``top_n`` is negative.
    """
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, Integral):
        raise TypeError("top_n must be an integer")
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")

    n_items = ld_result.q3_matrix.shape[0]
    rows, columns = np.triu_indices(n_items, k=1)
    q3_values = ld_result.q3_matrix[rows, columns]
    chi2_values = ld_result.ld_chi2_matrix[rows, columns]
    if ld_result.chi2_adjusted_p_value_matrix is None:
        _, p_values = _chi2_pair_p_values(chi2_values, ld_result.p_adjustment)
    else:
        p_values = ld_result.chi2_adjusted_p_value_matrix[rows, columns]

    selected = _top_absolute_pair_indices(q3_values, min(int(top_n), rows.size))
    rows = rows[selected]
    columns = columns[selected]
    q3_values = q3_values[selected]
    chi2_values = chi2_values[selected]
    p_values = p_values[selected]
    adjusted_values = ld_result.adj_residual_corr[rows, columns]

    p_value_heading = "p-value" if ld_result.p_adjustment == "none" else "adj p"

    lines = [
        f"{'Item i':<8} {'Item j':<8} {'Q3':>8} {'Adj r':>8} "
        f"{'LD χ²':>10} {p_value_heading:>10}",
        "-" * 62,
    ]

    for i, j, q3, adj_r, chi2, p_value in zip(
        rows,
        columns,
        q3_values,
        adjusted_values,
        chi2_values,
        p_values,
        strict=True,
    ):
        if ld_result.item_names:
            item_i = ld_result.item_names[i][:7]
            item_j = ld_result.item_names[j][:7]
        else:
            item_i = str(i + 1)
            item_j = str(j + 1)

        q3_str = f"{q3:.4f}"
        adj_str = f"{adj_r:.4f}"
        chi2_str = f"{chi2:.2f}" if not np.isnan(chi2) else "NA"
        p_str = f"{p_value:.4f}" if not np.isnan(p_value) else "NA"

        lines.append(
            f"{item_i:<8} {item_j:<8} {q3_str:>8} {adj_str:>8} {chi2_str:>10} {p_str:>10}"
        )

    return "\n".join(lines)


def _chi2_pair_p_values(
    chi2_values: NDArray[np.float64],
    p_adjust: PValueAdjustment,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return raw and adjusted p-values for one upper-triangle family."""
    raw = np.full(chi2_values.shape, np.nan)
    valid = ~np.isnan(chi2_values)
    raw[valid] = stats.chi2.sf(chi2_values[valid], df=1)
    return raw, adjust_p_values(raw, p_adjust)


def _symmetric_pair_matrix(
    n_items: int,
    rows: NDArray[np.intp],
    columns: NDArray[np.intp],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Place upper-triangle pair values into a symmetric matrix."""
    matrix = np.full((n_items, n_items), np.nan)
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


def _validate_finite_pair_scalar(
    value: float,
    *,
    name: str,
    lower: float,
    upper: float | None = None,
    lower_inclusive: bool = False,
) -> float:
    """Validate a finite scalar used to select locally dependent pairs."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    validated = float(value)
    lower_valid = validated >= lower if lower_inclusive else validated > lower
    upper_valid = upper is None or validated < upper
    if not np.isfinite(validated) or not lower_valid or not upper_valid:
        if upper is None:
            bound = f"at least {lower}" if lower_inclusive else f"greater than {lower}"
        else:
            bound = f"strictly between {lower} and {upper}"
        raise ValueError(f"{name} must be finite and {bound}")
    return validated


def _top_absolute_pair_indices(
    values: NDArray[np.float64],
    count: int,
) -> NDArray[np.intp]:
    """Select the largest absolute values with stable row-major tie handling."""
    if count == 0:
        return np.empty(0, dtype=np.intp)

    magnitudes = np.abs(values)
    magnitudes = np.where(np.isnan(magnitudes), -np.inf, magnitudes)
    if count == values.size:
        return np.argsort(-magnitudes, kind="stable")

    cutoff = np.partition(magnitudes, -count)[-count]
    greater = np.flatnonzero(magnitudes > cutoff)
    tied = np.flatnonzero(magnitudes == cutoff)[: count - greater.size]
    selected = np.concatenate((greater, tied))
    order = np.argsort(-magnitudes[selected], kind="stable")
    return selected[order]
