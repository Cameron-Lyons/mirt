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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


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
        List of (item_i, item_j, chi2_value, p_value) tuples flagged for LD
    item_names : list of str or None
        Item names for labeling
    """

    q3_matrix: NDArray[np.float64]
    ld_chi2_matrix: NDArray[np.float64]
    g2_matrix: NDArray[np.float64]
    adj_residual_corr: NDArray[np.float64]
    q3_flagged: list[tuple[int, int, float]]
    chi2_flagged: list[tuple[int, int, float, float]]
    item_names: list[str] | None = None

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
        lines.extend(
            [
                "LD Chi-Square Statistics:",
                f"  Mean χ²:        {np.nanmean(chi2_upper):.4f}",
                f"  Max χ²:         {np.nanmax(chi2_upper):.4f}",
                f"  Pairs p < 0.05: {len(self.chi2_flagged)}",
                "",
            ]
        )

        if self.chi2_flagged:
            lines.append("Flagged item pairs (p < 0.05):")
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
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
    q3_threshold: float = 0.2,
    alpha: float = 0.05,
) -> LDResult:
    """Compute local dependence statistics for all item pairs.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
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

    Returns
    -------
    LDResult
        Object containing all LD statistics and flagged pairs
    """
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP", n_quadpts=n_quadpts)
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    residuals = _compute_residuals(model, responses, theta)

    q3_matrix = _compute_q3(residuals, responses)
    adj_residual_corr = _adjust_q3(q3_matrix)

    ld_chi2_matrix, g2_matrix = _compute_ld_chi2_g2(model, responses, theta, n_quadpts)

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
    finite = np.isfinite(chi2_values)
    p_values = np.full(chi2_values.shape, np.nan)
    p_values[finite] = stats.chi2.sf(chi2_values[finite], df=1)
    chi2_mask = finite & (p_values < alpha)
    chi2_flagged = [
        (int(i), int(j), float(value), float(p_value))
        for i, j, value, p_value in zip(
            rows[chi2_mask],
            columns[chi2_mask],
            chi2_values[chi2_mask],
            p_values[chi2_mask],
            strict=True,
        )
    ]

    item_names = model.item_names if hasattr(model, "item_names") else None

    return LDResult(
        q3_matrix=q3_matrix,
        ld_chi2_matrix=ld_chi2_matrix,
        g2_matrix=g2_matrix,
        adj_residual_corr=adj_residual_corr,
        q3_flagged=q3_flagged,
        chi2_flagged=chi2_flagged,
        item_names=item_names,
    )


def compute_q3(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute Yen's Q3 statistics for all item pairs.

    Q3 is the correlation between residuals for pairs of items.
    Under local independence, Q3 should be approximately -1/(n_items - 1).

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates

    Returns
    -------
    NDArray
        Matrix of Q3 statistics
    """
    from mirt._backend_config import should_use_rust
    from mirt._rust_backend import compute_q3_matrix as rust_compute_q3

    responses = np.asarray(responses)
    n_persons = responses.shape[0]

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP")
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if should_use_rust() and not model.is_polytomous:
        disc = model.parameters.get("discrimination")
        diff = model.parameters.get("difficulty")
        if disc is not None and diff is not None:
            theta_flat = theta.ravel() if theta.ndim > 1 else theta
            return rust_compute_q3(responses, theta_flat, disc.ravel(), diff.ravel())

    residuals = _compute_residuals(model, responses, theta)
    return _compute_q3(residuals, responses)


def compute_ld_chi2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Chen & Thissen's LD χ² statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates
    n_quadpts : int
        Number of quadrature points

    Returns
    -------
    chi2_matrix : NDArray
        Matrix of LD χ² statistics
    p_value_matrix : NDArray
        Matrix of p-values
    """
    from mirt._backend_config import should_use_rust
    from mirt._rust_backend import compute_ld_chi2_matrix as rust_compute_chi2

    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    if theta is None:
        from mirt.scoring import fscores

        result = fscores(model, responses, method="EAP", n_quadpts=n_quadpts)
        theta = result.theta

    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and n_persons > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if should_use_rust() and not model.is_polytomous:
        disc = model.parameters.get("discrimination")
        diff = model.parameters.get("difficulty")
        if disc is not None and diff is not None:
            theta_flat = theta.ravel() if theta.ndim > 1 else theta
            chi2_matrix = rust_compute_chi2(
                responses, theta_flat, disc.ravel(), diff.ravel()
            )
        else:
            chi2_matrix, _ = _compute_ld_chi2_g2(model, responses, theta, n_quadpts)
    else:
        chi2_matrix, _ = _compute_ld_chi2_g2(model, responses, theta, n_quadpts)

    p_value_matrix = np.zeros_like(chi2_matrix)
    rows, columns = np.triu_indices(n_items, k=1)
    values = chi2_matrix[rows, columns]
    finite = np.isfinite(values)
    p_values = stats.chi2.sf(values[finite], df=1)
    finite_rows = rows[finite]
    finite_columns = columns[finite]
    p_value_matrix[finite_rows, finite_columns] = p_values
    p_value_matrix[finite_columns, finite_rows] = p_values

    return chi2_matrix, p_value_matrix


def _compute_residuals(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute standardized residuals for each person-item combination."""
    from mirt._backend_config import should_use_rust
    from mirt._rust_backend import compute_standardized_residuals

    if should_use_rust() and not model.is_polytomous:
        disc = model.parameters.get("discrimination")
        diff = model.parameters.get("difficulty")
        if disc is not None and diff is not None:
            theta_flat = theta.ravel() if theta.ndim > 1 else theta
            return compute_standardized_residuals(
                responses, theta_flat, disc.ravel(), diff.ravel()
            )

    n_persons, n_items = responses.shape
    residuals = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        probs = model.probability(theta, j)

        if probs.ndim == 2:
            n_cats = probs.shape[1]
            expected = np.sum(probs * np.arange(n_cats), axis=1)
            variance = np.sum(probs * (np.arange(n_cats) ** 2), axis=1) - expected**2
        else:
            expected = probs
            variance = probs * (1 - probs)

        valid = responses[:, j] >= 0
        residuals[valid, j] = (responses[valid, j] - expected[valid]) / np.sqrt(
            variance[valid] + PROB_EPSILON
        )

    return residuals


def _compute_q3(
    residuals: NDArray[np.float64],
    responses: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute Q3 (residual correlation) matrix."""
    n_items = residuals.shape[1]
    q3_matrix = np.zeros((n_items, n_items))

    valid = (responses >= 0) & ~np.isnan(residuals)
    valid_float = valid.astype(np.float64)
    values = np.where(valid, residuals, 0.0)

    pair_counts = valid_float.T @ valid_float
    safe_counts = np.where(pair_counts > 0, pair_counts, 1.0)
    pair_sums = values.T @ valid_float
    pair_square_sums = (values * values).T @ valid_float
    pair_cross_products = values.T @ values

    covariance = pair_cross_products - (pair_sums * pair_sums.T) / safe_counts
    variance_rows = pair_square_sums - (pair_sums * pair_sums) / safe_counts
    variance_columns = pair_square_sums.T - (pair_sums.T * pair_sums.T) / safe_counts
    np.maximum(variance_rows, 0.0, out=variance_rows)
    np.maximum(variance_columns, 0.0, out=variance_columns)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = covariance / np.sqrt(variance_rows * variance_columns)

    rows, columns = np.triu_indices(n_items, k=1)
    eligible = pair_counts[rows, columns] > 2
    eligible_rows = rows[eligible]
    eligible_columns = columns[eligible]
    q3_values = correlations[eligible_rows, eligible_columns]
    q3_matrix[eligible_rows, eligible_columns] = q3_values
    q3_matrix[eligible_columns, eligible_rows] = q3_values

    return q3_matrix


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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute LD χ² and G² statistics for item pairs.

    Uses the Chen & Thissen (1997) approach comparing observed and
    expected cross-classification frequencies.
    """
    del n_quadpts  # Retained for compatibility with the public call path.
    n_persons, n_items = responses.shape

    positive_probabilities = np.empty((n_persons, n_items), dtype=np.float64)
    for item_idx in range(n_items):
        probabilities = np.asarray(model.probability(theta, item_idx))
        if probabilities.ndim == 2:
            probabilities = 1.0 - probabilities[:, 0]
        positive_probabilities[:, item_idx] = probabilities

    valid = responses >= 0
    valid_float = valid.astype(np.float64)
    observed_positive = ((responses > 0) & valid).astype(np.float64)
    observed_zero = valid_float - observed_positive
    expected_positive = np.where(valid, positive_probabilities, 0.0)
    expected_zero = valid_float - expected_positive

    chi2_values = np.zeros((n_items, n_items), dtype=np.float64)
    g2_values = np.zeros((n_items, n_items), dtype=np.float64)
    tables = (
        (observed_zero, observed_zero, expected_zero, expected_zero),
        (observed_zero, observed_positive, expected_zero, expected_positive),
        (observed_positive, observed_zero, expected_positive, expected_zero),
        (
            observed_positive,
            observed_positive,
            expected_positive,
            expected_positive,
        ),
    )
    for observed_left, observed_right, expected_left, expected_right in tables:
        observed_counts = observed_left.T @ observed_right
        expected_counts = expected_left.T @ expected_right
        np.maximum(expected_counts, 0.5, out=expected_counts)
        chi2_values += (observed_counts - expected_counts) ** 2 / expected_counts
        g2_values += (
            2.0
            * observed_counts
            * np.log(observed_counts / expected_counts + PROB_EPSILON)
        )

    pair_counts = valid_float.T @ valid_float
    rows, columns = np.triu_indices(n_items, k=1)
    eligible = pair_counts[rows, columns] >= 10
    eligible_rows = rows[eligible]
    eligible_columns = columns[eligible]

    chi2_matrix = np.full((n_items, n_items), np.nan)
    g2_matrix = np.full((n_items, n_items), np.nan)
    pair_chi2 = chi2_values[eligible_rows, eligible_columns]
    pair_g2 = g2_values[eligible_rows, eligible_columns]
    chi2_matrix[eligible_rows, eligible_columns] = pair_chi2
    chi2_matrix[eligible_columns, eligible_rows] = pair_chi2
    g2_matrix[eligible_rows, eligible_columns] = pair_g2
    g2_matrix[eligible_columns, eligible_rows] = pair_g2

    return chi2_matrix, g2_matrix


def flag_ld_pairs(
    ld_result: LDResult,
    q3_threshold: float = 0.2,
    chi2_alpha: float = 0.05,
    method: str = "q3",
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

    Returns
    -------
    list of tuple
        List of (item_i, item_j) pairs flagged for LD
    """
    pairs = set()

    if method in ("q3", "both"):
        for i, j, _ in ld_result.q3_flagged:
            if np.abs(ld_result.q3_matrix[i, j]) > q3_threshold:
                pairs.add((min(i, j), max(i, j)))

    if method in ("chi2", "both"):
        n_items = ld_result.ld_chi2_matrix.shape[0]
        for i in range(n_items):
            for j in range(i + 1, n_items):
                chi2 = ld_result.ld_chi2_matrix[i, j]
                if not np.isnan(chi2):
                    p_value = 1 - stats.chi2.cdf(chi2, df=1)
                    if p_value < chi2_alpha:
                        pairs.add((i, j))

    return sorted(pairs)


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
    """
    n_items = ld_result.q3_matrix.shape[0]

    pairs_data = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            q3 = ld_result.q3_matrix[i, j]
            chi2 = ld_result.ld_chi2_matrix[i, j]
            adj_r = ld_result.adj_residual_corr[i, j]

            if not np.isnan(chi2):
                p_val = 1 - stats.chi2.cdf(chi2, df=1)
            else:
                p_val = np.nan

            pairs_data.append(
                {
                    "i": i,
                    "j": j,
                    "q3": q3,
                    "adj_r": adj_r,
                    "chi2": chi2,
                    "p": p_val,
                }
            )

    pairs_data.sort(key=lambda x: -abs(x["q3"]))

    lines = [
        f"{'Item i':<8} {'Item j':<8} {'Q3':>8} {'Adj r':>8} {'LD χ²':>10} {'p-value':>10}",
        "-" * 62,
    ]

    for data in pairs_data[:top_n]:
        if ld_result.item_names:
            item_i = ld_result.item_names[data["i"]][:7]
            item_j = ld_result.item_names[data["j"]][:7]
        else:
            item_i = str(data["i"] + 1)
            item_j = str(data["j"] + 1)

        q3_str = f"{data['q3']:.4f}"
        adj_str = f"{data['adj_r']:.4f}"
        chi2_str = f"{data['chi2']:.2f}" if not np.isnan(data["chi2"]) else "NA"
        p_str = f"{data['p']:.4f}" if not np.isnan(data["p"]) else "NA"

        lines.append(
            f"{item_i:<8} {item_j:<8} {q3_str:>8} {adj_str:>8} {chi2_str:>10} {p_str:>10}"
        )

    return "\n".join(lines)
