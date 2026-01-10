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

    adj_residual_corr = _compute_adjusted_residual_corr(
        model, responses, theta, n_quadpts
    )

    ld_chi2_matrix, g2_matrix = _compute_ld_chi2_g2(model, responses, theta, n_quadpts)

    q3_flagged = []
    chi2_flagged = []

    for i in range(n_items):
        for j in range(i + 1, n_items):
            if np.abs(q3_matrix[i, j]) > q3_threshold:
                q3_flagged.append((i, j, float(q3_matrix[i, j])))

            if not np.isnan(ld_chi2_matrix[i, j]):
                p_value = 1 - stats.chi2.cdf(ld_chi2_matrix[i, j], df=1)
                if p_value < alpha:
                    chi2_flagged.append(
                        (i, j, float(ld_chi2_matrix[i, j]), float(p_value))
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
    from mirt._rust_backend import RUST_AVAILABLE
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

    if RUST_AVAILABLE and not model.is_polytomous:
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
    from mirt._rust_backend import RUST_AVAILABLE
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

    if RUST_AVAILABLE and not model.is_polytomous:
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
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if not np.isnan(chi2_matrix[i, j]):
                p_value_matrix[i, j] = 1 - stats.chi2.cdf(chi2_matrix[i, j], df=1)
                p_value_matrix[j, i] = p_value_matrix[i, j]

    return chi2_matrix, p_value_matrix


def _compute_residuals(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute standardized residuals for each person-item combination."""
    from mirt._rust_backend import RUST_AVAILABLE, compute_standardized_residuals

    if RUST_AVAILABLE and not model.is_polytomous:
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
            variance[valid] + 1e-10
        )

    return residuals


def _compute_q3(
    residuals: NDArray[np.float64],
    responses: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute Q3 (residual correlation) matrix."""
    n_items = residuals.shape[1]
    q3_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            valid &= ~np.isnan(residuals[:, i]) & ~np.isnan(residuals[:, j])

            if valid.sum() > 2:
                r_i = residuals[valid, i]
                r_j = residuals[valid, j]

                q3 = np.corrcoef(r_i, r_j)[0, 1]
                q3_matrix[i, j] = q3
                q3_matrix[j, i] = q3

    return q3_matrix


def _compute_adjusted_residual_corr(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    n_quadpts: int,
) -> NDArray[np.float64]:
    """Compute adjusted residual correlations.

    Adjusts for the expected negative correlation under local independence.
    """
    n_items = responses.shape[1]

    residuals = _compute_residuals(model, responses, theta)
    q3 = _compute_q3(residuals, responses)

    expected_q3 = -1.0 / (n_items - 1)

    adj_corr = q3 - expected_q3

    return adj_corr


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
    n_persons, n_items = responses.shape

    chi2_matrix = np.full((n_items, n_items), np.nan)
    g2_matrix = np.full((n_items, n_items), np.nan)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            n_valid = valid.sum()

            if n_valid < 10:
                continue

            resp_i = responses[valid, i]
            resp_j = responses[valid, j]
            theta_valid = theta[valid]

            prob_i = model.probability(theta_valid, i)
            prob_j = model.probability(theta_valid, j)

            if prob_i.ndim == 2:
                prob_i = 1 - prob_i[:, 0]
            if prob_j.ndim == 2:
                prob_j = 1 - prob_j[:, 0]

            resp_i_bin = (resp_i > 0).astype(int)
            resp_j_bin = (resp_j > 0).astype(int)

            obs_00 = np.sum((resp_i_bin == 0) & (resp_j_bin == 0))
            obs_01 = np.sum((resp_i_bin == 0) & (resp_j_bin == 1))
            obs_10 = np.sum((resp_i_bin == 1) & (resp_j_bin == 0))
            obs_11 = np.sum((resp_i_bin == 1) & (resp_j_bin == 1))

            exp_00 = np.sum((1 - prob_i) * (1 - prob_j))
            exp_01 = np.sum((1 - prob_i) * prob_j)
            exp_10 = np.sum(prob_i * (1 - prob_j))
            exp_11 = np.sum(prob_i * prob_j)

            observed = np.array([obs_00, obs_01, obs_10, obs_11])
            expected = np.array([exp_00, exp_01, exp_10, exp_11])

            expected = np.maximum(expected, 0.5)

            chi2 = np.sum((observed - expected) ** 2 / expected)
            chi2_matrix[i, j] = chi2
            chi2_matrix[j, i] = chi2

            g2 = 2 * np.sum(observed * np.log(observed / expected + 1e-10))
            g2_matrix[i, j] = g2
            g2_matrix[j, i] = g2

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
