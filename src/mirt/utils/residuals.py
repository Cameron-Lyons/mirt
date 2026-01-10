"""Residual analysis functions for IRT models.

Provides functions for computing various types of residuals
for model diagnostics. Uses Rust backend for performance when available.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

try:
    from mirt._rust_backend import (
        RUST_AVAILABLE,
        compute_ld_chi2_matrix,
        compute_q3_matrix,
        compute_standardized_residuals,
    )
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class ResidualResult:
    """Container for residual analysis results.

    Attributes
    ----------
    raw : NDArray[np.float64]
        Raw residuals (observed - expected).
    standardized : NDArray[np.float64]
        Standardized residuals.
    ld_matrix : NDArray[np.float64] | None
        Local dependence matrix (Q3 or other).
    summary : dict
        Summary statistics for residuals.
    """

    raw: NDArray[np.float64]
    standardized: NDArray[np.float64]
    ld_matrix: NDArray[np.float64] | None
    summary: dict


def residuals(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    type: Literal["raw", "standardized", "pearson", "deviance"] = "standardized",
    suppress_abs: float | None = None,
    use_rust: bool = True,
) -> ResidualResult:
    """Compute model residuals.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, n_dims).
    type : str
        Type of residuals:
        - "raw": observed - expected
        - "standardized": raw / sqrt(variance)
        - "pearson": (obs - exp) / sqrt(exp * (1 - exp))
        - "deviance": Signed deviance residuals
    suppress_abs : float, optional
        Suppress residuals with absolute value below this threshold
        when computing LD matrix.
    use_rust : bool
        Use Rust backend for performance. Default True.

    Returns
    -------
    ResidualResult
        Container with residuals and local dependence matrix.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> resid = residuals(result.model, responses, result.theta)
    >>> print(f"Mean absolute residual: {np.mean(np.abs(resid.raw)):.3f}")
    """
    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_1d(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    theta_1d = theta[:, 0].astype(np.float64)

    can_use_rust = (
        use_rust
        and RUST_AVAILABLE
        and hasattr(model, "discrimination")
        and hasattr(model, "difficulty")
        and type in ("standardized", "pearson")
    )

    if can_use_rust:
        disc = np.asarray(model.discrimination, dtype=np.float64)
        diff = np.asarray(model.difficulty, dtype=np.float64)
        if disc.ndim > 1:
            disc = disc[:, 0]

        responses_int = np.where(np.isnan(responses), -1, responses).astype(np.int32)

        standardized = compute_standardized_residuals(
            responses_int, theta_1d, disc, diff
        )

        expected = model.probability(theta)
        expected = np.clip(expected, 1e-10, 1 - 1e-10)
        raw = responses - expected

        mask = ~np.isnan(responses)
        raw = np.where(mask, raw, np.nan)

        ld_matrix = compute_q3_matrix(responses_int, theta_1d, disc, diff)
        np.fill_diagonal(ld_matrix, 1.0)

        if suppress_abs is not None:
            ld_matrix = np.where(np.abs(ld_matrix) < suppress_abs, 0.0, ld_matrix)
            np.fill_diagonal(ld_matrix, 1.0)

    else:
        expected = model.probability(theta)
        expected = np.clip(expected, 1e-10, 1 - 1e-10)

        raw = responses - expected
        variance = expected * (1 - expected)

        if type == "raw":
            standardized = raw
        elif type == "standardized":
            standardized = raw / np.sqrt(variance)
        elif type == "pearson":
            standardized = raw / np.sqrt(variance)
        elif type == "deviance":
            sign = np.sign(raw)
            dev_sq = -2 * (
                responses * np.log(expected + 1e-10)
                + (1 - responses) * np.log(1 - expected + 1e-10)
            )
            standardized = sign * np.sqrt(np.maximum(dev_sq, 0))
        else:
            raise ValueError(f"Unknown residual type: {type}")

        mask = ~np.isnan(responses)
        standardized = np.where(mask, standardized, np.nan)
        raw = np.where(mask, raw, np.nan)

        ld_matrix = _compute_ld_matrix(standardized, suppress_abs)

    mean_resid = np.nanmean(raw, axis=0)
    std_resid = np.nanstd(standardized, axis=0)
    max_abs_resid = np.nanmax(np.abs(standardized), axis=0)

    summary = {
        "mean_raw": mean_resid,
        "std_standardized": std_resid,
        "max_abs_standardized": max_abs_resid,
        "n_large": np.sum(np.abs(standardized) > 2, axis=0),
    }

    return ResidualResult(
        raw=raw,
        standardized=standardized,
        ld_matrix=ld_matrix,
        summary=summary,
    )


def _compute_ld_matrix(
    standardized_residuals: NDArray[np.float64],
    suppress_abs: float | None = None,
) -> NDArray[np.float64]:
    """Compute local dependence matrix (Q3 statistic).

    The Q3 statistic is the correlation between standardized residuals
    for pairs of items.
    """
    n_items = standardized_residuals.shape[1]
    ld_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            mask = ~(
                np.isnan(standardized_residuals[:, i])
                | np.isnan(standardized_residuals[:, j])
            )
            if np.sum(mask) > 2:
                corr = np.corrcoef(
                    standardized_residuals[mask, i], standardized_residuals[mask, j]
                )[0, 1]
                if suppress_abs is not None and abs(corr) < suppress_abs:
                    corr = 0.0
                ld_matrix[i, j] = corr
                ld_matrix[j, i] = corr

    np.fill_diagonal(ld_matrix, 1.0)
    return ld_matrix


def Q3(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    use_rust: bool = True,
) -> NDArray[np.float64]:
    """Compute Yen's Q3 statistic for local dependence.

    Q3 is the correlation between standardized residuals for pairs
    of items. Large positive values suggest local dependence.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.
    use_rust : bool
        Use Rust backend for performance. Default True.

    Returns
    -------
    NDArray[np.float64]
        Q3 correlation matrix. Shape: (n_items, n_items).

    Notes
    -----
    Values above 0.2 may indicate local dependence between items.
    The average Q3 should be close to zero if model fits well.

    References
    ----------
    Yen, W. M. (1984). Effects of local item dependence on the fit and
    equating performance of the three-parameter logistic model.
    Applied Psychological Measurement, 8(2), 125-145.
    """
    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_1d(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    can_use_rust = (
        use_rust
        and RUST_AVAILABLE
        and hasattr(model, "discrimination")
        and hasattr(model, "difficulty")
    )

    if can_use_rust:
        disc = np.asarray(model.discrimination, dtype=np.float64)
        diff = np.asarray(model.difficulty, dtype=np.float64)
        if disc.ndim > 1:
            disc = disc[:, 0]

        theta_1d = theta[:, 0].astype(np.float64)
        responses_int = np.where(np.isnan(responses), -1, responses).astype(np.int32)

        q3_matrix = compute_q3_matrix(responses_int, theta_1d, disc, diff)
        np.fill_diagonal(q3_matrix, 1.0)
        return q3_matrix

    resid_result = residuals(
        model, responses, theta, type="standardized", use_rust=False
    )
    return resid_result.ld_matrix


def LD_X2(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    use_rust: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Chen & Thissen LD chi-square statistic.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.
    use_rust : bool
        Use Rust backend for performance. Default True.

    Returns
    -------
    ld_x2 : NDArray[np.float64]
        LD chi-square matrix. Shape: (n_items, n_items).
    p_values : NDArray[np.float64]
        P-value matrix.

    References
    ----------
    Chen, W. H., & Thissen, D. (1997). Local dependence indexes for item
    pairs using item response theory. Journal of Educational and
    Behavioral Statistics, 22(3), 265-289.
    """
    from scipy import stats

    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_1d(theta)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    can_use_rust = (
        use_rust
        and RUST_AVAILABLE
        and hasattr(model, "discrimination")
        and hasattr(model, "difficulty")
    )

    if can_use_rust:
        disc = np.asarray(model.discrimination, dtype=np.float64)
        diff = np.asarray(model.difficulty, dtype=np.float64)
        if disc.ndim > 1:
            disc = disc[:, 0]

        theta_1d = theta[:, 0].astype(np.float64)
        responses_int = np.where(np.isnan(responses), -1, responses).astype(np.int32)

        ld_x2 = compute_ld_chi2_matrix(responses_int, theta_1d, disc, diff)

        n_items = responses.shape[1]
        p_values = np.ones((n_items, n_items))
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if ld_x2[i, j] > 0:
                    p_values[i, j] = 1 - stats.chi2.cdf(ld_x2[i, j], 1)
                    p_values[j, i] = p_values[i, j]

        return ld_x2, p_values

    n_persons, n_items = responses.shape
    probs = model.probability(theta)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    ld_x2 = np.zeros((n_items, n_items))
    p_values = np.ones((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            mask = ~(np.isnan(responses[:, i]) | np.isnan(responses[:, j]))
            n_valid = np.sum(mask)

            if n_valid < 5:
                continue

            obs_00 = np.sum((responses[mask, i] == 0) & (responses[mask, j] == 0))
            obs_01 = np.sum((responses[mask, i] == 0) & (responses[mask, j] == 1))
            obs_10 = np.sum((responses[mask, i] == 1) & (responses[mask, j] == 0))
            obs_11 = np.sum((responses[mask, i] == 1) & (responses[mask, j] == 1))

            p_i = probs[mask, i]
            p_j = probs[mask, j]

            exp_00 = np.sum((1 - p_i) * (1 - p_j))
            exp_01 = np.sum((1 - p_i) * p_j)
            exp_10 = np.sum(p_i * (1 - p_j))
            exp_11 = np.sum(p_i * p_j)

            chi2 = 0.0
            for obs, exp in [
                (obs_00, exp_00),
                (obs_01, exp_01),
                (obs_10, exp_10),
                (obs_11, exp_11),
            ]:
                if exp > 0:
                    chi2 += (obs - exp) ** 2 / exp

            ld_x2[i, j] = chi2
            ld_x2[j, i] = chi2
            p_values[i, j] = 1 - stats.chi2.cdf(chi2, 1)
            p_values[j, i] = p_values[i, j]

    return ld_x2, p_values
