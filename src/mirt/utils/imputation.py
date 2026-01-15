"""Missing data imputation methods for IRT.

This module provides methods for handling missing responses:
- Simple imputation (mean, mode, random)
- Model-based imputation (EM)
- Multiple imputation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    pass

LARGE_DF = 1e10


def impute_responses(
    responses: NDArray[np.int_],
    method: Literal["mean", "mode", "random", "EM", "multiple"] = "EM",
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] | None = None,
    n_imputations: int = 5,
    missing_code: int = -1,
    seed: int | None = None,
) -> NDArray[np.int_] | list[NDArray[np.int_]]:
    """Impute missing responses in a response matrix.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items) with missing values coded as missing_code
    method : str
        Imputation method:
        - 'mean': Replace with item mean (rounded)
        - 'mode': Replace with item mode
        - 'random': Random draw from item distribution
        - 'EM': Model-based imputation using IRT
        - 'multiple': Multiple imputation (returns list)
    model : str, optional
        IRT model to use for EM imputation (default: '2PL')
    n_imputations : int
        Number of imputations for multiple imputation
    missing_code : int
        Code used to indicate missing values (default: -1)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    NDArray or list of NDArray
        Imputed response matrix (or list for multiple imputation)
    """
    rng = np.random.default_rng(seed)
    responses = np.asarray(responses).copy()
    n_persons, n_items = responses.shape

    missing_mask = responses == missing_code

    if not np.any(missing_mask):
        if method == "multiple":
            return [responses.copy() for _ in range(n_imputations)]
        return responses

    if method == "mean":
        return _impute_mean(responses, missing_mask)

    elif method == "mode":
        return _impute_mode(responses, missing_mask)

    elif method == "random":
        return _impute_random(responses, missing_mask, rng)

    elif method == "EM":
        if model is None:
            model = "2PL"
        return _impute_em(responses, missing_mask, model, rng)

    elif method == "multiple":
        if model is None:
            model = "2PL"
        return _impute_multiple(responses, missing_mask, model, n_imputations, rng)

    else:
        raise ValueError(f"Unknown imputation method: {method}")


def _impute_mean(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Impute with item means (rounded)."""
    imputed = responses.copy()
    n_items = responses.shape[1]

    for j in range(n_items):
        valid = ~missing_mask[:, j]
        if valid.any():
            mean_val = np.round(np.mean(responses[valid, j])).astype(np.int_)
            imputed[missing_mask[:, j], j] = mean_val

    return imputed


def _impute_mode(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Impute with item modes."""
    imputed = responses.copy()
    n_items = responses.shape[1]

    for j in range(n_items):
        valid = ~missing_mask[:, j]
        if valid.any():
            values, counts = np.unique(responses[valid, j], return_counts=True)
            mode_val = values[np.argmax(counts)]
            imputed[missing_mask[:, j], j] = mode_val

    return imputed


def _impute_random(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Impute by random draw from item distribution."""
    imputed = responses.copy()
    n_items = responses.shape[1]

    for j in range(n_items):
        valid = ~missing_mask[:, j]
        n_missing = missing_mask[:, j].sum()

        if valid.any() and n_missing > 0:
            observed = responses[valid, j]
            imputed[missing_mask[:, j], j] = rng.choice(
                observed, size=n_missing, replace=True
            )

    return imputed


def _impute_em(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    model: str,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Model-based imputation using EM algorithm."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    imputed = responses.copy()
    n_persons, n_items = responses.shape

    imputed = _impute_mode(responses, missing_mask)

    for iteration in range(10):
        try:
            result = fit_mirt(imputed, model=model, verbose=False)
            scores = fscores(result.model, imputed, method="EAP")
            theta = scores.theta
        except Exception:
            return _impute_random(responses, missing_mask, rng)

        old_imputed = imputed.copy()

        for j in range(n_items):
            item_missing = missing_mask[:, j]
            if not item_missing.any():
                continue

            theta_missing = theta[item_missing]
            if theta_missing.ndim == 1:
                theta_missing = theta_missing.reshape(-1, 1)

            probs = result.model.probability(theta_missing, j)

            if probs.ndim > 1:
                n_missing = probs.shape[0]
                for i in range(n_missing):
                    imputed[np.where(item_missing)[0][i], j] = rng.choice(
                        len(probs[i]), p=probs[i]
                    )
            else:
                imputed[item_missing, j] = (
                    rng.random(item_missing.sum()) < probs
                ).astype(np.int_)

        if np.allclose(old_imputed, imputed):
            break

    return imputed


def _impute_multiple(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    model: str,
    n_imputations: int,
    rng: np.random.Generator,
) -> list[NDArray[np.int_]]:
    """Multiple imputation using proper imputation."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    n_persons, n_items = responses.shape
    imputations: list[NDArray[np.int_]] = []

    initial = _impute_mode(responses, missing_mask)

    try:
        result = fit_mirt(initial, model=model, verbose=False)
    except Exception:
        return [
            _impute_random(responses, missing_mask, rng) for _ in range(n_imputations)
        ]

    for m in range(n_imputations):
        imputed = responses.copy()

        scores = fscores(result.model, initial, method="EAP")
        theta_mean = scores.theta
        theta_se = scores.standard_error

        if theta_mean.ndim == 1:
            theta_draw = theta_mean + rng.standard_normal(n_persons) * theta_se
            theta_draw = theta_draw.reshape(-1, 1)
        else:
            theta_draw = theta_mean + rng.standard_normal(theta_mean.shape) * theta_se

        for j in range(n_items):
            item_missing = missing_mask[:, j]
            if not item_missing.any():
                continue

            theta_missing = theta_draw[item_missing]
            probs = result.model.probability(theta_missing, j)

            if probs.ndim > 1:
                n_missing = probs.shape[0]
                for i in range(n_missing):
                    p = np.clip(probs[i], PROB_EPSILON, 1 - PROB_EPSILON)
                    p = p / p.sum()
                    imputed[np.where(item_missing)[0][i], j] = rng.choice(len(p), p=p)
            else:
                imputed[item_missing, j] = (
                    rng.random(item_missing.sum()) < probs
                ).astype(np.int_)

        imputations.append(imputed)

    return imputations


def analyze_missing(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> dict[str, NDArray[np.float64] | float]:
    """Analyze missing data patterns.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    dict
        Dictionary with:
        - 'total_missing_rate': Overall missing rate
        - 'item_missing_rate': Missing rate per item
        - 'person_missing_rate': Missing rate per person
        - 'n_complete_cases': Number of persons with no missing
        - 'n_complete_items': Number of items with no missing
    """
    responses = np.asarray(responses)
    missing_mask = responses == missing_code
    n_persons, n_items = responses.shape

    return {
        "total_missing_rate": float(missing_mask.mean()),
        "item_missing_rate": missing_mask.mean(axis=0).astype(np.float64),
        "person_missing_rate": missing_mask.mean(axis=1).astype(np.float64),
        "n_complete_cases": int((~missing_mask.any(axis=1)).sum()),
        "n_complete_items": int((~missing_mask.any(axis=0)).sum()),
    }


def listwise_deletion(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> NDArray[np.int_]:
    """Remove all rows (persons) with any missing values.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    NDArray
        Response matrix with complete cases only
    """
    responses = np.asarray(responses)
    missing_mask = responses == missing_code
    complete_mask = ~missing_mask.any(axis=1)
    return responses[complete_mask]


def pairwise_available(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Get counts for pairwise available analysis.

    Returns counts of valid pairs for each item pair.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    tuple
        (n_available, joint_available) where:
        - n_available: (n_items,) count of valid responses per item
        - joint_available: (n_items, n_items) count of valid pairs
    """
    responses = np.asarray(responses)
    n_items = responses.shape[1]
    valid = responses != missing_code

    n_available = valid.sum(axis=0)
    joint_available = np.zeros((n_items, n_items), dtype=np.int_)

    for i in range(n_items):
        for j in range(n_items):
            joint_available[i, j] = (valid[:, i] & valid[:, j]).sum()

    return n_available, joint_available


@dataclass
class MIResult:
    """Container for combined multiple imputation results.

    Attributes
    ----------
    estimate : float or NDArray[np.float64]
        Combined point estimate.
    within_variance : float or NDArray[np.float64]
        Average within-imputation variance.
    between_variance : float or NDArray[np.float64]
        Between-imputation variance.
    total_variance : float or NDArray[np.float64]
        Total variance (within + between + correction).
    standard_error : float or NDArray[np.float64]
        Standard error of combined estimate.
    df : float or NDArray[np.float64]
        Degrees of freedom for inference.
    fmi : float or NDArray[np.float64]
        Fraction of missing information.
    lambda_hat : float or NDArray[np.float64]
        Proportion of variation due to missingness.
    """

    estimate: float | NDArray[np.float64]
    within_variance: float | NDArray[np.float64]
    between_variance: float | NDArray[np.float64]
    total_variance: float | NDArray[np.float64]
    standard_error: float | NDArray[np.float64]
    df: float | NDArray[np.float64]
    fmi: float | NDArray[np.float64]
    lambda_hat: float | NDArray[np.float64]


def averageMI(
    estimates: list[float | NDArray[np.float64]],
    variances: list[float | NDArray[np.float64]] | None = None,
    standard_errors: list[float | NDArray[np.float64]] | None = None,
) -> MIResult:
    """Combine results from multiple imputation using Rubin's rules.

    Combines point estimates and standard errors/variances from multiple
    imputations into a single inference, properly accounting for both
    within-imputation and between-imputation variability.

    Parameters
    ----------
    estimates : list of float or ndarray
        Point estimates from each imputation. Each element should be
        the same shape (scalar, 1D array, or 2D array).
    variances : list of float or ndarray, optional
        Variance estimates from each imputation. If None, must provide
        standard_errors.
    standard_errors : list of float or ndarray, optional
        Standard errors from each imputation. Squared to get variances.
        If None, must provide variances.

    Returns
    -------
    MIResult
        Combined results with:
        - estimate: Combined point estimate
        - within_variance: Average within-imputation variance
        - between_variance: Between-imputation variance
        - total_variance: Total variance (Rubin's formula)
        - standard_error: Square root of total variance
        - df: Degrees of freedom for t-distribution
        - fmi: Fraction of missing information
        - lambda_hat: Proportion of variation due to missingness

    Examples
    --------
    >>> # Fit model on multiple imputations
    >>> imputations = impute_responses(data, method='multiple', n_imputations=5)
    >>> estimates = []
    >>> se_list = []
    >>> for imp in imputations:
    ...     result = fit_mirt(imp, model='2PL')
    ...     estimates.append(result.model.parameters['difficulty'])
    ...     se_list.append(result.standard_errors['difficulty'])
    >>> combined = averageMI(estimates, standard_errors=se_list)
    >>> print(f"Combined estimate: {combined.estimate}")
    >>> print(f"SE: {combined.standard_error}")

    Notes
    -----
    Uses Rubin's (1987) combining rules:
    - Combined estimate: Q_bar = mean(Q_m)
    - Within variance: U_bar = mean(U_m)
    - Between variance: B = var(Q_m)
    - Total variance: T = U_bar + (1 + 1/m) * B
    - Degrees of freedom: Based on Barnard & Rubin (1999)
    """
    if variances is None and standard_errors is None:
        raise ValueError("Must provide either variances or standard_errors")

    m = len(estimates)
    if m < 2:
        raise ValueError("Need at least 2 imputations to combine")

    estimates_arr = [np.asarray(e) for e in estimates]

    if variances is not None:
        variances_arr = [np.asarray(v) for v in variances]
    else:
        assert standard_errors is not None
        variances_arr = [np.asarray(se) ** 2 for se in standard_errors]

    q_bar = np.mean(estimates_arr, axis=0)

    u_bar = np.mean(variances_arr, axis=0)

    deviations = [e - q_bar for e in estimates_arr]
    b = np.sum([d**2 for d in deviations], axis=0) / (m - 1)

    total_var = u_bar + (1 + 1 / m) * b

    se = np.sqrt(total_var)

    r = (1 + 1 / m) * b / (u_bar + PROB_EPSILON)

    lambda_hat = (1 + 1 / m) * b / (total_var + PROB_EPSILON)

    df_old = (m - 1) * (1 + 1 / r) ** 2
    df_old = np.where(np.isinf(df_old) | np.isnan(df_old), LARGE_DF, df_old)

    fmi = (r + 2 / (df_old + 3)) / (r + 1)
    fmi = np.clip(fmi, 0, 1)

    if np.isscalar(q_bar):
        return MIResult(
            estimate=float(q_bar),
            within_variance=float(u_bar),
            between_variance=float(b),
            total_variance=float(total_var),
            standard_error=float(se),
            df=float(df_old),
            fmi=float(fmi),
            lambda_hat=float(lambda_hat),
        )

    return MIResult(
        estimate=q_bar,
        within_variance=u_bar,
        between_variance=b,
        total_variance=total_var,
        standard_error=se,
        df=df_old,
        fmi=fmi,
        lambda_hat=lambda_hat,
    )
