"""Shared numeric utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def logsumexp(
    a: NDArray[np.float64],
    axis: int | None = None,
    keepdims: bool = False,
) -> NDArray[np.float64]:
    """Compute log(sum(exp(a))) in a numerically stable way."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return result


def logsumexp_axis1(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute logsumexp along axis 1, returning a 1D array."""
    a_max = np.max(a, axis=1, keepdims=True)
    return (a_max + np.log(np.sum(np.exp(a - a_max), axis=1, keepdims=True))).ravel()


def compute_hessian_se(
    func: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute standard errors from diagonal of Hessian using finite differences.

    Parameters
    ----------
    func : callable
        Function to compute Hessian of (should be negative log-likelihood or similar).
    x : array
        Point at which to compute Hessian.
    h : float
        Step size for finite differences.

    Returns
    -------
    se : array
        Standard errors (sqrt of diagonal of inverse Hessian).
    """
    n = len(x)
    se = np.zeros(n)
    f_center = func(x)

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        x_minus = x.copy()
        x_minus[j] -= h

        f_plus = func(x_plus)
        f_minus = func(x_minus)

        hessian_jj = (f_plus - 2 * f_center + f_minus) / (h**2)

        if hessian_jj > 0:
            se[j] = np.sqrt(1.0 / hessian_jj)
        else:
            se[j] = np.nan

    return se


def compute_expected_variance(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_items: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected values and variances for all items.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    theta : array of shape (n_persons, n_factors)
        Person ability estimates.
    n_items : int
        Number of items.

    Returns
    -------
    expected : array of shape (n_persons, n_items)
        Expected scores for each person-item combination.
    variance : array of shape (n_persons, n_items)
        Variance of scores for each person-item combination.
    """
    n_persons = theta.shape[0]
    expected = np.zeros((n_persons, n_items))
    variance = np.zeros((n_persons, n_items))

    for i in range(n_items):
        probs = model.probability(theta, i)
        if probs.ndim == 1:
            expected[:, i] = probs
            variance[:, i] = probs * (1 - probs)
        else:
            n_cat = probs.shape[1]
            categories = np.arange(n_cat)
            expected[:, i] = np.sum(probs * categories, axis=1)
            expected_sq = np.sum(probs * (categories**2), axis=1)
            variance[:, i] = expected_sq - expected[:, i] ** 2

    return expected, variance


def compute_fit_stats(
    responses: NDArray[np.int_],
    expected: NDArray[np.float64],
    variance: NDArray[np.float64],
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute infit and outfit statistics.

    Parameters
    ----------
    responses : array
        Observed responses.
    expected : array
        Expected responses.
    variance : array
        Variance of responses.
    axis : int
        Axis along which to compute statistics (0 for items, 1 for persons).

    Returns
    -------
    infit : array
        Infit mean square statistics.
    outfit : array
        Outfit mean square statistics.
    """
    valid_mask = responses >= 0
    residuals = np.where(valid_mask, responses - expected, np.nan)
    std_residuals_sq = np.where(
        valid_mask & (variance > PROB_EPSILON),
        (residuals**2) / (variance + PROB_EPSILON),
        np.nan,
    )

    with np.errstate(all="ignore"):
        outfit = np.nanmean(std_residuals_sq, axis=axis)

    residuals_sq = np.where(valid_mask, residuals**2, 0.0)
    var_valid = np.where(valid_mask, variance, 0.0)

    numerator = np.sum(residuals_sq, axis=axis)
    denominator = np.sum(var_valid, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        infit = np.where(denominator > PROB_EPSILON, numerator / denominator, np.nan)

    return infit, outfit
