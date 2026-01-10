"""Reliability functions for IRT models.

Provides functions for computing marginal and empirical reliability
coefficients based on IRT model parameters.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def marginal_rxx(
    model: "BaseItemModel",
    theta_range: tuple[float, float] = (-6.0, 6.0),
    n_points: int = 61,
    density: str = "norm",
) -> float:
    """Compute marginal reliability coefficient.

    Marginal reliability is computed as:
        rxx = 1 - E[1/I(theta)] / Var(theta)

    where I(theta) is the test information function and the expectation
    is taken with respect to the ability distribution.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta_range : tuple of float
        Range of theta values for integration. Default (-6, 6).
    n_points : int
        Number of quadrature points. Default 61.
    density : str
        Ability distribution. Options: "norm" (standard normal).
        Default "norm".

    Returns
    -------
    float
        Marginal reliability coefficient.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> rxx = marginal_rxx(result.model)
    >>> print(f"Marginal reliability: {rxx:.3f}")
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta.reshape(-1, 1)

    item_info = model.information(theta_2d)
    test_info = np.sum(item_info, axis=1)

    if density == "norm":
        weights = stats.norm.pdf(theta)
    else:
        weights = stats.norm.pdf(theta)

    weights = weights / np.sum(weights)

    se_theta = 1.0 / np.sqrt(np.maximum(test_info, 1e-10))
    expected_var_error = np.sum(weights * se_theta**2)

    rxx = 1.0 - expected_var_error
    return float(np.clip(rxx, 0.0, 1.0))


def empirical_rxx(
    model: "BaseItemModel",
    theta_estimates: NDArray[np.float64],
    method: str = "posterior_variance",
) -> float:
    """Compute empirical reliability from theta estimates.

    Computes reliability using the observed variance of theta estimates
    and the average standard error of measurement.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta_estimates : NDArray[np.float64]
        Estimated ability values for examinees. Shape: (n_persons,) or (n_persons, n_dims).
    method : str
        Method for computing reliability:
        - "posterior_variance": Uses average posterior variance
        - "information": Uses test information at each theta

    Returns
    -------
    float
        Empirical reliability coefficient.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> rxx = empirical_rxx(result.model, result.theta)
    >>> print(f"Empirical reliability: {rxx:.3f}")
    """
    theta = np.atleast_1d(theta_estimates)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    observed_var = np.var(theta[:, 0])

    if observed_var < 1e-10:
        return 0.0

    if method == "posterior_variance":
        item_info = model.information(theta)
        test_info = np.sum(item_info, axis=1)
        se_theta = 1.0 / np.sqrt(np.maximum(test_info, 1e-10))
        avg_error_var = np.mean(se_theta**2)
    else:
        item_info = model.information(theta)
        test_info = np.sum(item_info, axis=1)
        se_theta = 1.0 / np.sqrt(np.maximum(test_info, 1e-10))
        avg_error_var = np.mean(se_theta**2)

    true_var = observed_var - avg_error_var
    true_var = max(true_var, 0.0)

    rxx = true_var / observed_var
    return float(np.clip(rxx, 0.0, 1.0))


def sem(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
) -> NDArray[np.float64]:
    """Compute standard error of measurement at given theta values.

    The SEM is computed as 1/sqrt(I(theta)) where I(theta) is the
    test information function.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values at which to compute SEM.

    Returns
    -------
    NDArray[np.float64]
        Standard error of measurement at each theta point.

    Examples
    --------
    >>> theta = np.array([[-2], [0], [2]])
    >>> se = sem(result.model, theta)
    >>> print(f"SEM at theta=0: {se[1]:.3f}")
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

    item_info = model.information(theta_arr)
    test_info = np.sum(item_info, axis=1)

    return 1.0 / np.sqrt(np.maximum(test_info, 1e-10))
