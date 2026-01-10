"""Statistical tests for IRT models.

Provides Wald and Lagrange (score) tests for parameter constraints
and model comparison.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class WaldTestResult:
    """Result of a Wald test.

    Attributes
    ----------
    statistic : float
        Wald chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value for the test.
    parameter_estimates : NDArray[np.float64]
        Parameter estimates being tested.
    standard_errors : NDArray[np.float64]
        Standard errors of estimates.
    constraint_values : NDArray[np.float64]
        Values under null hypothesis.
    """

    statistic: float
    df: int
    p_value: float
    parameter_estimates: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    constraint_values: NDArray[np.float64]


@dataclass
class LagrangeTestResult:
    """Result of a Lagrange (score) test.

    Attributes
    ----------
    statistic : float
        Score chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value for the test.
    scores : NDArray[np.float64]
        Score vector (gradient) for constrained parameters.
    """

    statistic: float
    df: int
    p_value: float
    scores: NDArray[np.float64]


def wald(
    model: "BaseItemModel",
    param_indices: list[int] | NDArray[np.intp],
    constraint_values: NDArray[np.float64] | list[float] | None = None,
    vcov: NDArray[np.float64] | None = None,
) -> WaldTestResult:
    """Perform Wald test on model parameters.

    Tests H0: theta = constraint_values using the Wald statistic:
        W = (theta - c)' V^{-1} (theta - c)

    where V is the variance-covariance matrix of the parameter estimates.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    param_indices : array-like of int
        Indices of parameters to test.
    constraint_values : array-like of float, optional
        Values under null hypothesis. Default is zeros.
    vcov : NDArray[np.float64], optional
        Variance-covariance matrix. If None, uses information matrix.

    Returns
    -------
    WaldTestResult
        Test results including statistic, df, and p-value.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> # Test if discrimination of item 0 equals 1.0
    >>> test = wald(result.model, param_indices=[0], constraint_values=[1.0])
    >>> print(f"Wald chi-sq = {test.statistic:.3f}, p = {test.p_value:.4f}")
    """
    param_indices = np.asarray(param_indices)
    n_params = len(param_indices)

    if hasattr(model, "parameters"):
        params = model.parameters
        all_params_list: list[float] = []
        for key in sorted(params.keys()):
            all_params_list.extend(np.asarray(params[key]).ravel())
        all_params = np.array(all_params_list)
    else:
        all_params = np.concatenate(
            [
                np.asarray(model.discrimination).ravel(),
                np.asarray(model.difficulty).ravel(),
            ]
        )
    estimates = all_params[param_indices]

    if constraint_values is None:
        constraint_values = np.zeros(n_params)
    else:
        constraint_values = np.asarray(constraint_values, dtype=np.float64)

    if vcov is None:
        if hasattr(model, "vcov") and model.vcov is not None:
            vcov = model.vcov
        elif hasattr(model, "information_matrix"):
            info = model.information_matrix()
            try:
                vcov = np.linalg.inv(info)
            except np.linalg.LinAlgError:
                vcov = np.linalg.pinv(info)
        else:
            vcov = np.eye(len(all_params)) * 0.01

    vcov_subset = vcov[np.ix_(param_indices, param_indices)]
    se = np.sqrt(np.diag(vcov_subset))

    diff = estimates - constraint_values

    try:
        vcov_inv = np.linalg.inv(vcov_subset)
    except np.linalg.LinAlgError:
        vcov_inv = np.linalg.pinv(vcov_subset)

    statistic = float(diff @ vcov_inv @ diff)
    p_value = 1 - stats.chi2.cdf(statistic, n_params)

    return WaldTestResult(
        statistic=statistic,
        df=n_params,
        p_value=float(p_value),
        parameter_estimates=estimates,
        standard_errors=se,
        constraint_values=constraint_values,
    )


def lagrange(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    param_indices: list[int] | NDArray[np.intp],
    vcov: NDArray[np.float64] | None = None,
) -> LagrangeTestResult:
    """Perform Lagrange (score) test for parameter constraints.

    Tests whether constrained parameters should be freed using
    the score statistic:
        LM = S' V S

    where S is the score (gradient) vector evaluated at the constrained
    estimates.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model (under constraints).
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons, n_dims).
    param_indices : array-like of int
        Indices of constrained parameters to test.
    vcov : NDArray[np.float64], optional
        Variance-covariance matrix for full model.

    Returns
    -------
    LagrangeTestResult
        Test results including statistic, df, and p-value.

    Examples
    --------
    >>> # Fit constrained model (e.g., Rasch with equal discriminations)
    >>> result = fit_mirt(responses, model="1PL")
    >>> # Test if discriminations should be freed
    >>> test = lagrange(result.model, responses, result.theta, param_indices=[0, 2, 4])
    >>> print(f"LM chi-sq = {test.statistic:.3f}, p = {test.p_value:.4f}")
    """
    param_indices = np.asarray(param_indices)
    n_params = len(param_indices)

    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_2d(theta)
    if theta.shape[0] == 1 and theta.shape[1] > 1:
        theta = theta.T
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    probs = model.probability(theta)
    residuals = responses - probs

    if hasattr(model, "parameters"):
        params = model.parameters
        all_params_list: list[float] = []
        for key in sorted(params.keys()):
            all_params_list.extend(np.asarray(params[key]).ravel())
        all_params = np.array(all_params_list)
    else:
        all_params = np.concatenate(
            [
                np.asarray(model.discrimination).ravel(),
                np.asarray(model.difficulty).ravel(),
            ]
        )
    n_total_params = len(all_params)

    scores = np.zeros(n_total_params)

    if hasattr(model, "score_function"):
        scores = model.score_function(responses, theta)
    else:
        n_items = model.n_items

        for j in range(n_items):
            item_residuals = residuals[:, j]

            scores[j] = np.sum(item_residuals * theta[:, 0])
            scores[n_items + j] = np.sum(item_residuals)

    score_subset = scores[param_indices]

    if vcov is None:
        if hasattr(model, "vcov") and model.vcov is not None:
            vcov = model.vcov
        else:
            vcov = np.eye(n_total_params) * 0.01

    vcov_subset = vcov[np.ix_(param_indices, param_indices)]

    statistic = float(score_subset @ vcov_subset @ score_subset)
    p_value = 1 - stats.chi2.cdf(statistic, n_params)

    return LagrangeTestResult(
        statistic=statistic,
        df=n_params,
        p_value=float(p_value),
        scores=score_subset,
    )


def _compute_log_likelihood(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """Compute log-likelihood for responses given model and theta."""
    probs = model.probability(theta)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    mask = ~np.isnan(responses)
    ll = np.sum(
        responses[mask] * np.log(probs[mask])
        + (1 - responses[mask]) * np.log(1 - probs[mask])
    )
    return float(ll)


def likelihood_ratio(
    ll_full: float,
    ll_reduced: float,
    df_diff: int,
) -> tuple[float, float]:
    """Compute likelihood ratio test statistic.

    Parameters
    ----------
    ll_full : float
        Log-likelihood of full (less constrained) model.
    ll_reduced : float
        Log-likelihood of reduced (more constrained) model.
    df_diff : int
        Difference in degrees of freedom (number of constraints).

    Returns
    -------
    statistic : float
        Chi-square statistic (-2 * (ll_reduced - ll_full)).
    p_value : float
        P-value from chi-square distribution.
    """
    statistic = -2 * (ll_reduced - ll_full)
    p_value = 1 - stats.chi2.cdf(statistic, df_diff)
    return float(statistic), float(p_value)
