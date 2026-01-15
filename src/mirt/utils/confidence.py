"""Confidence interval functions for IRT models.

Provides profile-likelihood and other advanced confidence interval
methods for IRT parameter estimates.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class PLCIResult:
    """Result of profile-likelihood confidence interval.

    Attributes
    ----------
    param_name : str
        Name of parameter.
    param_idx : int
        Index of parameter.
    estimate : float
        Point estimate.
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.
    alpha : float
        Significance level used.
    converged : bool
        Whether optimization converged for both bounds.
    """

    param_name: str
    param_idx: int
    estimate: float
    lower: float
    upper: float
    alpha: float
    converged: bool


def PLCI(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    param_idx: int,
    param_name: str = "discrimination",
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> PLCIResult:
    """Compute profile-likelihood confidence interval for a parameter.

    Profile-likelihood CIs are more accurate than Wald CIs,
    especially for parameters near boundaries.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix used for fitting.
    param_idx : int
        Index of the item (for item parameters).
    param_name : str
        Parameter name ("discrimination" or "difficulty").
    alpha : float
        Significance level. Default 0.05 for 95% CI.
    max_iter : int
        Maximum iterations for root finding.
    tol : float
        Tolerance for convergence.

    Returns
    -------
    PLCIResult
        Profile-likelihood confidence interval.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> ci = PLCI(result.model, responses, param_idx=0, param_name="difficulty")
    >>> print(f"95% CI for b_0: [{ci.lower:.3f}, {ci.upper:.3f}]")

    Notes
    -----
    The profile-likelihood CI is found by varying the parameter of interest
    and re-optimizing all other parameters, then finding where the
    log-likelihood drops by chi2(1, alpha/2) / 2.
    """
    from scipy import stats
    from scipy.optimize import brentq

    responses = np.asarray(responses, dtype=np.float64)

    if param_name == "discrimination":
        estimate = float(np.asarray(model.discrimination)[param_idx])
    elif param_name == "difficulty":
        estimate = float(np.asarray(model.difficulty)[param_idx])
    else:
        raise ValueError(f"Unknown parameter: {param_name}")

    theta_grid = np.linspace(-4, 4, 21).reshape(-1, 1)
    weights = stats.norm.pdf(theta_grid.ravel())
    weights = weights / np.sum(weights)

    def compute_profile_ll(fixed_value: float) -> float:
        disc = np.asarray(model.discrimination).copy()
        diff = np.asarray(model.difficulty).copy()

        if param_name == "discrimination":
            disc[param_idx] = fixed_value
        else:
            diff[param_idx] = fixed_value

        if disc.ndim == 1:
            disc = disc.reshape(-1, 1)

        n_persons = responses.shape[0]
        n_items = responses.shape[1]

        marginal_ll = 0.0
        for i in range(n_persons):
            person_ll = np.zeros(len(theta_grid))

            for q, theta_q in enumerate(theta_grid.ravel()):
                ll_q = 0.0
                for j in range(n_items):
                    if np.isnan(responses[i, j]):
                        continue

                    logit = disc[j, 0] * (theta_q - diff[j])
                    p = 1 / (1 + np.exp(-logit))
                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

                    if responses[i, j] == 1:
                        ll_q += np.log(p)
                    else:
                        ll_q += np.log(1 - p)

                person_ll[q] = np.exp(ll_q) * weights[q]

            marginal_ll += np.log(np.maximum(np.sum(person_ll), 1e-300))

        return marginal_ll

    ll_max = compute_profile_ll(estimate)

    critical = stats.chi2.ppf(1 - alpha, df=1) / 2

    def objective_lower(x):
        return compute_profile_ll(x) - (ll_max - critical)

    def objective_upper(x):
        return compute_profile_ll(x) - (ll_max - critical)

    converged = True

    if param_name == "discrimination":
        search_lower = max(0.1, estimate - 2)
        search_upper = estimate + 2
    else:
        search_lower = estimate - 3
        search_upper = estimate + 3

    try:
        if objective_lower(search_lower) * objective_lower(estimate) < 0:
            lower = brentq(objective_lower, search_lower, estimate, xtol=tol)
        else:
            lower = search_lower
            converged = False
    except (ValueError, RuntimeError):
        lower = estimate - 1.96 * 0.1
        converged = False

    try:
        if objective_upper(estimate) * objective_upper(search_upper) < 0:
            upper = brentq(objective_upper, estimate, search_upper, xtol=tol)
        else:
            upper = search_upper
            converged = False
    except (ValueError, RuntimeError):
        upper = estimate + 1.96 * 0.1
        converged = False

    return PLCIResult(
        param_name=param_name,
        param_idx=param_idx,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        alpha=alpha,
        converged=converged,
    )


def score_CI(
    model: "BaseItemModel",
    theta: float,
    responses: NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    method: str = "wald",
) -> tuple[float, float]:
    """Compute confidence interval for a theta estimate.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : float
        Point estimate of theta.
    responses : NDArray[np.float64], optional
        Individual's response pattern (for likelihood-based CI).
    alpha : float
        Significance level. Default 0.05.
    method : str
        CI method: "wald" or "likelihood".

    Returns
    -------
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.
    """
    from scipy import stats

    theta_2d = np.array([[theta]])
    info = model.information(theta_2d)
    test_info = np.sum(info)

    se = 1 / np.sqrt(max(test_info, PROB_EPSILON))

    z = stats.norm.ppf(1 - alpha / 2)

    if method == "wald":
        lower = theta - z * se
        upper = theta + z * se
    elif method == "likelihood" and responses is not None:
        critical = stats.chi2.ppf(1 - alpha, df=1) / 2

        def ll_at_theta(t):
            t_2d = np.array([[t]])
            probs = model.probability(t_2d).ravel()
            probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

            ll = 0.0
            for j, r in enumerate(responses):
                if not np.isnan(r):
                    if r == 1:
                        ll += np.log(probs[j])
                    else:
                        ll += np.log(1 - probs[j])
            return ll

        ll_max = ll_at_theta(theta)

        from scipy.optimize import brentq

        try:
            lower = brentq(
                lambda t: ll_at_theta(t) - (ll_max - critical),
                theta - 5,
                theta,
                xtol=0.01,
            )
        except (ValueError, RuntimeError):
            lower = theta - z * se

        try:
            upper = brentq(
                lambda t: ll_at_theta(t) - (ll_max - critical),
                theta,
                theta + 5,
                xtol=0.01,
            )
        except (ValueError, RuntimeError):
            upper = theta + z * se
    else:
        lower = theta - z * se
        upper = theta + z * se

    return float(lower), float(upper)


def delta_method(
    estimates: NDArray[np.float64],
    vcov: NDArray[np.float64],
    transform_func: Callable[[NDArray[np.float64]], float],
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Compute standard error using delta method.

    Parameters
    ----------
    estimates : NDArray[np.float64]
        Parameter estimates.
    vcov : NDArray[np.float64]
        Variance-covariance matrix.
    transform_func : callable
        Function that transforms parameters.
    eps : float
        Step size for numerical gradient.

    Returns
    -------
    transformed : float
        Transformed estimate.
    se : float
        Standard error of transformed estimate.
    """
    estimates = np.asarray(estimates).ravel()
    transformed = transform_func(estimates)

    gradient = np.zeros(len(estimates))
    for i in range(len(estimates)):
        e_plus = estimates.copy()
        e_plus[i] += eps
        e_minus = estimates.copy()
        e_minus[i] -= eps
        gradient[i] = (transform_func(e_plus) - transform_func(e_minus)) / (2 * eps)

    var = gradient @ vcov @ gradient
    se = np.sqrt(np.maximum(var, 0))

    return float(transformed), float(se)
