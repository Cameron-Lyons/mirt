"""Plausible Value Generation for IRT.

Plausible values are multiple imputations of latent trait scores,
drawn from the posterior distribution of theta given the responses.
They are used in large-scale assessments (e.g., PISA, NAEP) to
properly account for measurement error in secondary analyses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def generate_plausible_values(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    n_plausible: int = 5,
    method: Literal["posterior", "mcmc"] = "posterior",
    n_quadpts: int = 21,
    n_iter: int = 50,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate plausible values for latent abilities.

    Plausible values are random draws from the posterior distribution
    of theta given the observed responses. They provide a way to
    properly account for measurement error in subsequent analyses.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted IRT model
    responses : NDArray
        Response matrix (n_persons, n_items)
    n_plausible : int
        Number of plausible values to generate per person
    method : str
        Generation method:
        - 'posterior': Direct sampling from posterior using quadrature
        - 'mcmc': MCMC sampling (slower but more flexible)
    n_quadpts : int
        Number of quadrature points (for posterior method)
    n_iter : int
        Number of MCMC iterations (for mcmc method)
    seed : int, optional
        Random seed

    Returns
    -------
    NDArray
        Plausible values with shape (n_persons, n_factors, n_plausible)
        For unidimensional models: (n_persons, 1, n_plausible)
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        model = model.model

    rng = np.random.default_rng(seed)
    responses = np.asarray(responses)
    responses.shape[0]

    if method == "posterior":
        pvs = _generate_pv_posterior(model, responses, n_plausible, n_quadpts, rng)
    elif method == "mcmc":
        pvs = _generate_pv_mcmc(model, responses, n_plausible, rng, n_iter)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pvs


def _generate_pv_posterior(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_plausible: int,
    n_quadpts: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate PVs by sampling from posterior using quadrature."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    n_persons = responses.shape[0]
    n_factors = model.n_factors

    quad = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=n_factors)
    nodes = quad.nodes
    weights = quad.weights

    pvs = np.zeros((n_persons, n_factors, n_plausible))

    for i in range(n_persons):
        resp_i = responses[i : i + 1]

        log_likes = np.zeros(len(nodes))
        for q, node in enumerate(nodes):
            theta_q = node.reshape(1, -1)
            log_likes[q] = model.log_likelihood(resp_i, theta_q)[0]

        log_posterior = log_likes + np.log(weights + 1e-300)

        log_posterior = log_posterior - np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior = posterior / posterior.sum()

        for p in range(n_plausible):
            idx = rng.choice(len(nodes), p=posterior)
            theta_sample = nodes[idx].copy()

            jitter_sd = 0.3
            theta_sample += rng.normal(0, jitter_sd, n_factors)

            pvs[i, :, p] = theta_sample

    return pvs


def _generate_pv_mcmc(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_plausible: int,
    rng: np.random.Generator,
    n_iter: int = 50,
) -> NDArray[np.float64]:
    """Generate PVs using MCMC sampling."""
    n_persons = responses.shape[0]
    n_factors = model.n_factors

    proposal_sd = 0.5

    pvs = np.zeros((n_persons, n_factors, n_plausible))

    for i in range(n_persons):
        resp_i = responses[i : i + 1]

        theta = np.zeros(n_factors)

        for p in range(n_plausible):
            for iteration in range(n_iter):
                proposal = theta + rng.normal(0, proposal_sd, n_factors)

                ll_current = model.log_likelihood(resp_i, theta.reshape(1, -1))[0]
                ll_proposal = model.log_likelihood(resp_i, proposal.reshape(1, -1))[0]

                prior_current = stats.norm.logpdf(theta).sum()
                prior_proposal = stats.norm.logpdf(proposal).sum()

                log_alpha = (ll_proposal + prior_proposal) - (
                    ll_current + prior_current
                )

                if np.log(rng.random()) < log_alpha:
                    theta = proposal

            if p == 0:
                continue

            pvs[i, :, p] = theta.copy()

        pvs[i, :, 0] = theta.copy()

    return pvs


def combine_plausible_values(
    estimates: list[float | NDArray],
    variances: list[float | NDArray] | None = None,
) -> dict[str, float | NDArray]:
    """Combine estimates from analyses using plausible values.

    Uses Rubin's combining rules for multiple imputation:
    - Combined estimate = mean of estimates
    - Total variance = within-imputation variance + between-imputation variance

    Parameters
    ----------
    estimates : list
        Estimates from each plausible value (e.g., regression coefficients)
    variances : list, optional
        Variance estimates for each PV analysis.
        If None, only combines point estimates.

    Returns
    -------
    dict
        Dictionary with:
        - 'estimate': Combined estimate
        - 'variance': Total variance (if variances provided)
        - 'se': Standard error (if variances provided)
        - 'between_var': Between-imputation variance
        - 'within_var': Within-imputation variance (if provided)
    """
    m = len(estimates)
    estimates = np.array(estimates)

    combined = np.mean(estimates, axis=0)

    between_var = np.var(estimates, axis=0, ddof=1)

    result = {
        "estimate": combined,
        "between_var": between_var,
        "n_imputations": m,
    }

    if variances is not None:
        variances = np.array(variances)

        within_var = np.mean(variances, axis=0)

        total_var = within_var + (1 + 1 / m) * between_var

        result["within_var"] = within_var
        result["variance"] = total_var
        result["se"] = np.sqrt(total_var)

        if np.all(within_var > 0):
            r = (1 + 1 / m) * between_var / within_var
            df = (m - 1) * (1 + 1 / r) ** 2
            result["df"] = df

    return result


def plausible_value_regression(
    pvs: NDArray[np.float64],
    y: NDArray[np.float64],
    X: NDArray[np.float64] | None = None,
) -> dict[str, float | NDArray]:
    """Perform regression using plausible values as predictor.

    Runs regression with each set of plausible values and combines
    results using Rubin's rules.

    Parameters
    ----------
    pvs : NDArray
        Plausible values (n_persons, n_factors, n_plausible)
    y : NDArray
        Outcome variable (n_persons,)
    X : NDArray, optional
        Additional covariates (n_persons, n_covariates)

    Returns
    -------
    dict
        Combined regression results:
        - 'coefficients': Combined regression coefficients
        - 'se': Standard errors
        - 'pvalues': P-values
    """
    from scipy import stats as sp_stats

    n_persons, n_factors, n_plausible = pvs.shape

    coef_list = []
    var_list = []

    for p in range(n_plausible):
        theta_p = pvs[:, :, p]

        if X is not None:
            design = np.column_stack([np.ones(n_persons), theta_p, X])
        else:
            design = np.column_stack([np.ones(n_persons), theta_p])

        try:
            coef, residuals, rank, s = np.linalg.lstsq(design, y, rcond=None)

            if len(residuals) > 0:
                mse = residuals[0] / (n_persons - design.shape[1])
            else:
                mse = np.var(y - design @ coef)

            var_coef = mse * np.linalg.inv(design.T @ design).diagonal()

            coef_list.append(coef)
            var_list.append(var_coef)
        except np.linalg.LinAlgError:
            continue

    if len(coef_list) < 2:
        return {"coefficients": np.nan, "se": np.nan, "pvalues": np.nan}

    combined = combine_plausible_values(coef_list, var_list)

    if "se" in combined and "df" in combined:
        t_stats = combined["estimate"] / combined["se"]
        pvalues = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), combined["df"]))
    else:
        t_stats = combined["estimate"] / combined.get(
            "se", np.sqrt(combined["between_var"])
        )
        pvalues = 2 * (1 - sp_stats.norm.cdf(np.abs(t_stats)))

    return {
        "coefficients": combined["estimate"],
        "se": combined.get("se", np.sqrt(combined["between_var"])),
        "pvalues": pvalues,
        "n_plausible": n_plausible,
    }


def plausible_value_statistics(
    pvs: NDArray[np.float64],
    statistic: str = "mean",
) -> dict[str, float]:
    """Compute population statistics using plausible values.

    Parameters
    ----------
    pvs : NDArray
        Plausible values (n_persons, n_factors, n_plausible)
    statistic : str
        Statistic to compute: 'mean', 'variance', 'percentile_10', etc.

    Returns
    -------
    dict
        Combined statistic with standard error
    """
    n_persons, n_factors, n_plausible = pvs.shape

    estimates = []

    for p in range(n_plausible):
        theta_p = pvs[:, 0, p]

        if statistic == "mean":
            est = np.mean(theta_p)
        elif statistic == "variance":
            est = np.var(theta_p, ddof=1)
        elif statistic == "sd":
            est = np.std(theta_p, ddof=1)
        elif statistic.startswith("percentile_"):
            pct = float(statistic.split("_")[1])
            est = np.percentile(theta_p, pct)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        estimates.append(est)

    combined = combine_plausible_values(estimates)

    return {
        "estimate": float(combined["estimate"]),
        "se": float(np.sqrt(combined["between_var"] * (1 + 1 / n_plausible))),
        "n_plausible": n_plausible,
    }
