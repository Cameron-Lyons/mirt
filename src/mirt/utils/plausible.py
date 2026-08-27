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

from mirt.exceptions import MirtDataError
from mirt.utils.data import validate_responses

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
    burn_in: int = 0,
    proposal_scale: float = 0.5,
    chunk_size: int = 4096,
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
        Positive number of plausible values to generate per person
    method : str
        Generation method:
        - 'posterior': Direct sampling from posterior using quadrature
        - 'mcmc': MCMC sampling (slower but more flexible)
    n_quadpts : int
        Positive number of quadrature points (for posterior method)
    n_iter : int
        Positive number of MCMC iterations between draws (for mcmc method)
    seed : int, optional
        Random seed
    burn_in : int
        Non-negative number of MCMC iterations to discard before the first
        retained draw. Default 0.
    proposal_scale : float
        Positive standard deviation of the MCMC random-walk proposal.
        Default 0.5.
    chunk_size : int
        Positive number of people evaluated in each MCMC likelihood batch.
        Smaller values reduce peak memory use. Default 4096.

    Returns
    -------
    NDArray
        Plausible values with shape (n_persons, n_factors, n_plausible)
        For unidimensional models: (n_persons, 1, n_plausible)

    Raises
    ------
    ValueError
        If the model is unfitted or a generation parameter is invalid.
    MirtDataError
        If the response matrix shape or category codes are invalid.
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model, FitResult):
        model = model.model

    if not model.is_fitted:
        raise ValueError("Model must be fitted before generating plausible values")
    if isinstance(n_plausible, bool) or not isinstance(n_plausible, (int, np.integer)):
        raise ValueError("n_plausible must be a positive integer")
    if n_plausible < 1:
        raise ValueError("n_plausible must be a positive integer")
    if method not in ("posterior", "mcmc"):
        raise ValueError(f"Unknown method: {method}")
    if method == "posterior" and (
        isinstance(n_quadpts, bool)
        or not isinstance(n_quadpts, (int, np.integer))
        or n_quadpts < 1
    ):
        raise ValueError("n_quadpts must be a positive integer")
    if method == "mcmc" and (
        isinstance(n_iter, bool)
        or not isinstance(n_iter, (int, np.integer))
        or n_iter < 1
    ):
        raise ValueError("n_iter must be a positive integer")
    if method == "mcmc" and (
        isinstance(burn_in, bool)
        or not isinstance(burn_in, (int, np.integer))
        or burn_in < 0
    ):
        raise ValueError("burn_in must be a non-negative integer")
    if method == "mcmc" and (
        isinstance(proposal_scale, bool)
        or not isinstance(proposal_scale, (int, float, np.integer, np.floating))
        or not np.isfinite(proposal_scale)
        or proposal_scale <= 0
    ):
        raise ValueError("proposal_scale must be a positive finite number")
    if method == "mcmc" and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")

    responses = validate_responses(responses, n_items=model.n_items)
    observed = responses >= 0
    if model.is_polytomous:
        categories = np.asarray(model.n_categories)
        invalid = observed & (responses >= categories[None, :])
        if np.any(invalid):
            item_idx = int(np.flatnonzero(np.any(invalid, axis=0))[0])
            raise MirtDataError(
                f"responses for item {item_idx} must be below {categories[item_idx]}"
            )
    elif np.any(responses[observed] > 1):
        raise MirtDataError("dichotomous responses must be coded as 0 or 1")

    rng = np.random.default_rng(seed)

    if method == "posterior":
        pvs = _generate_pv_posterior(model, responses, n_plausible, n_quadpts, rng)
    else:
        pvs = _generate_pv_mcmc(
            model,
            responses,
            n_plausible,
            rng,
            n_iter,
            burn_in,
            float(proposal_scale),
            int(chunk_size),
        )

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

    log_likes = model.log_likelihood_batch(responses, nodes)
    log_posterior = log_likes + np.log(weights + 1e-300)[None, :]
    log_posterior -= np.max(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum(axis=1, keepdims=True)

    cumulative = np.cumsum(posterior, axis=1)
    cumulative[:, -1] = 1.0

    pvs = np.empty((n_persons, n_factors, n_plausible))
    for p in range(n_plausible):
        uniforms = rng.random(n_persons)
        indices = np.sum(uniforms[:, None] > cumulative, axis=1)
        jitter = rng.normal(0, 0.3, size=(n_persons, n_factors))
        pvs[:, :, p] = nodes[indices] + jitter

    return pvs


def _paired_log_density(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    chunk_size: int,
) -> NDArray[np.float64]:
    """Evaluate paired person/ability log densities in bounded batches."""
    n_persons = responses.shape[0]
    density = np.empty(n_persons, dtype=np.float64)
    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        theta_chunk = theta[start:stop]
        log_likelihood = np.asarray(
            model.log_likelihood(responses[start:stop], theta_chunk),
            dtype=np.float64,
        )
        if log_likelihood.shape != (stop - start,):
            raise ValueError(
                "model.log_likelihood must return one value per response row"
            )
        density[start:stop] = log_likelihood - 0.5 * np.einsum(
            "ij,ij->i", theta_chunk, theta_chunk, optimize=True
        )
    return density


def _generate_pv_mcmc(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_plausible: int,
    rng: np.random.Generator,
    n_iter: int = 50,
    burn_in: int = 0,
    proposal_scale: float = 0.5,
    chunk_size: int = 4096,
) -> NDArray[np.float64]:
    """Generate PVs using batched random-walk Metropolis sampling."""
    n_persons = responses.shape[0]
    n_factors = model.n_factors

    theta = np.zeros((n_persons, n_factors), dtype=np.float64)
    current_log_density = _paired_log_density(model, responses, theta, chunk_size)
    pvs = np.empty((n_persons, n_factors, n_plausible), dtype=np.float64)

    draw = 0
    total_iterations = burn_in + n_plausible * n_iter
    for completed in range(1, total_iterations + 1):
        proposal = theta + rng.normal(0.0, proposal_scale, size=theta.shape)
        proposal_log_density = _paired_log_density(
            model, responses, proposal, chunk_size
        )
        accepted = (
            np.log(rng.random(n_persons)) < proposal_log_density - current_log_density
        )
        theta[accepted] = proposal[accepted]
        current_log_density[accepted] = proposal_log_density[accepted]

        if completed > burn_in and (completed - burn_in) % n_iter == 0:
            pvs[:, :, draw] = theta
            draw += 1

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
            coef, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)

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
