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
        Positive number of people evaluated in each likelihood batch. Smaller
        values reduce peak memory use for posterior and MCMC generation.
        Default 4096.

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
    if (
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
        pvs = _generate_pv_posterior(
            model,
            responses,
            n_plausible,
            n_quadpts,
            rng,
            int(chunk_size),
        )
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
    chunk_size: int,
) -> NDArray[np.float64]:
    """Generate PVs from a quadrature posterior in bounded row batches."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    n_persons = responses.shape[0]
    n_factors = model.n_factors

    quad = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=n_factors)
    nodes = quad.nodes
    weights = quad.weights

    # Draw in the historical order before batching the deterministic likelihood
    # work. This keeps seeded results independent of chunk_size.
    uniforms = np.empty((n_persons, n_plausible), dtype=np.float64)
    pvs = np.empty((n_persons, n_factors, n_plausible), dtype=np.float64)
    for p in range(n_plausible):
        uniforms[:, p] = rng.random(n_persons)
        pvs[:, :, p] = rng.normal(0, 0.3, size=(n_persons, n_factors))

    log_weights = np.log(weights + 1e-300)
    n_nodes = nodes.shape[0]
    for start in range(0, n_persons, chunk_size):
        stop = min(start + chunk_size, n_persons)
        log_likes = np.asarray(
            model.log_likelihood_batch(responses[start:stop], nodes),
            dtype=np.float64,
        )
        expected_shape = (stop - start, n_nodes)
        if log_likes.shape != expected_shape:
            raise ValueError(
                "model.log_likelihood_batch must return one value per "
                "response row and quadrature node"
            )

        log_posterior = log_likes + log_weights[None, :]
        row_maximum = np.max(log_posterior, axis=1, keepdims=True)
        if not np.all(np.isfinite(row_maximum)):
            raise ValueError("posterior likelihood must have positive finite mass")
        posterior = np.exp(log_posterior - row_maximum)
        posterior_mass = posterior.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(posterior_mass)) or np.any(posterior_mass <= 0.0):
            raise ValueError("posterior likelihood must have positive finite mass")
        posterior /= posterior_mass

        cumulative = np.cumsum(posterior, axis=1)
        cumulative[:, -1] = 1.0
        indices = _inverse_cdf_rows(cumulative, uniforms[start:stop])
        pvs[start:stop] += np.moveaxis(nodes[indices], 1, 2)

    return pvs


def _inverse_cdf_rows(
    cumulative: NDArray[np.float64],
    uniforms: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Search independent row CDFs without a rows-by-draws-by-nodes array."""
    n_rows, n_nodes = cumulative.shape
    if uniforms.shape[0] != n_rows:
        raise ValueError("uniform draws must align with cumulative rows")

    # Separate adjacent row CDFs with a unit gap before flattening them into a
    # single monotonic sequence. The gap also handles an exact uniform draw of
    # zero without matching the preceding row's terminal probability.
    row_offsets = (2.0 * np.arange(n_rows, dtype=np.float64))[:, None]
    shifted_cdf = (cumulative + row_offsets).ravel()
    shifted_uniforms = (uniforms + row_offsets).ravel()
    flat_indices = np.searchsorted(shifted_cdf, shifted_uniforms, side="left")
    row_starts = (np.arange(n_rows, dtype=np.intp) * n_nodes)[:, None]
    return flat_indices.reshape(uniforms.shape) - row_starts


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
    factor: int | Literal["all"] = 0,
) -> dict[str, float | int | NDArray[np.float64]]:
    """Compute population statistics using plausible values.

    Parameters
    ----------
    pvs : NDArray
        Plausible values (n_persons, n_factors, n_plausible)
    statistic : str
        Statistic to compute: ``"mean"``, ``"variance"``, ``"sd"``, or
        ``"percentile_<value>``.
    factor : int or "all", default=0
        Zero-based latent factor to summarize. Use ``"all"`` to return one
        estimate and standard error per factor.

    Returns
    -------
    dict
        Combined estimate, standard error, between-draw variance, and number
        of plausible values. Values are scalars for one factor and arrays with
        shape ``(n_factors,)`` when ``factor="all"``.

    Notes
    -----
    With one plausible value, the estimate is returned while ``se`` and
    ``between_var`` are NaN because between-draw uncertainty is undefined.
    """
    try:
        values = np.asarray(pvs, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pvs must contain numeric plausible values") from exc
    if values.ndim != 3:
        raise ValueError("pvs must have shape (n_persons, n_factors, n_plausible)")
    n_persons, n_factors, n_plausible = values.shape
    if n_persons < 1 or n_factors < 1:
        raise ValueError("pvs must contain at least one person and one factor")
    if n_plausible < 1:
        raise ValueError("pvs must contain at least one plausible value")
    if not np.all(np.isfinite(values)):
        raise ValueError("pvs must contain only finite plausible values")
    if not isinstance(statistic, str):
        raise ValueError("statistic must be a string")

    if factor == "all":
        selected = values
        return_arrays = True
    else:
        if isinstance(factor, (bool, np.bool_)) or not isinstance(
            factor, (int, np.integer)
        ):
            raise ValueError("factor must be an integer or 'all'")
        factor_index = int(factor)
        if factor_index < 0 or factor_index >= n_factors:
            raise ValueError(f"factor must be between 0 and {n_factors - 1}")
        selected = values[:, factor_index : factor_index + 1, :]
        return_arrays = False

    if statistic == "mean":
        estimates = np.mean(selected, axis=0)
    elif statistic in {"variance", "sd"}:
        if n_persons < 2:
            raise ValueError(f"{statistic} requires at least two people")
        if statistic == "variance":
            estimates = np.var(selected, axis=0, ddof=1)
        else:
            estimates = np.std(selected, axis=0, ddof=1)
    elif isinstance(statistic, str) and statistic.startswith("percentile_"):
        suffix = statistic.removeprefix("percentile_")
        try:
            percentile = float(suffix)
        except ValueError as exc:
            raise ValueError(
                "percentile statistic must end with a number from 0 to 100"
            ) from exc
        if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
            raise ValueError("percentile must be a finite number from 0 to 100")
        estimates = np.percentile(selected, percentile, axis=0)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    combined = np.mean(estimates, axis=1)
    if n_plausible == 1:
        between_variance = np.full(n_factors if return_arrays else 1, np.nan)
        standard_error = np.full_like(between_variance, np.nan)
    else:
        between_variance = np.var(estimates, axis=1, ddof=1)
        standard_error = np.sqrt(between_variance * (1.0 + 1.0 / n_plausible))
    if return_arrays:
        return {
            "estimate": combined,
            "se": standard_error,
            "between_var": between_variance,
            "n_plausible": n_plausible,
        }
    return {
        "estimate": float(combined[0]),
        "se": float(standard_error[0]),
        "between_var": float(between_variance[0]),
        "n_plausible": n_plausible,
    }
