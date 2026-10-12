"""Shared, bounded NumPy evaluation of the Gaussian variational logistic bound."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

_MAX_WORKING_ELEMENTS = 1_000_000


def variational_elbo(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    lambda_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> float:
    """Sum the logistic bound and Gaussian prior/entropy terms in row blocks.

    Scratch space accounts for item and factor counts, with at least one full
    respondent per block. Inputs are never modified. Missing responses contribute
    no likelihood, but every respondent contributes a Gaussian KL term.
    """
    n_persons, n_items = responses.shape
    n_factors = loadings.shape[1]
    prior_precision = np.linalg.solve(prior_cov, np.eye(n_factors))
    log_det_prior = float(np.linalg.slogdet(prior_cov)[1])
    entries_per_person = (
        n_items * (n_factors + 8) + 3 * n_factors**2 + 3 * n_factors + 8
    )
    block_size = max(1, _MAX_WORKING_ELEMENTS // max(1, entries_per_person))

    expected_log_likelihood = 0.0
    divergence = 0.0
    for start in range(0, n_persons, block_size):
        stop = min(start + block_size, n_persons)
        data, block_xi = responses[start:stop], xi[start:stop]
        block_mu, block_sigma = mu[start:stop], sigma[start:stop]

        lam = lambda_function(block_xi)
        eta_mean = block_mu @ loadings.T
        eta_mean += intercepts
        eta_second = np.einsum(
            "jf,ifg,jg->ij", loadings, block_sigma, loadings, optimize=True
        )
        eta_second += eta_mean**2

        terms = np.negative(block_xi)
        np.logaddexp(0.0, terms, out=terms)
        np.negative(terms, out=terms)
        eta_mean *= data - 0.5
        terms += eta_mean
        # Reuse the predictor buffer after adding its linear contribution.
        np.multiply(block_xi, 0.5, out=eta_mean)
        terms -= eta_mean
        np.square(block_xi, out=eta_mean)
        eta_second -= eta_mean
        eta_second *= lam
        terms -= eta_second
        # Zero missing entries in owned storage so the reduction can use NumPy's
        # pairwise summation without allocating a gathered observation array.
        np.copyto(terms, 0.0, where=~(data >= 0))
        expected_log_likelihood += float(np.sum(terms))
        del lam, eta_mean, eta_second, terms

        diff = block_mu - prior_mean
        kl = np.einsum("if,fg,ig->i", diff, prior_precision, diff, optimize=True)
        kl *= 0.5
        kl += 0.5 * np.einsum("fg,igf->i", prior_precision, block_sigma, optimize=True)
        kl += 0.5 * (log_det_prior - np.linalg.slogdet(block_sigma)[1])
        kl -= 0.5 * n_factors
        divergence += float(np.sum(kl))

    return expected_log_likelihood - divergence


# Factors up to which the marginal likelihood is integrated on EM's default
# grid (61, 31 or 15 points per dimension); more use importance sampling.
_MAX_QUADRATURE_FACTORS = 3
_IMPORTANCE_DRAWS = 500
# Share of importance draws from the prior. This defensive component bounds
# every weight by 1 / share, even where the variational posterior is too narrow.
_DEFENSIVE_SHARE = 0.1
# Variational posteriors understate posterior spread; widen them for proposals.
_PROPOSAL_INFLATION = 1.5


def _gaussian_log_density(
    points: NDArray[np.float64],
    mean: NDArray[np.float64],
    precision: NDArray[np.float64],
    log_det: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Evaluate ``log N(points; mean, precision^-1)`` over the last axis."""
    centered = points - mean
    if precision.ndim == 2:
        quadratic = np.einsum("...f,fg,...g->...", centered, precision, centered)
    else:
        quadratic = np.einsum("bkf,bfg,bkg->bk", centered, precision, centered)
    n_factors = points.shape[-1]
    return -0.5 * (quadratic + log_det + n_factors * np.log(2.0 * np.pi))


def _person_log_likelihoods(
    responses: NDArray[np.int_],
    logits: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum Bernoulli log-likelihoods of ``(rows, points, items)`` logits."""
    correct = (responses == 1)[:, None, :]
    observed = (responses >= 0)[:, None, :]
    terms = np.where(correct, logits, 0.0)
    terms -= np.where(observed, np.logaddexp(0.0, logits), 0.0)
    return terms.sum(axis=2)


def affine_marginal_log_likelihood(
    responses: NDArray[np.int_],
    loadings: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    *,
    mu: NDArray[np.float64] | None = None,
    sigma: NDArray[np.float64] | None = None,
    method: str = "auto",
    n_draws: int = _IMPORTANCE_DRAWS,
    seed: int = 0,
) -> tuple[float, str]:
    """Return the marginal log-likelihood of slope-intercept logistic items.

    The value is ``sum_i log integral prod_j P(y_ij | theta) N(theta;
    prior_mean, prior_cov) dtheta`` with ``P(y = 1 | theta) =
    sigmoid(loadings_j' theta + intercepts_j)``; missing responses (``-1``)
    contribute nothing. Up to three factors are integrated on the default
    grid of :class:`~mirt.estimation.em.EMEstimator` (61, 31 or 15
    rectangular nodes per dimension), so the value is comparable with an EM
    log-likelihood. More factors use importance sampling with ``n_draws``
    draws per person from a defensive mixture: with probability 0.9 a draw
    comes from the variational posterior ``N(mu_i, 1.5 * sigma_i)`` and
    otherwise from the prior, which bounds every importance weight. The
    draws come from a fixed ``seed``, so the estimate is reproducible; its
    Monte Carlo error shrinks as ``1 / sqrt(n_draws)``.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Binary responses with ``-1`` for missing values.
    loadings : ndarray of shape (n_items, n_factors)
        Item slopes.
    intercepts : ndarray of shape (n_items,)
        Item intercepts.
    prior_mean, prior_cov : ndarray
        Mean and covariance of the Gaussian latent population.
    mu, sigma : ndarray, optional
        Variational posterior means ``(n_persons, n_factors)`` and
        covariances ``(n_persons, n_factors, n_factors)``, the importance
        proposals. Without them every draw comes from the prior.
    method : {"auto", "quadrature", "importance"}, default="auto"
        Integration rule; ``"auto"`` chooses by the number of factors.
    n_draws : int, default=500
        Importance draws per person.
    seed : int, default=0
        Seed of the importance draws.

    Returns
    -------
    tuple of (float, str)
        The log-likelihood and ``"quadrature"`` or ``"importance"``.
    """
    from mirt.estimation.latent_density import GaussianDensity
    from mirt.estimation.quadrature import build_quadrature, default_quadpts
    from mirt.utils.numeric import logsumexp

    n_persons, n_items = responses.shape
    n_factors = loadings.shape[1]
    if method == "auto":
        method = "quadrature" if n_factors <= _MAX_QUADRATURE_FACTORS else "importance"

    if method == "quadrature":
        grid = build_quadrature(default_quadpts(n_factors), n_factors)
        nodes = grid.nodes
        density = GaussianDensity(
            mean=prior_mean, cov=prior_cov, n_dimensions=n_factors
        )
        log_mass = density.log_quadrature_mass(nodes, grid.weights)
        logits = nodes @ loadings.T + intercepts
        log_correct = -np.logaddexp(0.0, -logits)
        log_incorrect = -np.logaddexp(0.0, logits)
        block_size = max(1, _MAX_WORKING_ELEMENTS // max(1, len(nodes) + n_items))
        total = 0.0
        for start in range(0, n_persons, block_size):
            block = responses[start : start + block_size]
            joint = (block == 1) @ log_correct.T + (block == 0) @ log_incorrect.T
            joint += log_mass
            total += float(np.sum(logsumexp(joint, axis=1)))
        return total, "quadrature"

    rng = np.random.default_rng(seed)
    prior_chol = np.linalg.cholesky(prior_cov)
    prior_precision = np.linalg.solve(prior_cov, np.eye(n_factors))
    prior_log_det = 2.0 * float(np.sum(np.log(np.diag(prior_chol))))
    log_draws = np.log(n_draws)
    block_size = max(
        1, _MAX_WORKING_ELEMENTS // max(1, n_draws * (n_items + 2 * n_factors))
    )
    total = 0.0
    for start in range(0, n_persons, block_size):
        stop = min(start + block_size, n_persons)
        block = responses[start:stop]
        standard = rng.standard_normal((stop - start, n_draws, n_factors))
        from_prior = prior_mean + standard @ prior_chol.T
        log_prior = _gaussian_log_density(
            from_prior, prior_mean, prior_precision, prior_log_det
        )
        if mu is None or sigma is None:
            draws, log_proposal = from_prior, log_prior
        else:
            proposal_cov = _PROPOSAL_INFLATION * sigma[start:stop]
            proposal_chol = np.linalg.cholesky(proposal_cov)
            proposal_precision = np.linalg.inv(proposal_cov)
            proposal_log_det = 2.0 * np.sum(
                np.log(np.diagonal(proposal_chol, axis1=1, axis2=2)), axis=1
            )
            means = mu[start:stop, None, :]
            local = means + np.einsum("bfg,bkg->bkf", proposal_chol, standard)
            defensive = rng.random((stop - start, n_draws)) < _DEFENSIVE_SHARE
            draws = np.where(defensive[..., None], from_prior, local)
            log_prior = _gaussian_log_density(
                draws, prior_mean, prior_precision, prior_log_det
            )
            log_local = _gaussian_log_density(
                draws, means, proposal_precision, proposal_log_det[:, None]
            )
            log_proposal = np.logaddexp(
                np.log1p(-_DEFENSIVE_SHARE) + log_local,
                np.log(_DEFENSIVE_SHARE) + log_prior,
            )
        logits = draws @ loadings.T + intercepts
        log_weights = _person_log_likelihoods(block, logits)
        log_weights += log_prior - log_proposal
        total += float(np.sum(logsumexp(log_weights, axis=1) - log_draws))
    return total, "importance"
