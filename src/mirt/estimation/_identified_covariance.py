"""Covariance draws for Gibbs samplers identified by a unit leading variance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def sample_unit_leading_covariance(
    centered: NDArray[np.float64],
    prior_df: float,
    prior_scale: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw a covariance matrix whose leading variance is fixed at one.

    Parameters
    ----------
    centered : NDArray
        Rows are independent ``N(0, Sigma)`` draws, shape ``(n, p)``.
    prior_df : float
        Degrees of freedom of the inverse-Wishart prior on ``Sigma``.
    prior_scale : NDArray
        Scale matrix of the inverse-Wishart prior, shape ``(p, p)``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    NDArray
        Covariance matrix with ``Sigma[0, 0] == 1``.

    Notes
    -----
    The prior is ``IW(prior_df, prior_scale)`` conditioned on
    ``Sigma[0, 0] = 1``. Partition ``Sigma`` by its first coordinate and write
    ``gamma = Sigma[0, 1:] / Sigma[0, 0]`` and
    ``Omega = Sigma[1:, 1:] - Sigma[1:, 0] gamma``. Under the inverse-Wishart,
    ``Sigma[0, 0]`` is independent of ``(gamma, Omega)``, with
    ``Omega ~ IW(prior_df, Psi[1:, 1:] - Psi[1:, 0] Psi[0, 1:] / Psi[0, 0])``
    and ``gamma | Omega ~ N(Psi[0, 1:] / Psi[0, 0], Omega / Psi[0, 0])``. So
    the conditioned prior is that same pair, and given the draws the full
    conditional is the conjugate regression of the remaining coordinates on
    the leading one.
    """
    values = np.asarray(centered, dtype=np.float64)
    n_draws, dimension = values.shape
    leading = values[:, 0]
    remaining = values[:, 1:]

    prior_weight = prior_scale[0, 0]
    prior_slope = prior_scale[0, 1:] / prior_weight
    posterior_weight = prior_weight + leading @ leading
    posterior_slope = (prior_weight * prior_slope + leading @ remaining) / (
        posterior_weight
    )
    posterior_scale = (
        prior_scale[1:, 1:]
        + remaining.T @ remaining
        - posterior_weight * np.outer(posterior_slope, posterior_slope)
    )
    posterior_scale = 0.5 * (posterior_scale + posterior_scale.T)

    residual_cov = np.atleast_2d(
        stats.invwishart.rvs(
            df=prior_df + n_draws,
            scale=posterior_scale,
            random_state=rng,
        )
    )
    slope = posterior_slope + np.linalg.cholesky(
        residual_cov / posterior_weight
    ) @ rng.standard_normal(dimension - 1)

    sigma = np.empty((dimension, dimension), dtype=np.float64)
    sigma[0, 0] = 1.0
    sigma[0, 1:] = slope
    sigma[1:, 0] = slope
    sigma[1:, 1:] = residual_cov + np.outer(slope, slope)
    return sigma
