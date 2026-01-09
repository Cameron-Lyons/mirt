from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.results.score_result import ScoreResult
from mirt.utils.numeric import logsumexp_axis1

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class EAPScorer:
    def __init__(
        self,
        n_quadpts: int = 49,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> None:
        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        prior_mean = self.prior_mean
        prior_cov = self.prior_cov

        if prior_mean is None:
            prior_mean = np.zeros(n_factors)
        if prior_cov is None:
            prior_cov = np.eye(n_factors)

        quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=n_factors,
            mean=prior_mean,
            cov=prior_cov,
        )

        quad_points = quadrature.nodes
        quad_weights = quadrature.weights
        n_quad = len(quad_weights)

        log_likes = np.zeros((n_persons, n_quad))
        for q in range(n_quad):
            # Use broadcasting instead of np.tile for efficiency
            theta_q = quad_points[q : q + 1]  # Shape (1, n_factors), broadcasts
            log_likes[:, q] = model.log_likelihood(responses, theta_q)

        log_posterior = log_likes + np.log(quad_weights + 1e-300)[None, :]

        log_norm = logsumexp_axis1(log_posterior)
        posterior = np.exp(log_posterior - log_norm[:, None])

        theta_eap = np.dot(posterior, quad_points)

        deviation = quad_points[None, :, :] - theta_eap[:, None, :]
        variance = np.sum(posterior[:, :, None] * (deviation**2), axis=1)
        theta_se = np.sqrt(variance)

        if n_factors == 1:
            theta_eap = theta_eap.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAP",
        )

    def __repr__(self) -> str:
        return f"EAPScorer(n_quadpts={self.n_quadpts})"
