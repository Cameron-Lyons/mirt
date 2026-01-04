"""Expected A Posteriori (EAP) scoring."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.results.score_result import ScoreResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class EAPScorer:
    """Expected A Posteriori (EAP) ability estimation.

    EAP scoring computes the posterior mean of theta given the response pattern:

    θ_EAP = E[θ|X] = ∫ θ × L(X|θ) × π(θ) dθ / ∫ L(X|θ) × π(θ) dθ

    where L is the likelihood and π is the prior.

    The posterior standard deviation (PSD) is returned as the standard error:

    PSD = sqrt(E[(θ - θ_EAP)²|X])

    Parameters
    ----------
    n_quadpts : int, default=49
        Number of quadrature points per dimension.
    prior_mean : ndarray, optional
        Mean of the prior distribution. Default: zeros.
    prior_cov : ndarray, optional
        Covariance of the prior. Default: identity.

    Attributes
    ----------
    n_quadpts : int
        Number of quadrature points.

    Examples
    --------
    >>> scorer = EAPScorer(n_quadpts=49)
    >>> result = scorer.score(model, responses)
    >>> print(result.theta)
    """

    def __init__(
        self,
        n_quadpts: int = 49,
        prior_mean: Optional[NDArray[np.float64]] = None,
        prior_cov: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

    def score(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Compute EAP scores for all persons.

        Parameters
        ----------
        model : BaseItemModel
            Fitted IRT model.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix.

        Returns
        -------
        ScoreResult
            Theta estimates with standard errors.
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        # Set up prior
        prior_mean = self.prior_mean
        prior_cov = self.prior_cov

        if prior_mean is None:
            prior_mean = np.zeros(n_factors)
        if prior_cov is None:
            prior_cov = np.eye(n_factors)

        # Set up quadrature
        quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=n_factors,
            mean=prior_mean,
            cov=prior_cov,
        )

        quad_points = quadrature.nodes  # (n_quad, n_factors)
        quad_weights = quadrature.weights  # (n_quad,)
        n_quad = len(quad_weights)

        # Initialize output arrays
        theta_eap = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        # Score each person
        for i in range(n_persons):
            person_responses = responses[i : i + 1, :]

            # Compute log-likelihood at each quadrature point
            log_likes = np.zeros(n_quad)
            for q in range(n_quad):
                theta_q = quad_points[q].reshape(1, -1)
                log_likes[q] = model.log_likelihood(person_responses, theta_q)[0]

            # Log posterior (prior already incorporated in quadrature weights)
            log_posterior = log_likes + np.log(quad_weights + 1e-300)

            # Normalize using log-sum-exp
            log_norm = self._logsumexp(log_posterior)
            posterior = np.exp(log_posterior - log_norm)

            # EAP = weighted mean of quadrature points
            theta_eap[i] = np.sum(posterior[:, None] * quad_points, axis=0)

            # PSD = posterior standard deviation
            deviation = quad_points - theta_eap[i]
            variance = np.sum(posterior[:, None] * (deviation ** 2), axis=0)
            theta_se[i] = np.sqrt(variance)

        # Flatten if unidimensional
        if n_factors == 1:
            theta_eap = theta_eap.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAP",
        )

    @staticmethod
    def _logsumexp(a: NDArray[np.float64]) -> float:
        """Compute log(sum(exp(a))) in a numerically stable way."""
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    def __repr__(self) -> str:
        return f"EAPScorer(n_quadpts={self.n_quadpts})"
