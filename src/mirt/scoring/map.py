"""Maximum A Posteriori (MAP) scoring."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

from mirt.results.score_result import ScoreResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MAPScorer:
    """Maximum A Posteriori (MAP) ability estimation.

    MAP scoring finds the mode of the posterior distribution:

    θ_MAP = argmax P(θ|X) = argmax L(X|θ) × π(θ)

    where L is the likelihood and π is the prior.

    Standard errors are computed from the curvature of the posterior
    at the mode (inverse of observed information).

    Parameters
    ----------
    prior_mean : ndarray, optional
        Mean of the prior distribution. Default: zeros.
    prior_cov : ndarray, optional
        Covariance of the prior. Default: identity.
    theta_bounds : tuple, optional
        Bounds for theta search. Default: (-6, 6).

    Examples
    --------
    >>> scorer = MAPScorer()
    >>> result = scorer.score(model, responses)
    >>> print(result.theta)
    """

    def __init__(
        self,
        prior_mean: Optional[NDArray[np.float64]] = None,
        prior_cov: Optional[NDArray[np.float64]] = None,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.theta_bounds = theta_bounds

    def score(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Compute MAP scores for all persons.

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

        # Precompute prior precision and log-det
        prior_prec = np.linalg.inv(prior_cov)
        sign, log_det = np.linalg.slogdet(prior_cov)

        # Initialize output
        theta_map = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        # Score each person
        for i in range(n_persons):
            person_responses = responses[i : i + 1, :]

            if n_factors == 1:
                # Unidimensional: use scalar optimization
                theta_est, se_est = self._score_unidimensional(
                    model, person_responses, prior_mean[0], prior_cov[0, 0]
                )
                theta_map[i, 0] = theta_est
                theta_se[i, 0] = se_est
            else:
                # Multidimensional: use multivariate optimization
                theta_est, se_est = self._score_multidimensional(
                    model, person_responses, prior_mean, prior_prec
                )
                theta_map[i] = theta_est
                theta_se[i] = se_est

        # Flatten if unidimensional
        if n_factors == 1:
            theta_map = theta_map.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_map,
            standard_error=theta_se,
            method="MAP",
        )

    def _score_unidimensional(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        prior_mean: float,
        prior_var: float,
    ) -> tuple[float, float]:
        """Score a single person with unidimensional model."""

        def neg_log_posterior(theta: float) -> float:
            theta_arr = np.array([[theta]])
            ll = model.log_likelihood(responses, theta_arr)[0]
            log_prior = -0.5 * ((theta - prior_mean) ** 2) / prior_var
            return -(ll + log_prior)

        # Optimize
        result = minimize_scalar(
            neg_log_posterior,
            bounds=self.theta_bounds,
            method='bounded',
        )

        theta_est = result.x

        # Compute SE from numerical second derivative
        h = 1e-5
        f_plus = neg_log_posterior(theta_est + h)
        f_minus = neg_log_posterior(theta_est - h)
        f_center = neg_log_posterior(theta_est)

        hessian = (f_plus - 2 * f_center + f_minus) / (h ** 2)

        if hessian > 0:
            se_est = np.sqrt(1.0 / hessian)
        else:
            se_est = np.nan

        return theta_est, se_est

    def _score_multidimensional(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_prec: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Score a single person with multidimensional model."""
        n_factors = len(prior_mean)

        def neg_log_posterior(theta: NDArray[np.float64]) -> float:
            theta_arr = theta.reshape(1, -1)
            ll = model.log_likelihood(responses, theta_arr)[0]
            diff = theta - prior_mean
            log_prior = -0.5 * np.dot(diff, np.dot(prior_prec, diff))
            return -(ll + log_prior)

        # Optimize
        result = minimize(
            neg_log_posterior,
            x0=prior_mean,
            method='L-BFGS-B',
            bounds=[(self.theta_bounds[0], self.theta_bounds[1])] * n_factors,
        )

        theta_est = result.x

        # Compute SE from numerical Hessian diagonal
        h = 1e-5
        se_est = np.zeros(n_factors)

        for j in range(n_factors):
            theta_plus = theta_est.copy()
            theta_plus[j] += h
            theta_minus = theta_est.copy()
            theta_minus[j] -= h

            f_plus = neg_log_posterior(theta_plus)
            f_minus = neg_log_posterior(theta_minus)
            f_center = neg_log_posterior(theta_est)

            hessian_jj = (f_plus - 2 * f_center + f_minus) / (h ** 2)

            if hessian_jj > 0:
                se_est[j] = np.sqrt(1.0 / hessian_jj)
            else:
                se_est[j] = np.nan

        return theta_est, se_est

    def __repr__(self) -> str:
        return f"MAPScorer(bounds={self.theta_bounds})"
