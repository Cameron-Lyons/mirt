"""Maximum Likelihood (ML) scoring."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

from mirt.results.score_result import ScoreResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MLScorer:
    """Maximum Likelihood (ML) ability estimation.

    ML scoring finds the theta that maximizes the likelihood:

    θ_ML = argmax L(X|θ)

    No prior distribution is used. This can lead to undefined estimates
    for extreme response patterns (all correct or all incorrect).

    Standard errors are computed from the observed information
    (negative Hessian of log-likelihood at the MLE).

    Parameters
    ----------
    theta_bounds : tuple, optional
        Bounds for theta search. Default: (-6, 6).

    Notes
    -----
    ML estimates may be undefined or at the boundary for extreme
    response patterns. Consider using MAP or EAP for more stable estimates.

    Examples
    --------
    >>> scorer = MLScorer()
    >>> result = scorer.score(model, responses)
    >>> print(result.theta)
    """

    def __init__(
        self,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        self.theta_bounds = theta_bounds

    def score(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Compute ML scores for all persons.

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

        # Initialize output
        theta_ml = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        # Score each person
        for i in range(n_persons):
            person_responses = responses[i : i + 1, :]

            if n_factors == 1:
                theta_est, se_est = self._score_unidimensional(model, person_responses)
                theta_ml[i, 0] = theta_est
                theta_se[i, 0] = se_est
            else:
                theta_est, se_est = self._score_multidimensional(model, person_responses)
                theta_ml[i] = theta_est
                theta_se[i] = se_est

        # Flatten if unidimensional
        if n_factors == 1:
            theta_ml = theta_ml.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_ml,
            standard_error=theta_se,
            method="ML",
        )

    def _score_unidimensional(
        self,
        model: "BaseItemModel",
        responses: NDArray[np.int_],
    ) -> tuple[float, float]:
        """Score a single person with unidimensional model."""

        def neg_log_likelihood(theta: float) -> float:
            theta_arr = np.array([[theta]])
            ll = model.log_likelihood(responses, theta_arr)[0]
            return -ll

        # Check for extreme patterns
        valid_responses = responses[responses >= 0]
        if len(valid_responses) == 0:
            return 0.0, np.inf

        prop_correct = valid_responses.mean()
        if prop_correct == 0:
            return self.theta_bounds[0], np.inf
        if prop_correct == 1:
            return self.theta_bounds[1], np.inf

        # Optimize
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=self.theta_bounds,
            method='bounded',
        )

        theta_est = result.x

        # Compute SE from information
        theta_arr = np.array([[theta_est]])
        info = model.information(theta_arr).sum()

        if info > 0:
            se_est = 1.0 / np.sqrt(info)
        else:
            # Fall back to numerical Hessian
            h = 1e-5
            f_plus = neg_log_likelihood(theta_est + h)
            f_minus = neg_log_likelihood(theta_est - h)
            f_center = neg_log_likelihood(theta_est)
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
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Score a single person with multidimensional model."""
        n_factors = model.n_factors

        def neg_log_likelihood(theta: NDArray[np.float64]) -> float:
            theta_arr = theta.reshape(1, -1)
            ll = model.log_likelihood(responses, theta_arr)[0]
            return -ll

        # Check for extreme patterns
        valid_responses = responses[responses >= 0]
        if len(valid_responses) == 0:
            return np.zeros(n_factors), np.full(n_factors, np.inf)

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=np.zeros(n_factors),
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

            f_plus = neg_log_likelihood(theta_plus)
            f_minus = neg_log_likelihood(theta_minus)
            f_center = neg_log_likelihood(theta_est)

            hessian_jj = (f_plus - 2 * f_center + f_minus) / (h ** 2)

            if hessian_jj > 0:
                se_est[j] = np.sqrt(1.0 / hessian_jj)
            else:
                se_est[j] = np.nan

        return theta_est, se_est

    def __repr__(self) -> str:
        return f"MLScorer(bounds={self.theta_bounds})"
