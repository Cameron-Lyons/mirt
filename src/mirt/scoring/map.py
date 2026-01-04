from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MAPScorer:
    def __init__(
        self,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
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

        prior_prec = np.linalg.inv(prior_cov)
        sign, log_det = np.linalg.slogdet(prior_cov)

        theta_map = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        for i in range(n_persons):
            person_responses = responses[i : i + 1, :]

            if n_factors == 1:
                theta_est, se_est = self._score_unidimensional(
                    model, person_responses, prior_mean[0], prior_cov[0, 0]
                )
                theta_map[i, 0] = theta_est
                theta_se[i, 0] = se_est
            else:
                theta_est, se_est = self._score_multidimensional(
                    model, person_responses, prior_mean, prior_prec
                )
                theta_map[i] = theta_est
                theta_se[i] = se_est

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
        def neg_log_posterior(theta: float) -> float:
            theta_arr = np.array([[theta]])
            ll = model.log_likelihood(responses, theta_arr)[0]
            log_prior = -0.5 * ((theta - prior_mean) ** 2) / prior_var
            return -(ll + log_prior)

        result = minimize_scalar(
            neg_log_posterior,
            bounds=self.theta_bounds,
            method="bounded",
        )

        theta_est = result.x

        h = 1e-5
        f_plus = neg_log_posterior(theta_est + h)
        f_minus = neg_log_posterior(theta_est - h)
        f_center = neg_log_posterior(theta_est)

        hessian = (f_plus - 2 * f_center + f_minus) / (h**2)

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
        n_factors = len(prior_mean)

        def neg_log_posterior(theta: NDArray[np.float64]) -> float:
            theta_arr = theta.reshape(1, -1)
            ll = model.log_likelihood(responses, theta_arr)[0]
            diff = theta - prior_mean
            log_prior = -0.5 * np.dot(diff, np.dot(prior_prec, diff))
            return -(ll + log_prior)

        result = minimize(
            neg_log_posterior,
            x0=prior_mean,
            method="L-BFGS-B",
            bounds=[(self.theta_bounds[0], self.theta_bounds[1])] * n_factors,
        )

        theta_est = result.x

        h = 1e-5
        se_est = np.zeros(n_factors)
        f_center = neg_log_posterior(theta_est)

        for j in range(n_factors):
            theta_plus = theta_est.copy()
            theta_plus[j] += h
            theta_minus = theta_est.copy()
            theta_minus[j] -= h

            f_plus = neg_log_posterior(theta_plus)
            f_minus = neg_log_posterior(theta_minus)

            hessian_jj = (f_plus - 2 * f_center + f_minus) / (h**2)

            if hessian_jj > 0:
                se_est[j] = np.sqrt(1.0 / hessian_jj)
            else:
                se_est[j] = np.nan

        return theta_est, se_est

    def __repr__(self) -> str:
        return f"MAPScorer(bounds={self.theta_bounds})"
