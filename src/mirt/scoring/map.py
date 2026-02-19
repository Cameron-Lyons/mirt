from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    finite_difference_se,
    resolve_prior_distribution,
    score_responses_parallel,
)
from mirt.utils.numeric import compute_hessian_se

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MAPScorer:
    def __init__(
        self,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
        n_jobs: int = 1,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.theta_bounds = theta_bounds
        self.n_jobs = n_jobs

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_factors = model.n_factors

        prior_mean, prior_cov = resolve_prior_distribution(
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )

        prior_prec = np.linalg.inv(prior_cov)

        def score_person(
            i: int,
        ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
            person_responses = responses[i : i + 1, :]
            if n_factors == 1:
                return self._score_unidimensional(
                    model, person_responses, prior_mean[0], prior_cov[0, 0]
                )
            return self._score_multidimensional(
                model, person_responses, prior_mean, prior_prec
            )

        theta_map, theta_se = score_responses_parallel(
            model=model,
            responses=responses,
            n_jobs=self.n_jobs,
            score_person=score_person,
        )

        return ScoreResult(
            theta=theta_map,
            standard_error=theta_se,
            method="MAP",
        )

    def _score_unidimensional(
        self,
        model: BaseItemModel,
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
        se_est = finite_difference_se(neg_log_posterior, theta_est)

        return theta_est, se_est

    def _score_multidimensional(
        self,
        model: BaseItemModel,
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
        se_est = compute_hessian_se(neg_log_posterior, theta_est)

        return theta_est, se_est

    def __repr__(self) -> str:
        return f"MAPScorer(bounds={self.theta_bounds})"
