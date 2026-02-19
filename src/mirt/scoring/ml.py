from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult
from mirt.scoring._common import finite_difference_se, score_responses_parallel
from mirt.utils.numeric import compute_hessian_se

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MLScorer:
    def __init__(
        self,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
        n_jobs: int = 1,
    ) -> None:
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

        def score_person(
            i: int,
        ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
            person_responses = responses[i : i + 1, :]
            if n_factors == 1:
                return self._score_unidimensional(model, person_responses)
            return self._score_multidimensional(model, person_responses)

        theta_ml, theta_se = score_responses_parallel(
            model=model,
            responses=responses,
            n_jobs=self.n_jobs,
            score_person=score_person,
        )

        return ScoreResult(
            theta=theta_ml,
            standard_error=theta_se,
            method="ML",
        )

    def _score_unidimensional(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[float, float]:
        def neg_log_likelihood(theta: float) -> float:
            theta_arr = np.array([[theta]])
            ll = model.log_likelihood(responses, theta_arr)[0]
            return -ll

        valid_responses = responses[responses >= 0]
        if len(valid_responses) == 0:
            return 0.0, np.inf

        prop_correct = valid_responses.mean()
        if prop_correct == 0:
            return self.theta_bounds[0], np.inf
        if prop_correct == 1:
            return self.theta_bounds[1], np.inf

        result = minimize_scalar(
            neg_log_likelihood,
            bounds=self.theta_bounds,
            method="bounded",
        )

        theta_est = result.x

        theta_arr = np.array([[theta_est]])
        info = model.information(theta_arr).sum()

        if info > 0:
            se_est = 1.0 / np.sqrt(info)
        else:
            se_est = finite_difference_se(neg_log_likelihood, theta_est)

        return theta_est, se_est

    def _score_multidimensional(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_factors = model.n_factors

        def neg_log_likelihood(theta: NDArray[np.float64]) -> float:
            theta_arr = theta.reshape(1, -1)
            ll = model.log_likelihood(responses, theta_arr)[0]
            return -ll

        valid_responses = responses[responses >= 0]
        if len(valid_responses) == 0:
            return np.zeros(n_factors), np.full(n_factors, np.inf)

        result = minimize(
            neg_log_likelihood,
            x0=np.zeros(n_factors),
            method="L-BFGS-B",
            bounds=[(self.theta_bounds[0], self.theta_bounds[1])] * n_factors,
        )

        theta_est = result.x
        se_est = compute_hessian_se(neg_log_likelihood, theta_est)

        return theta_est, se_est

    def __repr__(self) -> str:
        return f"MLScorer(bounds={self.theta_bounds})"
