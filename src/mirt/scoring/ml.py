from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult
from mirt.utils.numeric import compute_hessian_se

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MLScorer:
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
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        theta_ml = np.zeros((n_persons, n_factors))
        theta_se = np.zeros((n_persons, n_factors))

        for i in range(n_persons):
            person_responses = responses[i : i + 1, :]

            if n_factors == 1:
                theta_est, se_est = self._score_unidimensional(model, person_responses)
                theta_ml[i, 0] = theta_est
                theta_se[i, 0] = se_est
            else:
                theta_est, se_est = self._score_multidimensional(
                    model, person_responses
                )
                theta_ml[i] = theta_est
                theta_se[i] = se_est

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
            h = 1e-5
            f_plus = neg_log_likelihood(theta_est + h)
            f_minus = neg_log_likelihood(theta_est - h)
            f_center = neg_log_likelihood(theta_est)
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
