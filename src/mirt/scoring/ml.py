from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    finite_difference_se,
    finite_difference_se_rows,
    observed_test_information,
    resolve_n_jobs,
    score_pattern_chunks,
    score_responses_parallel,
    supports_row_batched_scoring,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.scoring._optimization import bounded_scalar_minimize, validate_theta_bounds
from mirt.utils.numeric import compute_hessian_se

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MLScorer:
    """Maximum likelihood ability scoring.

    Built-in unidimensional models are scored for every distinct response
    pattern with one row-batched bounded Brent search. Subclasses and
    instances that override the per-pattern hook keep the per-pattern path.

    Parameters
    ----------
    theta_bounds : tuple of float, default=(-6.0, 6.0)
        Lower and upper bounds for every theta coordinate.
    n_jobs : int, default=1
        Number of response patterns to optimize in parallel on the
        per-pattern path. ``-1`` uses all available CPU cores.
    """

    def __init__(
        self,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
        n_jobs: int = 1,
    ) -> None:
        self.theta_bounds = validate_theta_bounds(theta_bounds)
        resolve_n_jobs(n_jobs)
        self.n_jobs = int(n_jobs)

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = validate_scoring_responses(model, responses)
        patterns, inverse = unique_response_patterns(responses)
        n_factors = model.n_factors

        from mirt.backends.rust.optimization_scoring import try_optimized_scores

        native = None
        if type(self) is MLScorer and "_score_unidimensional" not in vars(self):
            native = try_optimized_scores(
                model,
                patterns,
                bounds=self.theta_bounds,
                method="ML",
                n_jobs=self.n_jobs,
            )
        if native is not None:
            return ScoreResult(
                theta=native[0][inverse], standard_error=native[1][inverse], method="ML"
            )

        if (
            n_factors == 1
            and type(self) is MLScorer
            and "_score_unidimensional" not in vars(self)
            and supports_row_batched_scoring(model)
        ):
            theta_ml, theta_se = score_pattern_chunks(
                patterns,
                lambda chunk: self._score_unidimensional_batch(model, chunk),
            )
            return ScoreResult(
                theta=theta_ml[inverse], standard_error=theta_se[inverse], method="ML"
            )

        def score_person(
            i: int,
        ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
            person_responses = patterns[i : i + 1, :]
            if n_factors == 1:
                return self._score_unidimensional(model, person_responses)
            return self._score_multidimensional(model, person_responses)

        theta_ml, theta_se = score_responses_parallel(
            model=model,
            responses=patterns,
            n_jobs=self.n_jobs,
            score_person=score_person,
        )
        theta_ml = theta_ml[inverse]
        theta_se = theta_se[inverse]

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

        if not model.is_polytomous:
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
        info = observed_test_information(model, theta_arr, responses[0] >= 0)[0]

        if info > 0:
            se_est = 1.0 / np.sqrt(info)
        else:
            se_est = finite_difference_se(neg_log_likelihood, theta_est)

        return theta_est, se_est

    def _score_unidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Score every pattern as :meth:`_score_unidimensional` would.

        All patterns share one bounded Brent search, so each optimizer step
        needs a single stacked log-likelihood evaluation.
        """
        lower, upper = self.theta_bounds
        observed = patterns >= 0
        n_observed = observed.sum(axis=1)
        theta = np.zeros(patterns.shape[0], dtype=np.float64)
        standard_error = np.full(patterns.shape[0], np.inf, dtype=np.float64)
        optimize = n_observed > 0
        if not model.is_polytomous:
            n_correct = np.where(observed, patterns, 0).sum(axis=1)
            all_incorrect = optimize & (n_correct == 0)
            all_correct = optimize & (n_correct == n_observed)
            theta[all_incorrect] = lower
            theta[all_correct] = upper
            optimize &= ~(all_incorrect | all_correct)

        rows = np.flatnonzero(optimize)
        if rows.size == 0:
            return theta, standard_error

        def neg_log_likelihood(
            subset: NDArray[np.intp],
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return -model.log_likelihood(patterns[rows[subset]], values[:, None])

        estimate, _ = bounded_scalar_minimize(
            neg_log_likelihood, rows.size, lower, upper
        )
        info = observed_test_information(model, estimate[:, None], observed[rows])
        estimate_se = np.empty_like(estimate)
        positive = info > 0
        estimate_se[positive] = 1.0 / np.sqrt(info[positive])
        if not np.all(positive):
            fallback = np.flatnonzero(~positive)
            estimate_se[fallback] = finite_difference_se_rows(
                lambda values: neg_log_likelihood(fallback, values),
                estimate[fallback],
            )
        theta[rows] = estimate
        standard_error[rows] = estimate_se
        return theta, standard_error

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
