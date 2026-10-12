from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    extreme_response_rows,
    finite_difference_se,
    finite_difference_se_rows,
    measured_factors,
    observed_test_information,
    score_pattern_chunks,
    score_responses_parallel,
    supports_row_batched_scoring,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.scoring._optimization import (
    RowObjective,
    batched_newton_minimize,
    bounded_scalar_minimize,
    finite_difference_derivatives,
    free_block_standard_errors,
    validate_theta_bounds,
)
from mirt.utils._parallel import resolve_n_jobs

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _bounded_ml_standard_errors(
    objective: RowObjective,
    theta: NDArray[np.float64],
    measured: NDArray[np.bool_],
    lower: float,
    upper: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ML estimates and SEs with the extreme-coordinate convention.

    ``objective(rows, x)`` is each row's negative log-likelihood. A
    coordinate that no observed item measures is reported as zero, and one on
    a bound whose likelihood still increases beyond it is snapped to that
    bound; both get infinite standard errors. The remaining coordinates take
    standard errors from the inverse of their finite-difference Hessian block.
    """
    theta = np.where(measured, theta, 0.0)
    span = 1e-7 * max(1.0, abs(lower), abs(upper))
    theta = np.where(theta <= lower + span, lower, theta)
    theta = np.where(theta >= upper - span, upper, theta)
    standard_error = np.full(theta.shape, np.inf, dtype=np.float64)
    if theta.shape[0] == 0:
        return theta, standard_error
    _, gradient, hessian, finite = finite_difference_derivatives(
        objective, np.arange(theta.shape[0], dtype=np.intp), theta, h=1e-5
    )
    unbounded = ((theta <= lower) & (gradient > 0.0)) | (
        (theta >= upper) & (gradient < 0.0)
    )
    free = measured & ~unbounded
    hessian[~finite] = np.nan
    block_se = free_block_standard_errors(hessian, free)
    standard_error[free] = block_se[free]
    return theta, standard_error


class MLScorer:
    """Maximum likelihood ability scoring.

    Built-in unidimensional models are scored for every distinct response
    pattern with one row-batched bounded Brent search, and built-in
    multidimensional models (whose log-concave likelihoods have a unique
    maximum) with one row-batched projected Newton search whose unconverged
    rows fall back to L-BFGS-B. Other models, subclasses and instances that
    override the per-pattern hooks keep the per-pattern optimizers.

    Parameters
    ----------
    theta_bounds : tuple of float, default=(-6.0, 6.0)
        Lower and upper bounds for every theta coordinate.
    n_jobs : int, default=1
        Number of response patterns to optimize in parallel on the
        per-pattern path. ``-1`` uses all available CPU cores.

    Notes
    -----
    A response pattern whose observed responses are all in the lowest
    category, or all in each item's highest category, has no finite maximum
    likelihood estimate. Every model family reports it at the lower or upper
    bound in every coordinate with an infinite standard error (R's mirt
    reports ``-Inf``/``Inf`` with a missing standard error). In
    multidimensional models, a coordinate whose likelihood still increases at
    a bound, such as a factor whose items are all answered in their lowest or
    highest category in a simple-structure model, is likewise reported at
    that bound with an infinite standard error, and the other coordinates take
    standard errors conditional on it. A pattern without observed items, and
    a factor that no observed item measures, receive zero with an infinite
    standard error.
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

        per_pattern_hook = (
            "_score_unidimensional" if n_factors == 1 else "_score_multidimensional"
        )
        if (
            type(self) is MLScorer
            and per_pattern_hook not in vars(self)
            and supports_row_batched_scoring(model)
            # Newton search is local; built-in multidimensional likelihoods
            # are log-concave, while noncompensatory and custom curves can
            # have several maxima.
            and (n_factors == 1 or uses_builtin_model_hooks(model, likelihood=True))
        ):

            def score_chunk(
                chunk: NDArray[np.int_],
            ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
                if n_factors == 1:
                    return self._score_unidimensional_batch(model, chunk)
                return self._score_multidimensional_batch(model, chunk)

            theta_ml, theta_se = score_pattern_chunks(patterns, score_chunk)
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

        all_lowest, all_highest = extreme_response_rows(model, responses)
        if all_lowest[0]:
            return self.theta_bounds[0], np.inf
        if all_highest[0]:
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
        all_lowest, all_highest = extreme_response_rows(model, patterns)
        theta[all_lowest] = lower
        theta[all_highest] = upper
        optimize = (n_observed > 0) & ~(all_lowest | all_highest)

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
        """Score one ``(1, n_items)`` pattern with L-BFGS-B from the origin."""
        n_factors = model.n_factors
        lower, upper = self.theta_bounds

        def neg_log_likelihood(theta: NDArray[np.float64]) -> float:
            theta_arr = theta.reshape(1, -1)
            ll = model.log_likelihood(responses, theta_arr)[0]
            return -ll

        observed = responses >= 0
        if not observed.any():
            return np.zeros(n_factors), np.full(n_factors, np.inf)
        all_lowest, all_highest = extreme_response_rows(model, responses)
        if all_lowest[0] or all_highest[0]:
            bound = lower if all_lowest[0] else upper
            return np.full(n_factors, bound), np.full(n_factors, np.inf)

        result = minimize(
            neg_log_likelihood,
            x0=np.zeros(n_factors),
            method="L-BFGS-B",
            bounds=[(lower, upper)] * n_factors,
        )

        def objective(
            rows: NDArray[np.intp], values: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return np.array([neg_log_likelihood(value) for value in values])

        theta, standard_error = _bounded_ml_standard_errors(
            objective,
            np.asarray(result.x, dtype=np.float64)[None, :],
            measured_factors(model, observed),
            lower,
            upper,
        )
        return theta[0], standard_error[0]

    def _score_multidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find every pattern's likelihood maximum with a shared Newton search.

        Rows for which the projected Newton search does not converge are
        re-solved by :meth:`_score_multidimensional`.
        """
        lower, upper = self.theta_bounds
        n_patterns, n_factors = patterns.shape[0], model.n_factors
        observed = patterns >= 0
        theta = np.zeros((n_patterns, n_factors), dtype=np.float64)
        standard_error = np.full((n_patterns, n_factors), np.inf, dtype=np.float64)
        all_lowest, all_highest = extreme_response_rows(model, patterns)
        theta[all_lowest] = lower
        theta[all_highest] = upper
        rows = np.flatnonzero(observed.any(axis=1) & ~(all_lowest | all_highest))
        if rows.size == 0:
            return theta, standard_error

        def neg_log_likelihood(
            subset: NDArray[np.intp],
            values: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return -model.log_likelihood(patterns[rows[subset]], values)

        estimate, _, converged = batched_newton_minimize(
            neg_log_likelihood, np.zeros((rows.size, n_factors)), lower, upper
        )
        solved = np.flatnonzero(converged)
        theta[rows[solved]], standard_error[rows[solved]] = _bounded_ml_standard_errors(
            lambda subset, values: neg_log_likelihood(solved[subset], values),
            estimate[solved],
            measured_factors(model, observed[rows[solved]]),
            lower,
            upper,
        )
        for row in rows[~converged]:
            theta[row], standard_error[row] = self._score_multidimensional(
                model, patterns[row : row + 1]
            )
        return theta, standard_error

    def __repr__(self) -> str:
        return f"MLScorer(bounds={self.theta_bounds})"
