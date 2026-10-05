from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    finite_difference_se,
    finite_difference_se_rows,
    resolve_n_jobs,
    resolve_prior_distribution,
    score_pattern_chunks,
    score_responses_parallel,
    supports_row_batched_scoring,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.scoring._optimization import (
    batched_hessian_se,
    batched_newton_minimize,
    bounded_scalar_minimize,
    validate_theta_bounds,
)
from mirt.utils.numeric import compute_hessian_se

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class MAPScorer:
    """Maximum a posteriori ability scoring under a normal prior.

    Distinct response patterns are scored together when the model's
    likelihood treats theta rows independently: unidimensional models with a
    row-batched bounded Brent search, and the built-in multidimensional models
    (whose log-concave likelihoods give a unique posterior mode) with a
    row-batched projected Newton search whose unconverged rows fall back to
    L-BFGS-B. Other models, scorer subclasses and instances that override the
    per-pattern hooks keep the per-pattern optimizers.

    Parameters
    ----------
    prior_mean : ndarray, optional
        Prior mean for theta. Default zeros.
    prior_cov : ndarray, optional
        Prior covariance for theta. Default identity.
    theta_bounds : tuple of float, default=(-6.0, 6.0)
        Lower and upper bounds for every theta coordinate.
    n_jobs : int, default=1
        Number of response patterns to optimize in parallel on the
        per-pattern path. ``-1`` uses all available CPU cores.
    """

    def __init__(
        self,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        theta_bounds: tuple[float, float] = (-6.0, 6.0),
        n_jobs: int = 1,
    ) -> None:
        self.prior_mean = (
            None
            if prior_mean is None
            else np.array(prior_mean, dtype=np.float64, copy=True)
        )
        self.prior_cov = (
            None
            if prior_cov is None
            else np.array(prior_cov, dtype=np.float64, copy=True)
        )
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

        prior_mean, prior_cov = resolve_prior_distribution(
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )

        prior_prec = np.linalg.inv(prior_cov)

        from mirt.backends.rust.optimization_scoring import try_optimized_scores

        native = None
        if type(self) is MAPScorer and "_score_unidimensional" not in vars(self):
            native = try_optimized_scores(
                model,
                patterns,
                bounds=self.theta_bounds,
                method="MAP",
                n_jobs=self.n_jobs,
                prior_mean=prior_mean[0],
                prior_var=prior_cov[0, 0],
            )
        if native is not None:
            return ScoreResult(
                theta=native[0][inverse],
                standard_error=native[1][inverse],
                method="MAP",
            )

        per_pattern_hook = (
            "_score_unidimensional" if n_factors == 1 else "_score_multidimensional"
        )
        if (
            type(self) is MAPScorer
            and per_pattern_hook not in vars(self)
            and supports_row_batched_scoring(model)
            # Newton search is local. Built-in multidimensional models have
            # log-concave likelihoods, so their posterior mode is unique;
            # noncompensatory and custom curves can be multimodal.
            and (n_factors == 1 or uses_builtin_model_hooks(model, likelihood=True))
        ):

            def score_chunk(
                chunk: NDArray[np.int_],
            ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
                if n_factors == 1:
                    return self._score_unidimensional_batch(
                        model, chunk, prior_mean[0], prior_cov[0, 0]
                    )
                return self._score_multidimensional_batch(
                    model, chunk, prior_mean, prior_prec
                )

            theta_map, theta_se = score_pattern_chunks(patterns, score_chunk)
            return ScoreResult(
                theta=theta_map[inverse],
                standard_error=theta_se[inverse],
                method="MAP",
            )

        def score_person(
            i: int,
        ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
            person_responses = patterns[i : i + 1, :]
            if n_factors == 1:
                return self._score_unidimensional(
                    model, person_responses, prior_mean[0], prior_cov[0, 0]
                )
            return self._score_multidimensional(
                model, person_responses, prior_mean, prior_prec
            )

        theta_map, theta_se = score_responses_parallel(
            model=model,
            responses=patterns,
            n_jobs=self.n_jobs,
            score_person=score_person,
        )
        theta_map = theta_map[inverse]
        theta_se = theta_se[inverse]

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

        def objective_with_cached_optimum(theta: float) -> float:
            if theta == theta_est:
                return float(result.fun)
            return neg_log_posterior(theta)

        se_est = finite_difference_se(objective_with_cached_optimum, theta_est)

        return theta_est, se_est

    def _score_unidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
        prior_mean: float,
        prior_var: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Score every pattern as :meth:`_score_unidimensional` would.

        All patterns share one bounded Brent search, so each optimizer step
        needs a single stacked log-likelihood evaluation.
        """

        def neg_log_posterior(
            rows: NDArray[np.intp],
            theta: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            ll = model.log_likelihood(patterns[rows], theta[:, None])
            log_prior = -0.5 * ((theta - prior_mean) ** 2) / prior_var
            return -(ll + log_prior)

        theta_est, f_optimum = bounded_scalar_minimize(
            neg_log_posterior, patterns.shape[0], *self.theta_bounds
        )
        all_rows = np.arange(patterns.shape[0], dtype=np.intp)
        se_est = finite_difference_se_rows(
            lambda theta: neg_log_posterior(all_rows, theta),
            theta_est,
            center=f_optimum,
        )
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

    def _score_multidimensional_batch(
        self,
        model: BaseItemModel,
        patterns: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_prec: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find every pattern's posterior mode with a shared Newton search.

        Rows for which the projected Newton search does not converge are
        re-solved by :meth:`_score_multidimensional`.
        """

        def neg_log_posterior(
            rows: NDArray[np.intp],
            theta: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            ll = model.log_likelihood(patterns[rows], theta)
            diff = theta - prior_mean
            log_prior = -0.5 * np.einsum("ij,jk,ik->i", diff, prior_prec, diff)
            return -(ll + log_prior)

        n_patterns = patterns.shape[0]
        theta, _, converged = batched_newton_minimize(
            neg_log_posterior,
            np.broadcast_to(prior_mean, (n_patterns, prior_mean.size)),
            *self.theta_bounds,
        )
        standard_error = np.empty_like(theta)
        solved = np.flatnonzero(converged)
        standard_error[solved] = batched_hessian_se(
            lambda rows, values: neg_log_posterior(solved[rows], values),
            theta[solved],
        )
        for row in np.flatnonzero(~converged):
            theta[row], standard_error[row] = self._score_multidimensional(
                model, patterns[row : row + 1], prior_mean, prior_prec
            )
        return theta, standard_error

    def __repr__(self) -> str:
        return f"MAPScorer(bounds={self.theta_bounds})"
