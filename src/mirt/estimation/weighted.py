"""Weighted estimation with survey weights support.

This module extends the EM estimator to support person-level sampling weights,
enabling analysis of complex survey data (e.g., PISA, NAEP, TIMSS).

Survey weights allow proper inference when the sample is not a simple
random sample from the population.

References:
    Mislevy, R. J. (1991). Randomization-based inference about latent
        variables from complex samples. Psychometrika, 56(2), 177-196.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from mirt.constants import PROB_EPSILON
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class WeightedEMEstimator(EMEstimator):
    """EM estimator with support for survey weights.

    Extends the standard EM algorithm to incorporate person-level weights
    in both the E-step and M-step. This enables valid estimation when
    analyzing data from complex survey designs.

    Parameters
    ----------
    n_quadpts : int
        Number of Gauss-Hermite quadrature points.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood change.
    verbose : bool
        Whether to print iteration progress.
    normalize_weights : bool
        Whether to normalize weights to sum to sample size.

    Notes
    -----
    The weighted log-likelihood is:
        WLL = sum_i w_i * log L_i

    where w_i is the weight for person i and L_i is their marginal likelihood.

    Standard errors computed under weighted estimation are design-based
    and may require additional corrections for proper inference.
    """

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__(n_quadpts, max_iter, tol, verbose)
        self.normalize_weights = normalize_weights

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        weights: NDArray[np.float64] | None = None,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Fit model with survey weights.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit
        responses : ndarray of shape (n_persons, n_items)
            Response matrix
        weights : ndarray of shape (n_persons,), optional
            Person-level sampling weights. If None, equal weights are used.
        prior_mean : ndarray, optional
            Prior mean for latent abilities
        prior_cov : ndarray, optional
            Prior covariance for latent abilities

        Returns
        -------
        FitResult
            Fitted model with estimates and diagnostics
        """
        from mirt.results.fit_result import FitResult

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]

        if weights is None:
            weights = np.ones(n_persons)
        else:
            weights = np.asarray(weights).ravel()
            if len(weights) != n_persons:
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"number of persons ({n_persons})"
                )
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")

        if self.normalize_weights:
            weights = weights * n_persons / weights.sum()

        self._weights = weights

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        if prior_mean is None:
            prior_mean = np.zeros(model.n_factors)
        if prior_cov is None:
            prior_cov = np.eye(model.n_factors)

        if not model._is_fitted:
            model._initialize_parameters()

        self._convergence_history = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            posterior_weights, marginal_ll = self._e_step_weighted(
                model, responses, prior_mean, prior_cov, weights
            )

            current_ll = np.sum(weights * np.log(marginal_ll + 1e-300))
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step_weighted(model, responses, posterior_weights, weights)

        model._is_fitted = True

        standard_errors = self._compute_weighted_standard_errors(
            model, responses, posterior_weights, weights
        )

        n_params = model.n_parameters
        effective_n = weights.sum() ** 2 / np.sum(weights**2)
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, effective_n)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=iteration < self.max_iter - 1,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _e_step_weighted(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step with survey weights."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights
        n_persons = responses.shape[0]
        n_quad = len(quad_weights)

        log_likelihoods = np.zeros((n_persons, n_quad))

        for q in range(n_quad):
            theta_q = np.tile(quad_points[q], (n_persons, 1))
            log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)

        log_prior = self._log_multivariate_normal(quad_points, prior_mean, prior_cov)

        log_joint = log_likelihoods + log_prior[None, :] + np.log(quad_weights)[None, :]

        log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
        log_posterior = log_joint - log_marginal

        posterior_weights = np.exp(log_posterior)
        marginal_ll = np.exp(log_marginal.ravel())

        return posterior_weights, marginal_ll

    def _m_step_weighted(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        survey_weights: NDArray[np.float64],
    ) -> None:
        """M-step with survey weights incorporated."""
        quad_points = self._quadrature.nodes
        n_items = model.n_items

        weighted_posterior = posterior_weights * survey_weights[:, None]
        n_k = weighted_posterior.sum(axis=0)

        for item_idx in range(n_items):
            self._optimize_item_weighted(
                model, item_idx, responses, weighted_posterior, quad_points, n_k
            )

    def _optimize_item_weighted(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        weighted_posterior: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        n_k: NDArray[np.float64],
    ) -> None:
        """Optimize item parameters using weighted expected counts."""
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        n_k_valid = np.sum(weighted_posterior[valid_mask], axis=0)

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)

        if model.is_polytomous:
            n_categories = model._n_categories[item_idx]
            n_quad = len(n_k)

            r_kc = np.zeros((n_quad, n_categories))
            for c in range(n_categories):
                cat_mask = valid_mask & (item_responses == c)
                r_kc[:, c] = np.sum(weighted_posterior[cat_mask, :], axis=0)

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

                ll = np.sum(r_kc * np.log(probs))

                return -ll

        else:
            r_k = np.sum(
                item_responses[valid_mask, None] * weighted_posterior[valid_mask, :],
                axis=0,
            )

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                probs = model.probability(quad_points, item_idx)
                probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

                ll = np.sum(r_k * np.log(probs) + (n_k_valid - r_k) * np.log(1 - probs))

                return -ll

        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        self._set_item_params(model, item_idx, result.x)

    def _compute_weighted_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        survey_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute design-based standard errors."""

        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model.parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.zeros_like(values)

            for item_idx in range(model.n_items):
                item_se = self._compute_item_se(
                    model,
                    item_idx,
                    name,
                    responses,
                    posterior_weights * survey_weights[:, None],
                )
                if values.ndim == 1:
                    se[item_idx] = item_se
                else:
                    se[item_idx] = item_se

            standard_errors[name] = se

        return standard_errors


def compute_effective_sample_size(weights: NDArray[np.float64]) -> float:
    """Compute effective sample size from survey weights.

    Parameters
    ----------
    weights : ndarray
        Person-level survey weights

    Returns
    -------
    float
        Effective sample size, which is smaller than actual N when
        weights vary substantially
    """
    weights = np.asarray(weights)
    return weights.sum() ** 2 / np.sum(weights**2)


def compute_design_effect(weights: NDArray[np.float64]) -> float:
    """Compute design effect (DEFF) from survey weights.

    Parameters
    ----------
    weights : ndarray
        Person-level survey weights

    Returns
    -------
    float
        Design effect, ratio of actual variance to SRS variance.
        DEFF = n / effective_n
    """
    n = len(weights)
    effective_n = compute_effective_sample_size(weights)
    return n / effective_n
