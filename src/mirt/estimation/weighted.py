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

from mirt._prior_mass import gaussian_log_quadrature_mass
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._patterns import supports_pattern_compression
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation.base import (
    StartValues,
    _apply_starting_values,
    _validate_start,
)
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def _validate_weights(
    weights: NDArray[np.float64],
    *,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    """Return a validated one-dimensional survey-weight vector."""
    raw_weights = np.asarray(weights)
    if raw_weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if raw_weights.dtype.kind not in "iuf":
        raise ValueError("weights must be numeric")

    validated = np.asarray(raw_weights, dtype=np.float64)
    if expected_size is not None and validated.size != expected_size:
        raise ValueError(
            f"weights length ({validated.size}) must match "
            f"number of persons ({expected_size})"
        )
    if not np.all(np.isfinite(validated)):
        raise ValueError("weights must be finite")
    if np.any(validated < 0.0):
        raise ValueError("weights must be non-negative")
    if not np.any(validated > 0.0):
        raise ValueError("weights must contain at least one positive value")
    return validated


def _normalize_weights_to_total(
    weights: NDArray[np.float64], total: float
) -> NDArray[np.float64]:
    scaled = weights / np.max(weights)
    return scaled * (total / np.sum(scaled))


def _effective_sample_size(weights: NDArray[np.float64]) -> float:
    scaled = weights / np.max(weights)
    return float(np.sum(scaled) ** 2 / np.sum(scaled**2))


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

    Standard errors use itemwise complete-data curvature with survey-weighted
    posterior counts. They do not account for clustering or stratification.
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
        if not isinstance(normalize_weights, (bool, np.bool_)):
            raise ValueError("normalize_weights must be a boolean")
        self.normalize_weights = bool(normalize_weights)

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        weights: NDArray[np.float64] | None = None,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit model with survey weights.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix
        weights : ndarray of shape (n_persons,), optional
            Person-level sampling weights. If None, equal weights are used.
        prior_mean : ndarray, optional
            Prior mean for latent abilities
        prior_cov : ndarray, optional
            Prior covariance for latent abilities
        start : {"default", "model"} or mapping, default="default"
            Starting values, as for :meth:`EMEstimator.fit`.

        Returns
        -------
        FitResult
            Fitted model with estimates and diagnostics
        """
        start = _validate_start(start)
        responses = self._validate_responses(responses, model.n_items)
        previous_context = self._fit_context
        with EMFitContext(responses) as context:
            self._fit_context = context
            try:
                return self._fit_weighted_prepared(
                    model, responses, weights, prior_mean, prior_cov, start
                )
            finally:
                self._fit_context = previous_context

    def _fit_weighted_prepared(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        weights: NDArray[np.float64] | None,
        prior_mean: NDArray[np.float64] | None,
        prior_cov: NDArray[np.float64] | None,
        start: StartValues = "default",
    ) -> FitResult:
        from mirt.results.fit_result import FitResult

        n_persons = responses.shape[0]

        if weights is None:
            weights = np.ones(n_persons, dtype=np.float64)
        else:
            weights = _validate_weights(weights, expected_size=n_persons)

        if self.normalize_weights:
            weights = _normalize_weights_to_total(weights, float(n_persons))

        self._weights = weights

        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        if prior_mean is None:
            prior_mean = np.zeros(model.n_factors)
        if prior_cov is None:
            prior_cov = np.eye(model.n_factors)

        _apply_starting_values(model, start)

        self._convergence_history = []
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            posterior_weights, log_marginal = self._e_step_weighted(
                model, responses, prior_mean, prior_cov, weights
            )

            current_ll = float(weights @ log_marginal)
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step_weighted(model, responses, posterior_weights, weights)
            del posterior_weights

        if not converged:
            posterior_weights, log_marginal = self._e_step_weighted(
                model, responses, prior_mean, prior_cov, weights
            )
            current_ll = float(weights @ log_marginal)
            self._convergence_history[-1] = current_ll

        model._is_fitted = True

        standard_errors = self._compute_weighted_standard_errors(
            model, responses, posterior_weights, weights
        )

        n_params = model.n_parameters
        effective_n = _effective_sample_size(weights)
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, effective_n)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=converged,
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
        """Return individual posterior weights and per-person log marginals.

        Survey weights enter the fit objective and expected counts, leaving each
        person's posterior unchanged.
        """
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights

        # The same exact-type check used for response compression identifies
        # built-in, exchangeable likelihoods with owned batch outputs. Preserve
        # custom person-specific likelihoods and their per-person theta inputs.
        if supports_pattern_compression(model):
            log_joint = model.log_likelihood_batch(responses, quad_points)
        else:
            n_persons = responses.shape[0]
            log_joint = np.empty((n_persons, len(quad_weights)))
            for q, point in enumerate(quad_points):
                theta_q = np.tile(point, (n_persons, 1))
                log_joint[:, q] = model.log_likelihood(responses, theta_q)

        log_prior_mass = gaussian_log_quadrature_mass(
            quad_points, quad_weights, prior_mean, prior_cov
        )

        return normalize_log_posterior(log_joint, log_prior_mass)

    def _m_step_weighted(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        survey_weights: NDArray[np.float64],
    ) -> None:
        """Optimize items from bounded survey-weighted expected counts."""
        quad_points = self._quadrature.nodes
        context = self._fit_context
        if context is None or context.responses is not responses:
            context = EMFitContext(responses)
        correct = observed = None
        if not model.is_polytomous:
            correct, observed = context.expected_counts(
                posterior_weights, survey_weights
            )

        for item_idx in range(model.n_items):
            params, _ = self._get_item_params_and_bounds(model, item_idx)
            if not params.size:
                continue
            category_counts = None
            if model.is_polytomous:
                category_counts = context.expected_category_counts(
                    item_idx,
                    model.n_categories[item_idx],
                    posterior_weights,
                    survey_weights,
                )
                item_observed = category_counts.sum(axis=1)
                item_correct = None
            else:
                item_observed = observed[item_idx]
                item_correct = correct[item_idx]
            if not np.any(item_observed):
                continue
            optimal = self._optimize_item_params(
                model,
                item_idx,
                responses,
                posterior_weights,
                quad_points,
                r_k=item_correct,
                n_k_valid=item_observed,
                r_kc=category_counts,
            )
            self._set_item_params(model, item_idx, optimal)

    def _compute_weighted_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        survey_weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute itemwise curvature standard errors using survey weights."""

        return self._compute_standard_errors(
            model, responses, posterior_weights, person_weights=survey_weights
        )


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
    weights = _validate_weights(weights)
    return _effective_sample_size(weights)


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
    weights = _validate_weights(weights)
    n = len(weights)
    effective_n = _effective_sample_size(weights)
    return n / effective_n
