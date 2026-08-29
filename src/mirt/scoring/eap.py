from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.results.ability_posterior import AbilityPosteriorResult
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import (
    build_quadrature,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.utils.numeric import logsumexp_axis1

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_TARGET_WORKING_BYTES = 32 * 1024 * 1024
_PATTERN_SAMPLE_SIZE = 1_024
# Require a projected fourfold reduction so sorting also pays off for the
# fastest native likelihood implementations.
_MAX_SAMPLE_UNIQUE_FRACTION = 0.25


def _eap_response_patterns(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.intp]]:
    """Compress rows when a bounded sample predicts useful likelihood reuse."""
    n_persons = responses.shape[0]
    if n_persons <= _PATTERN_SAMPLE_SIZE:
        return unique_response_patterns(responses)

    sample_indices = np.linspace(
        0,
        n_persons - 1,
        _PATTERN_SAMPLE_SIZE,
        dtype=np.intp,
    )
    sample_patterns, _ = unique_response_patterns(responses[sample_indices])
    if sample_patterns.shape[0] > (_MAX_SAMPLE_UNIQUE_FRACTION * _PATTERN_SAMPLE_SIZE):
        return responses, np.arange(n_persons, dtype=np.intp)
    return unique_response_patterns(responses)


class EAPScorer:
    def __init__(
        self,
        n_quadpts: int = 49,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        batch_size: int | None = None,
    ) -> None:
        if (
            isinstance(n_quadpts, (bool, np.bool_))
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 5
        ):
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = int(n_quadpts)
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
        if batch_size is not None and (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self.batch_size = None if batch_size is None else int(batch_size)

    def _resolve_batch_size(
        self,
        *,
        n_patterns: int,
        n_items: int,
        n_quad: int,
    ) -> int:
        """Choose a pattern batch that bounds temporary likelihood storage."""
        if self.batch_size is not None:
            return min(self.batch_size, n_patterns)

        # Generic likelihood evaluation uses boolean and float response matrices,
        # while posterior normalization holds several theta-grid matrices. This
        # conservative estimate keeps their combined working set near 32 MiB.
        bytes_per_pattern = 17 * n_items + 32 * n_quad
        automatic_size = max(1, _TARGET_WORKING_BYTES // bytes_per_pattern)
        return min(automatic_size, n_patterns)

    @staticmethod
    def _posterior_batch(
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
        log_weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize one likelihood batch and return its log marginal values."""
        posterior = np.array(
            model.log_likelihood_batch(responses, quad_points),
            dtype=np.float64,
            copy=True,
        )
        expected_shape = (responses.shape[0], quad_points.shape[0])
        if posterior.shape != expected_shape:
            raise ValueError(
                f"model log-likelihood batch has shape {posterior.shape}, "
                f"expected {expected_shape}"
            )

        posterior += log_weights[None, :]
        log_marginal = logsumexp_axis1(posterior)
        if not np.all(np.isfinite(log_marginal)):
            raise ValueError("model likelihoods must produce finite posterior mass")
        posterior -= log_marginal[:, None]
        np.exp(posterior, out=posterior)
        return posterior, log_marginal

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = validate_scoring_responses(model, responses)
        n_factors = model.n_factors

        quad_points, quad_weights = build_quadrature(
            n_quadpts=self.n_quadpts,
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )
        if responses.shape[0] == 0:
            shape = (0,) if n_factors == 1 else (0, n_factors)
            return ScoreResult(
                theta=np.empty(shape, dtype=np.float64),
                standard_error=np.empty(shape, dtype=np.float64),
                method="EAP",
            )

        patterns, inverse = _eap_response_patterns(responses)

        center = quad_weights @ quad_points
        centered_points = quad_points - center
        centered_points_squared = centered_points**2
        log_weights = np.log(quad_weights + 1e-300)
        n_patterns = patterns.shape[0]
        pattern_theta = np.empty((n_patterns, n_factors), dtype=np.float64)
        pattern_se = np.empty_like(pattern_theta)
        batch_size = self._resolve_batch_size(
            n_patterns=n_patterns,
            n_items=model.n_items,
            n_quad=quad_points.shape[0],
        )

        for start in range(0, n_patterns, batch_size):
            stop = min(start + batch_size, n_patterns)
            posterior, _ = self._posterior_batch(
                model,
                patterns[start:stop],
                quad_points,
                log_weights,
            )

            centered_mean = posterior @ centered_points
            pattern_theta[start:stop] = centered_mean + center
            variance = posterior @ centered_points_squared - centered_mean**2
            np.maximum(variance, 0.0, out=variance)
            np.sqrt(variance, out=pattern_se[start:stop])

        theta_eap = pattern_theta[inverse]
        theta_se = pattern_se[inverse]

        if n_factors == 1:
            theta_eap = theta_eap.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAP",
        )

    def posterior(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        *,
        person_ids: list[Any] | NDArray[Any] | None = None,
    ) -> AbilityPosteriorResult:
        """Return normalized ability distributions on the EAP quadrature grid.

        Unlike :meth:`score`, this method intentionally retains one probability
        per respondent and grid point. Likelihood evaluation remains batched,
        but callers should account for the returned ``n_persons * n_points``
        array when choosing ``n_quadpts`` for multidimensional models.
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = validate_scoring_responses(model, responses)
        quad_points, quad_weights = build_quadrature(
            n_quadpts=self.n_quadpts,
            n_factors=model.n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )
        n_persons = responses.shape[0]
        posterior_weights = np.empty(
            (n_persons, quad_points.shape[0]),
            dtype=np.float64,
        )
        log_marginal = np.empty(n_persons, dtype=np.float64)
        if n_persons:
            log_weights = np.log(quad_weights + 1e-300)
            batch_size = self._resolve_batch_size(
                n_patterns=n_persons,
                n_items=model.n_items,
                n_quad=quad_points.shape[0],
            )
            for start in range(0, n_persons, batch_size):
                stop = min(start + batch_size, n_persons)
                batch_weights, batch_log_marginal = self._posterior_batch(
                    model,
                    responses[start:stop],
                    quad_points,
                    log_weights,
                )
                posterior_weights[start:stop] = batch_weights
                log_marginal[start:stop] = batch_log_marginal

        return AbilityPosteriorResult._from_owned_arrays(
            points=quad_points,
            weights=posterior_weights,
            log_marginal_likelihood=log_marginal,
            person_ids=person_ids,
        )

    def __repr__(self) -> str:
        if self.batch_size is None:
            return f"EAPScorer(n_quadpts={self.n_quadpts})"
        return f"EAPScorer(n_quadpts={self.n_quadpts}, batch_size={self.batch_size})"


def ability_posterior(
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    *,
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    batch_size: int | None = None,
    person_ids: list[Any] | NDArray[Any] | None = None,
) -> AbilityPosteriorResult:
    """Compute normalized posterior ability distributions for respondents.

    ``model_or_result`` may be either a fitted item model or the ``FitResult``
    returned by :func:`mirt.fit_mirt`.
    """
    from mirt.results.fit_result import FitResult

    model = (
        model_or_result.model
        if isinstance(model_or_result, FitResult)
        else model_or_result
    )
    scorer = EAPScorer(
        n_quadpts=n_quadpts,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        batch_size=batch_size,
    )
    return scorer.posterior(model, responses, person_ids=person_ids)
