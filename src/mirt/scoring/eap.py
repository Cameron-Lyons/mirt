from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.results.score_result import ScoreResult
from mirt.scoring._common import build_quadrature, validate_scoring_responses
from mirt.utils.numeric import logsumexp_axis1

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class EAPScorer:
    def __init__(
        self,
        n_quadpts: int = 49,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
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

        posterior += np.log(quad_weights + 1e-300)[None, :]
        posterior -= logsumexp_axis1(posterior)[:, None]
        np.exp(posterior, out=posterior)

        center = quad_weights @ quad_points
        centered_points = quad_points - center
        centered_mean = posterior @ centered_points
        theta_eap = centered_mean + center
        variance = posterior @ (centered_points**2) - centered_mean**2
        np.maximum(variance, 0.0, out=variance)
        theta_se = np.sqrt(variance)

        if n_factors == 1:
            theta_eap = theta_eap.ravel()
            theta_se = theta_se.ravel()

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAP",
        )

    def __repr__(self) -> str:
        return f"EAPScorer(n_quadpts={self.n_quadpts})"
