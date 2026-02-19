"""EAPsum (Expected A Posteriori based on Sum Scores) scoring.

EAPsum estimates theta using only the sum score rather than the full
response pattern. This is computationally efficient and useful for:
- Computer Adaptive Testing (CAT) stopping rules
- Quick ability estimates when response patterns are not available
- Large-scale assessments where full EAP is too slow

References
----------
Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. (1995).
    Item response theory for scores on tests including polytomous items
    with ordered responses. Applied Psychological Measurement, 19(1), 39-49.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.results.score_result import ScoreResult
from mirt.scoring._common import build_quadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class EAPSumScorer:
    """EAP scoring based on sum scores only.

    This scorer computes expected a posteriori estimates using only the
    total sum score, not the full response pattern. This is done by
    pre-computing the probability of each sum score at each quadrature
    point, creating a lookup table.

    Parameters
    ----------
    n_quadpts : int
        Number of quadrature points. Default 49.
    prior_mean : ndarray, optional
        Prior mean for theta. Default zeros.
    prior_cov : ndarray, optional
        Prior covariance for theta. Default identity.
    """

    def __init__(
        self,
        n_quadpts: int = 49,
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> None:
        if n_quadpts < 5:
            raise ValueError("n_quadpts should be at least 5")

        self.n_quadpts = n_quadpts
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self._lookup_table: dict | None = None

    def score(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> ScoreResult:
        """Score responses using EAPsum method.

        Parameters
        ----------
        model : BaseItemModel
            Fitted IRT model
        responses : ndarray
            Response matrix (n_persons x n_items)

        Returns
        -------
        ScoreResult
            Scoring results with theta estimates and standard errors
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        if n_factors > 1:
            raise ValueError("EAPsum only supports unidimensional models")

        sum_scores = np.sum(np.maximum(responses, 0), axis=1)

        lookup = self._build_lookup_table(model)

        theta_eap = np.zeros(n_persons)
        theta_se = np.zeros(n_persons)

        for i in range(n_persons):
            s = int(sum_scores[i])
            if s in lookup:
                theta_eap[i] = lookup[s]["theta"]
                theta_se[i] = lookup[s]["se"]
            else:
                s_clipped = max(0, min(s, lookup["max_score"]))
                theta_eap[i] = lookup[s_clipped]["theta"]
                theta_se[i] = lookup[s_clipped]["se"]

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAPsum",
        )

    def _build_lookup_table(
        self,
        model: BaseItemModel,
    ) -> dict:
        """Build lookup table mapping sum scores to EAP estimates."""
        if self._lookup_table is not None:
            return self._lookup_table

        n_factors = model.n_factors
        quad_points, quad_weights = build_quadrature(
            n_quadpts=self.n_quadpts,
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )

        if model.is_polytomous:
            max_score = sum(model._n_categories[i] - 1 for i in range(model.n_items))
        else:
            max_score = model.n_items

        log_p_score_given_theta = self._compute_sum_score_distribution(
            model, quad_points, max_score
        )

        log_prior = np.log(quad_weights + 1e-300)

        lookup = {"max_score": max_score}

        for s in range(max_score + 1):
            log_posterior = log_p_score_given_theta[s, :] + log_prior
            log_norm = logsumexp(log_posterior)
            posterior = np.exp(log_posterior - log_norm)

            theta_s = np.dot(posterior, quad_points[:, 0])

            deviation = quad_points[:, 0] - theta_s
            variance = np.sum(posterior * (deviation**2))
            se_s = np.sqrt(variance)

            lookup[s] = {"theta": float(theta_s), "se": float(se_s)}

        self._lookup_table = lookup
        return lookup

    def _compute_sum_score_distribution(
        self,
        model: BaseItemModel,
        quad_points: NDArray[np.float64],
        max_score: int,
    ) -> NDArray[np.float64]:
        """Compute P(sum_score | theta) for all sum scores and theta points.

        Uses Lord-Wingersky recursion for efficiency.
        Uses Rust backend when available for ~10x speedup.
        """
        from mirt._rust_backend import lord_wingersky_recursion

        if not model.is_polytomous and model.model_name in ("2PL", "1PL"):
            params = model.parameters
            discrimination = params.get("discrimination", np.ones(model.n_items))
            difficulty = params["difficulty"]

            if discrimination.ndim == 1:
                result = lord_wingersky_recursion(
                    quad_points[:, 0] if quad_points.ndim > 1 else quad_points,
                    discrimination,
                    difficulty,
                )
                if result is not None:
                    return result

        n_quad = len(quad_points)
        n_items = model.n_items

        log_dist = np.full((max_score + 1, n_quad), -np.inf)
        log_dist[0, :] = 0.0

        for item_idx in range(n_items):
            probs = model.probability(quad_points, item_idx)

            if probs.ndim == 1:
                p1 = probs
                p0 = 1 - p1

                new_log_dist = np.full_like(log_dist, -np.inf)

                for s in range(max_score + 1):
                    log_stay = log_dist[s, :] + np.log(p0 + 1e-300)

                    if s > 0:
                        log_up = log_dist[s - 1, :] + np.log(p1 + 1e-300)
                        new_log_dist[s, :] = np.logaddexp(log_stay, log_up)
                    else:
                        new_log_dist[s, :] = log_stay

                log_dist = new_log_dist

            else:
                n_cats = probs.shape[1]
                log_probs = np.log(probs + 1e-300)

                new_log_dist = np.full_like(log_dist, -np.inf)

                for s in range(max_score + 1):
                    for c in range(n_cats):
                        if s >= c and s - c <= max_score:
                            contribution = log_dist[s - c, :] + log_probs[:, c]
                            new_log_dist[s, :] = np.logaddexp(
                                new_log_dist[s, :], contribution
                            )

                log_dist = new_log_dist

        return log_dist

    def get_lookup_table(self, model: BaseItemModel) -> dict:
        """Get the sum score to theta lookup table.

        Parameters
        ----------
        model : BaseItemModel
            Fitted IRT model

        Returns
        -------
        dict
            Dictionary mapping sum scores to theta estimates and SEs
        """
        return self._build_lookup_table(model)

    def clear_cache(self) -> None:
        """Clear the cached lookup table."""
        self._lookup_table = None

    def __repr__(self) -> str:
        return f"EAPSumScorer(n_quadpts={self.n_quadpts})"


def eapsum(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
) -> ScoreResult:
    """Convenience function for EAPsum scoring.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : ndarray
        Response matrix (n_persons x n_items)
    n_quadpts : int
        Number of quadrature points
    prior_mean : ndarray, optional
        Prior mean
    prior_cov : ndarray, optional
        Prior covariance

    Returns
    -------
    ScoreResult
        Scoring results
    """
    scorer = EAPSumScorer(
        n_quadpts=n_quadpts,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    return scorer.score(model, responses)


def sum_score_to_theta(
    model: BaseItemModel,
    sum_scores: NDArray[np.int_] | list[int],
    n_quadpts: int = 49,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert sum scores directly to theta estimates.

    Useful for quick conversions without full response data.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    sum_scores : array-like
        Sum scores to convert
    n_quadpts : int
        Number of quadrature points

    Returns
    -------
    theta : ndarray
        Theta estimates for each sum score
    se : ndarray
        Standard errors for each estimate
    """
    sum_scores = np.atleast_1d(sum_scores)

    scorer = EAPSumScorer(n_quadpts=n_quadpts)
    lookup = scorer.get_lookup_table(model)

    theta = np.zeros(len(sum_scores))
    se = np.zeros(len(sum_scores))

    for i, s in enumerate(sum_scores):
        s = int(s)
        if s in lookup:
            theta[i] = lookup[s]["theta"]
            se[i] = lookup[s]["se"]
        else:
            s_clipped = max(0, min(s, lookup["max_score"]))
            theta[i] = lookup[s_clipped]["theta"]
            se[i] = lookup[s_clipped]["se"]

    return theta, se
