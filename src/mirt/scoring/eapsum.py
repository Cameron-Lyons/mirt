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
        self.prior_mean = (
            None if prior_mean is None else np.asarray(prior_mean, dtype=np.float64)
        )
        self.prior_cov = (
            None if prior_cov is None else np.asarray(prior_cov, dtype=np.float64)
        )
        self._lookup_tables: dict[tuple[int, ...], dict] = {}
        self._lookup_values: dict[
            tuple[int, ...], tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        self._cached_model: BaseItemModel | None = None
        self._parameter_snapshot: dict[str, NDArray[np.float64]] = {}
        self._structure_snapshot: tuple[int, int, tuple[int, ...] | None] | None = None
        self._n_quadpts_snapshot: int | None = None
        self._prior_mean_snapshot: NDArray[np.float64] | None = None
        self._prior_cov_snapshot: NDArray[np.float64] | None = None

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
        n_factors = model.n_factors

        if n_factors > 1:
            raise ValueError("EAPsum only supports unidimensional models")

        responses = self._validate_responses(model, responses)
        self._ensure_cache_current(model)
        n_persons = responses.shape[0]
        theta_eap = np.empty(n_persons, dtype=np.float64)
        theta_se = np.empty(n_persons, dtype=np.float64)

        if n_persons == 0:
            return ScoreResult(
                theta=theta_eap,
                standard_error=theta_se,
                method="EAPsum",
            )

        missing = responses < 0
        if not np.any(missing):
            full_mask = tuple(range(model.n_items))
            theta_eap[:], theta_se[:] = self._score_response_group(
                model, responses, full_mask
            )
        else:
            observed = ~missing
            packed_masks = np.packbits(observed, axis=1)
            _, first_rows, group_ids = np.unique(
                packed_masks,
                axis=0,
                return_index=True,
                return_inverse=True,
            )
            grouped_rows = np.argsort(group_ids, kind="stable")
            group_sizes = np.bincount(group_ids, minlength=len(first_rows))
            group_starts = np.concatenate(([0], np.cumsum(group_sizes[:-1])))

            for first_row, group_start, group_size in zip(
                first_rows, group_starts, group_sizes, strict=True
            ):
                row_indices = grouped_rows[group_start : group_start + group_size]
                item_indices = tuple(np.flatnonzero(observed[first_row]).tolist())
                group_theta, group_se = self._score_response_group(
                    model,
                    responses[row_indices],
                    item_indices,
                )
                theta_eap[row_indices] = group_theta
                theta_se[row_indices] = group_se

        return ScoreResult(
            theta=theta_eap,
            standard_error=theta_se,
            method="EAPsum",
        )

    @staticmethod
    def _validate_responses(
        model: BaseItemModel,
        responses: NDArray,
    ) -> NDArray[np.int_]:
        """Validate response codes without changing negative missing values."""
        if responses.ndim != 2:
            raise ValueError(f"responses must be 2D, got {responses.ndim}D")
        if responses.shape[1] != model.n_items:
            raise ValueError(
                f"responses has {responses.shape[1]} items, expected {model.n_items}"
            )
        dtype_kind = responses.dtype.kind
        if dtype_kind not in "biuf":
            raise ValueError("responses must contain numeric values")
        if dtype_kind == "f":
            if not np.all(np.isfinite(responses)):
                raise ValueError("responses must contain finite values")
            if np.any(responses != np.trunc(responses)):
                raise ValueError("responses must contain integer category codes")
            int_bounds = np.iinfo(np.int_)
            if np.any(responses < int_bounds.min) or np.any(responses > int_bounds.max):
                raise ValueError("response codes exceed the supported integer range")

        observed = responses >= 0
        if model.is_polytomous:
            categories = np.asarray(model._n_categories)
            invalid = observed & (responses >= categories[None, :])
            if np.any(invalid):
                item_idx = int(np.flatnonzero(np.any(invalid, axis=0))[0])
                raise ValueError(
                    f"responses for item {item_idx} must be below {categories[item_idx]}"
                )
        elif np.any(responses[observed] > 1):
            raise ValueError("dichotomous responses must be coded as 0 or 1")

        return responses.astype(np.int_, copy=False)

    def _score_response_group(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        item_indices: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Score rows sharing the same observed-item mask."""
        self._build_current_lookup_table(model, item_indices)
        theta_values, se_values = self._lookup_values[item_indices]

        if item_indices:
            sum_scores = np.sum(responses[:, item_indices], axis=1, dtype=np.int64)
        else:
            sum_scores = np.zeros(responses.shape[0], dtype=np.int64)

        clipped_scores = np.clip(sum_scores, 0, len(theta_values) - 1)
        return theta_values[clipped_scores], se_values[clipped_scores]

    def _build_lookup_table(
        self,
        model: BaseItemModel,
        item_indices: tuple[int, ...] | None = None,
    ) -> dict:
        """Build lookup table mapping sum scores to EAP estimates."""
        self._ensure_cache_current(model)

        if item_indices is None:
            item_indices = tuple(range(model.n_items))
        return self._build_current_lookup_table(model, item_indices)

    def _build_current_lookup_table(
        self,
        model: BaseItemModel,
        item_indices: tuple[int, ...],
    ) -> dict:
        """Build a lookup after the model cache has been validated."""
        if item_indices in self._lookup_tables:
            return self._lookup_tables[item_indices]

        n_factors = model.n_factors
        quad_points, quad_weights = build_quadrature(
            n_quadpts=self.n_quadpts,
            n_factors=n_factors,
            prior_mean=self.prior_mean,
            prior_cov=self.prior_cov,
        )

        if model.is_polytomous:
            max_score = sum(model._n_categories[i] - 1 for i in item_indices)
        else:
            max_score = len(item_indices)

        log_p_score_given_theta = self._compute_sum_score_distribution(
            model,
            quad_points,
            max_score,
            item_indices,
        )

        log_prior = np.log(quad_weights + 1e-300)

        lookup = {"max_score": max_score}
        theta_values = np.empty(max_score + 1, dtype=np.float64)
        se_values = np.empty(max_score + 1, dtype=np.float64)

        for s in range(max_score + 1):
            log_posterior = log_p_score_given_theta[s, :] + log_prior
            log_norm = logsumexp(log_posterior)
            posterior = np.exp(log_posterior - log_norm)

            theta_s = np.dot(posterior, quad_points[:, 0])

            deviation = quad_points[:, 0] - theta_s
            variance = np.sum(posterior * (deviation**2))
            se_s = np.sqrt(variance)

            lookup[s] = {"theta": float(theta_s), "se": float(se_s)}
            theta_values[s] = theta_s
            se_values[s] = se_s

        self._lookup_tables[item_indices] = lookup
        self._lookup_values[item_indices] = (theta_values, se_values)
        return lookup

    @staticmethod
    def _optional_array_equal(
        left: NDArray[np.float64] | None,
        right: NDArray[np.float64] | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        return left_array.dtype == right_array.dtype and np.array_equal(
            left_array, right_array, equal_nan=True
        )

    @staticmethod
    def _model_structure(
        model: BaseItemModel,
    ) -> tuple[int, int, tuple[int, ...] | None]:
        categories = (
            tuple(int(value) for value in model._n_categories)
            if model.is_polytomous
            else None
        )
        return model.n_items, model.n_factors, categories

    def _cache_matches_model(self, model: BaseItemModel) -> bool:
        if self._cached_model is not model:
            return False
        if self._structure_snapshot != self._model_structure(model):
            return False
        if self._n_quadpts_snapshot != self.n_quadpts:
            return False
        if not self._optional_array_equal(
            self.prior_mean, self._prior_mean_snapshot
        ) or not self._optional_array_equal(self.prior_cov, self._prior_cov_snapshot):
            return False
        if self._parameter_snapshot.keys() != model._parameters.keys():
            return False

        return all(
            snapshot.dtype == model._parameters[name].dtype
            and np.array_equal(snapshot, model._parameters[name], equal_nan=True)
            for name, snapshot in self._parameter_snapshot.items()
        )

    def _ensure_cache_current(self, model: BaseItemModel) -> None:
        """Invalidate cached tables when the model or scorer changes."""
        if self._cache_matches_model(model):
            return

        self.clear_cache()
        self._cached_model = model
        self._parameter_snapshot = {
            name: values.copy() for name, values in model._parameters.items()
        }
        self._structure_snapshot = self._model_structure(model)
        self._n_quadpts_snapshot = self.n_quadpts
        self._prior_mean_snapshot = (
            None if self.prior_mean is None else np.asarray(self.prior_mean).copy()
        )
        self._prior_cov_snapshot = (
            None if self.prior_cov is None else np.asarray(self.prior_cov).copy()
        )

    def _compute_sum_score_distribution(
        self,
        model: BaseItemModel,
        quad_points: NDArray[np.float64],
        max_score: int,
        item_indices: tuple[int, ...] | None = None,
    ) -> NDArray[np.float64]:
        """Compute P(sum_score | theta) for all sum scores and theta points.

        Uses Lord-Wingersky recursion for efficiency.
        Uses Rust backend when available for ~10x speedup.
        """
        from mirt._rust_backend import lord_wingersky_recursion

        if item_indices is None:
            item_indices = tuple(range(model.n_items))

        if (
            item_indices
            and not model.is_polytomous
            and model.model_name in ("2PL", "1PL")
        ):
            params = model.parameters
            discrimination = params.get("discrimination", np.ones(model.n_items))
            difficulty = params["difficulty"]

            if discrimination.ndim == 1:
                result = lord_wingersky_recursion(
                    quad_points[:, 0] if quad_points.ndim > 1 else quad_points,
                    discrimination[list(item_indices)],
                    difficulty[list(item_indices)],
                )
                if result is not None:
                    return result

        n_quad = len(quad_points)

        log_dist = np.full((max_score + 1, n_quad), -np.inf)
        log_dist[0, :] = 0.0

        for item_idx in item_indices:
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
        """Clear all cached lookup tables and model snapshots."""
        self._lookup_tables.clear()
        self._lookup_values.clear()
        self._cached_model = None
        self._parameter_snapshot = {}
        self._structure_snapshot = None
        self._n_quadpts_snapshot = None
        self._prior_mean_snapshot = None
        self._prior_cov_snapshot = None

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
