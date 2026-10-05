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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError
from mirt.results.score_result import ScoreResult
from mirt.scoring._common import build_quadrature
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Item probabilities are floored before the recursion so every category keeps
# a finite log probability.
_PROBABILITY_FLOOR = 1e-300
# The probability-space recursion can lose only terms below the double
# underflow threshold (about 1e-308). When every sum score's largest
# unnormalized log posterior term exceeds this bound, such losses are far
# below double precision in its posterior; otherwise the score table is
# recomputed with the log-space recursion.
_MIN_PROBABILITY_SPACE_LOG_POSTERIOR = -600.0


@dataclass(frozen=True)
class SumScoreTable:
    """Sum-score to theta conversion table from EAPsum scoring.

    Attributes
    ----------
    sum_score : ndarray of int, shape (n_scores,)
        Every attainable sum score, from 0 to the maximum.
    theta : ndarray of shape (n_scores,)
        Posterior mean of theta given each sum score.
    standard_error : ndarray of shape (n_scores,)
        Posterior standard deviation of theta given each sum score.
    expected_proportion : ndarray of shape (n_scores,)
        Model-implied marginal probability of each sum score under the
        scoring prior.
    observed : ndarray of int, optional
        Number of respondents with each sum score, when responses were given.
    expected : ndarray, optional
        ``n_respondents * expected_proportion``, when responses were given.
    standardized_residual : ndarray, optional
        ``(observed - expected) / sqrt(expected)``; NaN where the expected
        count underflows to zero.
    """

    sum_score: NDArray[np.int_]
    theta: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    expected_proportion: NDArray[np.float64]
    observed: NDArray[np.int_] | None = None
    expected: NDArray[np.float64] | None = None
    standardized_residual: NDArray[np.float64] | None = None

    @property
    def n_scores(self) -> int:
        """Number of attainable sum scores."""
        return int(self.sum_score.shape[0])

    def _columns(self) -> dict[str, NDArray[Any]]:
        columns: dict[str, NDArray[Any]] = {
            "sum_score": self.sum_score,
            "theta": self.theta,
            "standard_error": self.standard_error,
            "expected_proportion": self.expected_proportion,
        }
        if self.observed is not None:
            columns["observed"] = self.observed
        if self.expected is not None:
            columns["expected"] = self.expected
        if self.standardized_residual is not None:
            columns["standardized_residual"] = self.standardized_residual
        return columns

    def to_dict(self) -> dict[str, list[Any]]:
        """Return the table as plain Python column lists."""
        return {name: values.tolist() for name, values in self._columns().items()}

    def to_dataframe(self) -> Any:
        """Return the table using the configured dataframe backend."""
        from mirt.utils.dataframe import create_dataframe

        return create_dataframe(self._columns())


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
        self._lookup_values: dict[
            tuple[int, ...],
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self._grid: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._item_tables: dict[int, NDArray[np.float64]] = {}
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
        theta_values, se_values, _ = self._lookup_arrays(model, item_indices)

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
        theta_values, se_values, _ = self._lookup_arrays(model, item_indices)
        lookup: dict = {"max_score": len(theta_values) - 1}
        for score, (theta_value, se_value) in enumerate(
            zip(theta_values, se_values, strict=True)
        ):
            lookup[score] = {
                "theta": float(theta_value),
                "se": float(se_value),
            }
        return lookup

    def _lookup_arrays(
        self,
        model: BaseItemModel,
        item_indices: tuple[int, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return per-score theta, SE and log marginal probability arrays.

        The model cache must already be validated. Results are cached by
        observed-item mask.
        """
        cached = self._lookup_values.get(item_indices)
        if cached is not None:
            return cached

        quad_points, log_prior = self._current_grid()

        if model.is_polytomous:
            max_score = sum(model._n_categories[i] - 1 for i in item_indices)
        else:
            max_score = len(item_indices)

        log_p_score_given_theta = self._compute_sum_score_distribution(
            model,
            quad_points,
            max_score,
            item_indices,
            log_prior=log_prior,
        )

        log_posterior = log_p_score_given_theta + log_prior[None, :]
        log_norm = logsumexp(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_norm)

        theta_points = quad_points[:, 0]
        theta_values = posterior @ theta_points
        deviations = theta_points[None, :] - theta_values[:, None]
        se_values = np.sqrt(np.sum(posterior * deviations**2, axis=1))

        values = (theta_values, se_values, log_norm.ravel())
        self._lookup_values[item_indices] = values
        return values

    def _current_grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return quadrature points and log weights for the cached state."""
        if self._grid is None:
            quad_points, quad_weights = build_quadrature(
                n_quadpts=self.n_quadpts,
                n_factors=1,
                prior_mean=self.prior_mean,
                prior_cov=self.prior_cov,
            )
            self._grid = (quad_points, np.log(quad_weights + _PROBABILITY_FLOOR))
        return self._grid

    def _item_probability_tables(
        self,
        model: BaseItemModel,
        quad_points: NDArray[np.float64],
        item_indices: tuple[int, ...],
    ) -> list[NDArray[np.float64]]:
        """Return floored ``(n_categories, n_quad)`` category probabilities.

        Tables on the scorer's own grid are cached until the model or scorer
        configuration changes, so each item is evaluated once per state.
        """
        cache = (
            self._item_tables
            if self._grid is not None
            and quad_points is self._grid[0]
            and self._cached_model is model
            else {}
        )
        tables = []
        for item_idx in item_indices:
            table = cache.get(item_idx)
            if table is None:
                probabilities = np.asarray(
                    model.probability(quad_points, item_idx), dtype=np.float64
                )
                if probabilities.ndim == 1:
                    probabilities = np.column_stack(
                        (1.0 - probabilities, probabilities)
                    )
                table = np.ascontiguousarray(probabilities.T) + _PROBABILITY_FLOOR
                cache[item_idx] = table
            tables.append(table)
        return tables

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
        *,
        log_prior: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute log P(sum_score | theta) for all sum scores and theta points.

        Uses the Lord-Wingersky recursion, in compiled log space for built-in
        1PL/2PL models and otherwise in probability space. A table with sum
        scores so improbable that their probabilities could underflow is
        recomputed in log space. Underflow is judged on the posterior terms
        when the grid's ``log_prior`` is given, else on the likelihoods alone.
        """
        if item_indices is None:
            item_indices = tuple(range(model.n_items))

        native = _native_sum_score_distribution(model, quad_points, item_indices)
        if native is not None:
            return native

        tables = self._item_probability_tables(model, quad_points, item_indices)
        log_dist = _probability_sum_score_distribution(tables, quad_points.shape[0])
        if log_dist.shape[0] != max_score + 1:
            raise RuntimeError(
                "sum-score distribution size does not match the model structure"
            )
        log_terms = log_dist if log_prior is None else log_dist + log_prior[None, :]
        if np.any(np.max(log_terms, axis=1) <= _MIN_PROBABILITY_SPACE_LOG_POSTERIOR):
            log_dist = _log_space_sum_score_distribution(tables, quad_points.shape[0])
        return log_dist

    def score_table(
        self,
        model: BaseItemModel,
        responses: ArrayLike | None = None,
    ) -> SumScoreTable:
        """Return the full-form sum-score conversion table.

        Parameters
        ----------
        model : BaseItemModel
            Fitted unidimensional IRT model.
        responses : array-like of shape (n_persons, n_items), optional
            Complete response matrix. When given, the table also reports
            observed and expected sum-score counts and standardized residuals
            ``(observed - expected) / sqrt(expected)``.

        Returns
        -------
        SumScoreTable
            Theta, standard error and model-implied proportion for every
            attainable sum score.

        Raises
        ------
        MirtValidationError
            If ``responses`` contain missing values, since observed
            frequencies of full-form sum scores are then undefined.
        """
        theta_values, se_values, log_marginal = self._full_form_arrays(model)
        expected_proportion = np.exp(log_marginal)
        expected_proportion /= expected_proportion.sum()
        n_scores = theta_values.shape[0]
        sum_score = np.arange(n_scores, dtype=np.int_)
        if responses is None:
            return SumScoreTable(
                sum_score=sum_score,
                theta=theta_values.copy(),
                standard_error=se_values.copy(),
                expected_proportion=expected_proportion,
            )

        validated = self._validate_responses(model, np.asarray(responses))
        if np.any(validated < 0):
            raise MirtValidationError(
                "score_table requires complete responses; sum-score "
                "frequencies are undefined when items are missing",
                parameter="responses",
            )
        observed = np.bincount(validated.sum(axis=1), minlength=n_scores)
        expected = validated.shape[0] * expected_proportion
        residual = np.divide(
            observed - expected,
            np.sqrt(expected),
            out=np.full(n_scores, np.nan),
            where=expected > 0.0,
        )
        return SumScoreTable(
            sum_score=sum_score,
            theta=theta_values.copy(),
            standard_error=se_values.copy(),
            expected_proportion=expected_proportion,
            observed=observed.astype(np.int_, copy=False),
            expected=expected,
            standardized_residual=residual,
        )

    def _full_form_arrays(
        self,
        model: BaseItemModel,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return lookup arrays for the form with every item observed."""
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        if model.n_factors > 1:
            raise ValueError("EAPsum only supports unidimensional models")
        self._ensure_cache_current(model)
        return self._lookup_arrays(model, tuple(range(model.n_items)))

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
        if not model.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        if model.n_factors > 1:
            raise ValueError("EAPsum only supports unidimensional models")
        return self._build_lookup_table(model)

    def clear_cache(self) -> None:
        """Clear all cached lookup tables and model snapshots."""
        self._lookup_values.clear()
        self._grid = None
        self._item_tables = {}
        self._cached_model = None
        self._parameter_snapshot = {}
        self._structure_snapshot = None
        self._n_quadpts_snapshot = None
        self._prior_mean_snapshot = None
        self._prior_cov_snapshot = None

    def __repr__(self) -> str:
        return f"EAPSumScorer(n_quadpts={self.n_quadpts})"


def _native_sum_score_distribution(
    model: BaseItemModel,
    quad_points: NDArray[np.float64],
    item_indices: tuple[int, ...],
) -> NDArray[np.float64] | None:
    """Use the compiled log-space recursion for built-in 1PL/2PL curves."""
    from mirt._model_defaults import uses_builtin_model_hooks
    from mirt.backends.rust.eapsum import lord_wingersky_recursion
    from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

    if (
        not item_indices
        or type(model) not in (OneParameterLogistic, TwoParameterLogistic)
        or not uses_builtin_model_hooks(model)
    ):
        return None
    params = model.parameters
    discrimination = params.get("discrimination", np.ones(model.n_items))
    if discrimination.ndim != 1:
        return None
    items = list(item_indices)
    return lord_wingersky_recursion(
        quad_points[:, 0] if quad_points.ndim > 1 else quad_points,
        discrimination[items],
        params["difficulty"][items],
    )


def _probability_sum_score_distribution(
    tables: Sequence[NDArray[np.float64]],
    n_quad: int,
) -> NDArray[np.float64]:
    """Lord-Wingersky recursion in probability space, returned as logs.

    ``tables`` holds one ``(n_categories, n_quad)`` probability table per
    item. Each column remains a probability distribution over sum scores, so
    its largest entry never falls below ``1 / (max_score + 1)``; only scores
    with negligible probability at a node can underflow.
    """
    width = 1 + sum(table.shape[0] - 1 for table in tables)
    current = np.zeros((width, n_quad), dtype=np.float64)
    updated = np.empty_like(current)
    work = np.empty_like(current)
    current[0] = 1.0
    current_width = 1
    for table in tables:
        next_width = current_width + table.shape[0] - 1
        conditional = current[:current_width]
        buffer = work[:current_width]
        np.multiply(conditional, table[0], out=updated[:current_width])
        updated[current_width:next_width] = 0.0
        for score in range(1, table.shape[0]):
            np.multiply(conditional, table[score], out=buffer)
            target = updated[score : score + current_width]
            np.add(target, buffer, out=target)
        current, updated = updated, current
        current_width = next_width

    with np.errstate(divide="ignore"):
        return np.log(current[:current_width])


def _log_space_sum_score_distribution(
    tables: Sequence[NDArray[np.float64]],
    n_quad: int,
) -> NDArray[np.float64]:
    """Lord-Wingersky recursion in log space, robust to extreme scores."""
    log_dist = np.zeros((1, n_quad), dtype=np.float64)
    for table in tables:
        log_probs = np.log(table)
        previous_scores = log_dist.shape[0]
        new_log_dist = np.full(
            (previous_scores + table.shape[0] - 1, n_quad),
            -np.inf,
            dtype=np.float64,
        )
        for category, category_log_probs in enumerate(log_probs):
            target = new_log_dist[category : category + previous_scores]
            np.logaddexp(target, log_dist + category_log_probs, out=target)
        log_dist = new_log_dist
    return log_dist


def _resolved_scorer(
    model_or_result: BaseItemModel | FitResult,
    n_quadpts: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> tuple[BaseItemModel, EAPSumScorer]:
    """Unwrap a fit result and default the prior to its latent covariance."""
    from mirt.results._common import resolve_latent_prior

    model, mean, cov = resolve_latent_prior(model_or_result, prior_mean, prior_cov)
    return model, EAPSumScorer(n_quadpts=n_quadpts, prior_mean=mean, prior_cov=cov)


def _validated_sum_scores(
    sum_scores: ArrayLike,
    max_score: int,
) -> NDArray[np.intp]:
    """Return sum scores as indices, rejecting values outside the score range."""
    values = np.atleast_1d(np.asarray(sum_scores))
    if values.dtype.kind not in "iuf":
        raise MirtValidationError(
            "sum_scores must contain integer sum scores",
            parameter="sum_scores",
            value=str(values.dtype),
            expected="integer values",
        )
    if values.dtype.kind == "f" and not np.all(
        np.isfinite(values) & (values == np.trunc(values))
    ):
        raise MirtValidationError(
            "sum_scores must contain finite integer values",
            parameter="sum_scores",
            expected="integer values",
        )
    if np.any(values < 0) or np.any(values > max_score):
        raise MirtValidationError(
            f"sum_scores must lie between 0 and {max_score}",
            parameter="sum_scores",
            expected=f"0 <= sum score <= {max_score}",
        )
    return values.astype(np.intp)


def eapsum(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
) -> ScoreResult:
    """Convenience function for EAPsum scoring.

    Parameters
    ----------
    model : BaseItemModel | FitResult
        Fitted IRT model, or the ``FitResult`` returned by
        :func:`mirt.fit_mirt`.
    responses : ndarray
        Response matrix (n_persons x n_items)
    n_quadpts : int
        Number of quadrature points
    prior_mean : ndarray, optional
        Prior mean. Default zero.
    prior_cov : ndarray, optional
        Prior variance, as a ``(1, 1)`` matrix. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to one
        otherwise.

    Returns
    -------
    ScoreResult
        Scoring results
    """
    item_model, scorer = _resolved_scorer(model, n_quadpts, prior_mean, prior_cov)
    return scorer.score(item_model, responses)


def eapsum_table(
    model_or_result: BaseItemModel | FitResult,
    responses: ArrayLike | None = None,
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
) -> SumScoreTable:
    """Tabulate EAPsum theta estimates for every attainable sum score.

    Parameters
    ----------
    model_or_result : BaseItemModel | FitResult
        Fitted unidimensional IRT model, or the ``FitResult`` returned by
        :func:`mirt.fit_mirt`.
    responses : array-like of shape (n_persons, n_items), optional
        Complete response matrix used for observed and expected sum-score
        counts and standardized residuals.
    n_quadpts : int, default=49
        Number of quadrature points.
    prior_mean : ndarray, optional
        Prior mean for theta. Default zero.
    prior_cov : ndarray, optional
        Prior variance for theta, as a ``(1, 1)`` matrix. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to one
        otherwise.

    Returns
    -------
    SumScoreTable
        Sum scores with their theta estimates, standard errors and
        model-implied proportions, plus frequencies when responses are given.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.scoring import eapsum_table
    >>> result = fit_mirt(data, model="2PL")
    >>> table = eapsum_table(result, data)
    >>> table.to_dataframe()
    """
    model, scorer = _resolved_scorer(model_or_result, n_quadpts, prior_mean, prior_cov)
    return scorer.score_table(model, responses)


def sum_score_to_theta(
    model: BaseItemModel | FitResult,
    sum_scores: ArrayLike,
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert sum scores directly to theta estimates.

    Useful for quick conversions without full response data.

    Parameters
    ----------
    model : BaseItemModel | FitResult
        Fitted unidimensional IRT model, or the ``FitResult`` returned by
        :func:`mirt.fit_mirt`.
    sum_scores : array-like
        Full-form sum scores to convert. Integral floats such as ``3.0`` are
        accepted.
    n_quadpts : int
        Number of quadrature points
    prior_mean : ndarray, optional
        Prior mean for theta. Default zero.
    prior_cov : ndarray, optional
        Prior variance for theta, as a ``(1, 1)`` matrix. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to one
        otherwise.

    Returns
    -------
    theta : ndarray
        Theta estimates for each sum score, with the shape of ``sum_scores``
        (at least one-dimensional).
    se : ndarray
        Standard errors for each estimate

    Raises
    ------
    MirtValidationError
        If a sum score is not a finite integer between 0 and the maximum
        attainable sum score.
    """
    item_model, scorer = _resolved_scorer(model, n_quadpts, prior_mean, prior_cov)
    theta_values, se_values, _ = scorer._full_form_arrays(item_model)
    indices = _validated_sum_scores(sum_scores, theta_values.shape[0] - 1)
    return theta_values[indices], se_values[indices]
