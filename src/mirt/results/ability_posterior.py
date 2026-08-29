"""Posterior ability distributions and exact grid-based summaries."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError
from mirt.results.score_result import ScoreResult

_HDI_TARGET_ELEMENTS = 500_000


@dataclass(slots=True)
class AbilityPosteriorResult:
    """Normalized ability probabilities evaluated on a quadrature grid.

    The same grid is shared by every respondent. ``weights`` contains one
    normalized probability distribution per row, while
    ``log_marginal_likelihood`` records each response pattern's integrated
    log likelihood under the configured prior.
    """

    points: NDArray[np.float64]
    weights: NDArray[np.float64]
    log_marginal_likelihood: NDArray[np.float64]
    person_ids: list[Any] | NDArray[Any] | None = None
    method: str = "EAP"

    @staticmethod
    def _validated_person_ids(
        person_ids: list[Any] | NDArray[Any] | None,
        n_persons: int,
    ) -> list[Any] | None:
        """Copy and validate optional respondent identifiers."""
        if person_ids is None:
            return None
        if isinstance(person_ids, np.ndarray) and person_ids.ndim != 1:
            raise MirtValidationError(
                "person_ids must be one-dimensional",
                parameter="person_ids",
                value=person_ids.shape,
                expected=f"({n_persons},)",
            )
        resolved = list(person_ids)
        if len(resolved) != n_persons:
            raise MirtValidationError(
                "person_ids must contain one identifier per posterior row",
                parameter="person_ids",
                value=len(resolved),
                expected=str(n_persons),
            )
        return resolved

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim == 1:
            points = points[:, None]
        weights = np.asarray(self.weights, dtype=np.float64)
        log_marginal = np.asarray(
            self.log_marginal_likelihood,
            dtype=np.float64,
        )

        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise MirtValidationError(
                "points must be a non-empty one- or two-dimensional array",
                parameter="points",
                value=points.shape,
                expected="(n_points,) or (n_points, n_factors)",
            )
        if not np.all(np.isfinite(points)):
            raise MirtValidationError(
                "points must contain only finite values",
                parameter="points",
                expected="finite quadrature coordinates",
            )
        if weights.ndim != 2 or weights.shape[1] != points.shape[0]:
            raise MirtValidationError(
                "weights must contain one value per respondent and grid point",
                parameter="weights",
                value=weights.shape,
                expected=f"(n_persons, {points.shape[0]})",
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise MirtValidationError(
                "weights must contain finite non-negative values",
                parameter="weights",
                expected="finite values >= 0",
            )
        row_sums = np.sum(weights, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            raise MirtValidationError(
                "each row of weights must sum to one",
                parameter="weights",
                value=row_sums,
                expected="row sums equal to 1",
            )
        if log_marginal.shape != (weights.shape[0],) or not np.all(
            np.isfinite(log_marginal)
        ):
            raise MirtValidationError(
                "log_marginal_likelihood must contain one finite value per respondent",
                parameter="log_marginal_likelihood",
                value=log_marginal.shape,
                expected=f"({weights.shape[0]},)",
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise MirtValidationError(
                "method must be a non-empty string",
                parameter="method",
                value=self.method,
                expected="non-empty string",
            )

        person_ids = self._validated_person_ids(self.person_ids, weights.shape[0])

        self.points = points.copy()
        self.weights = weights.copy()
        self.log_marginal_likelihood = log_marginal.copy()
        self.person_ids = person_ids
        self.method = self.method.strip()

    @classmethod
    def _from_owned_arrays(
        cls,
        *,
        points: NDArray[np.float64],
        weights: NDArray[np.float64],
        log_marginal_likelihood: NDArray[np.float64],
        person_ids: list[Any] | NDArray[Any] | None,
    ) -> Self:
        """Adopt validated internal arrays without doubling the output memory."""
        result = cls.__new__(cls)
        result.points = points
        result.weights = weights
        result.log_marginal_likelihood = log_marginal_likelihood
        result.person_ids = cls._validated_person_ids(person_ids, weights.shape[0])
        result.method = "EAP"
        return result

    @property
    def n_persons(self) -> int:
        """Number of posterior distributions."""
        return self.weights.shape[0]

    @property
    def n_points(self) -> int:
        """Number of points in the shared quadrature grid."""
        return self.points.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of latent factors represented by the grid."""
        return self.points.shape[1]

    def _restore_score_shape(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one-dimensional output for a unidimensional posterior."""
        if self.n_factors == 1:
            return values.ravel()
        return values

    @property
    def mean(self) -> NDArray[np.float64]:
        """Posterior mean ability for every respondent."""
        values = self.weights @ self.points
        return self._restore_score_shape(values)

    @property
    def standard_error(self) -> NDArray[np.float64]:
        """Posterior standard deviation for every respondent and factor."""
        center = np.mean(self.points, axis=0)
        centered_points = self.points - center
        centered_mean = self.weights @ centered_points
        variance = self.weights @ (centered_points**2) - centered_mean**2
        np.maximum(variance, 0.0, out=variance)
        return self._restore_score_shape(np.sqrt(variance))

    @property
    def map_estimate(self) -> NDArray[np.float64]:
        """Highest-probability grid point for every respondent."""
        indices = np.argmax(self.weights, axis=1)
        return self._restore_score_shape(self.points[indices])

    @property
    def entropy(self) -> NDArray[np.float64]:
        """Shannon entropy of each normalized grid distribution."""
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                self.weights > 0.0,
                self.weights * np.log(self.weights),
                0.0,
            )
        return -np.sum(terms, axis=1)

    @staticmethod
    def _validate_probability(value: float, *, parameter: str) -> float:
        """Validate a scalar probability strictly between zero and one."""
        if isinstance(value, bool) or not isinstance(value, Real):
            resolved = np.nan
        else:
            resolved = float(value)
        if not np.isfinite(resolved) or not 0.0 < resolved < 1.0:
            raise MirtValidationError(
                f"{parameter} must be a finite number strictly between 0 and 1",
                parameter=parameter,
                value=value,
                expected=f"0 < {parameter} < 1",
            )
        return resolved

    def _quantile(self, probability: float) -> NDArray[np.float64]:
        """Evaluate one marginal weighted quantile per respondent and factor."""
        quantiles = np.empty((self.n_persons, self.n_factors), dtype=np.float64)
        for factor in range(self.n_factors):
            order = np.argsort(self.points[:, factor], kind="stable")
            cumulative = np.cumsum(self.weights[:, order], axis=1)
            indices = np.sum(cumulative < probability, axis=1)
            np.minimum(indices, self.n_points - 1, out=indices)
            quantiles[:, factor] = self.points[order[indices], factor]
        return self._restore_score_shape(quantiles)

    def quantile(self, probability: float = 0.5) -> NDArray[np.float64]:
        """Return one marginal posterior quantile per respondent and factor."""
        resolved = self._validate_probability(
            probability,
            parameter="probability",
        )
        return self._quantile(resolved)

    @property
    def median(self) -> NDArray[np.float64]:
        """Marginal posterior median for every respondent and factor."""
        return self._quantile(0.5)

    def credible_intervals(
        self,
        level: float = 0.95,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return marginal equal-tail credible intervals from the exact grid."""
        resolved_level = self._validate_probability(level, parameter="level")
        tail = (1.0 - resolved_level) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    @staticmethod
    def _row_searchsorted(
        cumulative: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.intp]:
        """Search independent row CDFs without a three-dimensional array."""
        n_rows, n_boundaries = cumulative.shape
        row_offsets = (2.0 * np.arange(n_rows, dtype=np.float64))[:, None]
        shifted_cumulative = (cumulative + row_offsets).ravel()
        shifted_targets = (targets + row_offsets).ravel()
        flat_indices = np.searchsorted(
            shifted_cumulative,
            shifted_targets,
            side="left",
        )
        row_starts = (np.arange(n_rows, dtype=np.intp) * n_boundaries)[:, None]
        return flat_indices.reshape(targets.shape) - row_starts

    @classmethod
    def _shortest_mass_intervals(
        cls,
        coordinates: NDArray[np.float64],
        weights: NDArray[np.float64],
        level: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return shortest contiguous intervals for one marginal grid."""
        normalized = weights / np.sum(weights, axis=1, keepdims=True)
        cumulative = np.empty(
            (weights.shape[0], coordinates.size + 1),
            dtype=np.float64,
        )
        cumulative[:, 0] = 0.0
        np.cumsum(normalized, axis=1, out=cumulative[:, 1:])
        cumulative[:, -1] = 1.0

        targets = cumulative[:, :-1] + level
        end_boundaries = cls._row_searchsorted(cumulative, targets)
        valid = (targets <= 1.0) & (end_boundaries <= coordinates.size)
        safe_end_boundaries = np.clip(end_boundaries, 1, coordinates.size)

        interval_widths = coordinates[safe_end_boundaries - 1] - coordinates[None, :]
        interval_widths[~valid] = np.inf
        minimum_width = np.min(interval_widths, axis=1)
        coordinate_scale = max(1.0, float(np.ptp(coordinates)))
        width_tolerance = 16.0 * np.finfo(np.float64).eps * coordinate_scale
        shortest = interval_widths <= minimum_width[:, None] + width_tolerance

        rows = np.arange(weights.shape[0], dtype=np.intp)[:, None]
        enclosed_mass = cumulative[rows, safe_end_boundaries] - cumulative[:, :-1]
        candidate_mass = np.where(shortest, enclosed_mass, -np.inf)
        greatest_mass = np.max(candidate_mass, axis=1)
        mass_tolerance = 16.0 * np.finfo(np.float64).eps
        preferred = shortest & (
            enclosed_mass >= greatest_mass[:, None] - mass_tolerance
        )
        best_start = np.argmax(preferred, axis=1)
        best_end = (
            safe_end_boundaries[
                np.arange(weights.shape[0], dtype=np.intp),
                best_start,
            ]
            - 1
        )
        return coordinates[best_start], coordinates[best_end]

    def highest_density_intervals(
        self,
        level: float = 0.95,
        *,
        batch_size: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return shortest contiguous marginal posterior intervals.

        The result contains at least ``level`` posterior mass on each factor's
        discrete marginal grid. Duplicate coordinates are combined before the
        interval search. When several intervals have the same minimum width,
        the interval with greater enclosed mass is preferred, followed by the
        lower interval.

        Parameters
        ----------
        level : float, default=0.95
            Finite posterior probability strictly between zero and one.
        batch_size : int, optional
            Maximum respondents processed together. The default bounds
            temporary person-by-grid arrays automatically.

        Returns
        -------
        lower, upper : tuple of ndarray
            Marginal interval bounds with the standard score shape.
        """
        resolved_level = self._validate_probability(level, parameter="level")
        if batch_size is not None and (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise MirtValidationError(
                "batch_size must be a positive integer or None",
                parameter="batch_size",
                value=batch_size,
                expected="positive integer or None",
            )

        lower = np.empty((self.n_persons, self.n_factors), dtype=np.float64)
        upper = np.empty_like(lower)
        for factor in range(self.n_factors):
            order = np.argsort(self.points[:, factor], kind="stable")
            sorted_coordinates = self.points[order, factor]
            coordinates, group_starts = np.unique(
                sorted_coordinates,
                return_index=True,
            )
            resolved_batch_size = (
                max(1, _HDI_TARGET_ELEMENTS // coordinates.size)
                if batch_size is None
                else int(batch_size)
            )
            for start in range(0, self.n_persons, resolved_batch_size):
                stop = min(start + resolved_batch_size, self.n_persons)
                sorted_weights = self.weights[start:stop, order]
                marginal_weights = np.add.reduceat(
                    sorted_weights,
                    group_starts,
                    axis=1,
                )
                factor_lower, factor_upper = self._shortest_mass_intervals(
                    coordinates,
                    marginal_weights,
                    resolved_level,
                )
                lower[start:stop, factor] = factor_lower
                upper[start:stop, factor] = factor_upper

        return self._restore_score_shape(lower), self._restore_score_shape(upper)

    def _broadcast_cut_scores(self, cut_score: ArrayLike) -> NDArray[np.float64]:
        """Validate cut scores and broadcast them to the posterior mean shape."""
        shape = (
            (self.n_persons,)
            if self.n_factors == 1
            else (self.n_persons, self.n_factors)
        )
        raw_cuts = np.asarray(cut_score)
        if raw_cuts.dtype.kind not in {"i", "u", "f"}:
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {shape}",
            )
        try:
            cuts = np.asarray(cut_score, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {shape}",
            ) from exc
        if not np.all(np.isfinite(cuts)):
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {shape}",
            )
        try:
            return np.broadcast_to(cuts, shape)
        except ValueError as exc:
            raise MirtValidationError(
                "cut_score must be broadcastable to the score shape",
                parameter="cut_score",
                value=cuts.shape,
                expected=str(shape),
            ) from exc

    def classification_probabilities(
        self,
        cut_score: ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """Return posterior probabilities that ability exceeds each cut."""
        cuts = self._broadcast_cut_scores(cut_score)
        cuts_2d = cuts[:, None] if self.n_factors == 1 else cuts
        probabilities = np.empty(
            (self.n_persons, self.n_factors),
            dtype=np.float64,
        )
        for factor in range(self.n_factors):
            above = self.points[:, factor][None, :] > cuts_2d[:, factor, None]
            probabilities[:, factor] = np.sum(self.weights * above, axis=1)
        return self._restore_score_shape(probabilities)

    def classify(
        self,
        cut_score: ArrayLike = 0.0,
        *,
        confidence: float = 0.95,
    ) -> NDArray[np.str_]:
        """Classify abilities above or below cuts using posterior probability."""
        resolved = self._validate_probability(confidence, parameter="confidence")
        if resolved <= 0.5:
            raise MirtValidationError(
                "confidence must be greater than 0.5",
                parameter="confidence",
                value=confidence,
                expected="0.5 < confidence < 1",
            )
        probabilities = self.classification_probabilities(cut_score)
        classifications = np.full(probabilities.shape, "uncertain", dtype="U9")
        classifications[probabilities >= resolved] = "above"
        classifications[probabilities <= 1.0 - resolved] = "below"
        return classifications

    def to_score_result(self) -> ScoreResult:
        """Return posterior moments in the standard person-score container."""
        return ScoreResult(
            theta=self.mean,
            standard_error=self.standard_error,
            method=self.method,
            person_ids=self.person_ids,
        )

    def __repr__(self) -> str:
        return (
            f"AbilityPosteriorResult(n_persons={self.n_persons}, "
            f"n_points={self.n_points}, n_factors={self.n_factors})"
        )
