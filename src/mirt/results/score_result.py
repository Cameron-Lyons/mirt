"""Person-score result container, uncertainty, and scale helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError
from mirt.results._common import normal_critical_value


def _portable_person_id(value: Any) -> Any:
    """Convert NumPy scalar identifiers to standard Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ScoreResult:
    """Person scores and their uncertainty estimates."""

    theta: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    method: str
    person_ids: list[Any] | NDArray[Any] | None = None

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64)
        standard_error = np.asarray(self.standard_error, dtype=np.float64)
        if theta.ndim not in (1, 2):
            raise MirtValidationError(
                "theta must be a one- or two-dimensional array",
                parameter="theta",
                value=theta.shape,
                expected="(n_persons,) or (n_persons, n_factors)",
            )
        if theta.ndim == 2 and theta.shape[1] == 0:
            raise MirtValidationError(
                "theta must contain at least one factor",
                parameter="theta",
                value=theta.shape,
                expected="n_factors > 0",
            )
        if standard_error.shape != theta.shape:
            raise MirtValidationError(
                "standard_error must have the same shape as theta",
                parameter="standard_error",
                value=standard_error.shape,
                expected=str(theta.shape),
            )
        if np.any(standard_error < 0.0):
            raise MirtValidationError(
                "standard_error cannot contain negative values",
                parameter="standard_error",
                expected=">= 0, NaN, or infinity",
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise MirtValidationError(
                "method must be a non-empty string",
                parameter="method",
                value=self.method,
                expected="non-empty string",
            )

        if self.person_ids is None:
            resolved_ids = None
        else:
            if isinstance(self.person_ids, np.ndarray) and self.person_ids.ndim != 1:
                raise MirtValidationError(
                    "person_ids must be one-dimensional",
                    parameter="person_ids",
                    value=self.person_ids.shape,
                    expected=f"({theta.shape[0]},)",
                )
            resolved_ids = list(self.person_ids)
            if len(resolved_ids) != theta.shape[0]:
                raise MirtValidationError(
                    "person_ids must contain one identifier per score row",
                    parameter="person_ids",
                    value=len(resolved_ids),
                    expected=str(theta.shape[0]),
                )

        self.theta = theta.copy()
        self.standard_error = standard_error.copy()
        self.method = self.method.strip()
        self.person_ids = resolved_ids

    @property
    def n_persons(self) -> int:
        """Number of scored persons."""
        return self.theta.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of latent factors represented by each score."""
        if self.theta.ndim == 1:
            return 1
        return self.theta.shape[1]

    def confidence_intervals(
        self,
        alpha: float = 0.05,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return lower and upper normal-approximation score intervals."""
        z_crit = normal_critical_value(alpha)
        margin = z_crit * self.standard_error
        return self.theta - margin, self.theta + margin

    def linear_transform(
        self,
        multiplier: ArrayLike = 1.0,
        offset: ArrayLike = 0.0,
    ) -> Self:
        """Return scores on a linearly transformed reporting scale.

        The transformed score is ``multiplier * theta + offset`` and its
        standard error is ``abs(multiplier) * standard_error``. Parameters may
        be scalars or one-dimensional arrays with one value per factor, so a
        multidimensional result can use a different reporting scale for each
        factor. A T-score scale, for example, uses ``multiplier=10`` and
        ``offset=50``.

        The returned result retains the scoring method and person identifiers.
        This result is not modified.
        """
        factors = self._factor_parameters(
            multiplier,
            name="multiplier",
            require_nonzero=True,
        )
        offsets = self._factor_parameters(offset, name="offset")
        return type(self)(
            theta=self.theta * factors + offsets,
            standard_error=self.standard_error * np.abs(factors),
            method=self.method,
            person_ids=self.person_ids,
        )

    def normal_percentile_ranks(
        self,
        reference_mean: ArrayLike = 0.0,
        reference_sd: ArrayLike = 1.0,
    ) -> NDArray[np.float64]:
        """Return percentile ranks against normal reference distributions.

        Reference parameters may be scalars or one-dimensional arrays with one
        value per factor. The returned values range from 0 to 100 and preserve
        the score array shape. Unknown scores remain ``NaN``.
        """
        means = self._factor_parameters(reference_mean, name="reference_mean")
        standard_deviations = self._factor_parameters(
            reference_sd,
            name="reference_sd",
            require_positive=True,
        )

        from scipy import special

        standardized = (self.theta - means) / standard_deviations
        return np.asarray(100.0 * special.ndtr(standardized), dtype=np.float64)

    def classification_probabilities(
        self,
        cut_score: ArrayLike = 0.0,
    ) -> NDArray[np.float64]:
        """Return normal-approximation probabilities of exceeding cut scores.

        ``cut_score`` may be a scalar or any finite array that broadcasts to the
        score shape. For multidimensional scores, a one-dimensional array with
        one cut per factor is therefore supported directly.

        Zero uncertainty yields probabilities of zero or one away from the cut
        and 0.5 exactly on it. Infinite uncertainty yields 0.5 for finite scores,
        while unknown scores or uncertainty values yield ``NaN``.
        """
        cuts = self._broadcast_cut_scores(cut_score)
        difference = self.theta - cuts
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = difference / self.standard_error

        from scipy import special

        probabilities = np.asarray(special.ndtr(standardized), dtype=np.float64)
        zero_uncertainty = self.standard_error == 0.0
        probabilities[zero_uncertainty & (difference < 0.0)] = 0.0
        probabilities[zero_uncertainty & (difference == 0.0)] = 0.5
        probabilities[zero_uncertainty & (difference > 0.0)] = 1.0
        return probabilities

    def classify(
        self,
        cut_score: ArrayLike = 0.0,
        *,
        confidence: float = 0.95,
    ) -> NDArray[np.str_]:
        """Classify scores above or below cuts when confidence is sufficient.

        Results are ``"above"``, ``"below"``, or ``"uncertain"``. A decision is
        made only when the corresponding normal-approximation probability meets
        ``confidence``. Unknown probabilities remain uncertain.
        """
        confidence_value = self._validate_classification_confidence(confidence)
        probabilities = self.classification_probabilities(cut_score)
        classifications = np.full(probabilities.shape, "uncertain", dtype="U9")
        classifications[probabilities >= confidence_value] = "above"
        classifications[probabilities <= 1.0 - confidence_value] = "below"
        return classifications

    def _broadcast_cut_scores(self, cut_score: ArrayLike) -> NDArray[np.float64]:
        """Validate cut scores and broadcast them to the stored score shape."""
        raw_cuts = np.asarray(cut_score)
        if raw_cuts.dtype.kind not in {"i", "u", "f"}:
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {self.theta.shape}",
            )
        try:
            cuts = np.asarray(cut_score, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {self.theta.shape}",
            ) from exc
        if not np.all(np.isfinite(cuts)):
            raise MirtValidationError(
                "cut_score must contain only finite numbers",
                parameter="cut_score",
                value=cut_score,
                expected=f"finite values broadcastable to {self.theta.shape}",
            )
        try:
            return np.broadcast_to(cuts, self.theta.shape)
        except ValueError as exc:
            raise MirtValidationError(
                "cut_score must be broadcastable to the score shape",
                parameter="cut_score",
                value=cuts.shape,
                expected=str(self.theta.shape),
            ) from exc

    def _factor_parameters(
        self,
        value: ArrayLike,
        *,
        name: str,
        require_nonzero: bool = False,
        require_positive: bool = False,
    ) -> NDArray[np.float64]:
        """Validate a scalar or one finite value per score factor."""
        expected = f"a finite scalar or an array with shape ({self.n_factors},)"
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=value,
                expected=expected,
            ) from exc
        if raw.dtype.kind not in {"i", "u", "f"}:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=value,
                expected=expected,
            )
        try:
            parameters = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=value,
                expected=expected,
            ) from exc
        if parameters.ndim == 0:
            pass
        elif parameters.shape != (self.n_factors,):
            raise MirtValidationError(
                f"{name} must provide one value per factor",
                parameter=name,
                value=parameters.shape,
                expected=expected,
            )
        if not np.all(np.isfinite(parameters)):
            raise MirtValidationError(
                f"{name} must contain only finite values",
                parameter=name,
                value=value,
                expected=expected,
            )
        if require_nonzero and np.any(parameters == 0.0):
            raise MirtValidationError(
                f"{name} must contain only nonzero values",
                parameter=name,
                value=value,
                expected="nonzero scale values",
            )
        if require_positive and np.any(parameters <= 0.0):
            raise MirtValidationError(
                f"{name} must contain only positive values",
                parameter=name,
                value=value,
                expected="> 0",
            )
        return parameters

    @staticmethod
    def _validate_classification_confidence(confidence: float) -> float:
        """Validate the confidence required for a score-side decision."""
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            value = np.nan
        else:
            value = float(confidence)
        if not np.isfinite(value) or not 0.5 < value < 1.0:
            raise MirtValidationError(
                "confidence must be a finite number strictly between 0.5 and 1",
                parameter="confidence",
                value=confidence,
                expected="0.5 < confidence < 1",
            )
        return value

    def to_dataframe(self) -> Any:
        """Return scores using the configured dataframe backend."""
        from mirt.utils.dataframe import create_dataframe

        data: dict[str, Any] = {}
        if self.n_factors == 1:
            data["theta"] = self.theta.ravel()
            data["se"] = self.standard_error.ravel()
        else:
            for factor in range(self.n_factors):
                data[f"theta_{factor + 1}"] = self.theta[:, factor]
                data[f"se_{factor + 1}"] = self.standard_error[:, factor]

        return create_dataframe(
            data,
            index=self.person_ids,
            index_name="person" if self.person_ids is not None else None,
        )

    def to_array(self, include_se: bool = False) -> NDArray[np.float64]:
        """Return a defensive score array, optionally followed by uncertainty."""
        if not include_se:
            return self.theta.copy()
        if self.n_factors == 1:
            return np.column_stack([self.theta.ravel(), self.standard_error.ravel()])
        return np.column_stack([self.theta, self.standard_error])

    def to_dict(self) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible score representation."""
        return {
            "method": self.method,
            "n_persons": self.n_persons,
            "n_factors": self.n_factors,
            "theta": self.theta.tolist(),
            "standard_error": self.standard_error.tolist(),
            "person_ids": None
            if self.person_ids is None
            else [_portable_person_id(value) for value in self.person_ids],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct scores from :meth:`to_dict` output.

        Derived shape metadata is optional, but when supplied it must agree
        with the reconstructed arrays. Unknown fields are rejected so that
        misspelled input does not silently disappear.
        """
        if not isinstance(payload, Mapping):
            raise MirtValidationError(
                "score payload must be a mapping",
                parameter="payload",
                value=type(payload).__name__,
                expected="mapping",
            )

        allowed = {
            "method",
            "n_persons",
            "n_factors",
            "theta",
            "standard_error",
            "person_ids",
        }
        unknown = set(payload) - allowed
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise MirtValidationError(
                f"score payload contains unknown fields: {names}",
                parameter="payload",
                value=names,
                expected="fields produced by ScoreResult.to_dict()",
            )

        required = {"method", "theta", "standard_error"}
        missing = required - set(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise MirtValidationError(
                f"score payload is missing required fields: {names}",
                parameter="payload",
                value=names,
                expected="method, theta, and standard_error",
            )

        result = cls(
            theta=payload["theta"],
            standard_error=payload["standard_error"],
            method=payload["method"],
            person_ids=payload.get("person_ids"),
        )
        for name, actual in (
            ("n_persons", result.n_persons),
            ("n_factors", result.n_factors),
        ):
            if name not in payload:
                continue
            value = payload[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) != actual
            ):
                raise MirtValidationError(
                    f"{name} does not match the reconstructed score shape",
                    parameter=name,
                    value=value,
                    expected=str(actual),
                )
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize scores to JSON without a dataframe dependency."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Reconstruct scores from :meth:`to_json` output."""
        import json

        if not isinstance(value, (str, bytes, bytearray)):
            raise MirtValidationError(
                "score JSON must be a string or bytes",
                parameter="value",
                value=type(value).__name__,
                expected="str, bytes, or bytearray",
            )
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MirtValidationError(
                "score JSON must contain a valid JSON object",
                parameter="value",
                expected="JSON object produced by ScoreResult.to_json()",
            ) from error
        return cls.from_dict(payload)

    def summary(self) -> str:
        """Format factor-wise score and uncertainty summaries."""
        theta = self.theta.reshape(self.n_persons, self.n_factors)
        errors = self.standard_error.reshape(self.n_persons, self.n_factors)
        lines = [
            "Score Results",
            "=" * 58,
            f"Method: {self.method}",
            f"Persons: {self.n_persons}",
            f"Factors: {self.n_factors}",
        ]
        if self.n_persons == 0:
            return "\n".join(lines)

        lines.extend(
            [
                "-" * 58,
                f"{'Factor':<10} {'Mean':>10} {'SD':>10} {'Mean SE':>10}",
                "-" * 58,
            ]
        )
        for factor in range(self.n_factors):
            factor_theta = theta[:, factor]
            factor_se = errors[:, factor]
            finite_theta = factor_theta[np.isfinite(factor_theta)]
            finite_se = factor_se[np.isfinite(factor_se)]
            mean = float(np.mean(finite_theta)) if finite_theta.size else np.nan
            sd = float(np.std(finite_theta)) if finite_theta.size else np.nan
            mean_se = float(np.mean(finite_se)) if finite_se.size else np.nan
            lines.append(f"{factor + 1:<10} {mean:>10.4f} {sd:>10.4f} {mean_se:>10.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ScoreResult(n_persons={self.n_persons}, "
            f"n_factors={self.n_factors}, "
            f"method='{self.method}')"
        )
