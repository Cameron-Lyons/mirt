"""Person-score result container and uncertainty helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.results._common import normal_critical_value


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
            "person_ids": None if self.person_ids is None else list(self.person_ids),
        }

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
