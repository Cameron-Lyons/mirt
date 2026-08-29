"""Fitted-model result container and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError
from mirt.results._common import normal_critical_value, validate_alpha

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

ParameterStatistics = dict[str, dict[str, NDArray[np.float64]]]
FitStatistics = dict[str, float | int | bool]


def _compute_z_stats(
    est: float,
    err: float,
    z_crit: float,
) -> tuple[float, float, float, float]:
    """Compute a stable z-value, p-value, and confidence interval."""
    if err > 0 and np.isfinite(err) and np.isfinite(est):
        from scipy import special

        z = est / err
        p = float(2.0 * special.ndtr(-abs(z)))
        ci_low = est - z_crit * err
        ci_high = est + z_crit * err
        return z, p, ci_low, ci_high
    return np.nan, np.nan, np.nan, np.nan


def _array_statistics(
    estimates: NDArray[np.float64],
    errors: NDArray[np.float64],
    z_crit: float,
) -> dict[str, NDArray[np.float64]]:
    """Vectorize normal-approximation inference for one parameter array."""
    from scipy import special

    valid = np.isfinite(estimates) & np.isfinite(errors) & (errors > 0.0)
    z_values = np.full(estimates.shape, np.nan, dtype=np.float64)
    np.divide(estimates, errors, out=z_values, where=valid)

    p_values = np.full(estimates.shape, np.nan, dtype=np.float64)
    p_values[valid] = 2.0 * special.ndtr(-np.abs(z_values[valid]))

    ci_lower = np.full(estimates.shape, np.nan, dtype=np.float64)
    ci_upper = np.full(estimates.shape, np.nan, dtype=np.float64)
    ci_lower[valid] = estimates[valid] - z_crit * errors[valid]
    ci_upper[valid] = estimates[valid] + z_crit * errors[valid]
    return {
        "estimate": estimates.copy(),
        "standard_error": errors.copy(),
        "z": z_values,
        "p_value": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


@dataclass
class FitResult:
    """Results from fitting an item-response model.

    Result metadata and standard-error arrays are validated when the object is
    created. Missing standard errors remain unknown (``NaN``) instead of being
    reported as exact zeros.
    """

    model: BaseItemModel
    log_likelihood: float
    n_iterations: int
    converged: bool
    standard_errors: dict[str, NDArray[np.float64]]
    aic: float
    bic: float
    n_observations: int = 0
    n_parameters: int = 0

    def __post_init__(self) -> None:
        for name in ("n_iterations", "n_observations", "n_parameters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise MirtValidationError(
                    f"{name} must be a non-negative integer",
                    parameter=name,
                    value=value,
                    expected=">= 0",
                )
            setattr(self, name, int(value))

        if not isinstance(self.converged, (bool, np.bool_)):
            raise MirtValidationError(
                "converged must be a boolean",
                parameter="converged",
                value=self.converged,
                expected="bool",
            )
        self.converged = bool(self.converged)
        self.log_likelihood = float(self.log_likelihood)
        self.aic = float(self.aic)
        self.bic = float(self.bic)

        parameters = self.model.parameters
        normalized_errors: dict[str, NDArray[np.float64]] = {}
        for name, values in self.standard_errors.items():
            errors = np.asarray(values, dtype=np.float64)
            if name in parameters and errors.shape != parameters[name].shape:
                raise MirtValidationError(
                    f"standard errors for {name!r} must match its parameter shape",
                    parameter="standard_errors",
                    value=errors.shape,
                    expected=str(parameters[name].shape),
                )
            if np.any(errors < 0.0):
                raise MirtValidationError(
                    f"standard errors for {name!r} cannot be negative",
                    parameter="standard_errors",
                    expected=">= 0, NaN, or infinity",
                )
            normalized_errors[name] = errors.copy()
        self.standard_errors = normalized_errors

    def _errors_for(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        errors = self.standard_errors.get(name)
        if errors is None:
            return np.full(values.shape, np.nan, dtype=np.float64)
        if errors.shape != values.shape:
            raise MirtValidationError(
                f"standard errors for {name!r} must match its parameter shape",
                parameter="standard_errors",
                value=errors.shape,
                expected=str(values.shape),
            )
        return errors

    def parameter_statistics(self, alpha: float = 0.05) -> ParameterStatistics:
        """Return vectorized estimates, uncertainty, tests, and intervals.

        Parameters
        ----------
        alpha : float
            Two-sided significance level strictly between 0 and 1.

        Returns
        -------
        dict
            A mapping from parameter names to arrays named ``estimate``,
            ``standard_error``, ``z``, ``p_value``, ``ci_lower``, and
            ``ci_upper``. Array shapes match the model parameter shapes.
        """
        validated_alpha = validate_alpha(alpha)
        z_crit = normal_critical_value(validated_alpha)
        result: ParameterStatistics = {}
        for name, raw_values in self.model.parameters.items():
            values = np.asarray(raw_values, dtype=np.float64)
            errors = self._errors_for(name, values)
            result[name] = _array_statistics(values, errors, z_crit)
        return result

    def confidence_intervals(
        self,
        alpha: float = 0.05,
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Return lower and upper normal-approximation parameter intervals."""
        statistics = self.parameter_statistics(alpha)
        return {
            name: (values["ci_lower"].copy(), values["ci_upper"].copy())
            for name, values in statistics.items()
        }

    def _parameter_label(
        self,
        parameter_name: str,
        shape: tuple[int, ...],
        index: tuple[int, ...],
    ) -> str:
        if shape and shape[0] == self.model.n_items:
            item_index = index[0]
            item_name = self.model.item_names[item_index]
            if len(index) == 1:
                return item_name
            suffix = ",".join(str(value) for value in index[1:])
            return f"{item_name}[{suffix}]"
        if not index:
            return parameter_name
        suffix = ",".join(str(value) for value in index)
        return f"{parameter_name}[{suffix}]"

    def summary(self, alpha: float = 0.05) -> str:
        """Format model fit and parameter inference as a text table."""
        validated_alpha = validate_alpha(alpha)
        parameter_statistics = self.parameter_statistics(validated_alpha)
        lines: list[str] = []
        width = 80

        lines.append("=" * width)
        lines.append(f"{'IRT Model Results':^{width}}")
        lines.append("=" * width)
        lines.append(
            f"Model:              {self.model.model_name:<20} "
            f"Log-Likelihood:    {self.log_likelihood:>12.4f}"
        )
        lines.append(
            f"No. Items:          {self.model.n_items:<20} "
            f"AIC:               {self.aic:>12.4f}"
        )
        lines.append(
            f"No. Factors:        {self.model.n_factors:<20} "
            f"BIC:               {self.bic:>12.4f}"
        )
        lines.append(
            f"No. Persons:        {self.n_observations:<20} "
            f"No. Parameters:    {self.n_parameters:>12}"
        )
        lines.append(
            f"Converged:          {str(self.converged):<20} "
            f"Iterations:        {self.n_iterations:>12}"
        )
        lines.append("-" * width)

        ci_label = f"[{(1.0 - validated_alpha) * 100:.0f}%"
        for parameter_name, values in parameter_statistics.items():
            lines.append(f"\n{parameter_name}:")
            lines.append(
                f"{'Item':<15} {'Estimate':>10} {'Std.Err':>10} "
                f"{'z-value':>10} {'P>|z|':>10} "
                f"{ci_label:>8} {'CI]':>8}"
            )
            lines.append("-" * width)

            estimates = values["estimate"]
            for index in np.ndindex(estimates.shape):
                label = self._parameter_label(parameter_name, estimates.shape, index)
                lines.append(
                    f"{label:<15} {estimates[index]:>10.4f} "
                    f"{values['standard_error'][index]:>10.4f} "
                    f"{values['z'][index]:>10.3f} "
                    f"{values['p_value'][index]:>10.4f} "
                    f"{values['ci_lower'][index]:>8.4f} "
                    f"{values['ci_upper'][index]:>8.4f}"
                )

        lines.append("=" * width)
        return "\n".join(lines)

    def _coefficient_columns(self, *, include_se: bool) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for parameter_name, raw_values in self.model.parameters.items():
            values = np.asarray(raw_values, dtype=np.float64)
            if values.ndim not in (1, 2) or values.shape[0] != self.model.n_items:
                raise MirtValidationError(
                    "wide coefficient output requires per-item parameter arrays; "
                    "use parameter_statistics() or to_dict() for global parameters",
                    parameter=parameter_name,
                    value=values.shape,
                    expected=f"first dimension {self.model.n_items}",
                )
            errors = self._errors_for(parameter_name, values)
            if values.ndim == 1:
                data[parameter_name] = values
                if include_se:
                    data[f"{parameter_name}_se"] = errors
                continue
            for column in range(values.shape[1]):
                column_name = f"{parameter_name}_{column + 1}"
                data[column_name] = values[:, column]
                if include_se:
                    data[f"{column_name}_se"] = errors[:, column]

        if not data:
            raise MirtValidationError("model does not expose any coefficient arrays")
        return data

    def coef(self) -> Any:
        """Return per-item coefficients using the configured dataframe backend."""
        from mirt.utils.dataframe import create_dataframe

        return create_dataframe(
            self._coefficient_columns(include_se=False),
            index=self.model.item_names,
            index_name="item",
        )

    def coef_with_se(self) -> Any:
        """Return per-item coefficients and standard errors as a dataframe."""
        from mirt.utils.dataframe import create_dataframe

        return create_dataframe(
            self._coefficient_columns(include_se=True),
            index=self.model.item_names,
            index_name="item",
        )

    def fit_statistics(self) -> FitStatistics:
        """Return scalar fit statistics and convergence metadata."""
        return {
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }

    def to_dict(
        self,
        *,
        include_parameters: bool = True,
        include_standard_errors: bool = True,
    ) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible result representation."""
        result: dict[str, Any] = {
            "model": {
                "name": self.model.model_name,
                "n_items": self.model.n_items,
                "n_factors": self.model.n_factors,
                "item_names": list(self.model.item_names),
            },
            **self.fit_statistics(),
        }
        if include_parameters:
            result["parameters"] = {
                name: values.tolist() for name, values in self.model.parameters.items()
            }
        if include_standard_errors:
            result["standard_errors"] = {
                name: values.tolist() for name, values in self.standard_errors.items()
            }
        return result

    def to_json(
        self,
        *,
        include_parameters: bool = True,
        include_standard_errors: bool = True,
        indent: int | None = None,
    ) -> str:
        """Serialize the portable fit representation to JSON."""
        import json

        return json.dumps(
            self.to_dict(
                include_parameters=include_parameters,
                include_standard_errors=include_standard_errors,
            ),
            indent=indent,
        )

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model.model_name}, "
            f"LL={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )
