from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _compute_z_stats(
    est: float,
    err: float,
    z_crit: float,
) -> tuple[float, float, float, float]:
    """Compute z-value, p-value, and confidence interval."""
    if err > 0 and not np.isnan(err):
        z = est / err
        from scipy import stats

        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci_low = est - z_crit * err
        ci_high = est + z_crit * err
    else:
        z = np.nan
        p = np.nan
        ci_low = np.nan
        ci_high = np.nan
    return z, p, ci_low, ci_high


@dataclass
class FitResult:
    model: BaseItemModel
    log_likelihood: float
    n_iterations: int
    converged: bool
    standard_errors: dict[str, NDArray[np.float64]]
    aic: float
    bic: float
    n_observations: int = 0
    n_parameters: int = 0

    def summary(self, alpha: float = 0.05) -> str:
        from scipy import stats

        lines = []
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

        z_crit = stats.norm.ppf(1 - alpha / 2)

        for param_name, values in self.model.parameters.items():
            lines.append(f"\n{param_name}:")

            se = self.standard_errors.get(param_name, np.zeros_like(values))

            ci_label = f"[{(1 - alpha) * 100:.0f}%"
            lines.append(
                f"{'Item':<15} {'Estimate':>10} {'Std.Err':>10} "
                f"{'z-value':>10} {'P>|z|':>10} "
                f"{ci_label:>8} {'CI]':>8}"
            )
            lines.append("-" * width)

            if values.ndim == 1:
                for i in range(len(values)):
                    est = values[i]
                    err = se[i] if i < len(se) else 0.0
                    z, p, ci_low, ci_high = _compute_z_stats(est, err, z_crit)

                    item_name = (
                        self.model.item_names[i]
                        if i < len(self.model.item_names)
                        else f"Item_{i}"
                    )

                    lines.append(
                        f"{item_name:<15} {est:>10.4f} {err:>10.4f} "
                        f"{z:>10.3f} {p:>10.4f} "
                        f"{ci_low:>8.4f} {ci_high:>8.4f}"
                    )
            else:
                for i in range(values.shape[0]):
                    item_name = (
                        self.model.item_names[i]
                        if i < len(self.model.item_names)
                        else f"Item_{i}"
                    )

                    for j in range(values.shape[1]):
                        est = values[i, j]
                        err = se[i, j] if i < se.shape[0] and j < se.shape[1] else 0.0
                        z, p, ci_low, ci_high = _compute_z_stats(est, err, z_crit)

                        label = f"{item_name}[{j}]"
                        lines.append(
                            f"{label:<15} {est:>10.4f} {err:>10.4f} "
                            f"{z:>10.3f} {p:>10.4f} "
                            f"{ci_low:>8.4f} {ci_high:>8.4f}"
                        )

        lines.append("=" * width)
        return "\n".join(lines)

    def coef(self) -> Any:
        from mirt.utils.dataframe import create_dataframe

        data: dict[str, Any] = {}

        for param_name, values in self.model.parameters.items():
            if values.ndim == 1:
                data[param_name] = values
            else:
                for j in range(values.shape[1]):
                    data[f"{param_name}_{j + 1}"] = values[:, j]

        n_items = len(next(iter(data.values())))
        return create_dataframe(
            data, index=self.model.item_names[:n_items], index_name="item"
        )

    def coef_with_se(self) -> Any:
        from mirt.utils.dataframe import create_dataframe

        data: dict[str, Any] = {}

        for param_name, values in self.model.parameters.items():
            se = self.standard_errors.get(param_name, np.zeros_like(values))

            if values.ndim == 1:
                data[param_name] = values
                data[f"{param_name}_se"] = se
            else:
                for j in range(values.shape[1]):
                    data[f"{param_name}_{j + 1}"] = values[:, j]
                    if se.ndim > 1 and j < se.shape[1]:
                        data[f"{param_name}_{j + 1}_se"] = se[:, j]

        n_items = len(next(iter(data.values())))
        return create_dataframe(
            data, index=self.model.item_names[:n_items], index_name="item"
        )

    def fit_statistics(self) -> dict[str, float]:
        return {
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model.model_name}, "
            f"LL={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )
