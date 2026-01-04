"""Result container for model fitting."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from mirt.models.base import BaseItemModel


@dataclass
class FitResult:
    """Container for IRT model fitting results.

    This class holds the results of fitting an IRT model, including
    estimated parameters, standard errors, and fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        The fitted IRT model.
    log_likelihood : float
        Final log-likelihood value.
    n_iterations : int
        Number of iterations until convergence.
    converged : bool
        Whether the algorithm converged.
    standard_errors : dict
        Standard errors for each parameter.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    n_observations : int, optional
        Number of observations (persons).
    n_parameters : int, optional
        Number of free parameters.

    Examples
    --------
    >>> result = estimator.fit(model, responses)
    >>> print(result.summary())
    >>> params = result.coef()
    """

    model: Any  # BaseItemModel
    log_likelihood: float
    n_iterations: int
    converged: bool
    standard_errors: dict[str, NDArray[np.float64]]
    aic: float
    bic: float
    n_observations: int = 0
    n_parameters: int = 0

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a formatted summary of the results.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        str
            Formatted summary string.
        """
        from scipy import stats

        lines = []
        width = 80

        # Header
        lines.append("=" * width)
        lines.append(f"{'IRT Model Results':^{width}}")
        lines.append("=" * width)

        # Model info
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

        # Parameter tables
        z_crit = stats.norm.ppf(1 - alpha / 2)

        for param_name, values in self.model.parameters.items():
            lines.append(f"\n{param_name}:")

            se = self.standard_errors.get(param_name, np.zeros_like(values))

            # Header row
            ci_label = f"[{(1-alpha)*100:.0f}%"
            lines.append(
                f"{'Item':<15} {'Estimate':>10} {'Std.Err':>10} "
                f"{'z-value':>10} {'P>|z|':>10} "
                f"{ci_label:>8} {'CI]':>8}"
            )
            lines.append("-" * width)

            # Handle 1D and 2D parameter arrays
            if values.ndim == 1:
                for i in range(len(values)):
                    est = values[i]
                    err = se[i] if i < len(se) else 0.0

                    if err > 0 and not np.isnan(err):
                        z = est / err
                        p = 2 * (1 - stats.norm.cdf(abs(z)))
                        ci_low = est - z_crit * err
                        ci_high = est + z_crit * err
                    else:
                        z = np.nan
                        p = np.nan
                        ci_low = np.nan
                        ci_high = np.nan

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
                # 2D parameters (e.g., multidimensional discrimination)
                for i in range(values.shape[0]):
                    item_name = (
                        self.model.item_names[i]
                        if i < len(self.model.item_names)
                        else f"Item_{i}"
                    )

                    for j in range(values.shape[1]):
                        est = values[i, j]
                        err = se[i, j] if i < se.shape[0] and j < se.shape[1] else 0.0

                        if err > 0 and not np.isnan(err):
                            z = est / err
                            p = 2 * (1 - stats.norm.cdf(abs(z)))
                            ci_low = est - z_crit * err
                            ci_high = est + z_crit * err
                        else:
                            z = np.nan
                            p = np.nan
                            ci_low = np.nan
                            ci_high = np.nan

                        label = f"{item_name}[{j}]"
                        lines.append(
                            f"{label:<15} {est:>10.4f} {err:>10.4f} "
                            f"{z:>10.3f} {p:>10.4f} "
                            f"{ci_low:>8.4f} {ci_high:>8.4f}"
                        )

        lines.append("=" * width)
        return "\n".join(lines)

    def coef(self) -> "pd.DataFrame":
        """Return item parameters as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with item parameters as columns.
        """
        import pandas as pd

        data: dict[str, Any] = {}

        for param_name, values in self.model.parameters.items():
            if values.ndim == 1:
                data[param_name] = values
            else:
                # Multidimensional: split into columns
                for j in range(values.shape[1]):
                    data[f"{param_name}_{j+1}"] = values[:, j]

        df = pd.DataFrame(data)
        df.index = self.model.item_names[: len(df)]
        df.index.name = "item"

        return df

    def coef_with_se(self) -> "pd.DataFrame":
        """Return item parameters with standard errors as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with parameter estimates and their SEs.
        """
        import pandas as pd

        data: dict[str, Any] = {}

        for param_name, values in self.model.parameters.items():
            se = self.standard_errors.get(param_name, np.zeros_like(values))

            if values.ndim == 1:
                data[param_name] = values
                data[f"{param_name}_se"] = se
            else:
                for j in range(values.shape[1]):
                    data[f"{param_name}_{j+1}"] = values[:, j]
                    if se.ndim > 1 and j < se.shape[1]:
                        data[f"{param_name}_{j+1}_se"] = se[:, j]

        df = pd.DataFrame(data)
        df.index = self.model.item_names[: len(df)]
        df.index.name = "item"

        return df

    def fit_statistics(self) -> dict[str, float]:
        """Return fit statistics as a dictionary.

        Returns
        -------
        dict
            Dictionary containing AIC, BIC, log-likelihood, etc.
        """
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
