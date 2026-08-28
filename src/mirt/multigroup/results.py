from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.model import MultigroupModel


@dataclass
class MultigroupFitResult:
    """Results from multigroup IRT model fitting.

    Attributes
    ----------
    model : MultigroupModel
        The fitted multigroup model.
    invariance : str
        Invariance level used (configural, metric, scalar, strict).
    log_likelihood : float
        Combined log-likelihood across all groups.
    n_iterations : int
        Number of EM iterations until convergence.
    converged : bool
        Whether the algorithm converged.
    group_log_likelihoods : list[float]
        Per-group log-likelihood values.
    group_n_observations : list[int]
        Sample size per group.
    latent_distributions : list[GroupLatentDistribution]
        Estimated latent distributions per group.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    n_parameters : int
        Total number of free parameters.
    n_observations : int
        Total sample size across groups.
    standard_errors : dict, optional
        Standard errors organized by param_name -> group_idx -> array.
    """

    model: MultigroupModel
    invariance: str
    log_likelihood: float
    n_iterations: int
    converged: bool
    group_log_likelihoods: list[float]
    group_n_observations: list[int]
    latent_distributions: list[GroupLatentDistribution]
    aic: float
    bic: float
    n_parameters: int
    n_observations: int
    standard_errors: dict[str, dict[int, NDArray[np.float64]]] = field(
        default_factory=dict
    )

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return self.model.n_groups

    @property
    def group_labels(self) -> list[str]:
        """Group labels."""
        return self.model.group_labels.copy()

    def _resolve_group(self, group: int | str) -> int:
        """Resolve a group label or validate a numeric group index."""
        if isinstance(group, str):
            try:
                return self.group_labels.index(group)
            except ValueError as exc:
                raise ValueError(f"Unknown group label: {group}") from exc
        if isinstance(group, (bool, np.bool_)) or not isinstance(
            group, (int, np.integer)
        ):
            raise TypeError("group must be an integer index or string label")
        group_idx = int(group)
        if group_idx < 0 or group_idx >= self.n_groups:
            raise IndexError(
                f"group index {group_idx} out of range [0, {self.n_groups})"
            )
        return group_idx

    @staticmethod
    def _item_parameter_components(
        name: str,
        values: NDArray[np.float64],
        item_idx: int,
        n_items: int,
    ) -> list[tuple[str, float]]:
        """Flatten one item's parameter block into stable table columns."""
        array = np.asarray(values, dtype=np.float64)
        item_values = (
            array[item_idx] if array.ndim and array.shape[0] == n_items else array
        )
        item_values = np.asarray(item_values, dtype=np.float64)
        if item_values.ndim == 0:
            return [(name, float(item_values))]
        return [
            (
                f"{name}_{'_'.join(str(index) for index in component)}",
                float(item_values[component]),
            )
            for component in np.ndindex(item_values.shape)
        ]

    def _coefficient_rows(
        self,
        group_idx: int,
        *,
        include_standard_errors: bool,
    ) -> list[dict[str, str | float]]:
        """Build flattened item rows for one group."""
        params = self.model.get_group_parameters(group_idx)
        rows: list[dict[str, str | float]] = []
        for item_idx in range(self.model.n_items):
            row: dict[str, str | float] = {}
            for param_name, values in params.items():
                components = self._item_parameter_components(
                    param_name,
                    values,
                    item_idx,
                    self.model.n_items,
                )
                row.update(components)
                if not include_standard_errors:
                    continue

                group_errors = self.standard_errors.get(param_name, {})
                errors = group_errors.get(group_idx)
                if errors is None:
                    row.update({f"{column}_se": np.nan for column, _ in components})
                    continue
                errors_array = np.asarray(errors, dtype=np.float64)
                if errors_array.shape != np.asarray(values).shape:
                    raise ValueError(
                        f"standard errors for {param_name} in group {group_idx} "
                        "must match the parameter shape"
                    )
                error_components = self._item_parameter_components(
                    param_name,
                    errors_array,
                    item_idx,
                    self.model.n_items,
                )
                row.update(
                    {
                        f"{column}_se": error
                        for (column, _), (_, error) in zip(
                            components, error_components, strict=True
                        )
                    }
                )
            rows.append(row)
        return rows

    def coef(
        self,
        group: int | str | None = None,
        *,
        include_standard_errors: bool = False,
    ) -> Any:
        """Extract coefficients for one or all groups.

        Parameters
        ----------
        group : int, str, or None
            Group index, label, or None for all groups.
        include_standard_errors : bool
            Add ``_se`` columns. Missing standard errors are represented by NaN.

        Returns
        -------
        DataFrame
            Item parameters. If group is None, includes a 'group' column.
        """
        from mirt.utils.dataframe import create_dataframe

        if not isinstance(include_standard_errors, (bool, np.bool_)):
            raise TypeError("include_standard_errors must be a boolean")

        if group is None:
            rows = []
            for g in range(self.n_groups):
                group_rows = self._coefficient_rows(
                    g,
                    include_standard_errors=bool(include_standard_errors),
                )
                for item_idx, values in enumerate(group_rows):
                    row: dict[str, str | float] = {
                        "group": self.group_labels[g],
                        "item": self.model.item_names[item_idx],
                    }
                    row.update(values)
                    rows.append(row)
            return create_dataframe(rows)

        group_idx = self._resolve_group(group)
        rows = self._coefficient_rows(
            group_idx,
            include_standard_errors=bool(include_standard_errors),
        )
        return create_dataframe(rows, index=self.model.item_names, index_name="item")

    def latent_pars(self) -> Any:
        """Extract latent distribution parameters.

        Returns
        -------
        DataFrame
            Latent means and variances per group.
        """
        from mirt.utils.dataframe import create_dataframe

        rows = []
        for g, dist in enumerate(self.latent_distributions):
            row = {
                "group": self.group_labels[g],
                "is_reference": dist.is_reference,
            }
            if self.model.n_factors == 1:
                row["mean"] = dist.mean[0]
                row["variance"] = dist.cov[0, 0]
            else:
                for f in range(self.model.n_factors):
                    row[f"mean_{f}"] = dist.mean[f]
                for f1 in range(self.model.n_factors):
                    for f2 in range(f1, self.model.n_factors):
                        if f1 == f2:
                            row[f"var_{f1}"] = dist.cov[f1, f2]
                        else:
                            row[f"cov_{f1}_{f2}"] = dist.cov[f1, f2]
            rows.append(row)
        return create_dataframe(rows)

    def fit_statistics(self) -> dict[str, float | int | bool]:
        """Return fit statistics as dictionary.

        Returns
        -------
        dict
            Fit statistics including LL, AIC, BIC, n_parameters.
        """
        return {
            "log_likelihood": self.log_likelihood,
            "AIC": self.aic,
            "BIC": self.bic,
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }

    def summary(self) -> str:
        """Generate formatted summary string.

        Returns
        -------
        str
            Formatted summary of the multigroup analysis.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Multigroup IRT Analysis Results")
        lines.append("=" * 60)
        lines.append(f"Model: {self.model.model_name}")
        lines.append(f"Invariance: {self.invariance}")
        lines.append(f"Number of groups: {self.n_groups}")
        lines.append(f"Number of items: {self.model.n_items}")
        lines.append(f"Number of factors: {self.model.n_factors}")
        lines.append("")

        lines.append("Fit Statistics:")
        lines.append("-" * 40)
        lines.append(f"  Log-likelihood: {self.log_likelihood:.4f}")
        lines.append(f"  AIC: {self.aic:.4f}")
        lines.append(f"  BIC: {self.bic:.4f}")
        lines.append(f"  Free parameters: {self.n_parameters}")
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Iterations: {self.n_iterations}")
        lines.append("")

        lines.append("Group Information:")
        lines.append("-" * 40)
        for g in range(self.n_groups):
            ref_str = (
                " (reference)" if self.latent_distributions[g].is_reference else ""
            )
            lines.append(
                f"  {self.group_labels[g]}{ref_str}: "
                f"n={self.group_n_observations[g]}, "
                f"LL={self.group_log_likelihoods[g]:.4f}"
            )
        lines.append("")

        lines.append("Latent Distributions:")
        lines.append("-" * 40)
        for g in range(self.n_groups):
            dist = self.latent_distributions[g]
            if self.model.n_factors == 1:
                lines.append(
                    f"  {self.group_labels[g]}: "
                    f"mean={dist.mean[0]:.4f}, var={dist.cov[0, 0]:.4f}"
                )
            else:
                lines.append(f"  {self.group_labels[g]}:")
                lines.append(f"    mean: {np.array2string(dist.mean, precision=4)}")
                lines.append(
                    f"    cov diagonal: {np.array2string(np.diag(dist.cov), precision=4)}"
                )
        lines.append("")

        shared_params = [
            name
            for name in self.model.parameter_names
            if self.model.is_parameter_shared(name)
        ]
        if shared_params:
            lines.append(f"Shared parameters: {', '.join(shared_params)}")
        else:
            lines.append("Shared parameters: none (configural)")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultigroupFitResult("
            f"invariance={self.invariance}, "
            f"n_groups={self.n_groups}, "
            f"LL={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )
