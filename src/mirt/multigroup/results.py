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
        return self.model.group_labels

    def coef(self, group: int | str | None = None) -> Any:
        """Extract coefficients for one or all groups.

        Parameters
        ----------
        group : int, str, or None
            Group index, label, or None for all groups.

        Returns
        -------
        DataFrame
            Item parameters. If group is None, includes a 'group' column.
        """
        from mirt.utils.dataframe import create_dataframe

        if group is None:
            rows = []
            for g in range(self.n_groups):
                params = self.model.get_group_parameters(g)
                for item_idx in range(self.model.n_items):
                    row = {
                        "group": self.group_labels[g],
                        "item": self.model.item_names[item_idx],
                    }
                    for param_name, values in params.items():
                        if values.ndim == 1:
                            row[param_name] = values[item_idx]
                        else:
                            for j in range(values.shape[1]):
                                row[f"{param_name}_{j}"] = values[item_idx, j]
                    rows.append(row)
            return create_dataframe(rows)

        if isinstance(group, str):
            try:
                group_idx = self.group_labels.index(group)
            except ValueError:
                raise ValueError(f"Unknown group label: {group}")
        else:
            group_idx = group

        params = self.model.get_group_parameters(group_idx)
        rows = []
        for item_idx in range(self.model.n_items):
            row = {"item": self.model.item_names[item_idx]}
            for param_name, values in params.items():
                if values.ndim == 1:
                    row[param_name] = values[item_idx]
                else:
                    for j in range(values.shape[1]):
                        row[f"{param_name}_{j}"] = values[item_idx, j]
            rows.append(row)
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

    def fit_statistics(self) -> dict[str, float]:
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
