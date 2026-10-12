from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.multigroup.latent import GroupLatentDistribution
    from mirt.multigroup.model import MultigroupModel
    from mirt.results.score_result import ScoreResult

# Fit statistics that ``to_dict`` writes and ``from_dict`` requires.
_STATISTIC_FIELDS = (
    "invariance",
    "log_likelihood",
    "n_iterations",
    "converged",
    "group_log_likelihoods",
    "group_n_observations",
    "aic",
    "bic",
    "n_parameters",
    "n_observations",
)


def _payload_error(message: str, *, value: Any = None) -> MirtValidationError:
    return MirtValidationError(
        message,
        parameter="payload",
        value=value,
        expected="mapping produced by MultigroupFitResult.to_dict()",
    )


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
        Standard errors organized by param_name -> group_idx -> array, in
        the stored parameter shapes, with ``NaN`` for coordinates that were
        not estimated. Shared coordinates have the same error in every
        group. Empty unless the fit computed standard errors (see
        ``compute_standard_errors`` of :func:`fit_multigroup`).
    latent_standard_errors : list of dict, optional
        Per group, standard errors of the latent ``"mean"`` and ``"cov"``
        in their shapes, ``NaN`` for fixed moments; empty without standard
        errors.
    vcov : ndarray, optional
        Covariance of the joint free parameters, whose coordinates
        ``vcov_labels`` names: shared item coordinates once, group-specific
        coordinates per group, then the estimated latent moments.
        Coordinates held at an optimizer bound are ``NaN``.
    vcov_labels : list of str, optional
        Names of the ``vcov`` coordinates, such as
        ``"discrimination[Item_1]"`` for a shared coordinate and
        ``"B: difficulty[Item_1]"`` or ``"B: mean[0]"`` for group ``"B"``.
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
    latent_standard_errors: list[dict[str, NDArray[np.float64]]] = field(
        default_factory=list
    )
    vcov: NDArray[np.float64] | None = None
    vcov_labels: list[str] = field(default_factory=list)

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
        """Build flattened item rows for one group.

        Group-specific parameters of items the group never answered were not
        estimated and are reported as NaN.
        """
        params = self.model.get_group_parameters(group_idx)
        unobserved = set(self.model.unobserved_items(group_idx))
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
                if (
                    item_idx in unobserved
                    and self.model._parameter_is_item_major(param_name)
                    and not self.model.is_item_parameter_shared(param_name, item_idx)
                ):
                    components = [(column, np.nan) for column, _ in components]
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

    def latent_pars(self, *, include_standard_errors: bool = False) -> Any:
        """Extract latent distribution parameters.

        Parameters
        ----------
        include_standard_errors : bool
            Add an ``_se`` column after each mean, variance and covariance,
            as R's ``coef(..., printSE=TRUE)`` reports ``MEAN_1`` and
            ``COV_11`` errors. Fixed moments and fits without standard
            errors give NaN.

        Returns
        -------
        DataFrame
            Latent means and variances per group, with ``free_mean`` and
            ``free_variance`` flags for the moments that were estimated (see
            the ``free_means`` and ``free_variances`` options of
            :class:`~mirt.multigroup.InvarianceSpec`). Multidimensional
            models list every mean, variance and covariance, and
            ``free_correlations`` counts the estimated factor correlations.
        """
        from mirt.utils.dataframe import create_dataframe

        if not isinstance(include_standard_errors, (bool, np.bool_)):
            raise TypeError("include_standard_errors must be a boolean")
        rows = []
        n_factors = self.model.n_factors
        for g, dist in enumerate(self.latent_distributions):
            errors = (
                self.latent_standard_errors[g]
                if g < len(self.latent_standard_errors)
                else {
                    "mean": np.full(n_factors, np.nan),
                    "cov": np.full((n_factors, n_factors), np.nan),
                }
            )
            row: dict[str, Any] = {
                "group": self.group_labels[g],
                "is_reference": dist.is_reference,
                "free_mean": dist.estimate_mean,
                "free_variance": dist.estimate_cov,
            }

            def put(
                column: str, value: float, error: float, row: dict[str, Any] = row
            ) -> None:
                row[column] = value
                if include_standard_errors:
                    row[f"{column}_se"] = float(error)

            if n_factors == 1:
                put("mean", dist.mean[0], errors["mean"][0])
                put("variance", dist.cov[0, 0], errors["cov"][0, 0])
            else:
                free = dist.free_covariance_entries()
                row["free_correlations"] = (
                    0 if free is None else int(np.count_nonzero(np.triu(free, 1)))
                )
                for f in range(n_factors):
                    put(f"mean_{f}", dist.mean[f], errors["mean"][f])
                for f1 in range(n_factors):
                    for f2 in range(f1, n_factors):
                        column = f"var_{f1}" if f1 == f2 else f"cov_{f1}_{f2}"
                        put(column, dist.cov[f1, f2], errors["cov"][f1, f2])
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
            mean_state = "free" if dist.estimate_mean else "fixed"
            variance_state = "free" if dist.estimate_cov else "fixed"
            if self.model.n_factors == 1:
                lines.append(
                    f"  {self.group_labels[g]}: "
                    f"mean={dist.mean[0]:.4f} ({mean_state}), "
                    f"var={dist.cov[0, 0]:.4f} ({variance_state})"
                )
            else:
                lines.append(f"  {self.group_labels[g]}:")
                lines.append(
                    f"    mean: {np.array2string(dist.mean, precision=4)} "
                    f"({mean_state})"
                )
                lines.append(
                    f"    cov diagonal: {np.array2string(np.diag(dist.cov), precision=4)}"
                    f" ({variance_state})"
                )
                free = dist.free_covariance_entries()
                if free is not None and np.any(np.triu(free, 1)):
                    scale = np.sqrt(np.diag(dist.cov))
                    correlation = dist.cov / np.outer(scale, scale)
                    pairs = ", ".join(
                        f"r{first}{second}={correlation[first, second]:.4f}"
                        for first, second in zip(
                            *np.nonzero(np.triu(free, 1)), strict=True
                        )
                    )
                    lines.append(f"    correlations (free): {pairs}")
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

    def fscores(
        self,
        data: ArrayLike,
        groups: ArrayLike,
        method: Literal["EAP", "MAP", "ML", "WLE", "EAPsum"] = "EAP",
        n_quadpts: int | None = None,
        person_ids: list[Any] | NDArray[Any] | None = None,
        bounds: tuple[float, float] = (-6.0, 6.0),
        n_jobs: int = 1,
        batch_size: int | None = None,
    ) -> ScoreResult:
        """Score respondents with their own group's items and population.

        As R's ``fscores`` on a ``multipleGroup`` fit, each row is scored by
        :func:`mirt.fscores` with its group's fitted item parameters and,
        for the Bayesian methods (EAP, EAPsum and MAP), its group's fitted
        normal latent distribution as the prior. Scores are therefore on the
        reference group's metric. ML and WLE use no prior.

        Parameters
        ----------
        data : array-like of shape (n_persons, n_items)
            Responses, as for :func:`mirt.fscores`; DataFrame columns are
            matched to the item names.
        groups : array-like of shape (n_persons,)
            Group of each row, matched to :attr:`group_labels` by
            ``str(label)``, as :func:`fit_multigroup` labels groups.
        method, n_quadpts, person_ids, bounds, n_jobs, batch_size
            As for :func:`mirt.fscores`.

        Returns
        -------
        ScoreResult
            Scores and standard errors in the row order of ``data``.

        Raises
        ------
        MirtValidationError
            If ``groups`` does not have one known group label per row.
        """
        from mirt.multigroup import _missing_group_labels
        from mirt.results._common import validate_person_ids
        from mirt.results.score_result import ScoreResult
        from mirt.scoring import fscores
        from mirt.scoring._common import validate_scoring_responses

        responses = validate_scoring_responses(self.model.get_group_model(0), data)
        n_persons = responses.shape[0]
        labels = np.asarray(groups)
        if labels.ndim != 1 or labels.shape[0] != n_persons:
            raise MirtValidationError(
                "groups must contain one label per response row",
                parameter="groups",
                value=labels.shape,
                expected=f"({n_persons},)",
            )
        missing = _missing_group_labels(labels)
        known = self.group_labels
        names = np.array([str(label) for label in labels.tolist()], dtype=object)
        unknown = sorted({name for name in names[~missing] if name not in known})
        if np.any(missing) or unknown:
            raise MirtValidationError(
                "every row needs one of the fitted group labels "
                f"{known}; found missing or unknown labels {unknown}",
                parameter="groups",
                value=unknown,
                expected=", ".join(known),
            )
        person_ids = validate_person_ids(person_ids, n_persons, row_label="score")
        n_factors = self.model.n_factors
        theta = np.full((n_persons, n_factors), np.nan)
        errors = np.full((n_persons, n_factors), np.nan)
        scored_method = method
        for g, label in enumerate(known):
            rows = np.flatnonzero(names == label)
            if not rows.size:
                continue
            distribution = self.latent_distributions[g]
            score = fscores(
                self.model.get_group_model(g),
                responses[rows],
                method=method,
                n_quadpts=n_quadpts,
                prior_mean=distribution.mean,
                prior_cov=distribution.cov,
                bounds=bounds,
                n_jobs=n_jobs,
                batch_size=batch_size,
            )
            theta[rows] = np.reshape(score.theta, (rows.size, n_factors))
            errors[rows] = np.reshape(score.standard_error, (rows.size, n_factors))
            scored_method = score.method
        if n_factors == 1:
            theta, errors = theta[:, 0], errors[:, 0]
        return ScoreResult(
            theta=theta,
            standard_error=errors,
            method=scored_method,
            person_ids=person_ids,
        )

    def to_dict(
        self,
        *,
        include_standard_errors: bool = True,
    ) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible result representation.

        ``model`` describes the item family as
        :meth:`mirt.results.FitResult.to_dict` does. ``groups`` holds each
        group's label and item parameters, ``links`` the invariance links of
        every parameter (shared or not, with the items shared or freed),
        ``fixed_parameters`` the fixed item blocks and ``unobserved_items``
        each group's unadministered items. ``latent_distributions`` holds
        each group's mean, covariance and estimation flags. With standard
        errors, ``standard_errors`` (one list per group and parameter),
        ``latent_standard_errors`` and ``vcov`` are included when computed.
        Unknown values serialize as ``NaN``, which Python's ``json`` module
        reads and writes but strict JSON parsers may reject.
        """
        from mirt.results.fit_result import _category_counts, _model_structure

        model = self.model
        base = model.get_group_model(0)
        result: dict[str, Any] = {
            "model": {
                "name": base.model_name,
                "n_items": base.n_items,
                "n_factors": base.n_factors,
                "item_names": list(model.item_names),
                "n_categories": _category_counts(base),
                **_model_structure(base),
            },
            "groups": [
                {
                    "label": label,
                    "parameters": {
                        name: values.tolist()
                        for name, values in group.parameters.items()
                    },
                }
                for label, group in zip(
                    model.group_labels, model.group_models, strict=True
                )
            ],
            "links": {
                name: {
                    "is_shared": link.is_shared,
                    "shared_items": (
                        None
                        if link.shared_items is None
                        else sorted(int(item) for item in link.shared_items)
                    ),
                    "free_items": sorted(int(item) for item in link.free_items),
                }
                for name, link in model._parameter_links.items()
            },
            "fixed_parameters": {
                name: [[int(item), np.asarray(value).tolist()] for item, value in items]
                for name, items in (
                    (name, sorted(items.items()))
                    for name, items in model.fixed_item_parameters.items()
                )
            },
            "unobserved_items": [
                model.unobserved_items(g) for g in range(model.n_groups)
            ],
            "latent_distributions": [
                {
                    "mean": np.asarray(distribution.mean).tolist(),
                    "cov": np.asarray(distribution.cov).tolist(),
                    "is_reference": bool(distribution.is_reference),
                    "estimate_mean": bool(distribution.estimate_mean),
                    "estimate_cov": bool(distribution.estimate_cov),
                    "correlation_mask": (
                        None
                        if distribution.correlation_mask is None
                        else np.asarray(distribution.correlation_mask).tolist()
                    ),
                }
                for distribution in self.latent_distributions
            ],
            "invariance": self.invariance,
            "log_likelihood": float(self.log_likelihood),
            "n_iterations": int(self.n_iterations),
            "converged": bool(self.converged),
            "group_log_likelihoods": [float(v) for v in self.group_log_likelihoods],
            "group_n_observations": [int(v) for v in self.group_n_observations],
            "aic": float(self.aic),
            "bic": float(self.bic),
            "n_parameters": int(self.n_parameters),
            "n_observations": int(self.n_observations),
        }
        if include_standard_errors:
            result["standard_errors"] = {
                name: [
                    np.asarray(by_group[g]).tolist() if g in by_group else None
                    for g in range(model.n_groups)
                ]
                for name, by_group in self.standard_errors.items()
            }
            result["latent_standard_errors"] = [
                {key: np.asarray(value).tolist() for key, value in errors.items()}
                for errors in self.latent_standard_errors
            ]
            if self.vcov is not None:
                result["vcov"] = {
                    "labels": list(self.vcov_labels),
                    "matrix": np.asarray(self.vcov).tolist(),
                }
        return result

    def to_json(
        self, *, include_standard_errors: bool = True, indent: int | None = None
    ) -> str:
        """Serialize :meth:`to_dict` to JSON."""
        import json

        return json.dumps(
            self.to_dict(include_standard_errors=include_standard_errors),
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a fitted multigroup result from :meth:`to_dict` output.

        Group models of the families that :meth:`mirt.results.FitResult.from_dict`
        rebuilds are reconstructed with their invariance links, fixed
        parameters, unadministered items and latent distributions, and
        marked fitted, so the result can be scored with :meth:`fscores` or
        used as ``initial_latent`` and starting models of a refit.

        Raises
        ------
        MirtValidationError
            If the payload is malformed, incomplete, has unknown fields, or
            describes a model that cannot be rebuilt.
        """
        from mirt.multigroup.latent import GroupLatentDistribution
        from mirt.multigroup.model import MultigroupModel, ParameterLink
        from mirt.results.fit_result import (
            _covariance_from_payload,
            _float_arrays,
            _model_from_payload,
        )

        if not isinstance(payload, Mapping):
            raise _payload_error(
                "multigroup payload must be a mapping", value=type(payload).__name__
            )
        required = {
            "model",
            "groups",
            "links",
            "fixed_parameters",
            "unobserved_items",
            "latent_distributions",
            *_STATISTIC_FIELDS,
        }
        optional = {"standard_errors", "latent_standard_errors", "vcov"}
        unknown = set(payload) - required - optional
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise _payload_error(
                f"multigroup payload contains unknown fields: {names}", value=names
            )
        missing = required - set(payload)
        if missing:
            names = ", ".join(sorted(missing))
            raise _payload_error(
                f"multigroup payload is missing required fields: {names}",
                value=names,
            )
        groups = payload["groups"]
        if not isinstance(groups, list) or len(groups) < 2:
            raise _payload_error("groups must list at least two groups")
        models = []
        for entry in groups:
            if not isinstance(entry, Mapping) or set(entry) != {"label", "parameters"}:
                raise _payload_error("each group needs a label and parameters")
            models.append(
                _model_from_payload(
                    payload["model"],
                    _float_arrays(entry["parameters"], field="parameters"),
                )
            )
        try:
            model = MultigroupModel(
                models[0], len(models), [str(entry["label"]) for entry in groups]
            )
        except ValueError as error:
            raise _payload_error(str(error)) from error
        model._group_models = models
        links = payload["links"]
        if not isinstance(links, Mapping) or set(links) != set(model.parameter_names):
            raise _payload_error(
                "links must describe every parameter: "
                + ", ".join(model.parameter_names)
            )
        for name, link in links.items():
            if not isinstance(link, Mapping) or set(link) != {
                "is_shared",
                "shared_items",
                "free_items",
            }:
                raise _payload_error(f"links[{name!r}] is malformed", value=name)
            model._parameter_links[name] = ParameterLink(
                param_name=name,
                is_shared=bool(link["is_shared"]),
                shared_items=(
                    None
                    if link["shared_items"] is None
                    else {int(item) for item in link["shared_items"]}
                ),
                free_items={int(item) for item in link["free_items"]},
            )
        fixed = payload["fixed_parameters"]
        if not isinstance(fixed, Mapping):
            raise _payload_error("fixed_parameters must be a mapping")
        model._fixed_item_parameters = {
            str(name): {
                int(item): np.asarray(value, dtype=np.float64) for item, value in items
            }
            for name, items in fixed.items()
        }
        model._set_unobserved_items(payload["unobserved_items"])

        distributions = payload["latent_distributions"]
        if not isinstance(distributions, list) or len(distributions) != len(models):
            raise _payload_error("latent_distributions must list every group")
        latent = []
        for entry in distributions:
            if not isinstance(entry, Mapping):
                raise _payload_error("each latent distribution must be a mapping")
            mask = entry.get("correlation_mask")
            try:
                latent.append(
                    GroupLatentDistribution(
                        mean=np.asarray(entry["mean"], dtype=np.float64),
                        cov=np.asarray(entry["cov"], dtype=np.float64),
                        is_reference=bool(entry["is_reference"]),
                        estimate_mean=bool(entry["estimate_mean"]),
                        estimate_cov=bool(entry["estimate_cov"]),
                        correlation_mask=(
                            None if mask is None else np.asarray(mask, dtype=np.bool_)
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise _payload_error(
                    f"latent distribution is malformed: {error}"
                ) from error

        standard_errors: dict[str, dict[int, NDArray[np.float64]]] = {}
        for name, by_group in payload.get("standard_errors", {}).items():
            if name not in model.parameter_names or not isinstance(by_group, list):
                raise _payload_error(
                    f"standard_errors[{name!r}] is malformed", value=name
                )
            standard_errors[name] = {
                g: np.asarray(values, dtype=np.float64)
                for g, values in enumerate(by_group)
                if values is not None
            }
        latent_errors = [
            {key: np.asarray(value, dtype=np.float64) for key, value in errors.items()}
            for errors in payload.get("latent_standard_errors", [])
        ]
        vcov, vcov_labels = _covariance_from_payload(payload.get("vcov"))
        return cls(
            model=model,
            invariance=str(payload["invariance"]),
            log_likelihood=float(payload["log_likelihood"]),
            n_iterations=int(payload["n_iterations"]),
            converged=bool(payload["converged"]),
            group_log_likelihoods=[float(v) for v in payload["group_log_likelihoods"]],
            group_n_observations=[int(v) for v in payload["group_n_observations"]],
            latent_distributions=latent,
            aic=float(payload["aic"]),
            bic=float(payload["bic"]),
            n_parameters=int(payload["n_parameters"]),
            n_observations=int(payload["n_observations"]),
            standard_errors=standard_errors,
            latent_standard_errors=latent_errors,
            vcov=vcov,
            vcov_labels=list(vcov_labels or []),
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Rebuild a fitted multigroup result from :meth:`to_json` output."""
        import json

        if not isinstance(value, (str, bytes, bytearray)):
            raise MirtValidationError(
                "multigroup JSON must be a string or bytes",
                parameter="value",
                value=type(value).__name__,
                expected="str, bytes, or bytearray",
            )
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MirtValidationError(
                "multigroup JSON must contain a valid JSON object",
                parameter="value",
                expected="JSON object produced by MultigroupFitResult.to_json()",
            ) from error
        return cls.from_dict(payload)

    def __repr__(self) -> str:
        return (
            f"MultigroupFitResult("
            f"invariance={self.invariance}, "
            f"n_groups={self.n_groups}, "
            f"LL={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )
