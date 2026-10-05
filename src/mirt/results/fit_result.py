"""Fitted-model result container and inference helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.results._common import normal_critical_value, validate_alpha

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

ParameterStatistics = dict[str, dict[str, NDArray[np.float64]]]
FitStatistics = dict[str, float | int | bool]

_FIT_STATISTIC_FIELDS = (
    "log_likelihood",
    "aic",
    "bic",
    "n_parameters",
    "n_observations",
    "converged",
    "n_iterations",
)
_MODEL_FIELDS = {"name", "n_items", "n_factors", "item_names", "n_categories"}


def _payload_error(message: str, *, value: Any = None) -> MirtValidationError:
    return MirtValidationError(
        message,
        parameter="payload",
        value=value,
        expected="mapping produced by FitResult.to_dict()",
    )


def _float_arrays(values: Any, *, field: str) -> dict[str, NDArray[np.float64]]:
    """Convert a mapping of nested lists to float arrays."""
    if not isinstance(values, Mapping):
        raise _payload_error(f"{field} must be a mapping", value=type(values).__name__)
    arrays: dict[str, NDArray[np.float64]] = {}
    for name, raw in values.items():
        try:
            arrays[str(name)] = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise _payload_error(
                f"{field}[{name!r}] must contain numeric values", value=name
            ) from exc
    return arrays


def _category_counts(model: BaseItemModel) -> list[int] | None:
    """Return per-item category counts, or ``None`` for dichotomous models.

    Some models report one shared count (e.g. custom polytomous item types);
    it is repeated for every item.
    """
    if not model.is_polytomous:
        return None
    try:
        counts = np.broadcast_to(
            np.asarray(getattr(model, "n_categories", None), dtype=np.int64),
            (model.n_items,),
        )
    except (TypeError, ValueError):
        return None
    return [int(count) for count in counts]


def _covariance_from_payload(
    value: Any,
) -> tuple[NDArray[np.float64] | None, list[str] | None]:
    """Read the optional ``to_dict()['vcov']`` mapping."""
    if value is None:
        return None, None
    if not isinstance(value, Mapping) or set(value) != {"labels", "matrix"}:
        raise _payload_error("vcov must be a mapping with labels and matrix")
    labels = value["labels"]
    if not isinstance(labels, list):
        raise _payload_error("vcov labels must be a list", value=type(labels).__name__)
    try:
        matrix = np.asarray(value["matrix"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _payload_error("vcov matrix must contain numeric values") from exc
    return matrix, labels


def _model_from_payload(
    info: Any,
    parameters: dict[str, NDArray[np.float64]],
) -> BaseItemModel:
    """Rebuild a fitted built-in model from ``to_dict()['model']`` metadata."""
    from mirt.models._factory import (
        ITEM_MODEL_FAMILIES,
        POLYTOMOUS_FAMILIES,
        build_item_model,
    )

    if not isinstance(info, Mapping):
        raise _payload_error("model must be a mapping", value=type(info).__name__)
    unknown = set(info) - _MODEL_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise _payload_error(f"model contains unknown fields: {names}", value=names)
    missing = {"name", "n_items", "item_names"} - set(info)
    if missing:
        names = ", ".join(sorted(missing))
        raise _payload_error(f"model is missing required fields: {names}", value=names)

    name = info["name"]
    if not isinstance(name, str) or name not in ITEM_MODEL_FAMILIES:
        raise MirtValidationError(
            f"cannot rebuild model {name!r}; only fit_mirt model families are "
            "supported",
            parameter="model.name",
            value=name,
            expected=", ".join(ITEM_MODEL_FAMILIES),
        )
    n_items = info["n_items"]
    if isinstance(n_items, bool) or not isinstance(n_items, Integral) or n_items < 1:
        raise MirtValidationError(
            "model.n_items must be a positive integer",
            parameter="model.n_items",
            value=n_items,
            expected=">= 1",
        )
    item_names = info["item_names"]
    if not isinstance(item_names, list) or not all(
        isinstance(item, str) for item in item_names
    ):
        raise MirtValidationError(
            "model.item_names must be a list of strings",
            parameter="model.item_names",
            value=item_names,
        )
    n_categories = info.get("n_categories")
    if (name in POLYTOMOUS_FAMILIES) != (n_categories is not None):
        raise MirtValidationError(
            "model.n_categories must list category counts for polytomous models "
            "and be null for dichotomous models",
            parameter="model.n_categories",
            value=n_categories,
        )

    try:
        model = build_item_model(
            name,
            int(n_items),
            n_factors=info.get("n_factors", 1),
            n_categories=n_categories,
            item_names=item_names,
        )
    except MirtModelError as exc:
        raise MirtValidationError(exc.message, parameter="model", value=name) from exc

    stored = model.parameters
    if set(parameters) != set(stored):
        expected = ", ".join(sorted(stored))
        raise MirtValidationError(
            f"parameters must contain exactly: {expected}",
            parameter="parameters",
            value=sorted(parameters),
            expected=expected,
        )
    # Family-fixed arrays (1PL and PCM discriminations) cannot be set; they
    # must instead agree with the values the family imposes.
    free_masks = model.free_parameter_masks
    free: dict[str, NDArray[np.float64]] = {}
    for parameter, values in parameters.items():
        if np.any(free_masks[parameter]):
            free[parameter] = values
        elif not np.array_equal(values, stored[parameter]):
            raise MirtValidationError(
                f"{parameter} is fixed by the {name} family and cannot differ",
                parameter=parameter,
                value=values.shape,
                expected=str(stored[parameter].tolist()),
            )
    model.set_parameters(**free)
    model._is_fitted = True
    return model


def _item_labels(model: BaseItemModel) -> list[str]:
    """Item names, or zero-based positions when names are not unique."""
    names = [str(name) for name in model.item_names]
    if len(set(names)) == len(names):
        return names
    return [str(index) for index in range(model.n_items)]


def _parameter_rows(
    model: BaseItemModel, name: str
) -> Sequence[int] | NDArray[np.intp]:
    """Test items behind the rows of a per-item parameter array.

    The arrays of a mixed-format component follow that component's items.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        return model.parameter_items(name)
    return range(model.n_items)


def _row_labels(model: BaseItemModel, name: str, items: list[str]) -> list[str]:
    """Labels of the rows of ``name`` given every item's label."""
    return [items[item] for item in _parameter_rows(model, name)]


def _indexed_by_item(model: BaseItemModel, name: str, shape: tuple[int, ...]) -> bool:
    """Whether a parameter has one leading row per item it covers.

    Parameters shared by all items never do, even when their length happens
    to equal the number of items.
    """
    shared: frozenset[str] = getattr(model, "_shared_parameters", frozenset())
    rows = _parameter_rows(model, name)
    return bool(shape) and shape[0] == len(rows) and name not in shared


def _coordinate_label(
    name: str,
    index: tuple[int, ...],
    items: list[str],
    per_item: bool,
) -> str:
    if per_item:
        parts = [items[index[0]], *(str(value) for value in index[1:])]
    else:
        parts = [str(value) for value in index]
    return f"{name}[{','.join(parts)}]" if parts else name


def _coordinate_positions(model: BaseItemModel) -> dict[str, tuple[str, int]]:
    """Map each stored coordinate label to its parameter and flat index."""
    items = _item_labels(model)
    positions: dict[str, tuple[str, int]] = {}
    for name, values in model.parameters.items():
        rows = _row_labels(model, name, items)
        per_item = _indexed_by_item(model, name, values.shape)
        for flat, index in enumerate(np.ndindex(values.shape)):
            label = _coordinate_label(name, index, rows, per_item)
            positions[label] = (name, flat)
    return positions


def _free_parameter_labels(model: BaseItemModel) -> list[str]:
    """Label the free parameters in the order of a fitted ``vcov``.

    Labels read ``"name[item]"`` for per-item scalars and
    ``"name[item,column]"`` for per-item arrays, with zero-based columns.
    Item positions replace names when item names are not unique.
    """
    items = _item_labels(model)
    masks = model.free_parameter_masks
    labels = []
    for name, values in model.parameters.items():
        rows = _row_labels(model, name, items)
        per_item = _indexed_by_item(model, name, values.shape)
        for flat in np.flatnonzero(np.asarray(masks[name]).ravel()):
            index = tuple(int(i) for i in np.unravel_index(flat, values.shape))
            labels.append(_coordinate_label(name, index, rows, per_item))
    return labels


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
    reported as exact zeros. ``log_posterior`` is the log-likelihood plus the
    item log-prior of a Bayes modal fit, and ``None`` without item priors.

    Attributes
    ----------
    se_method : str, optional
        Estimator behind ``standard_errors``: ``"oakes"`` (observed
        information), ``"crossprod"``, ``"sandwich"``, ``"complete_data"``
        (itemwise complete-data curvature, which understates uncertainty),
        ``"hessian"`` (inverse Hessian of the marginal likelihood from
        ``BLEstimator``) or ``"mhrm_iterate_sd"`` (spread of the MH-RM
        iterates, which is not a sampling standard error). ``None`` when
        unrecorded.
    vcov : ndarray of shape (P, P), optional
        Covariance of the free parameters. Rows and columns of coordinates
        held at an optimizer bound, or whose variance is not estimable, are
        ``NaN``.
    vcov_labels : list of str, optional
        Labels of the ``vcov`` rows, such as ``"discrimination[Item_1]"`` or
        ``"thresholds[Item_1,0]"`` (zero-based columns). Derived from the
        model's free parameters when ``vcov`` is given without labels.
    latent_covariance : ndarray of shape (n_factors, n_factors), optional
        Estimated covariance of the latent factors, for example from a
        confirmatory ``fit_mirt(spec=...)`` fit with ``COV`` terms. ``None``
        when the factors are standard normal and uncorrelated. Scoring,
        plausible values, simulation and the fit diagnostics that accept a
        ``FitResult`` use it as the default latent population.
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
    log_posterior: float | None = None
    se_method: str | None = None
    vcov: NDArray[np.float64] | None = None
    vcov_labels: list[str] | None = None
    latent_covariance: NDArray[np.float64] | None = None

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
        if self.log_posterior is not None:
            self.log_posterior = float(self.log_posterior)
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

        if self.se_method is not None and not isinstance(self.se_method, str):
            raise MirtValidationError(
                "se_method must be a string or None",
                parameter="se_method",
                value=self.se_method,
                expected="str or None",
            )
        self._validate_covariance()
        self._validate_latent_covariance()

    def _validate_latent_covariance(self) -> None:
        if self.latent_covariance is None:
            return
        n_factors = self.model.n_factors
        try:
            matrix = np.array(self.latent_covariance, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "latent_covariance must be a numeric matrix",
                parameter="latent_covariance",
            ) from exc
        if matrix.shape != (n_factors, n_factors):
            raise MirtValidationError(
                f"latent_covariance must have shape ({n_factors}, {n_factors})",
                parameter="latent_covariance",
                value=matrix.shape,
            )
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix, matrix.T, rtol=1e-10, atol=1e-12
        ):
            raise MirtValidationError(
                "latent_covariance must be finite and symmetric",
                parameter="latent_covariance",
            )
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise MirtValidationError(
                "latent_covariance must be positive definite",
                parameter="latent_covariance",
            ) from exc
        self.latent_covariance = matrix

    @property
    def factor_correlation(self) -> NDArray[np.float64] | None:
        """Correlation matrix of :attr:`latent_covariance`, if estimated."""
        if self.latent_covariance is None:
            return None
        scale = np.sqrt(np.diag(self.latent_covariance))
        correlation = self.latent_covariance / np.outer(scale, scale)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    def _validate_covariance(self) -> None:
        if self.vcov is None:
            if self.vcov_labels is not None:
                raise MirtValidationError(
                    "vcov_labels require vcov", parameter="vcov_labels"
                )
            return
        try:
            matrix = np.array(self.vcov, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "vcov must be a numeric matrix", parameter="vcov"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MirtValidationError(
                "vcov must be a square matrix",
                parameter="vcov",
                value=matrix.shape,
                expected="(P, P)",
            )
        known = np.isfinite(matrix)
        if (
            np.any(np.isinf(matrix))
            or not np.array_equal(known, known.T)
            or not np.allclose(matrix[known], matrix.T[known], rtol=1e-8, atol=1e-12)
        ):
            raise MirtValidationError(
                "vcov must be symmetric with finite or NaN entries",
                parameter="vcov",
            )
        if np.any(np.diag(matrix) < 0.0):
            raise MirtValidationError(
                "vcov variances cannot be negative", parameter="vcov"
            )

        labels = self.vcov_labels
        if labels is None:
            labels = _free_parameter_labels(self.model)
            if len(labels) != matrix.shape[0]:
                raise MirtValidationError(
                    f"vcov has {matrix.shape[0]} rows but the model has "
                    f"{len(labels)} free parameters; pass vcov_labels",
                    parameter="vcov",
                    value=matrix.shape,
                )
        elif (
            isinstance(labels, str)
            or not isinstance(labels, Sequence)
            or not all(isinstance(label, str) for label in labels)
            or len(labels) != matrix.shape[0]
            or len(set(labels)) != len(labels)
        ):
            raise MirtValidationError(
                "vcov_labels must list one unique string per vcov row",
                parameter="vcov_labels",
            )
        unknown = sorted(set(labels) - set(_coordinate_positions(self.model)))
        if unknown:
            raise MirtValidationError(
                f"vcov_labels name unknown parameters: {', '.join(unknown[:5])}",
                parameter="vcov_labels",
                value=unknown[:5],
            )
        self.vcov = matrix
        self.vcov_labels = list(labels)

    def _parameter_covariance(
        self,
        names: Sequence[str],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], bool]:
        """Return the covariance of every stored coordinate of ``names``.

        Coordinates follow ``names`` order, each array raveled row-major.
        Returns the covariance, a mask of coordinates with a positive, finite
        variance (others have zero rows and columns), and whether cross-
        parameter covariances are available (otherwise the matrix is the
        diagonal of squared standard errors).

        Raises
        ------
        MirtValidationError
            If the result has neither a covariance nor standard errors.
        """
        parameters = self.model.parameters
        offsets: dict[str, int] = {}
        size = 0
        for name in names:
            offsets[name] = size
            size += parameters[name].size
        covariance = np.zeros((size, size), dtype=np.float64)

        if self.vcov is not None and self.vcov_labels is not None:
            positions = _coordinate_positions(self.model)
            rows, targets = [], []
            for row, label in enumerate(self.vcov_labels):
                name, flat = positions[label]
                if name in offsets:
                    rows.append(row)
                    targets.append(offsets[name] + flat)
            block = self.vcov[np.ix_(rows, rows)]
            variances = np.diag(block)
            usable = np.isfinite(variances) & (variances > 0.0)
            kept = np.asarray(targets, dtype=np.intp)[usable]
            covariance[np.ix_(kept, kept)] = block[np.ix_(usable, usable)]
            return covariance, np.diag(covariance) > 0.0, True

        if not self.standard_errors:
            raise MirtValidationError(
                "result has no parameter covariance or standard errors; fit with "
                "compute_standard_errors=True or pass vcov",
                parameter="result",
            )
        variances = np.concatenate(
            [
                self._errors_for(name, np.asarray(parameters[name])).ravel() ** 2
                for name in names
            ]
        )
        usable = np.isfinite(variances) & (variances > 0.0)
        covariance[np.diag_indices(size)] = np.where(usable, variances, 0.0)
        return covariance, usable, False

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
        if _indexed_by_item(self.model, parameter_name, shape):
            rows = _parameter_rows(self.model, parameter_name)
            item_name = self.model.item_names[rows[index[0]]]
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
        if self.log_posterior is not None:
            lines.append(f"{'':<41}Log-Posterior:     {self.log_posterior:>12.4f}")
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

        if self.latent_covariance is not None:
            labels = [f"F{index + 1}" for index in range(len(self.latent_covariance))]
            lines.append("\nLatent covariance:")
            lines.append(f"{'':<15}" + "".join(f"{label:>10}" for label in labels))
            for label, row in zip(labels, self.latent_covariance, strict=True):
                lines.append(
                    f"{label:<15}" + "".join(f"{value:>10.4f}" for value in row)
                )

        lines.append("=" * width)
        return "\n".join(lines)

    def _coefficient_columns(self, *, include_se: bool) -> dict[str, Any]:
        from mirt.models.mixed_format import MixedItemModel

        data: dict[str, Any] = {}
        parameters = self.model.parameters
        item_errors = None
        if isinstance(self.model, MixedItemModel):
            # One row per item, NaN where an item's family lacks a parameter.
            parameters = self.model.item_parameter_arrays()
            item_errors = self.model.item_parameter_arrays(self.standard_errors)
        for parameter_name, raw_values in parameters.items():
            values = np.asarray(raw_values, dtype=np.float64)
            # Mixed-format arrays from item_parameter_arrays are per item.
            per_item = item_errors is not None or _indexed_by_item(
                self.model, parameter_name, values.shape
            )
            if values.ndim not in (1, 2) or not per_item:
                raise MirtValidationError(
                    "wide coefficient output requires per-item parameter arrays; "
                    "use parameter_statistics() or to_dict() for global parameters",
                    parameter=parameter_name,
                    value=values.shape,
                    expected=f"first dimension {self.model.n_items}",
                )
            errors = (
                self._errors_for(parameter_name, values)
                if item_errors is None
                else item_errors[parameter_name]
            )
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
        """Return scalar fit statistics and convergence metadata.

        Bayes modal fits also report ``log_posterior``.
        """
        statistics: FitStatistics = {
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }
        if self.log_posterior is not None:
            statistics["log_posterior"] = self.log_posterior
        return statistics

    def to_dict(
        self,
        *,
        include_parameters: bool = True,
        include_standard_errors: bool = True,
    ) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible result representation.

        ``model`` records the family name, dimensions, item names, and
        per-item category counts (``None`` for dichotomous models), which
        together with the parameters let :meth:`from_dict` rebuild the model.
        Parameters include ``latent_covariance`` when it was estimated.
        With standard errors, ``se_method`` and a ``vcov`` mapping of row
        ``labels`` and ``matrix`` are included when recorded. Unknown standard
        errors and covariances serialize as ``NaN``, which Python's ``json``
        module reads and writes but strict JSON parsers may reject.
        """
        result: dict[str, Any] = {
            "model": {
                "name": self.model.model_name,
                "n_items": self.model.n_items,
                "n_factors": self.model.n_factors,
                "item_names": list(self.model.item_names),
                "n_categories": _category_counts(self.model),
            },
            **self.fit_statistics(),
        }
        if include_parameters:
            result["parameters"] = {
                name: values.tolist() for name, values in self.model.parameters.items()
            }
            if self.latent_covariance is not None:
                result["latent_covariance"] = self.latent_covariance.tolist()
        if include_standard_errors:
            result["standard_errors"] = {
                name: values.tolist() for name, values in self.standard_errors.items()
            }
            if self.se_method is not None:
                result["se_method"] = self.se_method
            if self.vcov is not None:
                result["vcov"] = {
                    "labels": list(self.vcov_labels or []),
                    "matrix": self.vcov.tolist(),
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild a fitted result from :meth:`to_dict` output.

        The payload must include parameters, so exports written with
        ``include_parameters=False`` cannot be reloaded. Models from the
        built-in ``fit_mirt`` families ("1PL", "2PL", "3PL", "4PL", "GRM",
        "GPCM", "PCM", "NRM") are reconstructed and marked fitted, so the
        result can be scored with :func:`mirt.fscores` directly. Omitted
        standard errors are restored as unknown. Unknown fields, including
        standard errors for parameters the model does not have, are rejected so
        that misspelled input does not silently disappear. Parameters that the
        family fixes (1PL and PCM discriminations) must equal their fixed
        values.

        Parameters
        ----------
        payload : mapping
            Output of :meth:`to_dict`.

        Returns
        -------
        FitResult
            The reconstructed result. Free-parameter restrictions applied
            with ``set_free_parameter_masks`` are not part of the payload.

        Raises
        ------
        MirtValidationError
            If the payload is malformed, incomplete, or describes a model that
            cannot be rebuilt.
        """
        if not isinstance(payload, Mapping):
            raise _payload_error(
                "fit payload must be a mapping", value=type(payload).__name__
            )
        allowed = {
            "model",
            "parameters",
            "standard_errors",
            "log_posterior",
            "se_method",
            "vcov",
            "latent_covariance",
            *_FIT_STATISTIC_FIELDS,
        }
        unknown = set(payload) - allowed
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise _payload_error(
                f"fit payload contains unknown fields: {names}", value=names
            )
        missing = {"model", "parameters", *_FIT_STATISTIC_FIELDS} - set(payload)
        if missing:
            names = ", ".join(sorted(missing))
            hint = (
                "; export with include_parameters=True"
                if "parameters" in missing
                else ""
            )
            raise _payload_error(
                f"fit payload is missing required fields: {names}{hint}", value=names
            )
        for name in ("log_likelihood", "aic", "bic", "log_posterior"):
            value = payload.get(name)
            if name == "log_posterior" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise MirtValidationError(
                    f"{name} must be a number",
                    parameter=name,
                    value=value,
                    expected="float",
                )

        parameters = _float_arrays(payload["parameters"], field="parameters")
        standard_errors = _float_arrays(
            payload.get("standard_errors", {}), field="standard_errors"
        )
        model = _model_from_payload(payload["model"], parameters)
        unknown_errors = set(standard_errors) - set(model.parameters)
        if unknown_errors:
            names = ", ".join(sorted(unknown_errors))
            raise MirtValidationError(
                f"standard_errors contains unknown parameters: {names}",
                parameter="standard_errors",
                value=names,
                expected=", ".join(sorted(model.parameters)),
            )
        vcov, vcov_labels = _covariance_from_payload(payload.get("vcov"))
        return cls(
            model=model,
            log_likelihood=payload["log_likelihood"],
            n_iterations=payload["n_iterations"],
            converged=payload["converged"],
            standard_errors=standard_errors,
            aic=payload["aic"],
            bic=payload["bic"],
            n_observations=payload["n_observations"],
            n_parameters=payload["n_parameters"],
            log_posterior=payload.get("log_posterior"),
            se_method=payload.get("se_method"),
            vcov=vcov,
            vcov_labels=vcov_labels,
            latent_covariance=payload.get("latent_covariance"),
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Rebuild a fitted result from :meth:`to_json` output."""
        import json

        if not isinstance(value, (str, bytes, bytearray)):
            raise MirtValidationError(
                "fit JSON must be a string or bytes",
                parameter="value",
                value=type(value).__name__,
                expected="str, bytes, or bytearray",
            )
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MirtValidationError(
                "fit JSON must contain a valid JSON object",
                parameter="value",
                expected="JSON object produced by FitResult.to_json()",
            ) from error
        return cls.from_dict(payload)

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model.model_name}, "
            f"LL={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )
