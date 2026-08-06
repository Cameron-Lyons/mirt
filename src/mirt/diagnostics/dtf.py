"""Differential Test Functioning analysis.

DTF summarizes differences between reference- and focal-group expected test
score curves.  Signed and unsigned summaries are averaged over a configurable
ability distribution so their scale remains interpretable as score points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import create_theta_grid, fit_group_models, split_groups

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


DTFMethod = Literal["signed", "unsigned", "expected_score"]
DTFWeighting = Literal["normal", "uniform"] | NDArray[np.float64]
_VALID_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_VALID_METHODS = frozenset({"signed", "unsigned", "expected_score"})
_BOOTSTRAP_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class _BootstrapSummary:
    standard_error: float
    p_value: float
    confidence_interval: tuple[float, float]
    n_successful: int
    n_failed: int


def compute_dtf(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    method: DTFMethod = "unsigned",
    theta_range: tuple[float, float] = (-4, 4),
    n_quadpts: int = 49,
    n_bootstrap: int = 100,
    *,
    focal_group: Any | None = None,
    weighting: DTFWeighting = "normal",
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = 42,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Compute Differential Test Functioning statistics.

    The reference and focal test-score functions are evaluated on a shared
    theta grid.  The reported statistic is an average score difference over
    the selected ability weighting, rather than an unnormalized geometric
    area that changes merely because ``theta_range`` is widened.

    Parameters
    ----------
    data
        Response matrix with shape ``(n_persons, n_items)``.
    groups
        One-dimensional group labels with exactly two unique values.
    model
        IRT model fitted independently in each group.
    method
        ``"signed"`` preserves the direction (reference minus focal).
        ``"unsigned"`` and ``"expected_score"`` average the absolute
        expected-score difference; the latter name is retained for API
        compatibility and emphasizes the returned pointwise difference curve.
    theta_range
        Strictly increasing finite integration limits.
    n_quadpts
        Number of grid points, at least two.
    n_bootstrap
        Number of stratified bootstrap replicates.  Set to zero to skip
        uncertainty estimation.
    focal_group
        Label to treat as focal.  By default the second sorted unique label is
        focal.
    weighting
        ``"normal"`` for a standard-normal ability density, ``"uniform"``
        for a uniform density, or nonnegative custom weights evaluated at the
        theta grid.
    confidence_level
        Percentile bootstrap confidence level.
    random_state
        Seed or NumPy random generator used for stratified resampling.
    **fit_kwargs
        Additional arguments passed to ``fit_mirt``.

    Returns
    -------
    dict
        Statistic, uncertainty estimates, expected-score curves, pointwise
        differences, group metadata, grid, and bootstrap diagnostics.

    References
    ----------
    van der Linden, W. J., Raju, N. S., & Fleer, P. F. (1995).
    IRT-based internal measures of differential functioning of items and tests.
    Applied Psychological Measurement, 19(4), 353-368.
    """
    values, labels, theta_limits = _validate_dtf_inputs(
        data=data,
        groups=groups,
        model=model,
        method=method,
        theta_range=theta_range,
        n_quadpts=n_quadpts,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )

    ref_data, focal_data, ref_mask, focal_mask, ref_group, selected_focal = (
        split_groups(values, labels, focal_group=focal_group)
    )
    theta_grid, _ = create_theta_grid(theta_limits, n_quadpts)
    integration_weights, weighting_name = _create_integration_weights(
        theta_grid, weighting
    )
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )

    expected_ref = _compute_expected_score(ref_result.model, theta_grid)
    expected_focal = _compute_expected_score(focal_result.model, theta_grid)
    difference = expected_ref - expected_focal
    statistic = _aggregate_dtf(difference, theta_grid, method, integration_weights)

    bootstrap = _bootstrap_dtf_statistics(
        data=values,
        groups=labels,
        model=model,
        method=method,
        theta_grid=theta_grid,
        integration_weights=integration_weights,
        observed_dtf=statistic,
        ref_group=ref_group,
        focal_group=selected_focal,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        fit_kwargs=fit_kwargs,
    )

    return {
        "DTF": statistic,
        "DTF_SE": bootstrap.standard_error,
        "p_value": bootstrap.p_value,
        "confidence_interval": np.asarray(
            bootstrap.confidence_interval, dtype=np.float64
        ),
        "method": method,
        "weighting": weighting_name,
        "expected_score_ref": expected_ref,
        "expected_score_focal": expected_focal,
        "expected_score_diff": difference,
        "theta_grid": theta_grid,
        "ability_weights": integration_weights,
        "ref_group": ref_group,
        "focal_group": selected_focal,
        "n_reference": int(np.count_nonzero(ref_mask)),
        "n_focal": int(np.count_nonzero(focal_mask)),
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_successful": bootstrap.n_successful,
        "n_bootstrap_failed": bootstrap.n_failed,
    }


def _validate_dtf_inputs(
    *,
    data: object,
    groups: object,
    model: str,
    method: str,
    theta_range: object,
    n_quadpts: int,
    n_bootstrap: int,
    confidence_level: float,
) -> tuple[NDArray[np.int_], NDArray[Any], tuple[float, float]]:
    if model not in _VALID_MODELS:
        valid = ", ".join(sorted(_VALID_MODELS))
        raise ValueError(f"model must be one of: {valid}")
    if method not in _VALID_METHODS:
        valid = ", ".join(sorted(_VALID_METHODS))
        raise ValueError(f"method must be one of: {valid}")
    if isinstance(n_quadpts, bool) or not isinstance(n_quadpts, (int, np.integer)):
        raise ValueError("n_quadpts must be an integer of at least 2")
    if n_quadpts < 2:
        raise ValueError("n_quadpts must be an integer of at least 2")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range must be strictly increasing")

    values = np.asarray(data)
    labels = np.asarray(groups)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty two-dimensional response matrix")
    if labels.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if labels.shape[0] != values.shape[0]:
        raise ValueError("groups length must match the number of response-matrix rows")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must not contain missing or non-finite labels")
    if labels.dtype.kind == "O" and any(label is None for label in labels):
        raise ValueError("groups must not contain missing labels")
    return values, labels, (float(limits[0]), float(limits[1]))


def _create_integration_weights(
    theta_grid: NDArray[np.float64],
    weighting: DTFWeighting,
) -> tuple[NDArray[np.float64], str]:
    if isinstance(weighting, str):
        if weighting == "normal":
            raw_weights = stats.norm.pdf(theta_grid)
        elif weighting == "uniform":
            raw_weights = np.ones_like(theta_grid)
        else:
            raise ValueError("weighting must be 'normal', 'uniform', or an array")
        weighting_name = weighting
    else:
        raw_weights = np.asarray(weighting, dtype=np.float64)
        if raw_weights.shape != theta_grid.shape:
            raise ValueError(
                f"custom weighting must have shape {theta_grid.shape}, "
                f"got {raw_weights.shape}"
            )
        weighting_name = "custom"

    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
        raise ValueError("ability weights must be finite and nonnegative")
    normalizer = float(integrate.trapezoid(raw_weights, theta_grid))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("ability weights must have a positive integral")
    return raw_weights / normalizer, weighting_name


def _compute_expected_score(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute expected total score with one full probability evaluation."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim != 1 or theta_values.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    probabilities = np.asarray(
        model.probability(theta_values.reshape(-1, 1)), dtype=np.float64
    )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    if np.any(probabilities < -PROB_EPSILON) or np.any(
        probabilities > 1.0 + PROB_EPSILON
    ):
        raise ValueError("model probabilities must lie in [0, 1]")

    n_theta = theta_values.size
    n_items = int(model.n_items)
    if probabilities.ndim == 1:
        if n_items != 1 or probabilities.shape != (n_theta,):
            raise ValueError("one-dimensional probabilities require a one-item model")
        return probabilities.copy()
    if probabilities.ndim == 2:
        if probabilities.shape != (n_theta, n_items):
            raise ValueError("dichotomous probability shape must be (n_theta, n_items)")
        return np.sum(probabilities, axis=1)
    if probabilities.ndim == 3:
        if probabilities.shape[:2] != (n_theta, n_items):
            raise ValueError(
                "polytomous probability shape must be (n_theta, n_items, n_categories)"
            )
        categories = np.arange(probabilities.shape[2], dtype=np.float64)
        return np.sum(probabilities * categories, axis=(1, 2))
    raise ValueError(
        "model probability output must be one-, two-, or three-dimensional"
    )


def _aggregate_dtf(
    difference: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    method: str,
    integration_weights: NDArray[np.float64],
) -> float:
    if method == "signed":
        values = difference
    elif method in {"unsigned", "expected_score"}:
        values = np.abs(difference)
    else:
        raise ValueError(f"Unknown DTF method: {method}")
    return float(integrate.trapezoid(values * integration_weights, theta_grid))


def _bootstrap_dtf_statistics(
    *,
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: str,
    method: str,
    theta_grid: NDArray[np.float64],
    integration_weights: NDArray[np.float64],
    observed_dtf: float,
    ref_group: Any,
    focal_group: Any,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int | np.random.Generator | None,
    fit_kwargs: dict[str, Any],
) -> _BootstrapSummary:
    if n_bootstrap == 0:
        return _BootstrapSummary(np.nan, np.nan, (np.nan, np.nan), 0, 0)

    rng = np.random.default_rng(random_state)
    ref_indices = np.flatnonzero(groups == ref_group)
    focal_indices = np.flatnonzero(groups == focal_group)
    bootstrap_statistics: list[float] = []

    for _ in range(n_bootstrap):
        sampled_ref = rng.choice(ref_indices, size=ref_indices.size, replace=True)
        sampled_focal = rng.choice(focal_indices, size=focal_indices.size, replace=True)
        try:
            ref_result, focal_result = fit_group_models(
                data[sampled_ref], data[sampled_focal], model=model, **fit_kwargs
            )
            expected_ref = _compute_expected_score(ref_result.model, theta_grid)
            expected_focal = _compute_expected_score(focal_result.model, theta_grid)
            statistic = _aggregate_dtf(
                expected_ref - expected_focal,
                theta_grid,
                method,
                integration_weights,
            )
            if np.isfinite(statistic):
                bootstrap_statistics.append(statistic)
        except _BOOTSTRAP_EXCEPTIONS:
            continue

    n_successful = len(bootstrap_statistics)
    n_failed = n_bootstrap - n_successful
    if n_successful < 2:
        return _BootstrapSummary(
            np.nan, np.nan, (np.nan, np.nan), n_successful, n_failed
        )

    estimates = np.asarray(bootstrap_statistics, dtype=np.float64)
    standard_error = float(np.std(estimates, ddof=1))
    if standard_error <= PROB_EPSILON:
        p_value = 1.0 if abs(observed_dtf) <= PROB_EPSILON else 0.0
    else:
        z_value = abs(observed_dtf) / standard_error
        p_value = float(2.0 * stats.norm.sf(z_value))

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail_probability, 1.0 - tail_probability])
    return _BootstrapSummary(
        standard_error,
        p_value,
        (float(lower), float(upper)),
        n_successful,
        n_failed,
    )


def _bootstrap_dtf_se(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: str,
    method: str,
    theta_range: tuple[float, float],
    n_quadpts: int,
    n_bootstrap: int = 100,
    *,
    observed_dtf: float | None = None,
    weighting: DTFWeighting = "normal",
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = 42,
    focal_group: Any | None = None,
    **fit_kwargs: Any,
) -> tuple[float, float]:
    """Return bootstrap standard error and approximate p-value.

    This compatibility wrapper retains the historical private helper's
    two-value return while using the validated bootstrap implementation.
    """
    values, labels, theta_limits = _validate_dtf_inputs(
        data=data,
        groups=groups,
        model=model,
        method=method,
        theta_range=theta_range,
        n_quadpts=n_quadpts,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )
    ref_data, focal_data, _, _, ref_group, selected_focal = split_groups(
        values, labels, focal_group=focal_group
    )
    theta_grid, _ = create_theta_grid(theta_limits, n_quadpts)
    integration_weights, _ = _create_integration_weights(theta_grid, weighting)

    if observed_dtf is None:
        ref_result, focal_result = fit_group_models(
            ref_data, focal_data, model=model, **fit_kwargs
        )
        difference = _compute_expected_score(
            ref_result.model, theta_grid
        ) - _compute_expected_score(focal_result.model, theta_grid)
        observed_dtf = _aggregate_dtf(
            difference, theta_grid, method, integration_weights
        )

    summary = _bootstrap_dtf_statistics(
        data=values,
        groups=labels,
        model=model,
        method=method,
        theta_grid=theta_grid,
        integration_weights=integration_weights,
        observed_dtf=observed_dtf,
        ref_group=ref_group,
        focal_group=selected_focal,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        fit_kwargs=fit_kwargs,
    )
    return summary.standard_error, summary.p_value


def plot_dtf(
    dtf_result: dict[str, Any],
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot reference and focal expected-score curves.

    Extra keyword arguments are forwarded to both line plots.  The returned
    axes object can be used for further customization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting") from None

    required = {
        "theta_grid",
        "expected_score_ref",
        "expected_score_focal",
        "ref_group",
        "focal_group",
        "DTF",
    }
    missing = sorted(required.difference(dtf_result))
    if missing:
        raise ValueError(f"dtf_result is missing required keys: {', '.join(missing)}")

    theta = np.asarray(dtf_result["theta_grid"], dtype=np.float64)
    expected_ref = np.asarray(dtf_result["expected_score_ref"], dtype=np.float64)
    expected_focal = np.asarray(dtf_result["expected_score_focal"], dtype=np.float64)
    if theta.ndim != 1 or theta.size < 2:
        raise ValueError("theta_grid must be a one-dimensional array of length >= 2")
    if expected_ref.shape != theta.shape or expected_focal.shape != theta.shape:
        raise ValueError("expected-score curves must match theta_grid shape")
    if not (
        np.all(np.isfinite(theta))
        and np.all(np.isfinite(expected_ref))
        and np.all(np.isfinite(expected_focal))
    ):
        raise ValueError("plot inputs must contain only finite values")
    dtf_value = float(dtf_result["DTF"])
    if not np.isfinite(dtf_value):
        raise ValueError("DTF must be finite")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    line_kwargs = {"linewidth": 2.0, **kwargs}
    reference_kwargs = {
        "label": f"Reference ({dtf_result['ref_group']})",
        **line_kwargs,
    }
    focal_kwargs = {
        "label": f"Focal ({dtf_result['focal_group']})",
        **line_kwargs,
    }
    ax.plot(theta, expected_ref, **reference_kwargs)
    ax.plot(theta, expected_focal, **focal_kwargs)
    ax.fill_between(theta, expected_ref, expected_focal, alpha=0.3, label="Difference")
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Expected Score")
    ax.set_title(f"Differential Test Functioning (DTF = {dtf_value:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
