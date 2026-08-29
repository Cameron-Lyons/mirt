"""Differential Response Functioning (DRF) analysis.

DRF examines differences in reliability and information functions
across groups, complementing DIF and DTF analyses.
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


DRFModel = Literal["1PL", "2PL", "3PL", "GRM", "GPCM"]
_VALID_MODELS = frozenset({"1PL", "2PL", "3PL", "GRM", "GPCM"})
_BOOTSTRAP_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True)
class _ReliabilityBootstrapSummary:
    standard_error: float
    p_value: float
    confidence_interval: tuple[float, float]
    n_successful: int
    n_failed: int


def compute_drf(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: DRFModel = "2PL",
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 49,
    *,
    focal_group: Any | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Compute Differential Response Functioning statistics.

    DRF examines whether the test provides different levels of measurement
    precision (information/reliability) for different groups.

    Parameters
    ----------
    data : NDArray
        Response matrix (n_persons, n_items)
    groups : NDArray
        Group membership (n_persons,)
    model : str
        IRT model to fit
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of theta points
    focal_group : optional
        Label to treat as focal. By default, the second sorted group is focal.
    **fit_kwargs
        Additional arguments for fit_mirt()

    Returns
    -------
    dict
        Dictionary with:
        - 'information_ref': Test information for reference group
        - 'information_focal': Test information for focal group
        - 'information_diff': Difference in information
        - 'DRF': Overall DRF statistic (integrated difference)
        - 'theta_grid': Theta values used
        - 'reliability_ref': Marginal reliability for reference group
        - 'reliability_focal': Marginal reliability for focal group
    """
    values, labels, theta_limits = _validate_drf_inputs(
        data=data,
        groups=groups,
        model=model,
        theta_range=theta_range,
        n_points=n_points,
    )

    ref_data, focal_data, _, _, ref_group, selected_focal = split_groups(
        values, labels, focal_group=focal_group
    )
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    theta_grid, _ = create_theta_grid(theta_limits, n_points)

    info_ref = _compute_test_information(ref_result.model, theta_grid)
    info_focal = _compute_test_information(focal_result.model, theta_grid)

    info_diff = info_ref - info_focal

    drf = float(integrate.trapezoid(np.abs(info_diff), theta_grid))

    rel_ref = _compute_marginal_reliability(
        ref_result.model, theta_limits, n_points=n_points
    )
    rel_focal = _compute_marginal_reliability(
        focal_result.model, theta_limits, n_points=n_points
    )

    return {
        "DRF": drf,
        "information_ref": info_ref,
        "information_focal": info_focal,
        "information_diff": info_diff,
        "theta_grid": theta_grid,
        "reliability_ref": rel_ref,
        "reliability_focal": rel_focal,
        "reliability_diff": rel_ref - rel_focal,
        "ref_group": ref_group,
        "focal_group": selected_focal,
    }


def _validate_drf_inputs(
    *,
    data: object,
    groups: object,
    model: str,
    theta_range: object,
    n_points: int,
) -> tuple[NDArray[np.int_], NDArray[Any], tuple[float, float]]:
    """Validate inputs shared by test-, item-, and reliability-level DRF."""
    if model not in _VALID_MODELS:
        valid = ", ".join(sorted(_VALID_MODELS))
        raise ValueError(f"model must be one of: {valid}")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer of at least 2")
    if n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")

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
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    return values, labels, (float(limits[0]), float(limits[1]))


def _validate_information(
    information: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float64]:
    values = np.asarray(information, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < -PROB_EPSILON):
        raise ValueError(f"{name} must be nonnegative")
    return np.maximum(values, 0.0)


def _compute_test_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute a validated test information curve."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim != 1 or theta_values.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    n_theta = theta_values.size
    n_items = int(model.n_items)
    raw_information = np.asarray(
        model.information(theta_values.reshape(-1, 1)), dtype=np.float64
    )
    if raw_information.ndim == 1:
        return _validate_information(
            raw_information,
            expected_shape=(n_theta,),
            name="test information",
        )
    item_information = _validate_information(
        raw_information,
        expected_shape=(n_theta, n_items),
        name="item information",
    )
    return np.sum(item_information, axis=1)


def _compute_item_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return item information as ``(n_theta, n_items)`` for every model family."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim != 1 or theta_values.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    theta_2d = theta_values.reshape(-1, 1)
    n_theta = theta_values.size
    n_items = int(model.n_items)
    raw_information = np.asarray(model.information(theta_2d), dtype=np.float64)
    if raw_information.ndim == 2:
        return _validate_information(
            raw_information,
            expected_shape=(n_theta, n_items),
            name="item information",
        )
    _validate_information(
        raw_information,
        expected_shape=(n_theta,),
        name="test information",
    )

    columns = [
        _validate_information(
            model.information(theta_2d, item_idx=item_index),
            expected_shape=(n_theta,),
            name=f"item {item_index} information",
        )
        for item_index in range(n_items)
    ]
    return np.column_stack(columns)


def _compute_marginal_reliability(
    model: BaseItemModel,
    theta_range: tuple[float, float],
    n_points: int = 49,
) -> float:
    """Compute marginal reliability coefficient.

    Uses the formula: rho = 1 - E[1/I(theta)] / Var(theta)
    where expectation is taken over the theta distribution.
    """
    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range must be strictly increasing")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer of at least 2")
    if n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")

    theta_grid = np.linspace(limits[0], limits[1], n_points)
    info = _compute_test_information(model, theta_grid)

    weights = stats.norm.pdf(theta_grid)
    weight_integral = float(integrate.trapezoid(weights, theta_grid))
    if not np.isfinite(weight_integral) or weight_integral <= 0.0:
        raise ValueError("theta_range must have positive standard-normal mass")

    se_sq = 1.0 / np.maximum(info, PROB_EPSILON)
    avg_se_sq = float(
        integrate.trapezoid(weights * se_sq, theta_grid) / weight_integral
    )
    reliability = 1.0 - avg_se_sq

    return float(np.clip(reliability, 0.0, 1.0))


def compute_item_drf(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: DRFModel = "2PL",
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 49,
    *,
    focal_group: Any | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Compute DRF for each item individually.

    Parameters
    ----------
    data : NDArray
        Response matrix
    groups : NDArray
        Group membership
    model : str
        IRT model
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of theta points
    focal_group : optional
        Label to treat as focal. By default, the second sorted group is focal.
    **fit_kwargs
        Additional arguments for fit_mirt()

    Returns
    -------
    dict
        Dictionary with:
        - 'item_drf': DRF statistic for each item
        - 'info_diff_max': Maximum absolute information difference per item
        - 'info_ref': Item information functions for reference (n_items x n_points)
        - 'info_focal': Item information functions for focal (n_items x n_points)
    """
    values, labels, theta_limits = _validate_drf_inputs(
        data=data,
        groups=groups,
        model=model,
        theta_range=theta_range,
        n_points=n_points,
    )

    ref_data, focal_data, _, _, ref_group, selected_focal = split_groups(
        values, labels, focal_group=focal_group
    )
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    theta_grid, _ = create_theta_grid(theta_limits, n_points)

    info_ref_all = _compute_item_information(ref_result.model, theta_grid)
    info_focal_all = _compute_item_information(focal_result.model, theta_grid)
    if info_ref_all.shape != info_focal_all.shape:
        raise ValueError("reference and focal information shapes must match")

    difference = np.abs(info_ref_all - info_focal_all)
    item_drf = np.asarray(
        integrate.trapezoid(difference, theta_grid, axis=0), dtype=np.float64
    )
    info_diff_max = np.max(difference, axis=0)

    return {
        "item_drf": item_drf,
        "info_diff_max": info_diff_max,
        "info_ref": info_ref_all.T,
        "info_focal": info_focal_all.T,
        "theta_grid": theta_grid,
        "ref_group": ref_group,
        "focal_group": selected_focal,
    }


def plot_drf(
    drf_result: dict[str, Any],
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot DRF results showing information functions.

    Parameters
    ----------
    drf_result : dict
        Result from compute_drf()
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional plotting arguments
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for plotting") from exc

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = [ax, ax.twinx()]

    theta = drf_result["theta_grid"]
    info_ref = drf_result["information_ref"]
    info_focal = drf_result["information_focal"]
    plot_kwargs = {"linewidth": 2, **kwargs}

    axes[0].plot(
        theta,
        info_ref,
        label=f"Reference ({drf_result['ref_group']})",
        **plot_kwargs,
    )
    axes[0].plot(
        theta,
        info_focal,
        label=f"Focal ({drf_result['focal_group']})",
        **plot_kwargs,
    )
    axes[0].fill_between(theta, info_ref, info_focal, alpha=0.3)
    axes[0].set_xlabel(r"$\theta$ (Ability)")
    axes[0].set_ylabel("Test Information")
    axes[0].set_title(f"Test Information Functions (DRF = {drf_result['DRF']:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    se_ref = 1 / np.sqrt(np.maximum(info_ref, PROB_EPSILON))
    se_focal = 1 / np.sqrt(np.maximum(info_focal, PROB_EPSILON))

    axes[1].plot(
        theta,
        se_ref,
        label=f"Reference ({drf_result['ref_group']})",
        **plot_kwargs,
    )
    axes[1].plot(
        theta,
        se_focal,
        label=f"Focal ({drf_result['focal_group']})",
        **plot_kwargs,
    )
    axes[1].set_xlabel(r"$\theta$ (Ability)")
    axes[1].set_ylabel("Standard Error")
    axes[1].set_title("Standard Error of Measurement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return axes


def _bootstrap_reliability_differences(
    *,
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    model: str,
    theta_range: tuple[float, float],
    n_points: int,
    observed_difference: float,
    n_bootstrap: int,
    confidence_level: float,
    seed: int | np.random.Generator | None,
    fit_kwargs: dict[str, Any],
) -> _ReliabilityBootstrapSummary:
    if n_bootstrap == 0:
        return _ReliabilityBootstrapSummary(np.nan, np.nan, (np.nan, np.nan), 0, 0)

    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    n_successful = 0
    for _ in range(n_bootstrap):
        ref_indices = rng.integers(0, len(ref_data), size=len(ref_data))
        focal_indices = rng.integers(0, len(focal_data), size=len(focal_data))
        try:
            ref_result, focal_result = fit_group_models(
                ref_data[ref_indices],
                focal_data[focal_indices],
                model=model,
                **fit_kwargs,
            )
            ref_reliability = _compute_marginal_reliability(
                ref_result.model, theta_range, n_points=n_points
            )
            focal_reliability = _compute_marginal_reliability(
                focal_result.model, theta_range, n_points=n_points
            )
        except _BOOTSTRAP_EXCEPTIONS:
            continue
        difference = ref_reliability - focal_reliability
        if np.isfinite(difference):
            differences[n_successful] = difference
            n_successful += 1

    n_failed = n_bootstrap - n_successful
    if n_successful < 2:
        return _ReliabilityBootstrapSummary(
            np.nan, np.nan, (np.nan, np.nan), n_successful, n_failed
        )

    estimates = differences[:n_successful]
    standard_error = float(np.std(estimates, ddof=1))
    if standard_error <= PROB_EPSILON:
        p_value = 1.0 if abs(observed_difference) <= PROB_EPSILON else 0.0
    else:
        z_value = abs(observed_difference) / standard_error
        p_value = float(2.0 * stats.norm.sf(z_value))

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail_probability, 1.0 - tail_probability])
    return _ReliabilityBootstrapSummary(
        standard_error,
        p_value,
        (float(lower), float(upper)),
        n_successful,
        n_failed,
    )


def reliability_invariance(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: DRFModel = "2PL",
    n_bootstrap: int = 100,
    seed: int | np.random.Generator | None = None,
    *,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 49,
    focal_group: Any | None = None,
    confidence_level: float = 0.95,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Test whether reliability is invariant across groups.

    Uses bootstrap to test the null hypothesis that marginal reliability
    is equal in both groups.

    Parameters
    ----------
    data : NDArray
        Response matrix
    groups : NDArray
        Group membership
    model : str
        IRT model
    n_bootstrap : int
        Number of bootstrap samples
    seed : int, optional
        Random seed
    theta_range : tuple
        Ability range used for marginal reliability integration.
    n_points : int
        Number of integration points, at least two.
    focal_group : optional
        Label to treat as focal. By default, the second sorted group is focal.
    confidence_level : float
        Percentile bootstrap confidence level.
    **fit_kwargs
        Additional arguments passed to the group model fits.

    Returns
    -------
    dict
        Dictionary with:
        - 'reliability_ref': Reliability for reference group
        - 'reliability_focal': Reliability for focal group
        - 'reliability_diff': Difference in reliability
        - 'reliability_diff_se': SE of difference
        - 'z': Z-statistic
        - 'p_value': P-value for test of equal reliability
        - 'reliability_diff_ci': Percentile bootstrap confidence interval
        - 'n_bootstrap_successful': Number of successful bootstrap fits
        - 'n_bootstrap_failed': Number of failed bootstrap fits
    """
    values, labels, theta_limits = _validate_drf_inputs(
        data=data,
        groups=groups,
        model=model,
        theta_range=theta_range,
        n_points=n_points,
    )
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    ref_data, focal_data, _, _, ref_group, selected_focal = split_groups(
        values, labels, focal_group=focal_group
    )
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )

    rel_ref = _compute_marginal_reliability(
        ref_result.model, theta_limits, n_points=n_points
    )
    rel_focal = _compute_marginal_reliability(
        focal_result.model, theta_limits, n_points=n_points
    )
    rel_diff = rel_ref - rel_focal
    bootstrap = _bootstrap_reliability_differences(
        ref_data=ref_data,
        focal_data=focal_data,
        model=model,
        theta_range=theta_limits,
        n_points=n_points,
        observed_difference=rel_diff,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        fit_kwargs=fit_kwargs,
    )
    if np.isnan(bootstrap.standard_error):
        z_value = np.nan
    else:
        z_value = rel_diff / max(bootstrap.standard_error, PROB_EPSILON)

    return {
        "reliability_ref": rel_ref,
        "reliability_focal": rel_focal,
        "reliability_diff": rel_diff,
        "reliability_diff_se": bootstrap.standard_error,
        "reliability_diff_ci": np.asarray(
            bootstrap.confidence_interval, dtype=np.float64
        ),
        "z": z_value,
        "p_value": bootstrap.p_value,
        "ref_group": ref_group,
        "focal_group": selected_focal,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_successful": bootstrap.n_successful,
        "n_bootstrap_failed": bootstrap.n_failed,
    }
