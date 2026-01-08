"""Differential Response Functioning (DRF) analysis.

DRF examines differences in reliability and information functions
across groups, complementing DIF and DTF analyses.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from mirt.diagnostics._utils import create_theta_grid, fit_group_models, split_groups


def compute_drf(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 49,
    **fit_kwargs,
) -> dict[str, NDArray[np.float64] | float]:
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
    data = np.asarray(data)
    groups = np.asarray(groups)

    ref_data, focal_data, _, _, ref_group, focal_group = split_groups(data, groups)
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    theta_grid, _ = create_theta_grid(theta_range, n_points)

    info_ref = _compute_test_information(ref_result.model, theta_grid)
    info_focal = _compute_test_information(focal_result.model, theta_grid)

    info_diff = info_ref - info_focal

    drf = float(integrate.trapezoid(np.abs(info_diff), theta_grid))

    rel_ref = _compute_marginal_reliability(ref_result.model, theta_range)
    rel_focal = _compute_marginal_reliability(focal_result.model, theta_range)

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
        "focal_group": focal_group,
    }


def _compute_test_information(
    model,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute test information function."""
    theta_2d = theta.reshape(-1, 1)
    info = model.information(theta_2d)

    if info.ndim == 2:
        return info.sum(axis=1)
    return info


def _compute_marginal_reliability(
    model,
    theta_range: tuple[float, float],
    n_points: int = 49,
) -> float:
    """Compute marginal reliability coefficient.

    Uses the formula: rho = 1 - E[1/I(theta)] / Var(theta)
    where expectation is taken over the theta distribution.
    """
    from scipy import stats

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_points)
    info = _compute_test_information(model, theta_grid)

    weights = stats.norm.pdf(theta_grid)
    weights = weights / weights.sum()

    se_sq = 1.0 / np.maximum(info, 1e-10)
    avg_se_sq = np.sum(weights * se_sq)

    var_theta = 1.0

    reliability = 1 - avg_se_sq / var_theta

    return float(np.clip(reliability, 0, 1))


def compute_item_drf(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 49,
    **fit_kwargs,
) -> dict[str, NDArray[np.float64]]:
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
    data = np.asarray(data)
    groups = np.asarray(groups)
    n_items = data.shape[1]

    ref_data, focal_data, _, _, _, _ = split_groups(data, groups)
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    theta_grid, theta_2d = create_theta_grid(theta_range, n_points)

    info_ref_all = ref_result.model.information(theta_2d)
    info_focal_all = focal_result.model.information(theta_2d)

    item_drf = np.zeros(n_items)
    info_diff_max = np.zeros(n_items)

    for j in range(n_items):
        diff = np.abs(info_ref_all[:, j] - info_focal_all[:, j])
        item_drf[j] = integrate.trapezoid(diff, theta_grid)
        info_diff_max[j] = diff.max()

    return {
        "item_drf": item_drf,
        "info_diff_max": info_diff_max,
        "info_ref": info_ref_all.T,
        "info_focal": info_focal_all.T,
        "theta_grid": theta_grid,
    }


def plot_drf(
    drf_result: dict,
    ax=None,
    **kwargs,
):
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
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        axes = [ax, ax.twinx()]

    theta = drf_result["theta_grid"]
    info_ref = drf_result["information_ref"]
    info_focal = drf_result["information_focal"]

    axes[0].plot(
        theta, info_ref, label=f"Reference ({drf_result['ref_group']})", linewidth=2
    )
    axes[0].plot(
        theta, info_focal, label=f"Focal ({drf_result['focal_group']})", linewidth=2
    )
    axes[0].fill_between(theta, info_ref, info_focal, alpha=0.3)
    axes[0].set_xlabel(r"$\theta$ (Ability)")
    axes[0].set_ylabel("Test Information")
    axes[0].set_title(f"Test Information Functions (DRF = {drf_result['DRF']:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    se_ref = 1 / np.sqrt(np.maximum(info_ref, 1e-10))
    se_focal = 1 / np.sqrt(np.maximum(info_focal, 1e-10))

    axes[1].plot(
        theta, se_ref, label=f"Reference ({drf_result['ref_group']})", linewidth=2
    )
    axes[1].plot(
        theta, se_focal, label=f"Focal ({drf_result['focal_group']})", linewidth=2
    )
    axes[1].set_xlabel(r"$\theta$ (Ability)")
    axes[1].set_ylabel("Standard Error")
    axes[1].set_title("Standard Error of Measurement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return axes


def reliability_invariance(
    data: NDArray[np.int_],
    groups: NDArray,
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    n_bootstrap: int = 100,
    seed: int | None = None,
) -> dict[str, float]:
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
    """
    from mirt import fit_mirt

    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    groups = np.asarray(groups)

    ref_data, focal_data, _, _, _, _ = split_groups(data, groups)
    ref_result, focal_result = fit_group_models(ref_data, focal_data, model=model)

    rel_ref = _compute_marginal_reliability(ref_result.model, (-4, 4))
    rel_focal = _compute_marginal_reliability(focal_result.model, (-4, 4))
    rel_diff = rel_ref - rel_focal

    boot_diffs = []

    for _ in range(n_bootstrap):
        ref_idx = rng.choice(len(ref_data), size=len(ref_data), replace=True)
        focal_idx = rng.choice(len(focal_data), size=len(focal_data), replace=True)

        try:
            boot_ref = fit_mirt(ref_data[ref_idx], model=model, verbose=False)
            boot_focal = fit_mirt(focal_data[focal_idx], model=model, verbose=False)

            boot_rel_ref = _compute_marginal_reliability(boot_ref.model, (-4, 4))
            boot_rel_focal = _compute_marginal_reliability(boot_focal.model, (-4, 4))

            boot_diffs.append(boot_rel_ref - boot_rel_focal)
        except Exception:
            continue

    if len(boot_diffs) < 10:
        se = np.nan
        z = np.nan
        p_value = np.nan
    else:
        se = float(np.std(boot_diffs, ddof=1))
        z = rel_diff / (se + 1e-10)
        from scipy import stats

        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "reliability_ref": rel_ref,
        "reliability_focal": rel_focal,
        "reliability_diff": rel_diff,
        "reliability_diff_se": se,
        "z": z,
        "p_value": p_value,
    }
