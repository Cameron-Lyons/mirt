"""Differential Test Functioning (DTF) analysis.

DTF examines whether the entire test functions differently across groups,
aggregating item-level DIF to a test-level measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

from mirt.constants import PROB_EPSILON
from mirt.diagnostics._utils import create_theta_grid, fit_group_models, split_groups

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def compute_dtf(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    method: Literal["signed", "unsigned", "expected_score"] = "unsigned",
    theta_range: tuple[float, float] = (-4, 4),
    n_quadpts: int = 49,
    n_bootstrap: int = 100,
    **fit_kwargs: Any,
) -> dict[str, float | NDArray[np.float64]]:
    """Compute Differential Test Functioning statistics.

    DTF measures the difference in expected total scores between groups
    at each ability level, then aggregates across the ability distribution.

    Parameters
    ----------
    data : NDArray
        Response matrix (n_persons, n_items)
    groups : NDArray
        Group membership (n_persons,) with exactly 2 unique values
    model : str
        IRT model to fit
    method : str
        DTF method:
        - 'signed': Signed area (can be positive or negative)
        - 'unsigned': Unsigned area (absolute difference)
        - 'expected_score': Expected score difference at each theta
    theta_range : tuple
        Range of theta values for integration
    n_quadpts : int
        Number of quadrature points
    **fit_kwargs
        Additional arguments passed to fit_mirt()

    Returns
    -------
    dict
        Dictionary with:
        - 'DTF': Overall DTF statistic
        - 'DTF_SE': Standard error (bootstrap)
        - 'p_value': Statistical significance
        - 'expected_score_ref': Expected score curve for reference group
        - 'expected_score_focal': Expected score curve for focal group
        - 'theta_grid': Theta values used
    """
    data = np.asarray(data)
    groups = np.asarray(groups)

    ref_data, focal_data, ref_mask, focal_mask, ref_group, focal_group = split_groups(
        data, groups
    )
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    theta_grid, _ = create_theta_grid(theta_range, n_quadpts)

    exp_score_ref = _compute_expected_score(ref_result.model, theta_grid)
    exp_score_focal = _compute_expected_score(focal_result.model, theta_grid)

    if method == "signed":
        diff = exp_score_ref - exp_score_focal
        dtf = float(integrate.trapezoid(diff, theta_grid))

    elif method == "unsigned":
        diff = np.abs(exp_score_ref - exp_score_focal)
        dtf = float(integrate.trapezoid(diff, theta_grid))

    elif method == "expected_score":
        diff = exp_score_ref - exp_score_focal
        dtf = float(np.mean(np.abs(diff)))

    else:
        raise ValueError(f"Unknown DTF method: {method}")

    dtf_se, p_value = _bootstrap_dtf_se(
        data,
        groups,
        model,
        method,
        theta_range,
        n_quadpts,
        n_bootstrap=n_bootstrap,
        **fit_kwargs,
    )

    return {
        "DTF": dtf,
        "DTF_SE": dtf_se,
        "p_value": p_value,
        "method": method,
        "expected_score_ref": exp_score_ref,
        "expected_score_focal": exp_score_focal,
        "theta_grid": theta_grid,
        "ref_group": ref_group,
        "focal_group": focal_group,
    }


def _compute_expected_score(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute expected total score at each theta."""
    theta_2d = theta.reshape(-1, 1)
    probs = model.probability(theta_2d)

    if probs.ndim == 2:
        return probs.sum(axis=1)
    else:
        expected = np.zeros(len(theta))
        for i in range(model.n_items):
            item_probs = model.probability(theta_2d, i)
            if item_probs.ndim == 2:
                n_cats = item_probs.shape[1]
                categories = np.arange(n_cats)
                expected += (item_probs * categories).sum(axis=1)
            else:
                expected += item_probs
        return expected


def _bootstrap_dtf_se(
    data: NDArray[np.int_],
    groups: NDArray[Any],
    model: str,
    method: str,
    theta_range: tuple[float, float],
    n_quadpts: int,
    n_bootstrap: int = 100,
    **fit_kwargs: Any,
) -> tuple[float, float]:
    """Bootstrap standard error for DTF."""
    from mirt import fit_mirt

    rng = np.random.default_rng(42)
    len(groups)

    unique_groups = np.unique(groups)
    ref_group, focal_group = unique_groups[0], unique_groups[1]

    boot_dtf = []

    for _ in range(n_bootstrap):
        ref_mask = groups == ref_group
        focal_mask = groups == focal_group

        ref_idx = np.where(ref_mask)[0]
        focal_idx = np.where(focal_mask)[0]

        boot_ref_idx = rng.choice(ref_idx, size=len(ref_idx), replace=True)
        boot_focal_idx = rng.choice(focal_idx, size=len(focal_idx), replace=True)

        boot_ref_data = data[boot_ref_idx]
        boot_focal_data = data[boot_focal_idx]

        try:
            ref_result = fit_mirt(
                boot_ref_data, model=model, verbose=False, **fit_kwargs
            )
            focal_result = fit_mirt(
                boot_focal_data, model=model, verbose=False, **fit_kwargs
            )

            theta_grid = np.linspace(theta_range[0], theta_range[1], n_quadpts)
            exp_ref = _compute_expected_score(ref_result.model, theta_grid)
            exp_focal = _compute_expected_score(focal_result.model, theta_grid)

            if method == "signed":
                diff = exp_ref - exp_focal
                dtf = float(integrate.trapezoid(diff, theta_grid))
            elif method == "unsigned":
                diff = np.abs(exp_ref - exp_focal)
                dtf = float(integrate.trapezoid(diff, theta_grid))
            else:
                dtf = float(np.mean(np.abs(exp_ref - exp_focal)))

            boot_dtf.append(dtf)

        except Exception:
            continue

    if len(boot_dtf) < 10:
        return np.nan, np.nan

    boot_dtf = np.array(boot_dtf)
    se = float(np.std(boot_dtf, ddof=1))

    z = np.mean(boot_dtf) / (se + PROB_EPSILON)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return se, p_value


def plot_dtf(
    dtf_result: dict[str, Any],
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Plot DTF results showing expected score curves.

    Parameters
    ----------
    dtf_result : dict
        Result from compute_dtf()
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
        _, ax = plt.subplots(figsize=(8, 6))

    theta = dtf_result["theta_grid"]
    exp_ref = dtf_result["expected_score_ref"]
    exp_focal = dtf_result["expected_score_focal"]

    ax.plot(theta, exp_ref, label=f"Reference ({dtf_result['ref_group']})", linewidth=2)
    ax.plot(theta, exp_focal, label=f"Focal ({dtf_result['focal_group']})", linewidth=2)

    ax.fill_between(theta, exp_ref, exp_focal, alpha=0.3, label="Difference")

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Expected Score")
    ax.set_title(f"Differential Test Functioning (DTF = {dtf_result['DTF']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
