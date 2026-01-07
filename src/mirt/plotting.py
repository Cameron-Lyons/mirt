"""Plotting functions for IRT models and diagnostics.

This module provides visualization tools for IRT analysis including:
- Item Characteristic Curves (ICC)
- Item/Test Information Functions
- Ability distributions
- Fit statistics plots
- Person-item maps (Wright maps)
- DIF visualizations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _check_matplotlib() -> Any:
    """Check if matplotlib is available and return pyplot."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def plot_icc(
    model: BaseItemModel,
    item_idx: int | list[int] | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    show_legend: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot Item Characteristic Curve(s).

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    item_idx : int, list of int, or None
        Item index(es) to plot. If None, plot all items.
    theta_range : tuple
        Range of theta values (min, max)
    n_points : int
        Number of points for smooth curve
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    show_legend : bool
        Whether to show legend
    **kwargs
        Additional arguments passed to plot()

    Returns
    -------
    matplotlib Axes
        The axes with the plot
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(theta_range[0], theta_range[1], n_points).reshape(-1, 1)

    if item_idx is None:
        item_idx = list(range(model.n_items))
    elif isinstance(item_idx, int):
        item_idx = [item_idx]

    for idx in item_idx:
        prob = model.probability(theta, idx)
        label = model.item_names[idx] if model.item_names else f"Item {idx + 1}"
        ax.plot(theta.ravel(), prob, label=label, **kwargs)

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("P(X = 1)")
    ax.set_title("Item Characteristic Curves")
    ax.set_ylim(0, 1)
    ax.set_xlim(theta_range)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    if show_legend and len(item_idx) <= 10:
        ax.legend(loc="best")

    return ax


def plot_category_curves(
    model: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot category response curves for polytomous items.

    Parameters
    ----------
    model : BaseItemModel
        Fitted polytomous IRT model
    item_idx : int
        Item index to plot
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of points for smooth curve
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to plot()

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(theta_range[0], theta_range[1], n_points).reshape(-1, 1)

    probs = model.probability(theta, item_idx)

    n_categories = probs.shape[1] if probs.ndim > 1 else 2

    if probs.ndim == 1:
        ax.plot(theta.ravel(), 1 - probs, label="Category 0", **kwargs)
        ax.plot(theta.ravel(), probs, label="Category 1", **kwargs)
    else:
        for cat in range(n_categories):
            ax.plot(theta.ravel(), probs[:, cat], label=f"Category {cat}", **kwargs)

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("P(X = k)")
    item_name = (
        model.item_names[item_idx] if model.item_names else f"Item {item_idx + 1}"
    )
    ax.set_title(f"Category Response Curves: {item_name}")
    ax.set_ylim(0, 1)
    ax.set_xlim(theta_range)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return ax


def plot_information(
    model: BaseItemModel,
    item_idx: int | list[int] | None = None,
    test_info: bool = True,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot item and/or test information functions.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    item_idx : int, list of int, or None
        Item index(es) to plot. If None and test_info=False, plot all items.
    test_info : bool
        Whether to plot test information (sum of item information)
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of points for smooth curve
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to plot()

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(theta_range[0], theta_range[1], n_points).reshape(-1, 1)

    all_info = model.information(theta)

    if test_info:
        total_info = all_info.sum(axis=1)
        ax.plot(
            theta.ravel(), total_info, label="Test Information", linewidth=2, **kwargs
        )

    if item_idx is not None or not test_info:
        if item_idx is None:
            item_idx = list(range(model.n_items))
        elif isinstance(item_idx, int):
            item_idx = [item_idx]

        for idx in item_idx:
            info = all_info[:, idx]
            label = model.item_names[idx] if model.item_names else f"Item {idx + 1}"
            ax.plot(theta.ravel(), info, label=label, alpha=0.7, **kwargs)

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Information")
    ax.set_title("Information Function")
    ax.set_xlim(theta_range)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return ax


def plot_ability_distribution(
    theta: NDArray[np.float64],
    se: NDArray[np.float64] | None = None,
    bins: int | str = "auto",
    ax: Any = None,
    show_density: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot distribution of ability estimates.

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,) or (n_persons, n_factors)
    se : NDArray, optional
        Standard errors for ability estimates
    bins : int or str
        Number of bins or binning method for histogram
    ax : matplotlib Axes, optional
        Axes to plot on
    show_density : bool
        Whether to show kernel density estimate
    **kwargs
        Additional arguments passed to hist()

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.asarray(theta)
    if theta.ndim == 2:
        theta = theta[:, 0]

    ax.hist(theta, bins=bins, density=True, alpha=0.7, edgecolor="black", **kwargs)

    if show_density:
        from scipy import stats

        x = np.linspace(theta.min() - 0.5, theta.max() + 0.5, 200)
        kde = stats.gaussian_kde(theta)
        ax.plot(x, kde(x), "r-", linewidth=2, label="KDE")

    x = np.linspace(-4, 4, 200)
    from scipy import stats

    ax.plot(x, stats.norm.pdf(x), "k--", alpha=0.5, label="N(0,1)")

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Ability Estimates")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if se is not None:
        mean_se = np.mean(se)
        ax.text(
            0.95,
            0.95,
            f"Mean SE = {mean_se:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return ax


def plot_itemfit(
    fit_stats: dict[str, NDArray[np.float64]],
    statistic: str = "infit",
    criterion: tuple[float, float] = (0.7, 1.3),
    item_names: list[str] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot item fit statistics.

    Parameters
    ----------
    fit_stats : dict
        Dictionary with fit statistics (from itemfit())
    statistic : str
        Which statistic to plot ('infit' or 'outfit')
    criterion : tuple
        Acceptable range for fit statistic
    item_names : list, optional
        Names for items
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments passed to bar()

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    values = fit_stats.get(statistic, fit_stats.get("infit"))
    n_items = len(values)

    if item_names is None:
        item_names = [f"Item {i + 1}" for i in range(n_items)]

    colors = []
    for v in values:
        if criterion[0] <= v <= criterion[1]:
            colors.append("steelblue")
        else:
            colors.append("tomato")

    ax.bar(range(n_items), values, color=colors, **kwargs)

    ax.axhline(y=criterion[0], color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=criterion[1], color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=1.0, color="green", linestyle="-", alpha=0.5)

    ax.set_xticks(range(n_items))
    ax.set_xticklabels(item_names, rotation=45, ha="right")
    ax.set_xlabel("Item")
    ax.set_ylabel(statistic.capitalize())
    ax.set_title(f"Item Fit: {statistic.capitalize()}")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", label="Acceptable fit"),
        Patch(facecolor="tomato", label="Misfit"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    return ax


def plot_person_item_map(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    ax: Any = None,
    bins: int = 30,
    **kwargs: Any,
) -> Any:
    """Plot Wright map (person-item map).

    Shows distribution of persons and item difficulties on same scale.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    theta : NDArray
        Ability estimates
    ax : matplotlib Axes, optional
        Axes to plot on
    bins : int
        Number of bins for person histogram
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    theta = np.asarray(theta)
    if theta.ndim == 2:
        theta = theta[:, 0]

    params = model.parameters
    if "difficulty" in params:
        difficulties = params["difficulty"]
    elif "intercepts" in params:
        difficulties = -params["intercepts"]
    else:
        difficulties = np.zeros(model.n_items)

    ax2 = ax.twinx()

    ax.hist(
        theta,
        bins=bins,
        orientation="horizontal",
        alpha=0.7,
        color="steelblue",
        label="Persons",
    )

    item_names = model.item_names or [f"Item {i + 1}" for i in range(model.n_items)]
    for i, (d, name) in enumerate(zip(difficulties, item_names)):
        ax2.plot(0.5, d, "ro", markersize=8)
        ax2.annotate(name, (0.55, d), fontsize=8, va="center")

    ax.set_ylabel(r"$\theta$ / Difficulty")
    ax.set_xlabel("Person Count")
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])

    ax.set_title("Person-Item Map (Wright Map)")
    ax.legend(loc="upper left")

    return ax


def plot_dif(
    dif_results: dict[str, NDArray[np.float64]],
    effect_size_key: str = "effect_size",
    classification_key: str = "classification",
    item_names: list[str] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot DIF effect sizes with ETS classification.

    Parameters
    ----------
    dif_results : dict
        Dictionary with DIF results (from dif())
    effect_size_key : str
        Key for effect size values
    classification_key : str
        Key for ETS classification
    item_names : list, optional
        Names for items
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    effect_sizes = dif_results.get(effect_size_key, np.zeros(1))
    classifications = dif_results.get(classification_key, ["A"] * len(effect_sizes))

    n_items = len(effect_sizes)
    if item_names is None:
        item_names = [f"Item {i + 1}" for i in range(n_items)]

    color_map = {"A": "green", "B": "gold", "C": "red"}
    colors = [color_map.get(str(c), "gray") for c in classifications]

    ax.bar(range(n_items), np.abs(effect_sizes), color=colors, **kwargs)

    ax.axhline(y=0.426, color="gold", linestyle="--", alpha=0.7, label="B threshold")
    ax.axhline(y=0.638, color="red", linestyle="--", alpha=0.7, label="C threshold")

    ax.set_xticks(range(n_items))
    ax.set_xticklabels(item_names, rotation=45, ha="right")
    ax.set_xlabel("Item")
    ax.set_ylabel("|Effect Size|")
    ax.set_title("DIF Effect Sizes (ETS Classification)")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="A (Negligible)"),
        Patch(facecolor="gold", label="B (Moderate)"),
        Patch(facecolor="red", label="C (Large)"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    return ax


def plot_expected_score(
    model: BaseItemModel,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot expected total score as function of theta.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of points for smooth curve
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(theta_range[0], theta_range[1], n_points).reshape(-1, 1)

    probs = model.probability(theta)
    if probs.ndim == 2:
        expected = probs.sum(axis=1)
    else:
        expected = probs

    ax.plot(theta.ravel(), expected, linewidth=2, **kwargs)

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Expected Score")
    ax.set_title("Test Characteristic Curve")
    ax.set_xlim(theta_range)
    ax.set_ylim(0, model.n_items)
    ax.grid(True, alpha=0.3)

    return ax


def plot_se(
    model: BaseItemModel,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot standard error of measurement as function of theta.

    SE = 1 / sqrt(Information)

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    theta_range : tuple
        Range of theta values
    n_points : int
        Number of points for smooth curve
    ax : matplotlib Axes, optional
        Axes to plot on
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib Axes
    """
    plt = _check_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    theta = np.linspace(theta_range[0], theta_range[1], n_points).reshape(-1, 1)

    info = model.information(theta).sum(axis=1)
    se = 1 / np.sqrt(np.maximum(info, 1e-10))

    ax.plot(theta.ravel(), se, linewidth=2, **kwargs)

    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Standard Error")
    ax.set_title("Standard Error of Measurement")
    ax.set_xlim(theta_range)
    ax.grid(True, alpha=0.3)

    return ax
