"""Utilities for embedding matplotlib plots as base64 images."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.models.base import BaseItemModel


def _get_pyplot() -> Any:
    """Get matplotlib.pyplot, raising helpful error if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for report generation. "
            "Install it with: pip install matplotlib"
        ) from None


def figure_to_base64(fig: Any, dpi: int = 100, fmt: str = "png") -> str:
    """Convert matplotlib figure to base64-encoded string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert.
    dpi : int
        Resolution in dots per inch.
    fmt : str
        Image format (png recommended for quality).

    Returns
    -------
    str
        Base64-encoded image data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def create_icc_plot_base64(
    model: BaseItemModel,
    item_idx: int | list[int] | None = None,
    dpi: int = 100,
) -> str:
    """Create ICC plot and return as base64.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    item_idx : int, list of int, or None
        Item indices to plot.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_icc

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_icc(model, item_idx=item_idx, ax=ax)
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_information_plot_base64(
    model: BaseItemModel,
    test_info: bool = True,
    dpi: int = 100,
) -> str:
    """Create information function plot and return as base64.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    test_info : bool
        Include test information.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_information

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_information(model, test_info=test_info, ax=ax)
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_itemfit_plot_base64(
    fit_stats: dict[str, Any],
    statistic: str = "infit",
    item_names: list[str] | None = None,
    dpi: int = 100,
) -> str:
    """Create item fit plot and return as base64.

    Parameters
    ----------
    fit_stats : dict
        Item fit statistics from compute_itemfit.
    statistic : str
        Which statistic to plot.
    item_names : list of str, optional
        Item labels.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_itemfit

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_itemfit(fit_stats, statistic=statistic, item_names=item_names, ax=ax)
    fig.tight_layout()
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_wright_map_base64(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    dpi: int = 100,
) -> str:
    """Create Wright map and return as base64.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    theta : ndarray
        Ability estimates.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_person_item_map

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_person_item_map(model, theta, ax=ax)
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_dif_plot_base64(
    dif_results: dict[str, Any],
    item_names: list[str] | None = None,
    dpi: int = 100,
) -> str:
    """Create DIF effect size plot and return as base64.

    Parameters
    ----------
    dif_results : dict
        DIF analysis results.
    item_names : list of str, optional
        Item labels.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_dif

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_dif(dif_results, item_names=item_names, ax=ax)
    fig.tight_layout()
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_ability_distribution_base64(
    theta: NDArray[np.float64],
    se: NDArray[np.float64] | None = None,
    dpi: int = 100,
) -> str:
    """Create ability distribution plot and return as base64.

    Parameters
    ----------
    theta : ndarray
        Ability estimates.
    se : ndarray, optional
        Standard errors.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_ability_distribution

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_ability_distribution(theta, se=se, ax=ax)
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result


def create_se_plot_base64(model: BaseItemModel, dpi: int = 100) -> str:
    """Create standard error of measurement plot and return as base64.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model.
    dpi : int
        Plot resolution.

    Returns
    -------
    str
        Base64-encoded PNG.
    """
    from mirt.plotting import plot_se

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_se(model, ax=ax)
    result = figure_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return result
