"""Utilities for embedding matplotlib plots as base64 images."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.models.base import BaseItemModel


@lru_cache(maxsize=1)
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


@contextmanager
def _figure(figsize: tuple[float, float]) -> Iterator[tuple[Any, Any]]:
    """Create a figure and guarantee that its resources are released."""
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


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
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    with io.BytesIO() as buf:
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        return base64.b64encode(buf.getbuffer()).decode("ascii")


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

    with _figure((10, 6)) as (fig, ax):
        plot_icc(model, item_idx=item_idx, ax=ax)
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((10, 6)) as (fig, ax):
        plot_information(model, test_info=test_info, ax=ax)
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((12, 6)) as (fig, ax):
        plot_itemfit(fit_stats, statistic=statistic, item_names=item_names, ax=ax)
        fig.tight_layout()
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((10, 8)) as (fig, ax):
        plot_person_item_map(model, theta, ax=ax)
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((12, 6)) as (fig, ax):
        plot_dif(dif_results, item_names=item_names, ax=ax)
        fig.tight_layout()
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((10, 6)) as (fig, ax):
        plot_ability_distribution(theta, se=se, ax=ax)
        return figure_to_base64(fig, dpi=dpi)


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

    with _figure((10, 6)) as (fig, ax):
        plot_se(model, ax=ax)
        return figure_to_base64(fig, dpi=dpi)
