"""Validated plotting helpers with call-specific numerical dependencies.

Importing :mod:`mirt` or this module does not load NumPy, SciPy, or Matplotlib.
NumPy is requested by the first plotting call, Matplotlib only when a function
must create its own axes, and SciPy only for a requested density estimate.
Callers may also supply an existing axes-compatible object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def _check_matplotlib() -> Any:
    """Return pyplot or raise an actionable optional-dependency error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required to create plotting axes; "
            "install it with `pip install 'mirt[plot]'`"
        ) from None
    return plt


def _resolve_axes(ax: Any, figsize: tuple[float, float]) -> Any:
    if ax is not None:
        return ax
    plt = _check_matplotlib()
    _, created_ax = plt.subplots(figsize=figsize)
    return created_ax


def _resolve_model(model: BaseItemModel | FitResult) -> BaseItemModel:
    """Accept a fitted result wherever an item model is plotted."""
    from mirt.results._common import resolve_item_model

    resolved: BaseItemModel = resolve_item_model(model)
    return resolved


def _model_size(model: BaseItemModel) -> tuple[int, int]:
    import numpy as np

    n_items = getattr(model, "n_items", None)
    n_factors = getattr(model, "n_factors", 1)
    if (
        isinstance(n_items, bool)
        or not isinstance(n_items, (int, np.integer))
        or n_items < 1
    ):
        raise ValueError("model.n_items must be a positive integer")
    if (
        isinstance(n_factors, bool)
        or not isinstance(n_factors, (int, np.integer))
        or n_factors < 1
    ):
        raise ValueError("model.n_factors must be a positive integer")
    return int(n_items), int(n_factors)


def _theta_grid(
    model: BaseItemModel,
    theta_range: tuple[float, float],
    n_points: int,
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]:
    import numpy as np

    _, n_factors = _model_size(model)
    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range must be strictly increasing")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer of at least 2")
    if n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError("factor must be an integer")
    if factor < 0 or factor >= n_factors:
        raise ValueError(f"factor must be in [0, {n_factors})")

    values = np.linspace(float(limits[0]), float(limits[1]), int(n_points))
    theta = np.zeros((int(n_points), n_factors), dtype=np.float64)
    theta[:, int(factor)] = values
    return values, theta, (float(limits[0]), float(limits[1]))


def _item_indices(
    model: BaseItemModel,
    item_idx: int | list[int] | None,
) -> list[int]:
    import numpy as np

    n_items, _ = _model_size(model)
    if item_idx is None:
        return list(range(n_items))
    if isinstance(item_idx, bool):
        raise ValueError("item indices must be integers")
    if isinstance(item_idx, (int, np.integer)):
        candidates: Sequence[Any] = [item_idx]
    else:
        try:
            candidates = list(item_idx)
        except TypeError:
            raise ValueError("item_idx must be an integer or a sequence") from None
    if not candidates:
        return []

    indices: list[int] = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("item indices must be integers")
        index = int(value)
        if index < 0 or index >= n_items:
            raise IndexError(f"item index {index} out of range [0, {n_items})")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("item indices must be unique")
    return indices


def _item_name(model: BaseItemModel, index: int) -> str:
    names = getattr(model, "item_names", None)
    if names is not None and len(names) == model.n_items:
        return str(names[index])
    return f"Item {index + 1}"


def _probability_curves(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    import numpy as np

    probabilities = np.asarray(model.probability(theta, item_idx), dtype=np.float64)
    n_points = theta.shape[0]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("item probabilities must contain only finite values")
    if np.any(probabilities < -PROB_EPSILON) or np.any(
        probabilities > 1.0 + PROB_EPSILON
    ):
        raise ValueError("item probabilities must lie in [0, 1]")
    if probabilities.ndim == 1:
        if probabilities.shape != (n_points,):
            raise ValueError("item probabilities must have shape (n_points,)")
    elif probabilities.ndim == 2:
        if probabilities.shape[0] != n_points or probabilities.shape[1] < 2:
            raise ValueError(
                "category probabilities must have shape (n_points, n_categories)"
            )
        if not np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("category probabilities must sum to one")
    else:
        raise ValueError("item probabilities must be one- or two-dimensional")
    return np.clip(probabilities, 0.0, 1.0)


def _curve(
    values: Any,
    n_points: int,
    name: str,
    *,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    import numpy as np

    result = np.asarray(values, dtype=np.float64)
    if result.shape != (n_points,):
        raise ValueError(f"{name} must have shape (n_points,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _full_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    import numpy as np

    n_items, _ = _model_size(model)
    n_points = theta.shape[0]
    information = np.asarray(model.information(theta), dtype=np.float64)
    if information.ndim == 1:
        return (
            _curve(
                information,
                n_points,
                "test information",
                nonnegative=True,
            ),
            None,
        )
    if information.shape != (n_points, n_items):
        raise ValueError(
            "full information must have shape (n_points,) or (n_points, n_items)"
        )
    if not np.all(np.isfinite(information)) or np.any(information < 0.0):
        raise ValueError("item information must be finite and nonnegative")
    return np.sum(information, axis=1), information


def _item_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    return _curve(
        model.information(theta, item_idx),
        theta.shape[0],
        f"information for item {item_idx}",
        nonnegative=True,
    )


def _category_columns(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return category probabilities, adding ``1 - P`` to binary curves."""
    import numpy as np

    if probabilities.ndim == 1:
        return np.column_stack((1.0 - probabilities, probabilities))
    return probabilities


def _numerical_factor_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
    factor: int,
) -> NDArray[np.float64]:
    """Fisher information of one item along one factor, by central differences.

    The information along the unit vector ``e_k`` is
    ``sum_c (dP_c / dtheta_k)^2 / P_c`` over the item's categories.
    """
    import numpy as np

    step = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(theta[:, factor]))
    upper = theta.copy()
    lower = theta.copy()
    upper[:, factor] += step
    lower[:, factor] -= step
    width = (upper[:, factor] - lower[:, factor])[:, None]
    center = _category_columns(_probability_curves(model, theta, item_idx))
    derivative = (
        _category_columns(_probability_curves(model, upper, item_idx))
        - _category_columns(_probability_curves(model, lower, item_idx))
    ) / width
    return np.sum(np.square(derivative) / np.maximum(center, PROB_EPSILON), axis=1)


def _factor_item_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
    factor: int,
) -> NDArray[np.float64]:
    """Information of one item along ``factor``: ``[I_j(theta)]_kk``.

    Models with ``item_information_matrix`` supply the diagonal element of
    their exact Fisher matrices; other models are differentiated
    numerically along the factor.
    """
    import numpy as np

    from mirt.exceptions import MirtModelError

    n_points, n_factors = theta.shape
    matrix_function = getattr(model, "item_information_matrix", None)
    matrices = None
    if callable(matrix_function):
        try:
            matrices = np.asarray(matrix_function(theta, item_idx), dtype=np.float64)
        except (MirtModelError, NotImplementedError):
            pass  # e.g. mixed-format components without exact matrices
    if matrices is None:
        values = _numerical_factor_information(model, theta, item_idx, factor)
    elif matrices.shape == (n_points, n_factors, n_factors):
        values = matrices[:, factor, factor]
    else:
        raise ValueError(
            "item information matrices must have shape (n_points, n_factors, n_factors)"
        )
    return _curve(
        values,
        n_points,
        f"information for item {item_idx}",
        nonnegative=True,
    )


def _factor_test_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    factor: int,
    item_curves: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Test information along ``factor``: ``[I(theta)]_kk``, summed over items.

    ``item_curves`` may hold every item's curve along the factor already.
    """
    import numpy as np

    n_items, n_factors = _model_size(model)
    n_points = theta.shape[0]
    matrix_function = getattr(model, "test_information_matrix", None)
    if item_curves is None and callable(matrix_function):
        matrices = np.asarray(matrix_function(theta), dtype=np.float64)
        if matrices.shape != (n_points, n_factors, n_factors):
            raise ValueError(
                "test information matrices must have shape "
                "(n_points, n_factors, n_factors)"
            )
        return _curve(
            matrices[:, factor, factor],
            n_points,
            "test information",
            nonnegative=True,
        )
    if item_curves is None:
        item_curves = np.column_stack(
            [
                _factor_item_information(model, theta, index, factor)
                for index in range(n_items)
            ]
        )
    return np.sum(item_curves, axis=1)


def _information_curves(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    indices: list[int],
    include_total: bool,
    factor: int = 0,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    import numpy as np

    n_items, n_factors = _model_size(model)
    if n_factors > 1:
        # Information along the plotted factor, not the trace over factors.
        item_curves = (
            np.column_stack(
                [
                    _factor_item_information(model, theta, index, factor)
                    for index in indices
                ]
            )
            if indices
            else None
        )
        if not include_total:
            return None, item_curves
        all_items = item_curves is not None and sorted(indices) == list(range(n_items))
        total = _factor_test_information(
            model, theta, factor, item_curves if all_items else None
        )
        return total, item_curves

    is_polytomous = bool(getattr(model, "is_polytomous", False))

    if is_polytomous and indices:
        item_curves = np.column_stack(
            [_item_information(model, theta, index) for index in indices]
        )
        if not include_total:
            return None, item_curves
        if len(indices) == n_items and set(indices) == set(range(n_items)):
            return np.sum(item_curves, axis=1), item_curves
        total, _ = _full_information(model, theta)
        return total, item_curves

    total, full_item_curves = _full_information(model, theta)
    if not indices:
        return total if include_total else None, None
    if full_item_curves is None:
        selected = np.column_stack(
            [_item_information(model, theta, index) for index in indices]
        )
    else:
        selected = full_item_curves[:, indices]
    return total if include_total else None, selected


def _factor_label(label: str, model: BaseItemModel, factor: int) -> str:
    """Name the plotted factor of a multidimensional model in an axis label."""
    _, n_factors = _model_size(model)
    return label if n_factors == 1 else f"{label} (factor {factor + 1})"


def _expected_score(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    import numpy as np

    expected = _curve(
        model.expected_score(theta),
        theta.shape[0],
        "expected score",
        nonnegative=True,
    )
    maximum = _maximum_score(model)
    if np.any(expected > maximum + PROB_EPSILON):
        raise ValueError("expected score exceeds the model maximum")
    return expected


def _maximum_score(model: BaseItemModel) -> float:
    import numpy as np

    n_items, _ = _model_size(model)
    category_counts = getattr(model, "n_categories", None)
    if isinstance(category_counts, Sequence) and not isinstance(
        category_counts, (str, bytes)
    ):
        counts = np.asarray(category_counts, dtype=np.int64)
        if counts.shape == (n_items,) and np.all(counts >= 2):
            return float(np.sum(counts - 1))
    return float(n_items)


def _factor_values(theta: Any, factor: int, name: str) -> NDArray[np.float64]:
    import numpy as np

    values = np.asarray(theta, dtype=np.float64)
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError("factor must be an integer")
    if values.ndim == 1:
        if factor != 0:
            raise ValueError(f"factor must be 0 for one-dimensional {name}")
        selected = values
    elif values.ndim == 2:
        if factor < 0 or factor >= values.shape[1]:
            raise ValueError(f"factor must be in [0, {values.shape[1]})")
        selected = values[:, int(factor)]
    else:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if selected.size == 0 or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must contain finite values")
    return selected


def _names(item_names: list[str] | None, n_items: int) -> list[str]:
    if item_names is None:
        return [f"Item {index + 1}" for index in range(n_items)]
    if len(item_names) != n_items:
        raise ValueError(f"item_names must contain {n_items} entries")
    return [str(name) for name in item_names]


def _mapping_values(
    mapping: Mapping[str, Any], key: str, *, allow_missing: bool = False
) -> NDArray[np.float64]:
    import numpy as np

    if key not in mapping:
        raise ValueError(f"results do not contain {key!r}")
    values = np.asarray(mapping[key], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{key} must be a nonempty one-dimensional array")
    finite = np.isfinite(values) | (allow_missing & np.isnan(values))
    if not np.all(finite):
        raise ValueError(f"{key} must contain only finite values")
    return values


def _result_columns(
    results: Any, name: str
) -> tuple[Mapping[str, Any], list[str] | None]:
    """Read a results mapping or a pandas/polars DataFrame of item rows.

    Returns the columns by name and the item labels of a frame: the index of
    a pandas frame unless it is the default ``0, 1, ...`` range, or the
    ``item`` column of a polars frame, as :func:`mirt.itemfit` and
    :func:`mirt.dif` label their rows.
    """
    import numpy as np

    if isinstance(results, Mapping):
        return results, None
    columns = getattr(results, "columns", None)
    if columns is None:
        raise ValueError(f"{name} must be a mapping or a pandas or polars DataFrame")
    get_column = getattr(results, "get_column", None)
    if callable(get_column):
        values = {str(column): get_column(column).to_numpy() for column in columns}
        labels = values.pop("item", None)
    else:
        values = {
            str(column): np.asarray(results[column].to_numpy()) for column in columns
        }
        index = getattr(results, "index", None)
        labels = None if index is None else list(index)
        if labels is not None and getattr(index, "name", None) is None:
            if labels == list(range(len(labels))):
                labels = None
    names = None if labels is None else [str(label) for label in labels]
    return values, names


def _ordered_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
    import numpy as np

    pair = np.asarray(values, dtype=np.float64)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must contain two finite increasing values")
    return float(pair[0]), float(pair[1])


def plot_icc(
    model: BaseItemModel | FitResult,
    item_idx: int | list[int] | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    show_legend: bool = True,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot item response curves, including every category for ordinal items."""
    import numpy as np

    model = _resolve_model(model)

    if not isinstance(show_legend, (bool, np.bool_)):
        raise ValueError("show_legend must be boolean")
    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    indices = _item_indices(model, item_idx)
    if not indices:
        raise ValueError("at least one item must be selected")

    prepared: list[tuple[NDArray[np.float64], str]] = []
    has_categories = False
    for index in indices:
        probabilities = _probability_curves(model, theta, index)
        label = _item_name(model, index)
        if probabilities.ndim == 1:
            prepared.append((probabilities, label))
        else:
            has_categories = True
            prepared.extend(
                (probabilities[:, category], f"{label} · Category {category}")
                for category in range(probabilities.shape[1])
            )

    ax = _resolve_axes(ax, (8, 6))
    for values, label in prepared:
        ax.plot(theta_values, values, **{"label": label, **kwargs})
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("P(X = k)" if has_categories else "P(X = 1)")
    ax.set_title(
        "Item Response Curves" if has_categories else "Item Characteristic Curves"
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(limits)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    if show_legend and len(prepared) <= 10:
        ax.legend(loc="best")
    return ax


def plot_category_curves(
    model: BaseItemModel | FitResult,
    item_idx: int,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot all response-category curves for one item."""
    import numpy as np

    model = _resolve_model(model)

    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    indices = _item_indices(model, item_idx)
    probabilities = _probability_curves(model, theta, indices[0])
    if probabilities.ndim == 1:
        curves = np.column_stack((1.0 - probabilities, probabilities))
    else:
        curves = probabilities

    ax = _resolve_axes(ax, (8, 6))
    for category in range(curves.shape[1]):
        ax.plot(
            theta_values,
            curves[:, category],
            **{"label": f"Category {category}", **kwargs},
        )
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("P(X = k)")
    ax.set_title(f"Category Response Curves: {_item_name(model, indices[0])}")
    ax.set_ylim(0, 1)
    ax.set_xlim(limits)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_information(
    model: BaseItemModel | FitResult,
    item_idx: int | list[int] | None = None,
    test_info: bool = True,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot test information and optional item-information curves.

    For a multidimensional model the curves vary ``factor`` while holding the
    other factors at zero and show the information along that factor: the
    diagonal element ``[I(theta)]_kk`` of the Fisher information matrix, as
    R's ``testinfo(degrees=)`` for the angle pointing along the factor.
    This is not the trace over all factors that :func:`mirt.testinfo` and
    :func:`mirt.iteminfo` return for multidimensional models.
    """
    import numpy as np

    model = _resolve_model(model)

    if not isinstance(test_info, (bool, np.bool_)):
        raise ValueError("test_info must be boolean")
    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    if item_idx is None:
        indices = [] if test_info else _item_indices(model, None)
    else:
        indices = _item_indices(model, item_idx)
    if not test_info and not indices:
        raise ValueError("select at least one item when test_info is False")
    total, item_curves = _information_curves(
        model, theta, indices, bool(test_info), int(factor)
    )

    ax = _resolve_axes(ax, (8, 6))
    if total is not None:
        ax.plot(
            theta_values,
            total,
            **{"label": "Test Information", "linewidth": 2.0, **kwargs},
        )
    if item_curves is not None:
        for column, index in enumerate(indices):
            ax.plot(
                theta_values,
                item_curves[:, column],
                **{"label": _item_name(model, index), "alpha": 0.7, **kwargs},
            )
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel(_factor_label("Information", model, int(factor)))
    ax.set_title("Information Function")
    ax.set_xlim(limits)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_ability_distribution(
    theta: NDArray[np.float64],
    se: NDArray[np.float64] | None = None,
    bins: int | str = "auto",
    ax: Any = None,
    show_density: bool = True,
    *,
    factor: int = 0,
    show_normal: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot one ability dimension with optional KDE and normal reference."""
    import numpy as np

    if not isinstance(show_density, (bool, np.bool_)):
        raise ValueError("show_density must be boolean")
    if not isinstance(show_normal, (bool, np.bool_)):
        raise ValueError("show_normal must be boolean")
    values = _factor_values(theta, factor, "theta")
    try:
        np.histogram_bin_edges(values, bins=bins)
    except (TypeError, ValueError):
        raise ValueError("bins must be a valid histogram bin specification") from None
    histogram_kwargs = {
        "density": True,
        "alpha": 0.7,
        "edgecolor": "black",
        **kwargs,
    }
    density_scale = histogram_kwargs["density"]
    if not isinstance(density_scale, (bool, np.bool_)):
        raise ValueError("density must be boolean")
    if not density_scale and (show_density or show_normal):
        raise ValueError("density overlays require density=True")

    selected_se: NDArray[np.float64] | None = None
    if se is not None:
        se_values = np.asarray(se)
        selected_se = _factor_values(se, 0 if se_values.ndim == 1 else factor, "se")
        if selected_se.shape != values.shape:
            raise ValueError("se must match theta shape")
        if np.any(selected_se < 0.0):
            raise ValueError("se must be nonnegative")

    margin = max(0.5, float(np.ptp(values)) * 0.05)
    lower = min(-4.0, float(np.min(values)) - margin)
    upper = max(4.0, float(np.max(values)) + margin)
    x_values = np.linspace(lower, upper, 200)

    ax = _resolve_axes(ax, (8, 6))
    ax.hist(values, bins=bins, **histogram_kwargs)
    has_legend = False
    if show_density and values.size >= 2 and float(np.ptp(values)) > 0.0:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        ax.plot(x_values, kde(x_values), "r-", linewidth=2, label="KDE")
        has_legend = True
    if show_normal:
        normal_density = np.exp(-0.5 * np.square(x_values)) / np.sqrt(2.0 * np.pi)
        ax.plot(x_values, normal_density, "k--", alpha=0.5, label="N(0,1)")
        has_legend = True
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Density" if density_scale else "Count")
    ax.set_title("Distribution of Ability Estimates")
    if has_legend:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if selected_se is not None:
        ax.text(
            0.95,
            0.95,
            f"Mean SE = {float(np.mean(selected_se)):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    return ax


def plot_itemfit(
    fit_stats: Mapping[str, Any] | Any,
    statistic: str = "infit",
    criterion: tuple[float, float] = (0.7, 1.3),
    item_names: list[str] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot item-fit statistics and flag values outside a criterion band.

    ``fit_stats`` is the mapping of
    :func:`mirt.diagnostics.itemfit.compute_itemfit` or the pandas or polars
    DataFrame of :func:`mirt.itemfit`, whose item labels name the bars
    unless ``item_names`` is given.
    """
    columns, frame_names = _result_columns(fit_stats, "fit_stats")
    values = _mapping_values(columns, statistic)
    lower, upper = _ordered_pair(criterion, "criterion")
    names = _names(frame_names if item_names is None else item_names, values.size)
    colors = ["steelblue" if lower <= value <= upper else "tomato" for value in values]
    custom_color = "color" in kwargs
    bar_kwargs = {"color": colors, **kwargs}

    ax = _resolve_axes(ax, (10, 6))
    ax.bar(range(values.size), values, **bar_kwargs)
    ax.axhline(y=lower, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=upper, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=1.0, color="green", linestyle="-", alpha=0.5)
    ax.set_xticks(range(values.size))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Item")
    ax.set_ylabel(statistic.capitalize())
    ax.set_title(f"Item Fit: {statistic.capitalize()}")
    ax.grid(True, alpha=0.3, axis="y")
    if not custom_color:
        ax.scatter([], [], marker="s", color="steelblue", label="Acceptable fit")
        ax.scatter([], [], marker="s", color="tomato", label="Misfit")
        ax.legend(loc="best")
    return ax


def _slope_matrix(
    model: BaseItemModel,
    params: Mapping[str, Any],
    n_items: int,
    n_factors: int,
) -> NDArray[np.float64] | None:
    """Return item slopes of shape ``(n_items, n_factors)``, if the model has any."""
    import numpy as np

    slopes = params.get("slopes", params.get("discrimination"))
    if slopes is None:
        loading_matrix = getattr(model, "get_loading_matrix", None)
        if not callable(loading_matrix):
            return None
        slopes = loading_matrix()
    slope_values = np.asarray(slopes, dtype=np.float64)
    if slope_values.shape == (n_items,) and n_factors == 1:
        return slope_values[:, None]
    if slope_values.shape == (n_items, n_factors):
        return slope_values
    raise ValueError("slope parameters have an unsupported shape")


def _active_means(
    model: BaseItemModel,
    values: NDArray[np.float64],
    n_items: int,
    name: str,
) -> NDArray[np.float64]:
    """Mean of each item's active thresholds or steps, ignoring padding."""
    import numpy as np

    if values.ndim != 2 or values.shape[0] != n_items:
        raise ValueError(f"{name} parameters have an unsupported shape")
    counts = getattr(model, "n_categories", None)
    if not (isinstance(counts, Sequence) and len(counts) == n_items):
        return np.mean(values, axis=1)
    category_counts = np.asarray(counts, dtype=np.int64)
    if np.any(category_counts < 2) or np.any(category_counts - 1 > values.shape[1]):
        raise ValueError(f"n_categories is inconsistent with {name}")
    return np.asarray(
        [
            np.mean(values[index, : int(category_counts[index]) - 1])
            for index in range(n_items)
        ],
        dtype=np.float64,
    )


def _item_locations(model: BaseItemModel, factor: int) -> NDArray[np.float64]:
    """Locations of the items on ``factor``, NaN where none is defined.

    A location is the point of the factor, with the other factors at zero,
    where the item's (mean) logit is zero: the difficulty of a dichotomous
    item, the mean threshold of a graded item and the mean step of a
    (generalized) partial credit item, or ``-d / a_k`` for slope-intercept
    items. In several dimensions an item stored with a difficulty and slope
    vector ``a`` is located at ``sum(a) / a_k`` times that difficulty. Items
    with a zero slope on the factor have no location there. Mixed-format
    models locate each component's items.
    """
    import numpy as np

    n_items, n_factors = _model_size(model)
    if factor < 0 or factor >= n_factors:
        raise ValueError(f"factor must be in [0, {n_factors})")

    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        locations = np.full(n_items, np.nan)
        for component, items in model.components:
            try:
                locations[np.asarray(items)] = _item_locations(component, factor)
            except ValueError:
                # Families without item locations, such as nominal items.
                continue
        if not np.any(np.isfinite(locations)):
            raise ValueError("no item of this model has an item location")
        return locations

    params = model.parameters
    if not isinstance(params, Mapping):
        raise ValueError("model.parameters must be a mapping")

    defined = np.ones(n_items, dtype=np.bool_)
    if "difficulty" in params:
        difficulty = np.asarray(params["difficulty"], dtype=np.float64)
        if difficulty.shape == (n_items,):
            locations = difficulty
            scale_by_slopes = True
        elif difficulty.shape == (n_items, n_factors):
            locations = difficulty[:, factor]
            scale_by_slopes = False
        else:
            raise ValueError("difficulty parameters have an unsupported shape")
    elif "thresholds" in params:
        thresholds = np.asarray(params["thresholds"], dtype=np.float64)
        if thresholds.ndim != 2 or thresholds.shape[0] != n_items:
            raise ValueError("threshold parameters have an unsupported shape")
        locations = _active_means(model, thresholds, n_items, "thresholds")
        scale_by_slopes = True
    elif "steps" in params:
        steps = np.asarray(params["steps"], dtype=np.float64)
        locations = _active_means(model, steps, n_items, "steps")
        scale_by_slopes = True
    elif "intercepts" in params:
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        if intercepts.shape != (n_items,):
            raise ValueError(
                "intercept parameters have an unsupported shape; nominal "
                "response items have no single item location"
            )
        slopes = _slope_matrix(model, params, n_items, n_factors)
        if slopes is None:
            locations = -intercepts
        else:
            denominator = slopes[:, factor]
            defined = np.abs(denominator) > PROB_EPSILON
            if not np.any(defined):
                raise ValueError("item locations are undefined for zero slopes")
            locations = np.full(n_items, np.nan)
            locations[defined] = -intercepts[defined] / denominator[defined]
        scale_by_slopes = False
    else:
        raise ValueError("model does not expose item location parameters")

    if scale_by_slopes and n_factors > 1:
        # A slope-weighted location: the logit a @ theta - sum(a) * b is zero
        # at theta_k = sum(a) * b / a_k when the other factors are zero.
        slopes = _slope_matrix(model, params, n_items, n_factors)
        if slopes is not None:
            denominator = slopes[:, factor]
            defined = np.abs(denominator) > PROB_EPSILON
            if not np.any(defined):
                raise ValueError("item locations are undefined for zero slopes")
            scaled = np.full(n_items, np.nan)
            scaled[defined] = (
                locations[defined] * slopes[defined].sum(axis=1) / denominator[defined]
            )
            locations = scaled

    if locations.shape != (n_items,) or not np.all(np.isfinite(locations[defined])):
        raise ValueError("item locations must be finite")
    return locations


def plot_person_item_map(
    model: BaseItemModel | FitResult,
    theta: NDArray[np.float64],
    ax: Any = None,
    bins: int | str = 30,
    *,
    factor: int = 0,
    item_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a Wright map for one factor with persons and item locations.

    Item locations are difficulties, mean GRM thresholds, mean GPCM or PCM
    steps, or ``-d / a`` for slope-intercept items; mixed-format models
    combine their components. For a multidimensional model the map shows
    the items that load on ``factor``, located where their (mean) logit is
    zero with the other factors at zero; items without a slope on the
    factor are left out and counted in a note.

    Raises
    ------
    ValueError
        If no item has a location on the factor, as for nominal response
        models, whose unordered categories have no single item location.
    """
    import numpy as np

    model = _resolve_model(model)

    values = _factor_values(theta, factor, "theta")
    try:
        np.histogram_bin_edges(values, bins=bins)
    except (TypeError, ValueError):
        raise ValueError("bins must be a valid histogram bin specification") from None
    locations = _item_locations(model, factor)
    located = np.flatnonzero(np.isfinite(locations))
    if item_kwargs is not None and not isinstance(item_kwargs, Mapping):
        raise ValueError("item_kwargs must be a mapping")
    marker_kwargs = {"color": "red", "s": 64.0, **dict(item_kwargs or {})}
    histogram_kwargs = {
        "orientation": "horizontal",
        "alpha": 0.7,
        "color": "steelblue",
        "label": "Persons",
        **kwargs,
    }

    ax = _resolve_axes(ax, (10, 8))
    item_ax = ax.twiny()
    ax.hist(values, bins=bins, **histogram_kwargs)
    item_ax.scatter(np.full(located.size, 0.5), locations[located], **marker_kwargs)
    for index in located:
        item_ax.annotate(
            _item_name(model, int(index)),
            (0.55, locations[index]),
            fontsize=8,
            va="center",
        )
    ax.set_ylabel(_factor_label(r"$\theta$ / Item Location", model, int(factor)))
    ax.set_xlabel("Person Count")
    item_ax.set_xlim(0, 1)
    item_ax.set_xticks([])
    n_omitted = locations.size - located.size
    if n_omitted:
        ax.text(
            0.99,
            0.01,
            f"{n_omitted} item(s) without a location on this factor not shown",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(_factor_label("Person-Item Map (Wright Map)", model, int(factor)))
    ax.legend(loc="upper left")
    return ax


_ETS_THRESHOLDS = {
    # Zwick (2012): |MH D-DIF| of 1 and 1.5 for binary items, and |SMD/SD|
    # above 0.17 and 0.25 for polytomous items, start classes B and C.
    "delta": (1.0, 1.5, "|MH D-DIF|"),
    "smd": (0.17, 0.25, "|SMD / SD|"),
}


def plot_dif(
    dif_results: Mapping[str, Any] | Any,
    effect_size_key: str | None = None,
    classification_key: str = "classification",
    item_names: list[str] | None = None,
    ax: Any = None,
    *,
    ets_scale: Literal["delta", "smd"] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot absolute DIF effects using ETS A/B/C classifications.

    Parameters
    ----------
    dif_results : mapping or DataFrame
        The mapping of :func:`mirt.diagnostics.compute_dif` or the pandas or
        polars DataFrame of :func:`mirt.dif`, whose item labels name the
        bars unless ``item_names`` is given.
    effect_size_key : str, optional
        Column to plot. Defaults to ``"ets_effect"``, the MH D-DIF or
        SMD/SD behind the ETS class, when present, otherwise
        ``"effect_size"``.
    classification_key : str, default="classification"
        Column of A/B/C classes that colors the bars.
    item_names : list of str, optional
        Bar labels.
    ax : matplotlib axes, optional
        Axes to draw on.
    ets_scale : {"delta", "smd"}, optional
        Scale of ``ets_effect``, which places the B and C threshold lines:
        ``"delta"`` (binary items, ``|MH D-DIF|`` of 1 and 1.5) or ``"smd"``
        (polytomous items, ``|SMD/SD|`` of 0.17 and 0.25). Defaults to the
        ``ets_scale`` entry of ``dif_results``. The DataFrame of
        :func:`mirt.dif` does not carry it, so pass it to draw the lines for
        a frame; without a scale, and for columns other than
        ``ets_effect``, no threshold lines are drawn.
    **kwargs
        Forwarded to ``ax.bar``.

    Notes
    -----
    Untested items, such as DIF anchors, have ``NaN`` effects and are drawn
    with zero height.
    """
    import numpy as np

    columns, frame_names = _result_columns(dif_results, "dif_results")
    if effect_size_key is None:
        effect_size_key = "ets_effect" if "ets_effect" in columns else "effect_size"
    effect_sizes = _mapping_values(columns, effect_size_key, allow_missing=True)
    if classification_key not in columns:
        raise ValueError(f"results do not contain {classification_key!r}")
    classifications = np.asarray(columns[classification_key])
    if classifications.shape != effect_sizes.shape:
        raise ValueError("classifications must match effect-size shape")
    labels = np.char.upper(classifications.astype(str))
    valid_labels = {"A", "B", "C"}
    if any(label not in valid_labels for label in labels):
        raise ValueError("classifications must contain only A, B, or C")
    if frame_names is None and isinstance(dif_results, Mapping):
        stored_names = dif_results.get("item_names")
        if stored_names is not None and len(stored_names) == effect_sizes.size:
            frame_names = [str(name) for name in stored_names]
    names = _names(frame_names if item_names is None else item_names, effect_sizes.size)
    thresholds = None
    if effect_size_key == "ets_effect":
        scale = ets_scale if ets_scale is not None else columns.get("ets_scale")
        if scale is not None:
            if scale not in _ETS_THRESHOLDS:
                raise ValueError("ets_scale must be 'delta' or 'smd'")
            thresholds = _ETS_THRESHOLDS[scale]
    elif ets_scale is not None:
        raise ValueError("ets_scale applies only to the 'ets_effect' column")
    color_map = {"A": "green", "B": "gold", "C": "red"}
    colors = [color_map[label] for label in labels]
    custom_color = "color" in kwargs

    ax = _resolve_axes(ax, (10, 6))
    ax.bar(
        range(effect_sizes.size),
        np.abs(np.nan_to_num(effect_sizes, nan=0.0)),
        **{"color": colors, **kwargs},
    )
    if thresholds is not None:
        low, high, axis_label = thresholds
        ax.axhline(
            y=low,
            color="gold",
            linestyle="--",
            alpha=0.7,
            label=f"B threshold ({low:g})",
        )
        ax.axhline(
            y=high,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"C threshold ({high:g})",
        )
    elif effect_size_key == "ets_effect":
        axis_label = "|ETS effect|"
    elif effect_size_key == "effect_size":
        axis_label = "|Effect Size|"
    else:
        axis_label = f"|{effect_size_key}|"
    ax.set_xticks(range(effect_sizes.size))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Item")
    ax.set_ylabel(axis_label)
    ax.set_title("DIF Effect Sizes (ETS Classification)")
    ax.grid(True, alpha=0.3, axis="y")
    if not custom_color:
        for label, color, description in (
            ("A", "green", "Negligible"),
            ("B", "gold", "Moderate"),
            ("C", "red", "Large"),
        ):
            ax.scatter(
                [], [], marker="s", color=color, label=f"{label} ({description})"
            )
    if thresholds is not None or not custom_color or "label" in kwargs:
        ax.legend(loc="best")
    return ax


def plot_expected_score(
    model: BaseItemModel | FitResult,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot a test characteristic curve for dichotomous or ordinal models."""
    import numpy as np

    model = _resolve_model(model)

    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    expected = _expected_score(model, theta)
    maximum = _maximum_score(model)

    ax = _resolve_axes(ax, (8, 6))
    ax.plot(theta_values, expected, **{"linewidth": 2.0, **kwargs})
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Expected Score")
    ax.set_title("Test Characteristic Curve")
    ax.set_xlim(limits)
    ax.set_ylim(0, max(maximum, float(np.max(expected))) * 1.02)
    ax.grid(True, alpha=0.3)
    return ax


def plot_se(
    model: BaseItemModel | FitResult,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot standard error of measurement from total information.

    The standard error is ``1 / sqrt(I(theta))``. For a multidimensional
    model ``I`` is the test information along ``factor``, the diagonal
    element ``[I(theta)]_kk`` that :func:`plot_information` draws, so the
    curve is the standard error of that factor when the others are known.
    """
    import numpy as np

    model = _resolve_model(model)

    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    _, n_factors = _model_size(model)
    if n_factors > 1:
        information = _factor_test_information(model, theta, int(factor))
    else:
        information, _ = _full_information(model, theta)
    standard_error = 1.0 / np.sqrt(np.maximum(information, PROB_EPSILON))

    ax = _resolve_axes(ax, (8, 6))
    ax.plot(theta_values, standard_error, **{"linewidth": 2.0, **kwargs})
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel(_factor_label("Standard Error", model, int(factor)))
    ax.set_title("Standard Error of Measurement")
    ax.set_xlim(limits)
    ax.grid(True, alpha=0.3)
    return ax
