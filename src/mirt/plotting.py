"""Validated plotting helpers with call-specific numerical dependencies.

Importing :mod:`mirt` or this module does not load NumPy, SciPy, or Matplotlib.
NumPy is requested by the first plotting call, Matplotlib only when a function
must create its own axes, and SciPy only for a requested density estimate.
Callers may also supply an existing axes-compatible object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.models.base import BaseItemModel


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


def _information_curves(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    indices: list[int],
    include_total: bool,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    import numpy as np

    n_items, _ = _model_size(model)
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


def _mapping_values(mapping: Mapping[str, Any], key: str) -> NDArray[np.float64]:
    import numpy as np

    if key not in mapping:
        raise ValueError(f"results do not contain {key!r}")
    values = np.asarray(mapping[key], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{key} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain only finite values")
    return values


def _ordered_pair(values: tuple[float, float], name: str) -> tuple[float, float]:
    import numpy as np

    pair = np.asarray(values, dtype=np.float64)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
        raise ValueError(f"{name} must contain two finite increasing values")
    return float(pair[0]), float(pair[1])


def plot_icc(
    model: BaseItemModel,
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
    model: BaseItemModel,
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
    model: BaseItemModel,
    item_idx: int | list[int] | None = None,
    test_info: bool = True,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot test information and optional item-information curves."""
    import numpy as np

    if not isinstance(test_info, (bool, np.bool_)):
        raise ValueError("test_info must be boolean")
    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    if item_idx is None:
        indices = [] if test_info else _item_indices(model, None)
    else:
        indices = _item_indices(model, item_idx)
    if not test_info and not indices:
        raise ValueError("select at least one item when test_info is False")
    total, item_curves = _information_curves(model, theta, indices, bool(test_info))

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
    ax.set_ylabel("Information")
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
    fit_stats: dict[str, NDArray[np.float64]],
    statistic: str = "infit",
    criterion: tuple[float, float] = (0.7, 1.3),
    item_names: list[str] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot item-fit statistics and flag values outside a criterion band."""
    if not isinstance(fit_stats, Mapping):
        raise ValueError("fit_stats must be a mapping")
    values = _mapping_values(fit_stats, statistic)
    lower, upper = _ordered_pair(criterion, "criterion")
    names = _names(item_names, values.size)
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


def _item_locations(model: BaseItemModel, factor: int) -> NDArray[np.float64]:
    import numpy as np

    n_items, n_factors = _model_size(model)
    if factor < 0 or factor >= n_factors:
        raise ValueError(f"factor must be in [0, {n_factors})")
    params = model.parameters
    if not isinstance(params, Mapping):
        raise ValueError("model.parameters must be a mapping")

    if "difficulty" in params:
        difficulty = np.asarray(params["difficulty"], dtype=np.float64)
        if difficulty.shape == (n_items,):
            locations = difficulty
        elif difficulty.shape == (n_items, n_factors):
            locations = difficulty[:, factor]
        else:
            raise ValueError("difficulty parameters have an unsupported shape")
    elif "thresholds" in params:
        thresholds = np.asarray(params["thresholds"], dtype=np.float64)
        if thresholds.ndim != 2 or thresholds.shape[0] != n_items:
            raise ValueError("threshold parameters have an unsupported shape")
        counts = getattr(model, "n_categories", None)
        if isinstance(counts, Sequence) and len(counts) == n_items:
            category_counts = np.asarray(counts, dtype=np.int64)
            if np.any(category_counts < 2) or np.any(
                category_counts - 1 > thresholds.shape[1]
            ):
                raise ValueError("n_categories is inconsistent with thresholds")
            locations = np.asarray(
                [
                    np.mean(thresholds[index, : int(category_counts[index]) - 1])
                    for index in range(n_items)
                ],
                dtype=np.float64,
            )
        else:
            locations = np.mean(thresholds, axis=1)
    elif "intercepts" in params:
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        if intercepts.shape != (n_items,):
            raise ValueError("intercept parameters have an unsupported shape")
        slopes = params.get("slopes", params.get("discrimination"))
        if slopes is None:
            locations = -intercepts
        else:
            slope_values = np.asarray(slopes, dtype=np.float64)
            if slope_values.shape == (n_items,):
                denominator = slope_values
            elif slope_values.shape == (n_items, n_factors):
                denominator = slope_values[:, factor]
            else:
                raise ValueError("slope parameters have an unsupported shape")
            if np.any(np.abs(denominator) <= PROB_EPSILON):
                raise ValueError("item locations are undefined for zero slopes")
            locations = -intercepts / denominator
    else:
        raise ValueError("model does not expose item location parameters")

    if locations.shape != (n_items,) or not np.all(np.isfinite(locations)):
        raise ValueError("item locations must be finite")
    return locations


def plot_person_item_map(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    ax: Any = None,
    bins: int | str = 30,
    *,
    factor: int = 0,
    item_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a Wright map for one factor with persons and item locations."""
    import numpy as np

    values = _factor_values(theta, factor, "theta")
    try:
        np.histogram_bin_edges(values, bins=bins)
    except (TypeError, ValueError):
        raise ValueError("bins must be a valid histogram bin specification") from None
    locations = _item_locations(model, factor)
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
    item_ax.scatter(np.full(locations.size, 0.5), locations, **marker_kwargs)
    for location, name in zip(
        locations,
        (_item_name(model, index) for index in range(model.n_items)),
        strict=True,
    ):
        item_ax.annotate(name, (0.55, location), fontsize=8, va="center")
    ax.set_ylabel(r"$\theta$ / Item Location")
    ax.set_xlabel("Person Count")
    item_ax.set_xlim(0, 1)
    item_ax.set_xticks([])
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
    """Plot absolute DIF effect sizes using ETS A/B/C classifications."""
    import numpy as np

    if not isinstance(dif_results, Mapping):
        raise ValueError("dif_results must be a mapping")
    effect_sizes = _mapping_values(dif_results, effect_size_key)
    if classification_key not in dif_results:
        raise ValueError(f"results do not contain {classification_key!r}")
    classifications = np.asarray(dif_results[classification_key])
    if classifications.shape != effect_sizes.shape:
        raise ValueError("classifications must match effect-size shape")
    labels = np.char.upper(classifications.astype(str))
    valid_labels = {"A", "B", "C"}
    if any(label not in valid_labels for label in labels):
        raise ValueError("classifications must contain only A, B, or C")
    names = _names(item_names, effect_sizes.size)
    color_map = {"A": "green", "B": "gold", "C": "red"}
    colors = [color_map[label] for label in labels]
    custom_color = "color" in kwargs

    ax = _resolve_axes(ax, (10, 6))
    ax.bar(
        range(effect_sizes.size), np.abs(effect_sizes), **{"color": colors, **kwargs}
    )
    ax.axhline(y=0.426, color="gold", linestyle="--", alpha=0.7, label="B threshold")
    ax.axhline(y=0.638, color="red", linestyle="--", alpha=0.7, label="C threshold")
    ax.set_xticks(range(effect_sizes.size))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Item")
    ax.set_ylabel("|Effect Size|")
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
    ax.legend(loc="best")
    return ax


def plot_expected_score(
    model: BaseItemModel,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot a test characteristic curve for dichotomous or ordinal models."""
    import numpy as np

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
    model: BaseItemModel,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: Any = None,
    *,
    factor: int = 0,
    **kwargs: Any,
) -> Any:
    """Plot standard error of measurement from total information."""
    import numpy as np

    theta_values, theta, limits = _theta_grid(model, theta_range, n_points, factor)
    information, _ = _full_information(model, theta)
    standard_error = 1.0 / np.sqrt(np.maximum(information, PROB_EPSILON))

    ax = _resolve_axes(ax, (8, 6))
    ax.plot(theta_values, standard_error, **{"linewidth": 2.0, **kwargs})
    ax.set_xlabel(r"$\theta$ (Ability)")
    ax.set_ylabel("Standard Error")
    ax.set_title("Standard Error of Measurement")
    ax.set_xlim(limits)
    ax.grid(True, alpha=0.3)
    return ax
