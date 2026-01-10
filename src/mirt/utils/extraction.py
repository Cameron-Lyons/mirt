"""Model extraction utilities for IRT models.

Provides functions for extracting and converting model parameters
to different formats.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ItemParameters:
    """Container for extracted item parameters.

    Attributes
    ----------
    item_idx : int
        Item index.
    model_type : str
        Model type (e.g., "2PL", "GRM").
    discrimination : NDArray[np.float64]
        Discrimination parameter(s). Shape depends on dimensionality.
    difficulty : float | NDArray[np.float64]
        Difficulty parameter(s). Scalar for dichotomous, array for polytomous.
    guessing : float | None
        Lower asymptote (for 3PL/4PL).
    slipping : float | None
        Upper asymptote (for 4PL).
    """

    item_idx: int
    model_type: str
    discrimination: NDArray[np.float64]
    difficulty: float | NDArray[np.float64]
    guessing: float | None = None
    slipping: float | None = None


@dataclass
class ModelValues:
    """Container for all model parameter values.

    Attributes
    ----------
    model_type : str
        Model type (e.g., "2PL", "GRM").
    n_items : int
        Number of items.
    n_dimensions : int
        Number of latent dimensions.
    discrimination : NDArray[np.float64]
        Discrimination matrix. Shape: (n_items, n_dims).
    difficulty : NDArray[np.float64]
        Difficulty parameters. Shape depends on model.
    guessing : NDArray[np.float64] | None
        Lower asymptotes if applicable.
    slipping : NDArray[np.float64] | None
        Upper asymptotes if applicable.
    """

    model_type: str
    n_items: int
    n_dimensions: int
    discrimination: NDArray[np.float64]
    difficulty: NDArray[np.float64]
    guessing: NDArray[np.float64] | None = None
    slipping: NDArray[np.float64] | None = None


def mod2values(model: "BaseItemModel") -> ModelValues:
    """Extract all parameter values from model.

    Converts model parameters to a standardized format for
    inspection and modification.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.

    Returns
    -------
    ModelValues
        Container with all model parameters.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> values = mod2values(result.model)
    >>> print(f"Discriminations shape: {values.discrimination.shape}")
    >>> print(f"Mean difficulty: {np.mean(values.difficulty):.2f}")
    """
    n_items = model.n_items

    model_type = model.__class__.__name__
    if "1PL" in model_type or "Rasch" in model_type.lower():
        model_type = "1PL"
    elif "4PL" in model_type:
        model_type = "4PL"
    elif "3PL" in model_type:
        model_type = "3PL"
    elif "2PL" in model_type:
        model_type = "2PL"
    elif "GRM" in model_type:
        model_type = "GRM"
    elif "GPCM" in model_type:
        model_type = "GPCM"
    elif "PCM" in model_type:
        model_type = "PCM"
    elif "NRM" in model_type:
        model_type = "NRM"
    else:
        model_type = "Unknown"

    n_dims = getattr(model, "n_factors", getattr(model, "n_dimensions", 1))

    discrimination = np.ones((n_items, n_dims))
    difficulty = np.zeros(n_items)
    guessing = None
    slipping = None

    if hasattr(model, "discrimination"):
        disc = np.asarray(model.discrimination)
        if disc.ndim == 1:
            discrimination = disc.reshape(-1, 1)
        else:
            discrimination = disc

    if hasattr(model, "difficulty"):
        difficulty = np.asarray(model.difficulty)

    if hasattr(model, "guessing"):
        g = model.guessing
        if g is not None:
            guessing = np.asarray(g)

    if hasattr(model, "slipping"):
        s = model.slipping
        if s is not None:
            slipping = np.asarray(s)

    return ModelValues(
        model_type=model_type,
        n_items=n_items,
        n_dimensions=n_dims,
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=guessing,
        slipping=slipping,
    )


def extract_item(
    model: "BaseItemModel",
    item_idx: int,
) -> ItemParameters:
    """Extract parameters for a single item.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int
        Index of the item to extract.

    Returns
    -------
    ItemParameters
        Container with item parameters.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> item = extract_item(result.model, item_idx=0)
    >>> print(f"Item 0 discrimination: {item.discrimination}")
    >>> print(f"Item 0 difficulty: {item.difficulty:.2f}")
    """
    values = mod2values(model)

    if item_idx < 0 or item_idx >= values.n_items:
        raise ValueError(f"item_idx {item_idx} out of range [0, {values.n_items})")

    discrimination = values.discrimination[item_idx]

    if values.difficulty.ndim == 1:
        difficulty = float(values.difficulty[item_idx])
    else:
        difficulty = values.difficulty[item_idx]

    guessing = None
    if values.guessing is not None:
        guessing = float(values.guessing[item_idx])

    slipping = None
    if values.slipping is not None:
        slipping = float(values.slipping[item_idx])

    return ItemParameters(
        item_idx=item_idx,
        model_type=values.model_type,
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=guessing,
        slipping=slipping,
    )


def coef(
    model: "BaseItemModel",
    irt_pars: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Extract model coefficients in dictionary format.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    irt_pars : bool
        If True, return IRT parameterization (a, b, c, d).
        If False, return slope-intercept form. Default True.

    Returns
    -------
    dict
        Dictionary with parameter arrays.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> params = coef(result.model)
    >>> print(params["discrimination"])
    >>> print(params["difficulty"])
    """
    values = mod2values(model)

    if irt_pars:
        result = {
            "discrimination": values.discrimination,
            "difficulty": values.difficulty,
        }
        if values.guessing is not None:
            result["guessing"] = values.guessing
        if values.slipping is not None:
            result["slipping"] = values.slipping
    else:
        a = values.discrimination
        b = values.difficulty

        if b.ndim == 1:
            d = -a[:, 0] * b
        else:
            d = -np.sum(a * b, axis=1)

        result = {
            "slope": a,
            "intercept": d,
        }
        if values.guessing is not None:
            result["guessing"] = values.guessing
        if values.slipping is not None:
            result["slipping"] = values.slipping

    return result


def itemplot_data(
    model: "BaseItemModel",
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
) -> dict[str, NDArray[np.float64]]:
    """Get data for item characteristic curve plot.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int
        Index of item.
    theta_range : tuple
        Range of theta values.
    n_points : int
        Number of points.

    Returns
    -------
    dict
        Dictionary with "theta", "probability", and "information" arrays.
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta.reshape(-1, 1)

    probs = model.probability(theta_2d, item_idx=item_idx)
    if probs.ndim > 1:
        probs = probs[:, 0] if probs.shape[1] == 1 else probs

    info = model.information(theta_2d)
    item_info = info[:, item_idx] if info.ndim > 1 else info

    return {
        "theta": theta,
        "probability": probs,
        "information": item_info,
    }
