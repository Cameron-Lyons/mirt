"""Model extraction utilities for IRT models.

Provides functions for extracting and converting model parameters
to different formats.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN, PROB_EPSILON

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


def estfun(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Extract empirical estimating functions from a fitted model.

    Computes the score function (gradient of log-likelihood) for each
    person. Used for sandwich estimators of standard errors and for
    detecting influential observations.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, n_factors).

    Returns
    -------
    NDArray[np.float64]
        Estimating functions (scores) for each person-parameter combination.
        Shape: (n_persons, n_parameters).

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> scores = fscores(result, responses)
    >>> ef = estfun(result.model, responses, scores.theta)
    >>> # Sum should be close to zero at MLE
    >>> print(f"Sum of estimating functions: {ef.sum(axis=0)}")

    Notes
    -----
    The estimating functions are the first derivatives of the log-likelihood
    with respect to the item parameters. At the MLE, these should sum to
    approximately zero across all persons.

    For the 2PL model, returns gradients with respect to both discrimination
    (a) and difficulty (b) parameters, giving shape (n_persons, 2*n_items).

    These can be used to compute:
    - Robust (sandwich) standard errors
    - Influence functions for individual observations
    - Model-based residuals
    """
    responses = np.asarray(responses, dtype=np.float64)
    theta = np.atleast_1d(theta)
    if theta.ndim == 2:
        theta = theta[:, 0]

    n_persons, n_items = responses.shape

    values = mod2values(model)

    param_vec = []
    param_names = []

    disc = values.discrimination
    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    for j in range(n_items):
        for d in range(disc.shape[1]):
            param_vec.append(disc[j, d])
            param_names.append(f"a_{j}_{d}")

    diff = values.difficulty
    if diff.ndim == 1:
        for j in range(n_items):
            param_vec.append(diff[j])
            param_names.append(f"b_{j}")
    else:
        for j in range(n_items):
            for k in range(diff.shape[1]):
                param_vec.append(diff[j, k])
                param_names.append(f"b_{j}_{k}")

    if values.guessing is not None:
        for j in range(n_items):
            param_vec.append(values.guessing[j])
            param_names.append(f"c_{j}")

    if values.slipping is not None:
        for j in range(n_items):
            param_vec.append(values.slipping[j])
            param_names.append(f"d_{j}")

    n_params = len(param_vec)

    ef = np.zeros((n_persons, n_params))

    probs = model.probability(theta.reshape(-1, 1))
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    probs = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)

    valid = responses >= 0
    for j in range(n_items):
        p = probs[:, j]
        x = responses[:, j]
        v = valid[:, j]

        residual = np.where(v, x - p, 0.0)

        if disc.shape[1] == 1:
            dp_da = p * (1 - p) * (theta - values.difficulty[j])
            dp_db = -p * (1 - p) * disc[j, 0]

            a_idx = j
            b_idx = n_items * disc.shape[1] + j

            ef[:, a_idx] = np.where(
                v, residual * dp_da / (p * (1 - p) + PROB_EPSILON), 0.0
            )
            ef[:, b_idx] = np.where(
                v, residual * dp_db / (p * (1 - p) + PROB_EPSILON), 0.0
            )

    return ef


def estfun_summary(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Compute summary statistics for estimating functions.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.

    Returns
    -------
    dict
        Dictionary containing:
        - sum: Sum of estimating functions (should be ~0)
        - mean: Mean estimating function
        - var: Variance of estimating functions
        - meat: "Meat" matrix for sandwich estimator (sum of outer products)
    """
    ef = estfun(model, responses, theta)

    n_persons = ef.shape[0]

    meat = ef.T @ ef / n_persons

    return {
        "sum": ef.sum(axis=0),
        "mean": ef.mean(axis=0),
        "var": ef.var(axis=0),
        "meat": meat,
    }
