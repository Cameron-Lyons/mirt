"""Multidimensional IRT indices.

Provides MDIFF and MDISC indices for summarizing multidimensional
item parameters in unidimensional terms.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def MDISC(
    model: "BaseItemModel",
    item_idx: int | list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute multidimensional discrimination (MDISC).

    MDISC is the length of the discrimination vector in the
    multidimensional space:
        MDISC = sqrt(sum(a_k^2))

    where a_k are the discrimination parameters for each dimension.

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    item_idx : int, list of int, or None
        Item index or indices. If None, returns MDISC for all items.

    Returns
    -------
    NDArray[np.float64]
        MDISC values. Shape: (n_items,) or scalar for single item.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_dimensions=2)
    >>> mdisc = MDISC(result.model)
    >>> print(f"Mean MDISC: {np.mean(mdisc):.3f}")

    Notes
    -----
    MDISC represents the overall discriminating power of an item
    across all dimensions. Higher values indicate greater sensitivity
    to differences in the latent trait space.
    """
    if hasattr(model, "discrimination"):
        disc = np.asarray(model.discrimination)
    else:
        params = model.get_params()
        disc = params

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    mdisc = np.sqrt(np.sum(disc**2, axis=1))

    if item_idx is None:
        return mdisc

    if isinstance(item_idx, int):
        return np.array([mdisc[item_idx]])

    return mdisc[item_idx]


def MDIFF(
    model: "BaseItemModel",
    item_idx: int | list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute multidimensional difficulty (MDIFF).

    MDIFF is the difficulty expressed as a distance in the
    multidimensional space:
        MDIFF = -d / MDISC

    where d is the intercept and MDISC is the multidimensional
    discrimination.

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    item_idx : int, list of int, or None
        Item index or indices. If None, returns MDIFF for all items.

    Returns
    -------
    NDArray[np.float64]
        MDIFF values. Shape: (n_items,) or scalar for single item.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_dimensions=2)
    >>> mdiff = MDIFF(result.model)
    >>> print(f"Mean MDIFF: {np.mean(mdiff):.3f}")

    Notes
    -----
    MDIFF represents the overall difficulty of an item in a
    multidimensional space. It can be interpreted as the distance
    from the origin in the direction of maximum discrimination.
    """
    mdisc = MDISC(model)

    if hasattr(model, "discrimination"):
        disc = np.asarray(model.discrimination)
    else:
        params = model.get_params()
        disc = params

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    if hasattr(model, "difficulty"):
        diff = np.asarray(model.difficulty)
        intercept = -np.sum(disc * diff.reshape(-1, 1), axis=1)
    elif hasattr(model, "intercept"):
        intercept = np.asarray(model.intercept)
    else:
        intercept = np.zeros(disc.shape[0])

    mdiff = -intercept / np.maximum(mdisc, 1e-10)

    if item_idx is None:
        return mdiff

    if isinstance(item_idx, int):
        return np.array([mdiff[item_idx]])

    return mdiff[item_idx]


def direction_cosines(
    model: "BaseItemModel",
    item_idx: int | list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute direction cosines for item discrimination vectors.

    Direction cosines indicate the angle between the item discrimination
    vector and each coordinate axis:
        cos(alpha_k) = a_k / MDISC

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    item_idx : int, list of int, or None
        Item index or indices. If None, returns for all items.

    Returns
    -------
    NDArray[np.float64]
        Direction cosines. Shape: (n_items, n_dims).

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_dimensions=2)
    >>> cosines = direction_cosines(result.model, item_idx=0)
    >>> angles_deg = np.arccos(cosines) * 180 / np.pi
    >>> print(f"Item 0 angles: {angles_deg}")
    """
    if hasattr(model, "discrimination"):
        disc = np.asarray(model.discrimination)
    else:
        params = model.get_params()
        disc = params

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    mdisc = np.sqrt(np.sum(disc**2, axis=1, keepdims=True))
    cosines = disc / np.maximum(mdisc, 1e-10)

    if item_idx is None:
        return cosines

    if isinstance(item_idx, int):
        return cosines[item_idx : item_idx + 1]

    return cosines[item_idx]


def composite_score_weights(
    model: "BaseItemModel",
    reference_direction: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute optimal weights for composite score.

    Computes weights that maximize information in a given reference
    direction (default: equal weight on all dimensions).

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    reference_direction : NDArray[np.float64], optional
        Direction in latent space. If None, uses (1, 1, ..., 1).

    Returns
    -------
    NDArray[np.float64]
        Optimal item weights for composite scoring.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_dimensions=2)
    >>> weights = composite_score_weights(result.model)
    >>> composite = responses @ weights
    """
    if hasattr(model, "discrimination"):
        disc = np.asarray(model.discrimination)
    else:
        params = model.get_params()
        disc = params

    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)

    n_dims = disc.shape[1]

    if reference_direction is None:
        reference_direction = np.ones(n_dims) / np.sqrt(n_dims)
    else:
        reference_direction = np.asarray(reference_direction)
        reference_direction = reference_direction / np.linalg.norm(reference_direction)

    weights = disc @ reference_direction
    weights = weights / np.sum(weights)

    return weights
