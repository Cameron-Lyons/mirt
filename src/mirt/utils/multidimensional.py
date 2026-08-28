"""Multidimensional IRT indices.

Provides MDIFF and MDISC indices for summarizing multidimensional
item parameters in unidimensional terms.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _named_parameter(
    model: "BaseItemModel",
    names: tuple[str, ...],
) -> Any | None:
    """Return the first available attribute or stored parameter by name."""
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value

    parameters = getattr(model, "parameters", None)
    if isinstance(parameters, Mapping):
        for name in names:
            if name in parameters:
                return parameters[name]
    return None


def _get_discrimination_matrix(model: "BaseItemModel") -> NDArray[np.float64]:
    """Extract discrimination parameters as a 2D array (n_items, n_dims)."""
    values = _named_parameter(model, ("discrimination", "slopes"))
    if values is None:
        get_loading_matrix = getattr(model, "get_loading_matrix", None)
        if callable(get_loading_matrix):
            values = get_loading_matrix()
    if values is None:
        raise ValueError(
            "model must provide discrimination, slopes, or a loading matrix"
        )

    disc = np.asarray(values, dtype=np.float64)
    if disc.ndim == 1:
        disc = disc.reshape(-1, 1)
    if disc.ndim != 2 or disc.shape[0] != model.n_items:
        raise ValueError(
            "model discrimination parameters must have shape "
            f"({model.n_items}, n_factors)"
        )
    if disc.shape[1] != model.n_factors:
        raise ValueError(
            "model discrimination parameters have "
            f"{disc.shape[1]} factors, expected {model.n_factors}"
        )
    if not np.all(np.isfinite(disc)):
        raise ValueError("model discrimination parameters must be finite")

    return disc


def _get_intercepts(
    model: "BaseItemModel",
    discrimination: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return logistic intercepts for either supported parameterization."""
    direct = _named_parameter(model, ("intercepts", "intercept"))
    if direct is not None:
        intercepts = np.asarray(direct, dtype=np.float64)
    else:
        difficulty = _named_parameter(model, ("difficulty",))
        if difficulty is None:
            raise ValueError("model must provide intercepts or difficulty parameters")
        difficulty = np.asarray(difficulty, dtype=np.float64)
        if difficulty.shape != (model.n_items,):
            raise ValueError(
                f"model difficulty parameters must have shape ({model.n_items},)"
            )
        intercepts = -np.sum(discrimination, axis=1) * difficulty

    if intercepts.shape != (model.n_items,):
        raise ValueError(f"model intercepts must have shape ({model.n_items},)")
    if not np.all(np.isfinite(intercepts)):
        raise ValueError("model intercepts must be finite")
    return intercepts


def _row_norms(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute stable Euclidean norms for item parameter rows."""
    return np.hypot.reduce(values, axis=1)


def _select_items(
    values: NDArray[np.float64],
    item_idx: int | Sequence[int] | None,
) -> NDArray[np.float64]:
    """Select valid item indices while preserving an item axis."""
    if item_idx is None:
        return values

    if isinstance(item_idx, (int, np.integer)) and not isinstance(
        item_idx, (bool, np.bool_)
    ):
        indices = np.array([int(item_idx)], dtype=np.intp)
    else:
        try:
            raw_indices = np.asarray(item_idx)
        except (TypeError, ValueError) as exc:
            raise TypeError("item_idx must contain integer indices") from exc
        if raw_indices.ndim != 1:
            raise TypeError("item_idx must contain integer indices")
        if raw_indices.size == 0:
            indices = np.empty(0, dtype=np.intp)
        elif np.issubdtype(raw_indices.dtype, np.integer):
            indices = raw_indices.astype(np.intp, copy=False)
        else:
            raise TypeError("item_idx must contain integer indices")

    invalid = indices[(indices < 0) | (indices >= values.shape[0])]
    if invalid.size:
        raise IndexError(
            f"item index {int(invalid[0])} out of range [0, {values.shape[0]})"
        )

    return values[indices]


def MDISC(
    model: "BaseItemModel",
    item_idx: int | Sequence[int] | None = None,
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
    item_idx : int, sequence of int, or None
        Item index or indices. If None, returns MDISC for all items.

    Returns
    -------
    NDArray[np.float64]
        MDISC values. Shape: (n_items,) or (1,) for a single item.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_factors=2)
    >>> mdisc = MDISC(result.model)
    >>> print(f"Mean MDISC: {np.mean(mdisc):.3f}")

    Notes
    -----
    MDISC represents the overall discriminating power of an item
    across all dimensions. Higher values indicate greater sensitivity
    to differences in the latent trait space.
    """
    disc = _get_discrimination_matrix(model)
    mdisc = _row_norms(disc)
    return _select_items(mdisc, item_idx)


def MDIFF(
    model: "BaseItemModel",
    item_idx: int | Sequence[int] | None = None,
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
    item_idx : int, sequence of int, or None
        Item index or indices. If None, returns MDIFF for all items.

    Returns
    -------
    NDArray[np.float64]
        MDIFF values. Shape: (n_items,) or (1,) for a single item. Items with
        a zero discrimination vector have undefined difficulty and return NaN.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_factors=2)
    >>> mdiff = MDIFF(result.model)
    >>> print(f"Mean MDIFF: {np.mean(mdiff):.3f}")

    Notes
    -----
    MDIFF represents the overall difficulty of an item in a
    multidimensional space. It can be interpreted as the distance
    from the origin in the direction of maximum discrimination.
    """
    disc = _get_discrimination_matrix(model)
    mdisc = _row_norms(disc)
    intercept = _get_intercepts(model, disc)
    mdiff = np.full(model.n_items, np.nan, dtype=np.float64)
    np.divide(-intercept, mdisc, out=mdiff, where=mdisc > 0.0)
    return _select_items(mdiff, item_idx)


def direction_cosines(
    model: "BaseItemModel",
    item_idx: int | Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Compute direction cosines for item discrimination vectors.

    Direction cosines indicate the angle between the item discrimination
    vector and each coordinate axis:
        cos(alpha_k) = a_k / MDISC

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    item_idx : int, sequence of int, or None
        Item index or indices. If None, returns for all items.

    Returns
    -------
    NDArray[np.float64]
        Direction cosines. Shape: (n_items, n_dims). Items with a zero
        discrimination vector have undefined direction and return NaN values.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_factors=2)
    >>> cosines = direction_cosines(result.model, item_idx=0)
    >>> angles_deg = np.arccos(cosines) * 180 / np.pi
    >>> print(f"Item 0 angles: {angles_deg}")
    """
    disc = _get_discrimination_matrix(model)
    mdisc = _row_norms(disc)[:, None]
    cosines = np.full_like(disc, np.nan, dtype=np.float64)
    np.divide(disc, mdisc, out=cosines, where=mdisc > 0.0)
    return _select_items(cosines, item_idx)


def composite_score_weights(
    model: "BaseItemModel",
    reference_direction: NDArray[np.float64] | Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Compute optimal weights for composite score.

    Computes weights that maximize information in a given reference
    direction (default: equal weight on all dimensions).

    Parameters
    ----------
    model : BaseItemModel
        A fitted multidimensional IRT model.
    reference_direction : array-like, optional
        Direction in latent space. If None, uses (1, 1, ..., 1).

    Returns
    -------
    NDArray[np.float64]
        Optimal item weights for composite scoring.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL", n_factors=2)
    >>> weights = composite_score_weights(result.model)
    >>> composite = responses @ weights
    """
    disc = _get_discrimination_matrix(model)
    n_dims = disc.shape[1]

    if reference_direction is None:
        reference_direction = np.ones(n_dims) / np.sqrt(n_dims)
    else:
        reference_direction = np.asarray(reference_direction, dtype=np.float64)
        if reference_direction.shape != (n_dims,):
            raise ValueError(f"reference_direction must have shape ({n_dims},)")
        if not np.all(np.isfinite(reference_direction)):
            raise ValueError("reference_direction must be finite")
        direction_norm = np.linalg.norm(reference_direction)
        if direction_norm == 0.0:
            raise ValueError("reference_direction must be nonzero")
        reference_direction = reference_direction / direction_norm

    weights = disc @ reference_direction
    weight_sum = float(np.sum(weights))
    zero_tolerance = np.finfo(np.float64).eps * max(1.0, float(np.sum(np.abs(weights))))
    if not np.isfinite(weight_sum) or abs(weight_sum) <= zero_tolerance:
        raise ValueError(
            "item projections must have a nonzero sum in the reference direction"
        )
    weights = weights / weight_sum

    return weights
