"""Shared ability validation and bounded information reductions."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_INFORMATION_CHUNK_ELEMENTS = 262_144


def _validate_information_values(values: NDArray[np.float64]) -> None:
    """Reject non-finite or negative information before any reduction."""
    if not np.all(np.isfinite(values)):
        raise ValueError("model information must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("model information must be non-negative")


def _theta_array(
    model: "BaseItemModel",
    theta: ArrayLike,
    *,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Normalize theta without confusing factors with respondents."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if values.size == 0 and allow_empty:
            values = values.reshape(0, model.n_factors)
        elif model.n_factors == 1:
            values = values.reshape(-1, 1)
        elif values.size == model.n_factors:
            values = values.reshape(1, -1)
        else:
            raise ValueError(
                f"theta must have {model.n_factors} columns for this model"
            )

    if values.ndim != 2:
        raise ValueError("theta must be a scalar, a one-dimensional array, or a matrix")
    if values.shape[0] == 0 and not allow_empty:
        raise ValueError("theta must contain at least one estimate")
    if values.shape[1] != model.n_factors:
        raise ValueError(
            f"theta has {values.shape[1]} factors, expected {model.n_factors}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain only finite values")
    return values


def _test_information(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return total information while preserving genuinely uninformative points."""
    test_information = np.empty(theta.shape[0])
    rows_per_block = max(1, _INFORMATION_CHUNK_ELEMENTS // model.n_items)
    for start in range(0, theta.shape[0], rows_per_block):
        block = theta[start : start + rows_per_block]
        information = np.asarray(model.information(block), dtype=np.float64)
        if information.ndim not in (1, 2):
            raise ValueError(
                "model.information() must return test information or item information"
            )
        if information.shape[0] != block.shape[0]:
            raise ValueError(
                "model.information() returned an incompatible number of theta points"
            )
        if information.ndim == 2 and information.shape[1] != model.n_items:
            raise ValueError(
                "model.information() returned an incompatible number of items"
            )
        _validate_information_values(information)
        if information.ndim == 2:
            with np.errstate(over="ignore"):
                information = np.sum(information, axis=1)
            if not np.all(np.isfinite(information)):
                raise ValueError(
                    "total model information must contain only finite values"
                )
        test_information[start : start + rows_per_block] = information
    return test_information
