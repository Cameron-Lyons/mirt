"""Item information for candidate items at a few ability points.

Selection, exposure control, information-based stopping and form assembly
all rank candidate items by Fisher information. Dichotomous models return
one value per item from a single ``information(theta)`` call. Polytomous
models define that call as total test information, so built-in polytomous
models are evaluated with their all-item kernel ``_information_by_item``,
which equals ``information(theta, item_idx=j)`` column by column. Other
models are evaluated one item at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._model_defaults import uses_original_model_hook

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def uses_all_item_kernel(model: BaseItemModel) -> bool:
    """Whether ``model._information_by_item`` gives its item information.

    The kernel describes ``information(theta, item_idx=j)`` when the model
    keeps the authored ``information`` hook; built-in kernels themselves fall
    back to the item-wise curves when those curves are customized.
    """
    from mirt.models.base import PolytomousItemModel

    return (
        isinstance(model, PolytomousItemModel)
        and "_information_by_item" not in vars(model)
        and uses_original_model_hook(model, "information")
    )


def polytomous_item_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    items: Sequence[int] | NDArray[np.intp],
) -> NDArray[np.float64] | None:
    """Return ``(n_theta, len(items))`` information from the all-item kernel.

    Returns None when the model has no usable kernel, so callers keep their
    item-wise evaluation.
    """
    if not model.is_polytomous or not uses_all_item_kernel(model):
        return None
    information = np.asarray(model._information_by_item(theta), dtype=np.float64)
    if information.shape != (theta.shape[0], model.n_items):
        return None
    return information[:, np.asarray(items, dtype=np.intp)]


def item_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    items: Sequence[int] | NDArray[np.intp],
) -> NDArray[np.float64]:
    """Return information of ``items`` with shape ``(n_theta, len(items))``.

    ``theta`` has shape ``(n_theta, n_factors)``. Models whose bulk output is
    not one value per item are evaluated item by item.
    """
    columns = np.asarray(items, dtype=np.intp)
    n_theta = theta.shape[0]
    if not model.is_polytomous:
        information = np.asarray(model.information(theta), dtype=np.float64)
        if information.size == n_theta * model.n_items:
            return information.reshape(n_theta, model.n_items)[:, columns]
    else:
        kernel = polytomous_item_information(model, theta, columns)
        if kernel is not None:
            return kernel
    result = np.empty((n_theta, columns.size), dtype=np.float64)
    for position, item_idx in enumerate(columns.tolist()):
        values = np.asarray(model.information(theta, item_idx=item_idx), dtype=float)
        result[:, position] = values.reshape(n_theta, -1).sum(axis=1)
    return result
