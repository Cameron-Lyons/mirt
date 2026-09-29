"""Exact diagonal curvature of the fixed variational logistic bound."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_MAX_INFORMATION_ELEMENTS = 262_144


def gvem_standard_errors(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    xi: NDArray[np.float64],
    lambda_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Evaluate item curvature without perturbing model or variational state.

    With q and xi held fixed, an item's parameter-dependent bound is
    (y - 1/2) * a'(mu - b) - lambda * (a'Sigma*a + (a'(mu - b))**2).
    Its negative diagonal second derivatives are therefore
    2 * sum(lambda * ((mu_f - b)**2 + Sigma_ff)) for each loading, and
    2 * sum(lambda) * sum(a)**2 for difficulty. Prior/entropy terms are constant.
    """
    params = model.parameters
    a = params["discrimination"].reshape(model.n_items, model.n_factors)
    b = params["difficulty"]
    masks = model.free_parameter_masks
    estimate_a = masks["discrimination"].reshape(a.shape)
    information_a = np.zeros_like(a)
    se_b = np.full_like(b, np.nan)
    variances = np.diagonal(sigma, axis1=1, axis2=2)
    block_size = max(1, _MAX_INFORMATION_ELEMENTS // (3 * model.n_factors + 5))

    for item in range(model.n_items):
        weight_sum = 0.0
        for start in range(0, len(responses), block_size):
            stop = min(start + block_size, len(responses))
            observed = responses[start:stop, item] >= 0
            if not np.any(observed):
                continue
            weights = 2.0 * lambda_function(xi[start:stop, item][observed])
            weight_sum += float(np.sum(weights))
            if np.any(estimate_a[item]):
                centered = mu[start:stop][observed]
                centered -= b[item]
                np.square(centered, out=centered)
                centered += variances[start:stop][observed]
                information_a[item] += weights @ centered

        slope_sum = abs(float(np.sum(a[item])))
        if weight_sum > 0.0 and slope_sum > 0.0:
            # Taking the square root before multiplying avoids squaring very
            # large or very small loadings just to undo it in the standard error.
            se_b[item] = (1.0 / np.sqrt(weight_sum)) / slope_sum

    se_a = np.full_like(information_a, np.nan)
    np.divide(1.0, np.sqrt(information_a), out=se_a, where=information_a > 0.0)
    result = {
        "discrimination": se_a.reshape(params["discrimination"].shape),
        "difficulty": se_b,
    }
    for name, free in masks.items():
        result[name][~free] = 0.0
    return result
