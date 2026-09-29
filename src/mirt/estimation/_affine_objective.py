"""Prepared, model-independent M-step objectives for affine logistic items."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlog1py, xlogy

from mirt._logistic import _affine_logits, _logistic_probability
from mirt.models.base import BaseItemModel

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def prepare_affine_objective(
    model: BaseItemModel,
    item_idx: int,
    theta: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
    epsilon: float,
    bounds: list[tuple[float, float]],
) -> _Objective | None:
    """Prepare a pure objective for built-in models and the supplied optimizer box.

    Trial parameters must lie within ``bounds``, as enforced by L-BFGS-B. This
    lets ordinary quadrature grids skip repeated overflow checks, while grids
    with large coordinates still use the shared exact-recovery logit kernel.
    Custom model implementations retain their probability-based objective.
    """
    from mirt.models.bifactor import BifactorModel
    from mirt.models.multidimensional import MultidimensionalModel

    if any(
        name in vars(model)
        for name in (
            "probability",
            "_logits",
            "_curve_parameters",
            "set_parameters",
            "set_item_parameter",
            "_canonical_parameter_values",
            "free_parameter_masks",
        )
    ):
        return None

    if type(model) is MultidimensionalModel:
        if tuple(model._parameters) != ("slopes", "intercepts"):
            return None
        pattern = model._loading_pattern[item_idx]
        if not np.all((pattern == 0.0) | (pattern == 1.0)):
            return None
        points = theta[:, pattern != 0.0]
    elif type(model) is BifactorModel:
        if tuple(model._parameters) != (
            "general_loadings",
            "specific_loadings",
            "intercepts",
        ):
            return None
        points = theta[:, [0, 1 + model._specific_factor_indices[item_idx]]]
    else:
        return None

    if len(bounds) != points.shape[1] + 1:
        return None
    limits = np.max(np.abs(bounds), axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        max_logit_bound = (
            np.abs(points).max(initial=0.0) * limits[:-1].sum() + limits[-1]
        )
    ordinary = max_logit_bound <= 1000.0
    incorrect = observed - correct

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        if ordinary:
            logits = points @ params[:-1]
            logits += params[-1]
        else:
            logits = _affine_logits(points, params[:-1], params[-1])
        probability = _logistic_probability(logits, None, None)
        interior = (probability > epsilon) & (probability < 1.0 - epsilon)
        np.clip(probability, epsilon, 1.0 - epsilon, out=probability)
        loss = -float(
            np.sum(xlogy(correct, probability) + xlog1py(incorrect, -probability))
        )
        # The clipped objective is constant beyond each probability boundary.
        residual = observed * probability - correct
        residual[~interior] = 0.0
        gradient = np.empty(params.size)
        gradient[:-1] = points.T @ residual
        gradient[-1] = residual.sum()
        return loss, gradient

    return objective
