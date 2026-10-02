"""Prepared clipped likelihoods and gradients for built-in 1PL–4PL items."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlog1py, xlogy

from mirt._logistic import _logistic_probability
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.models.base import BaseItemModel

_Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def prepare_dichotomous_objective(
    model: BaseItemModel,
    item_idx: int,
    theta: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
    epsilon: float,
    bounds: list[tuple[float, float]] | None = None,
) -> _Objective | None:
    """Prepare a pure objective, optionally restricted to an optimizer box.

    When bounds are supplied, trial parameters must lie within them. Ordinary
    quadrature grids can then skip repeated overflow checks. Without bounds,
    use the public curves' stable logit kernels for every evaluation.
    """
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    layouts = {
        OneParameterLogistic: ("discrimination", "difficulty"),
        TwoParameterLogistic: ("discrimination", "difficulty"),
        ThreeParameterLogistic: ("discrimination", "difficulty", "guessing"),
        FourParameterLogistic: ("discrimination", "difficulty", "guessing", "upper"),
    }
    if (
        tuple(model._parameters) != layouts.get(type(model))
        or not uses_builtin_model_hooks(model)
        or model._free_parameter_restrictions
    ):
        return None

    fixed_slope = type(model) is OneParameterLogistic
    fixed_a = (
        float(model._parameters["discrimination"][item_idx]) if fixed_slope else None
    )
    return prepare_logistic_objective(
        theta,
        observed,
        correct,
        epsilon,
        fixed_a=fixed_a,
        has_guessing="guessing" in model._parameters,
        has_upper="upper" in model._parameters,
        bounds=bounds,
    )


def prepare_logistic_objective(
    theta: NDArray[np.float64],
    observed: NDArray[np.float64],
    correct: NDArray[np.float64],
    epsilon: float,
    *,
    fixed_a: float | None = None,
    has_guessing: bool = False,
    has_upper: bool = False,
    bounds: list[tuple[float, float]] | None = None,
) -> _Objective | None:
    """Prepare the shared clipped logistic kernel without a model adapter."""
    from mirt.models.dichotomous import (
        _multidimensional_logits,
        _unidimensional_logits,
    )

    unidimensional = theta.shape[1] == 1
    fixed_slope = fixed_a is not None
    fixed_a = 1.0 if fixed_a is None else fixed_a
    n_slopes = 0 if fixed_slope else theta.shape[1]
    points = theta[:, 0] if unidimensional else theta
    incorrect = observed - correct
    ordinary = False
    if bounds is not None:
        if len(bounds) != n_slopes + 1 + has_guessing + has_upper:
            return None
        slope_limit = (
            abs(fixed_a)
            if fixed_slope
            else sum(max(abs(low), abs(high)) for low, high in bounds[:n_slopes])
        )
        location_limit = max(abs(value) for value in bounds[n_slopes])
        max_logit_bound = (
            float(np.abs(points).max(initial=0.0)) + float(location_limit)
        ) * float(slope_limit)
        # This also keeps every exponential normal and finite on the fast path.
        ordinary = max_logit_bound <= 700.0

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        a = (
            fixed_a
            if fixed_slope
            else params[0]
            if unidimensional
            else params[:n_slopes]
        )
        b = params[n_slopes]
        slope_sum = a if unidimensional else a.sum()
        if ordinary:
            z = a * (points - b) if unidimensional else points @ a - slope_sum * b
        elif unidimensional:
            z = _unidimensional_logits(points, a, b)
        else:
            z = _multidimensional_logits(points, a, b)

        if has_guessing:
            c = params[n_slopes + 1]
            d = params[-1] if has_upper else 1.0
            if ordinary:
                tail = np.exp(-np.abs(z))
            else:
                with np.errstate(under="ignore"):
                    tail = np.exp(-np.abs(z))
            tail /= 1.0 + tail
            positive = z >= 0.0
            failure = np.where(positive, tail, 1.0 - tail)
            p = np.where(positive, d - (d - c) * tail, c + (d - c) * tail)
        elif ordinary:
            np.negative(z, out=z)
            np.exp(z, out=z)
            z += 1.0
            np.reciprocal(z, out=z)
            p = z
        else:
            p = _logistic_probability(z, None, None)
        active = (p > epsilon) & (p < 1.0 - epsilon)
        np.clip(p, epsilon, 1.0 - epsilon, out=p)
        loss = -float(np.sum(xlogy(correct, p) + xlog1py(incorrect, -p)))
        residual = observed * p - correct
        if has_guessing:
            score = np.divide(
                residual, p * (1.0 - p), out=np.zeros_like(p), where=active
            )
            common = score * (d - c) * tail * (1.0 - tail)
        else:
            common = residual
            common[~active] = 0.0
        gradient = np.empty_like(params)
        if n_slopes:
            gradient[:n_slopes] = (points - b).T @ common
        gradient[n_slopes] = -slope_sum * common.sum()
        if has_guessing:
            gradient[n_slopes + 1] = failure @ score
            if has_upper:
                success = np.where(positive, 1.0 - tail, tail)
                gradient[-1] = success @ score
        return loss, gradient

    return objective
