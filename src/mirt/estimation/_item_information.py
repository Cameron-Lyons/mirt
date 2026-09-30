"""Exact diagonal complete-data curvature for built-in EM item models.

This preserves the itemwise EM uncertainty objective; it is distinct from the
full marginal information matrix in standard_errors.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.estimation._em_context import EMFitContext

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def item_standard_errors(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    weights: NDArray[np.float64],
    points: NDArray[np.float64],
    epsilon: float,
    *,
    person_weights: NDArray[np.float64] | None = None,
    context: EMFitContext | None = None,
) -> dict[str, NDArray[np.float64]] | None:
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        PartialCreditModel,
    )

    if (
        model.n_factors != 1
        or type(model)
        not in (
            OneParameterLogistic,
            TwoParameterLogistic,
            ThreeParameterLogistic,
            FourParameterLogistic,
            GradedResponseModel,
            GeneralizedPartialCredit,
            PartialCreditModel,
        )
        or "probability" in vars(model)
    ):
        return None
    params = model.parameters
    result = {name: np.zeros_like(value) for name, value in params.items()}
    theta = points.ravel()
    context = context or EMFitContext(responses)
    correct = observed = None
    if not model.is_polytomous:
        correct, observed = context.expected_counts(weights, person_weights)

    def se(curvature):
        curvature = np.asarray(curvature, dtype=np.float64)
        return np.sqrt(
            np.divide(
                -1.0,
                curvature,
                out=np.full_like(curvature, np.nan),
                where=curvature < 0,
            )
        )

    for j in range(model.n_items):
        a = params["discrimination"][j]
        if not model.is_polytomous:
            n = observed[j]
            r = correct[j]
            z = theta - params["difficulty"][j]
            s = sigmoid(a * z)
            c = params.get("guessing", np.zeros(model.n_items))[j]
            d = params.get("upper", np.ones(model.n_items))[j]
            raw = c + (d - c) * s
            p = np.clip(raw, epsilon, 1 - epsilon)
            active = (raw > epsilon) & (raw < 1 - epsilon)
            score = np.where(active, r / p - (n - r) / (1 - p), 0.0)
            info = np.where(active, r / p**2 + (n - r) / (1 - p) ** 2, 0.0)
            first = (d - c) * s * (1 - s)
            second = first * (1 - 2 * s)
            derivatives = {
                "discrimination": (first * z, second * z**2),
                "difficulty": (-a * first, a * a * second),
                "guessing": (1 - s, 0.0),
                "upper": (s, 0.0),
            }
            for name in params:
                dp, d2p = derivatives[name]
                result[name][j] = se(np.sum(score * d2p - info * dp**2))
            continue

        k = model.n_categories[j]
        counts = context.expected_category_counts(j, k, weights, person_weights)
        p = model.probability(points, j)
        active = (p > epsilon) & (p < 1 - epsilon)
        effective = np.where(active, counts, 0.0)
        if type(model) is not GradedResponseModel:
            increments = theta[:, None] - params["steps"][j, : k - 1]
            features = np.column_stack(
                (np.zeros(theta.size), np.cumsum(increments, axis=1))
            )
            mean = np.sum(p * features, axis=1)
            variance = np.sum(p * (features - mean[:, None]) ** 2, axis=1)
            total = effective.sum(axis=1)
            result["discrimination"][j] = se(-np.sum(total * variance))
            tails = np.cumsum(p[:, :0:-1], axis=1)[:, ::-1]
            result["steps"][j, : k - 1] = se(
                -a * a * np.sum(total[:, None] * tails * (1 - tails), axis=0)
            )
            continue

        centered = theta[:, None] - params["thresholds"][j, : k - 1]
        cumulative = sigmoid(a * centered)
        derivative = cumulative * (1 - cumulative)
        second = derivative * (1 - 2 * cumulative)
        clipped = np.clip(p, epsilon, 1 - epsilon)

        def curvature(dp, d2p):
            return np.sum(effective * (d2p / clipped - (dp / clipped) ** 2))

        def category_difference(values):
            padded = np.column_stack(
                (np.zeros(theta.size), values, np.zeros(theta.size))
            )
            return padded[:, :-1] - padded[:, 1:]

        result["discrimination"][j] = se(
            curvature(
                category_difference(derivative * centered),
                category_difference(second * centered**2),
            )
        )
        for t in range(k - 1):
            dp = np.zeros_like(p)
            d2p = np.zeros_like(p)
            dp[:, t], dp[:, t + 1] = a * derivative[:, t], -a * derivative[:, t]
            d2p[:, t], d2p[:, t + 1] = -a * a * second[:, t], a * a * second[:, t]
            result["thresholds"][j, t] = se(curvature(dp, d2p))

    for name, mask in model.free_parameter_masks.items():
        result[name][~mask] = 0.0
    return result
