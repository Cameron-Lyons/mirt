"""Diagonal loss curvature for built-in clipped category likelihoods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.models.base import BaseItemModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


def polytomous_item_curvature(
    model: BaseItemModel,
    item: int,
    theta: NDArray[np.float64],
    counts: NDArray[np.float64],
    epsilon: float,
) -> dict[str, NDArray[np.float64]]:
    """Compute exact diagonal curvature with category counts held fixed.

    Callers establish that model curves and parameter layouts are unchanged.
    Clipped categories contribute zero derivatives, matching the Monte Carlo
    objective's upper clipping bound of one. No model parameters are mutated.
    """
    if type(model) not in (
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
    ):
        raise TypeError("Unsupported built-in polytomous curvature model")
    parameters = model.parameters
    result = {name: np.zeros_like(values[item]) for name, values in parameters.items()}
    probability = model.probability(theta, item)
    effective = np.where((probability > epsilon) & (probability < 1.0), counts, 0.0)
    total = effective.sum(axis=1)
    categories = model.n_categories[item]

    if type(model) is NominalResponseModel:
        variance = total[:, None] * probability[:, 1:] * (1.0 - probability[:, 1:])
        result["intercepts"][1:categories] = variance.sum(axis=0)
        slopes = variance.T @ (theta * theta)
        result["slopes"][1:categories] = (
            slopes.ravel() if model.n_factors == 1 else slopes
        )
        return result

    a = np.asarray(parameters["discrimination"][item]).reshape(-1)
    if type(model) is GradedResponseModel:
        thresholds = parameters["thresholds"][item, : categories - 1]
        cumulative = sigmoid((theta @ a)[:, None] - a.sum() * thresholds)
        derivative = cumulative * (1.0 - cumulative)
        second = derivative * (1.0 - 2.0 * cumulative)
        score = np.divide(
            effective,
            probability,
            out=np.zeros_like(probability),
            where=effective != 0.0,
        )
        information = np.divide(
            score,
            probability,
            out=np.zeros_like(probability),
            where=effective != 0.0,
        )

        def category_difference(values):
            padded = np.column_stack(
                (np.zeros(len(theta)), values, np.zeros(len(theta)))
            )
            return padded[:, :-1] - padded[:, 1:]

        slopes = result["discrimination"].reshape(-1)
        for factor in range(model.n_factors):
            centered = theta[:, factor, None] - thresholds
            first = category_difference(derivative * centered)
            second_category = category_difference(second * centered**2)
            slopes[factor] = np.sum(information * first**2 - score * second_category)
        result["thresholds"][: categories - 1] = a.sum() ** 2 * np.sum(
            (information[:, :-1] + information[:, 1:]) * derivative**2
            + (score[:, :-1] - score[:, 1:]) * second,
            axis=0,
        )
        # Coincident or inverted thresholds cannot admit an ordered local trial.
        gaps = np.diff(thresholds)
        if np.any(gaps <= 0.0):
            invalid = np.zeros(categories - 1, dtype=bool)
            invalid[:-1] |= gaps <= 0.0
            invalid[1:] |= gaps <= 0.0
            result["thresholds"][: categories - 1][invalid] = np.nan
        return result

    scale = a.sum()
    tails = np.cumsum(probability[:, :0:-1], axis=1)[:, ::-1]
    result["steps"][: categories - 1] = scale**2 * np.sum(
        total[:, None] * tails * (1.0 - tails), axis=0
    )
    if type(model) is PartialCreditModel:
        return result

    category = np.arange(categories)
    offsets = np.r_[0.0, np.cumsum(parameters["steps"][item, : categories - 1])]
    for factor in range(model.n_factors):
        features = theta[:, factor, None] * category - offsets
        mean = np.sum(probability * features, axis=1)
        result["discrimination"].reshape(-1)[factor] = np.sum(
            total * np.sum(probability * (features - mean[:, None]) ** 2, axis=1)
        )
    return result
