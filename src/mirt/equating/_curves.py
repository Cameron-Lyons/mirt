"""Validated expected-score curves shared by curve-based equating tools."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10


@dataclass
class _ItemCurveSet:
    """Expected-score and category curves for selected items."""

    expected: NDArray[np.float64]
    categories: list[NDArray[np.float64]] | None


def _validated_probabilities(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate probability bounds and absorb harmless roundoff."""
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -_PROBABILITY_TOLERANCE)
        | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")
    return np.clip(probabilities, 0.0, 1.0, out=probabilities)


def _expected_item_score_curves(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.int_],
) -> _ItemCurveSet:
    """Evaluate selected items without scanning unused portions of large banks.

    Selections covering at least a quarter of the bank use one batched
    probability call; sparse selections of large banks are evaluated per item.
    """
    n_theta = theta.shape[0]
    if len(item_indices) * 4 >= model.n_items:
        probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    else:
        probabilities = None

    if model.is_polytomous and probabilities is not None:
        if probabilities.ndim != 3 or probabilities.shape[:2] != (
            n_theta,
            model.n_items,
        ):
            raise ValueError("model returned invalid polytomous probabilities")
        probabilities = _validated_probabilities(probabilities[:, item_indices, :])
        probability_mass = probabilities.sum(axis=2, keepdims=True)
        if np.any(np.abs(probability_mass - 1.0) > _PROBABILITY_TOLERANCE):
            raise ValueError("model category probabilities must sum to one")
        probabilities /= probability_mass
        categories = np.arange(probabilities.shape[2], dtype=np.float64)
        category_counts = model.n_categories
        return _ItemCurveSet(
            expected=probabilities @ categories,
            categories=[
                probabilities[:, column, : category_counts[int(index)]]
                for column, index in enumerate(item_indices)
            ],
        )

    if model.is_polytomous:
        expected = np.empty((n_theta, len(item_indices)), dtype=np.float64)
        category_curves: list[NDArray[np.float64]] = []
        category_counts = model.n_categories
        for column, item_index in enumerate(item_indices):
            n_categories = category_counts[int(item_index)]
            item_probabilities = np.array(
                model.probability(theta, int(item_index)),
                dtype=np.float64,
                copy=True,
            )
            if item_probabilities.shape != (n_theta, n_categories):
                raise ValueError("model returned invalid polytomous probabilities")
            item_probabilities = _validated_probabilities(item_probabilities)
            probability_mass = item_probabilities.sum(axis=1, keepdims=True)
            if np.any(np.abs(probability_mass - 1.0) > _PROBABILITY_TOLERANCE):
                raise ValueError("model category probabilities must sum to one")
            item_probabilities /= probability_mass
            category_curves.append(item_probabilities)
            expected[:, column] = item_probabilities @ np.arange(
                n_categories, dtype=np.float64
            )
        return _ItemCurveSet(expected=expected, categories=category_curves)

    if probabilities is None:
        item_curves = []
        for item_index in item_indices:
            item_probabilities = np.array(
                model.probability(theta, int(item_index)),
                dtype=np.float64,
                copy=True,
            )
            if item_probabilities.shape != (n_theta,):
                raise ValueError("model returned invalid dichotomous probabilities")
            item_curves.append(_validated_probabilities(item_probabilities))
        return _ItemCurveSet(expected=np.column_stack(item_curves), categories=None)

    if probabilities.shape != (n_theta, model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    return _ItemCurveSet(
        expected=_validated_probabilities(probabilities[:, item_indices]),
        categories=None,
    )
