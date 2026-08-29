"""Regression coverage for grouped all-item probability evaluation."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt.models.polytomous as polytomous
from mirt.models.base import PolytomousItemModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)

_MIXED_CATEGORIES = [2, 5, 3, 4, 2, 5, 3]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GradedResponseModel(7, _MIXED_CATEGORIES, n_factors=1),
        lambda: GradedResponseModel(7, _MIXED_CATEGORIES, n_factors=3),
        lambda: GeneralizedPartialCredit(7, _MIXED_CATEGORIES, n_factors=1),
        lambda: GeneralizedPartialCredit(7, _MIXED_CATEGORIES, n_factors=3),
        lambda: PartialCreditModel(7, _MIXED_CATEGORIES),
        lambda: NominalResponseModel(7, _MIXED_CATEGORIES, n_factors=1),
        lambda: NominalResponseModel(7, _MIXED_CATEGORIES, n_factors=3),
        lambda: RatingScaleModel(7, 5),
        lambda: GradedRatingScaleModel(7, 5),
    ],
    ids=[
        "grm-1d",
        "grm-3d",
        "gpcm-1d",
        "gpcm-3d",
        "pcm",
        "nrm-1d",
        "nrm-3d",
        "rsm",
        "grsm",
    ],
)
def test_all_item_probabilities_match_item_specific_curves(
    factory: Callable[[], PolytomousItemModel],
) -> None:
    model = factory()
    _set_distinct_parameters(model)
    theta = np.linspace(-2.5, 2.5, 39 * model.n_factors).reshape(39, model.n_factors)

    actual = model.probability(theta)
    expected = _itemwise_probabilities(model, theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    for item_idx, n_categories in enumerate(model.n_categories):
        np.testing.assert_allclose(
            actual[:, item_idx, :n_categories].sum(axis=1),
            1.0,
            rtol=1e-14,
            atol=1e-14,
        )
        assert np.count_nonzero(actual[:, item_idx, n_categories:]) == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GradedResponseModel(8, [3, 5] * 4, n_factors=2),
        lambda: GeneralizedPartialCredit(8, [3, 5] * 4, n_factors=2),
        lambda: NominalResponseModel(8, [3, 5] * 4, n_factors=2),
        lambda: RatingScaleModel(8, 5),
        lambda: GradedRatingScaleModel(8, 5),
    ],
    ids=["grm", "gpcm", "nrm", "rsm", "grsm"],
)
def test_all_item_probabilities_are_chunk_invariant(
    factory: Callable[[], PolytomousItemModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = factory()
    _set_distinct_parameters(model)
    theta = np.linspace(-3.0, 3.0, 82 * model.n_factors).reshape(82, model.n_factors)
    expected = model.probability(theta)

    monkeypatch.setattr(polytomous, "_MAX_PROBABILITY_CHUNK_ENTRIES", 1)
    actual = model.probability(theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize(
    "model",
    [
        GradedResponseModel(4, [2, 3, 4, 5], n_factors=2),
        GeneralizedPartialCredit(4, [2, 3, 4, 5], n_factors=2),
        NominalResponseModel(4, [2, 3, 4, 5], n_factors=2),
    ],
    ids=["grm", "gpcm", "nrm"],
)
def test_all_item_probabilities_remain_finite_at_extreme_theta(
    model: PolytomousItemModel,
) -> None:
    theta = np.array(
        [
            [-1_000.0, 1_000.0],
            [0.0, 0.0],
            [1_000.0, -1_000.0],
        ]
    )

    probabilities = model.probability(theta)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    for item_idx, n_categories in enumerate(model.n_categories):
        np.testing.assert_allclose(
            probabilities[:, item_idx, :n_categories].sum(axis=1),
            1.0,
            rtol=1e-14,
            atol=1e-14,
        )


def _itemwise_probabilities(
    model: PolytomousItemModel,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = np.zeros(
        (theta.shape[0], model.n_items, model.max_categories),
        dtype=np.float64,
    )
    for item_idx, n_categories in enumerate(model.n_categories):
        probabilities[:, item_idx, :n_categories] = model.probability(theta, item_idx)
    return probabilities


def _set_distinct_parameters(model: PolytomousItemModel) -> None:
    """Give every item distinct values so axis errors cannot cancel out."""
    rng = np.random.default_rng(9182)
    item_offsets = np.linspace(-0.25, 0.25, model.n_items)[:, None]

    if isinstance(model, GradedResponseModel):
        model.set_parameters(
            discrimination=rng.uniform(0.4, 1.8, size=model.discrimination.shape),
            thresholds=model.thresholds + item_offsets,
        )
    elif isinstance(model, PartialCreditModel):
        model.set_parameters(steps=model.steps + item_offsets)
    elif isinstance(model, GeneralizedPartialCredit):
        model.set_parameters(
            discrimination=rng.uniform(0.4, 1.8, size=model.discrimination.shape),
            steps=model.steps + item_offsets,
        )
    elif isinstance(model, NominalResponseModel):
        model.set_parameters(
            slopes=model.slopes + rng.normal(0.0, 0.2, size=model.slopes.shape),
            intercepts=model.intercepts + item_offsets,
        )
    elif isinstance(model, RatingScaleModel):
        model.set_parameters(
            difficulty=np.linspace(-0.8, 0.8, model.n_items),
            thresholds=np.linspace(-1.4, 1.1, model.max_categories - 1),
        )
    elif isinstance(model, GradedRatingScaleModel):
        model.set_parameters(
            discrimination=1.35,
            difficulty=np.linspace(-0.8, 0.8, model.n_items),
            thresholds=np.linspace(-1.8, 1.6, model.max_categories - 1),
        )
