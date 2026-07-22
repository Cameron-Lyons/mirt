"""Smoke tests for sequential / ordinal response models."""

import numpy as np

from mirt.models.sequential import (
    AdjacentCategoryModel,
    ContinuationRatioModel,
    SequentialResponseModel,
)


def _assert_category_probs_sum(model, n_persons=8, n_cats=4):
    theta = np.linspace(-2, 2, n_persons)
    probs = model.probability(theta)
    assert probs.shape == (n_persons, model.n_items, n_cats)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all((probs >= 0) & (probs <= 1))


def test_sequential_response_category_probs():
    model = SequentialResponseModel(n_items=3, n_categories=4)
    _assert_category_probs_sum(model)


def test_continuation_ratio_category_probs():
    model = ContinuationRatioModel(n_items=3, n_categories=4)
    _assert_category_probs_sum(model)


def test_adjacent_category_category_probs():
    model = AdjacentCategoryModel(n_items=3, n_categories=4)
    _assert_category_probs_sum(model)
