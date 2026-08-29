"""Regression coverage for grouped GGUM probability evaluation."""

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt.models.unfolding as unfolding
from mirt.models.unfolding import GeneralizedGradedUnfolding


def test_grouped_ggum_probabilities_match_itemwise_curves() -> None:
    categories = [2, 3, 4, 5] * 4
    model = GeneralizedGradedUnfolding(16, categories)
    _set_distinct_parameters(model)
    theta = np.linspace(-4.0, 4.0, 101)

    actual = model.probability(theta)
    expected = _itemwise_probabilities(model, theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=3e-15)
    for item_idx, n_categories in enumerate(categories):
        np.testing.assert_allclose(
            actual[:, item_idx, :n_categories].sum(axis=1),
            1.0,
            rtol=1e-14,
            atol=1e-14,
        )
        assert np.count_nonzero(actual[:, item_idx, n_categories:]) == 0


def test_grouped_ggum_probabilities_are_chunk_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GeneralizedGradedUnfolding(12, 5)
    _set_distinct_parameters(model)
    theta = np.linspace(-3.0, 3.0, 83)
    expected = model.probability(theta)

    monkeypatch.setattr(
        unfolding,
        "_MAX_GGUM_PROBABILITY_CHUNK_ENTRIES",
        theta.size * model.max_categories * 4,
    )

    def reject_itemwise_evaluation(*args: object, **kwargs: object) -> None:
        raise AssertionError("bounded groups should remain vectorized")

    monkeypatch.setattr(model, "_item_components", reject_itemwise_evaluation)
    actual = model.probability(theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=3e-15)


def test_grouped_ggum_extremes_use_stable_item_fallback() -> None:
    model = GeneralizedGradedUnfolding(8, 5)
    model.set_parameters(
        discrimination=np.linspace(0.5, 2.0, model.n_items),
        location=np.linspace(-1.0, 1.0, model.n_items),
    )
    theta = np.array([-1e308, 0.0, 1e308])

    with np.errstate(over="raise", invalid="raise"):
        actual = model.probability(theta)
        expected = _itemwise_probabilities(model, theta)

    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=2), 1.0, rtol=1e-14, atol=1e-14)


def _itemwise_probabilities(
    model: GeneralizedGradedUnfolding,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = np.zeros(
        (theta.size, model.n_items, model.max_categories),
        dtype=np.float64,
    )
    for item_idx, n_categories in enumerate(model.n_categories):
        probabilities[:, item_idx, :n_categories] = model.probability(theta, item_idx)
    return probabilities


def _set_distinct_parameters(model: GeneralizedGradedUnfolding) -> None:
    model.set_parameters(
        discrimination=np.linspace(0.5, 2.0, model.n_items),
        location=np.linspace(-1.5, 1.5, model.n_items),
    )
    for item_idx, n_categories in enumerate(model.n_categories):
        independent = np.linspace(-2.5, -0.4, n_categories - 1)
        model.set_independent_thresholds(
            item_idx,
            independent + 0.01 * item_idx,
        )
