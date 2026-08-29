"""Regression coverage for grouped nested-logit probability evaluation."""

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt.models.nested as nested
from mirt.models.nested import (
    FourPLNestedLogit,
    ThreePLNestedLogit,
    TwoPLNestedLogit,
)

MODEL_CLASSES = (TwoPLNestedLogit, ThreePLNestedLogit, FourPLNestedLogit)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_grouped_probabilities_match_itemwise_curves(
    model_class: type[TwoPLNestedLogit],
) -> None:
    categories = [3, 4, 5, 6] * 4
    correct = [item_idx % count for item_idx, count in enumerate(categories)]
    model = model_class(16, categories, correct_response=correct)
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


def test_grouped_probabilities_are_chunk_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FourPLNestedLogit(
        12,
        5,
        correct_response=[item_idx % 5 for item_idx in range(12)],
    )
    _set_distinct_parameters(model)
    theta = np.linspace(-3.0, 3.0, 83)
    expected = model.probability(theta)

    monkeypatch.setattr(
        nested,
        "_MAX_NESTED_PROBABILITY_CHUNK_ENTRIES",
        theta.size * model.max_categories * 3,
    )

    def reject_itemwise_evaluation(*args: object, **kwargs: object) -> None:
        raise AssertionError("bounded groups should remain vectorized")

    monkeypatch.setattr(model, "_item_curves_from_theta", reject_itemwise_evaluation)
    actual = model.probability(theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=3e-15)


def test_grouped_extreme_logits_remain_finite_and_match_itemwise() -> None:
    model = FourPLNestedLogit(
        8,
        5,
        correct_response=[item_idx % 5 for item_idx in range(8)],
    )
    slopes = np.array(
        [
            np.roll([-1000.0, -20.0, 700.0, 0.0, 1000.0], item_idx)
            for item_idx in range(model.n_items)
        ]
    )
    intercepts = np.array(
        [
            np.roll([900.0, -800.0, 300.0, 0.0, -700.0], -item_idx)
            for item_idx in range(model.n_items)
        ]
    )
    model.set_parameters(
        discrimination=np.linspace(0.5, 8.0, model.n_items),
        difficulty=np.linspace(-1.0, 1.0, model.n_items),
        guessing=np.linspace(0.03, 0.15, model.n_items),
        upper=np.linspace(0.85, 0.98, model.n_items),
        distractor_slopes=slopes,
        distractor_intercepts=intercepts,
    )
    theta = np.array([-100.0, 0.0, 100.0])

    actual = model.probability(theta)
    expected = _itemwise_probabilities(model, theta)

    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=2), 1.0, rtol=1e-14, atol=1e-14)


def test_grouped_probability_rejects_corrupted_asymptotes() -> None:
    model = FourPLNestedLogit(8, 5)
    model.upper[0] = 0.1

    with pytest.raises(ValueError, match="strictly less"):
        model.probability(np.array([0.0]))


def _itemwise_probabilities(
    model: TwoPLNestedLogit,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = np.zeros(
        (theta.size, model.n_items, model.max_categories),
        dtype=np.float64,
    )
    for item_idx, n_categories in enumerate(model.n_categories):
        probabilities[:, item_idx, :n_categories] = model.probability(theta, item_idx)
    return probabilities


def _set_distinct_parameters(model: TwoPLNestedLogit) -> None:
    parameter_count = model.n_items * model.max_categories
    parameters = {
        "discrimination": np.linspace(0.5, 2.0, model.n_items),
        "difficulty": np.linspace(-1.5, 1.5, model.n_items),
        "distractor_slopes": np.linspace(-1.5, 1.5, parameter_count).reshape(
            model.n_items,
            model.max_categories,
        ),
        "distractor_intercepts": np.linspace(-0.8, 0.8, parameter_count).reshape(
            model.n_items,
            model.max_categories,
        ),
    }
    if isinstance(model, ThreePLNestedLogit):
        parameters["guessing"] = np.linspace(0.03, 0.15, model.n_items)
    if isinstance(model, FourPLNestedLogit):
        parameters["upper"] = np.linspace(0.85, 0.98, model.n_items)
    model.set_parameters(**parameters)
