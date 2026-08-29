"""Tests for aligned respondent-item probability evaluation."""

from collections.abc import Callable

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel


def _configure_dichotomous(model: BaseItemModel) -> BaseItemModel:
    parameters = model.parameters
    parameters["difficulty"] = np.array([-1.2, -0.3, 0.5, 1.4])
    if model.model_name == "1PL":
        parameters.pop("discrimination")
    else:
        parameters["discrimination"] = np.array([0.6, 1.1, 1.5, 2.0])
    if "guessing" in parameters:
        parameters["guessing"] = np.array([0.05, 0.1, 0.2, 0.25])
    if "upper" in parameters:
        parameters["upper"] = np.array([0.9, 0.95, 0.85, 0.98])
    model.set_parameters(**parameters)
    return model


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OneParameterLogistic(4),
        lambda: TwoParameterLogistic(4),
        lambda: ThreeParameterLogistic(4),
        lambda: FourParameterLogistic(4),
    ],
)
def test_dichotomous_pairs_match_individual_item_evaluation(
    factory: Callable[[], BaseItemModel],
) -> None:
    model = _configure_dichotomous(factory())
    theta = np.array([[-2.0], [-0.5], [0.2], [1.0], [2.5]])
    item_indices = np.array([3, 0, 2, 2, 1])

    actual = model.probability_pairs(theta, item_indices)
    expected = np.array(
        [
            model.probability(theta[row : row + 1], int(item_idx))[0]
            for row, item_idx in enumerate(item_indices)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_multidimensional_2pl_pairs_match_item_evaluation() -> None:
    model = TwoParameterLogistic(3, n_factors=2)
    model.set_parameters(
        discrimination=np.array([[0.8, 0.2], [0.3, 1.4], [1.1, -0.4]]),
        difficulty=np.array([-0.7, 0.2, 1.0]),
    )
    theta = np.array([[-1.0, 0.5], [0.2, -0.4], [1.5, 0.8], [0.0, 1.2]])
    item_indices = np.array([2, 0, 1, 2])

    actual = model.probability_pairs(theta, item_indices)
    expected = np.array(
        [
            model.probability(theta[row : row + 1], int(item_idx))[0]
            for row, item_idx in enumerate(item_indices)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "model",
    [
        GradedResponseModel(3, [2, 4, 3]),
        GeneralizedPartialCredit(3, [2, 4, 3]),
    ],
)
def test_polytomous_pairs_match_items_with_zero_padding(model: BaseItemModel) -> None:
    theta = np.array([[-1.5], [0.0], [1.2], [0.6], [-0.3]])
    item_indices = np.array([1, 0, 2, 1, 0])

    actual = model.probability_pairs(theta, item_indices)
    expected = np.zeros((len(theta), 4))
    for row, item_idx in enumerate(item_indices):
        probabilities = model.probability(theta[row : row + 1], int(item_idx))[0]
        expected[row, : probabilities.size] = probabilities

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "model",
    [
        TwoParameterLogistic(3),
        ThreeParameterLogistic(3),
        GradedResponseModel(3, [2, 3, 4]),
        GeneralizedPartialCredit(3, [2, 3, 4]),
    ],
)
def test_standard_pair_implementations_avoid_item_dispatch(
    model: BaseItemModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta = np.linspace(-2.0, 2.0, 9).reshape(-1, 1)
    item_indices = np.tile(np.arange(3), 3)

    def fail_item_dispatch(*args: object, **kwargs: object) -> None:
        raise AssertionError("paired evaluation dispatched through probability()")

    monkeypatch.setattr(model, "probability", fail_item_dispatch)

    result = model.probability_pairs(theta, item_indices)

    assert result.shape[0] == len(theta)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "item_indices",
    [
        np.array([0.0, 1.0]),
        np.array([[0, 1]]),
        np.array([0]),
        np.array([0, 3]),
    ],
)
def test_probability_pairs_reject_invalid_item_indices(
    item_indices: np.ndarray,
) -> None:
    model = TwoParameterLogistic(3)

    with pytest.raises(MirtValidationError, match="item_indices"):
        model.probability_pairs(np.array([[-1.0], [1.0]]), item_indices)


def test_probability_pairs_support_empty_inputs() -> None:
    dichotomous = TwoParameterLogistic(2)
    polytomous = GradedResponseModel(2, [2, 3])
    theta = np.empty((0, 1))
    item_indices = np.empty(0, dtype=int)

    assert dichotomous.probability_pairs(theta, item_indices).shape == (0,)
    assert polytomous.probability_pairs(theta, item_indices).shape == (0, 3)
