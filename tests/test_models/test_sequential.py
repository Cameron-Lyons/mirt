"""Tests for sequential and adjacent-category ordinal models."""

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.sequential import (
    AdjacentCategoryModel,
    ContinuationRatioModel,
    SequentialResponseModel,
)

MODEL_CLASSES = (
    SequentialResponseModel,
    ContinuationRatioModel,
    AdjacentCategoryModel,
)


def _configured_model(model_class):
    model = model_class(n_items=2, n_categories=[3, 5], item_names=["A", "B"])
    model.set_parameters(
        discrimination=np.array([1.2, 0.8]),
        thresholds=np.array(
            [
                [-1.0, 0.4, 0.0, 0.0],
                [-1.5, -0.3, 0.7, 1.2],
            ]
        ),
    )
    return model


@pytest.fixture(params=MODEL_CLASSES, ids=["sequential", "continuation", "adjacent"])
def configured_model(request):
    return _configured_model(request.param)


def test_initialization_with_varying_category_counts(configured_model):
    assert configured_model.n_categories == [3, 5]
    assert configured_model.max_categories == 5
    assert configured_model.discrimination.shape == (2,)
    assert configured_model.thresholds.shape == (2, 4)
    np.testing.assert_array_equal(configured_model.thresholds_for_item(0), [-1.0, 0.4])
    np.testing.assert_array_equal(
        configured_model.thresholds_for_item(1), [-1.5, -0.3, 0.7, 1.2]
    )


def test_probability_shapes_ranges_and_padding(configured_model):
    theta = np.linspace(-2.0, 2.0, 8)
    probabilities = configured_model.probability(theta)

    assert probabilities.shape == (8, 2, 5)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)
    np.testing.assert_array_equal(probabilities[:, 0, 3:], 0.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert configured_model.probability(theta, item_idx=0).shape == (8, 3)
    assert configured_model.probability(theta, item_idx=1).shape == (8, 5)


def test_category_probability_and_curves_share_one_likelihood(configured_model):
    theta = np.array([-1.0, 0.0, 1.0])
    probabilities = configured_model.probability(theta, item_idx=1)

    np.testing.assert_allclose(
        configured_model.category_response_curves(theta, 1), probabilities
    )
    for category in range(5):
        np.testing.assert_allclose(
            configured_model.category_probability(theta, 1, category),
            probabilities[:, category],
        )


@pytest.mark.parametrize(
    "model_class", [SequentialResponseModel, ContinuationRatioModel]
)
def test_sequential_probability_factorization(model_class):
    model = _configured_model(model_class)
    theta = np.array([-0.7, 0.2, 1.1])
    steps = np.column_stack(
        [model.step_probability(theta, 1, step) for step in range(4)]
    )
    expected = np.column_stack(
        [
            1.0 - steps[:, 0],
            steps[:, 0] * (1.0 - steps[:, 1]),
            steps[:, 0] * steps[:, 1] * (1.0 - steps[:, 2]),
            np.prod(steps[:, :3], axis=1) * (1.0 - steps[:, 3]),
            np.prod(steps, axis=1),
        ]
    )

    np.testing.assert_allclose(model.probability(theta, 1), expected)
    for category in range(5):
        np.testing.assert_allclose(
            model.reach_probability(theta, 1, category),
            np.sum(expected[:, category:], axis=1),
        )


def test_sequential_and_continuation_models_are_forward_ratio_equivalents():
    sequential = _configured_model(SequentialResponseModel)
    continuation = _configured_model(ContinuationRatioModel)
    theta = np.linspace(-2.0, 2.0, 11)

    np.testing.assert_allclose(
        sequential.probability(theta), continuation.probability(theta)
    )
    np.testing.assert_allclose(
        sequential.information(theta), continuation.information(theta)
    )


def test_adjacent_category_log_odds():
    model = _configured_model(AdjacentCategoryModel)
    theta = np.array([-1.1, 0.0, 1.3])
    probabilities = model.probability(theta, item_idx=1)
    expected_log_odds = model.discrimination[1] * (
        theta[:, None] - model.thresholds_for_item(1)[None, :]
    )

    np.testing.assert_allclose(
        np.log(probabilities[:, 1:] / probabilities[:, :-1]), expected_log_odds
    )


def test_information_matches_category_probability_derivatives(configured_model):
    theta = np.array([-1.3, -0.2, 0.6, 1.5])
    step = 1e-6
    probability = configured_model.probability(theta, item_idx=1)
    derivative = (
        configured_model.probability(theta + step, item_idx=1)
        - configured_model.probability(theta - step, item_idx=1)
    ) / (2.0 * step)
    expected = np.sum(derivative**2 / probability, axis=1)

    np.testing.assert_allclose(
        configured_model.information(theta, item_idx=1),
        expected,
        rtol=2e-6,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "model_class", [SequentialResponseModel, ContinuationRatioModel]
)
def test_sequential_information_matches_conditional_step_sum(model_class):
    model = _configured_model(model_class)
    theta = np.array([-1.0, 0.0, 1.0])
    steps = np.column_stack(
        [model.step_probability(theta, 1, step) for step in range(4)]
    )
    reach = np.column_stack(
        [
            np.ones(theta.size),
            steps[:, 0],
            np.prod(steps[:, :2], axis=1),
            np.prod(steps[:, :3], axis=1),
        ]
    )
    expected = model.discrimination[1] ** 2 * np.sum(
        reach * steps * (1.0 - steps), axis=1
    )

    np.testing.assert_allclose(model.information(theta, 1), expected)


def test_total_information_is_sum_of_item_information(configured_model):
    theta = np.linspace(-1.5, 1.5, 7)
    expected = sum(
        configured_model.information(theta, item_idx=item_idx)
        for item_idx in range(configured_model.n_items)
    )
    np.testing.assert_allclose(configured_model.information(theta), expected)


def test_expected_scores_match_probability_moments(configured_model):
    theta = np.array([-1.0, 0.0, 1.0])
    probabilities = configured_model.probability(theta)
    categories = np.arange(configured_model.max_categories)
    item_expected = np.sum(probabilities[:, 1] * categories[None, :], axis=1)
    total_expected = np.sum(probabilities * categories[None, None, :], axis=(1, 2))

    np.testing.assert_allclose(
        configured_model.expected_score(theta, item_idx=1), item_expected
    )
    np.testing.assert_allclose(configured_model.expected_score(theta), total_expected)


def test_log_likelihood_matches_selected_category_probabilities(configured_model):
    theta = np.array([-0.8, 0.1, 1.0])
    responses = np.array([[0, 4], [2, 1], [1, 3]])
    probabilities = configured_model.probability(theta)
    expected = np.array(
        [
            np.log(probabilities[person, 0, responses[person, 0]])
            + np.log(probabilities[person, 1, responses[person, 1]])
            for person in range(3)
        ]
    )

    np.testing.assert_allclose(
        configured_model.log_likelihood(responses, theta), expected
    )


def test_log_likelihood_supports_independent_missing_responses(configured_model):
    theta = np.array([-0.8, 0.1, 1.0])
    responses = np.array([[-1.0, 4.0], [2.0, np.nan], [np.nan, -1.0]])
    probabilities = configured_model.probability(theta)

    np.testing.assert_allclose(
        configured_model.log_likelihood(responses, theta),
        [np.log(probabilities[0, 1, 4]), np.log(probabilities[1, 0, 2]), 0.0],
    )


def test_batch_log_likelihood_matches_person_likelihoods(configured_model):
    responses = np.array([[0, 4], [2, 1], [-1, 3]])
    theta_grid = np.array([-1.0, 0.2, 1.1])
    expected = np.column_stack(
        [
            configured_model.log_likelihood(
                responses, np.full(responses.shape[0], theta)
            )
            for theta in theta_grid
        ]
    )

    np.testing.assert_allclose(
        configured_model.log_likelihood_batch(responses, theta_grid), expected
    )


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"n_items": 1, "n_categories": 1}, "n_categories"),
        ({"n_items": 2, "n_categories": [3]}, "n_categories"),
        ({"n_items": 2, "n_categories": [3, 1]}, "n_categories"),
        ({"n_items": 2, "n_categories": [3, 2.5]}, "n_categories"),
        ({"n_items": 1, "n_categories": 3, "n_factors": 2}, "one factor"),
    ],
)
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_invalid_model_structure_is_rejected(model_class, parameters, match):
    with pytest.raises(MirtValidationError, match=match):
        model_class(**parameters)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    ("parameter", "values"),
    [
        ("discrimination", np.array([0.0, 1.0])),
        ("discrimination", np.array([np.nan, 1.0])),
        (
            "thresholds",
            np.array([[0.0, np.inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_parameter_domains_are_validated(model_class, parameter, values):
    model = _configured_model(model_class)
    with pytest.raises(MirtValidationError, match=parameter):
        model.set_parameters(**{parameter: values})


def test_parameter_updates_are_atomic_and_detached():
    model = _configured_model(SequentialResponseModel)
    original_thresholds = model.thresholds.copy()
    thresholds = np.full((2, 4), 0.5)

    with pytest.raises(MirtValidationError, match="discrimination"):
        model.set_parameters(
            thresholds=thresholds,
            discrimination=np.array([1.0, -1.0]),
        )
    np.testing.assert_array_equal(model.thresholds, original_thresholds)

    model.set_parameters(thresholds=thresholds)
    thresholds[0, 0] = 99.0
    np.testing.assert_array_equal(model.thresholds, np.full((2, 4), 0.5))


def test_item_parameter_updates_accept_active_thresholds():
    model = _configured_model(SequentialResponseModel)
    model.set_item_parameter(0, "thresholds", np.array([-0.5, 0.8]))
    model.set_item_parameter(1, "discrimination", 1.6)

    np.testing.assert_array_equal(model.thresholds_for_item(0), [-0.5, 0.8])
    assert model.discrimination[1] == 1.6
    with pytest.raises(MirtValidationError, match="thresholds"):
        model.set_item_parameter(0, "thresholds", np.array([0.2]))
    with pytest.raises(MirtValidationError, match="discrimination"):
        model.set_item_parameter(0, "discrimination", 0.0)


def test_mutated_invalid_state_is_rejected_at_evaluation():
    model = _configured_model(SequentialResponseModel)
    model.discrimination[0] = -1.0
    with pytest.raises(MirtValidationError, match="discrimination"):
        model.probability(np.array([0.0]))


@pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
def test_invalid_item_indices_are_rejected(item_idx):
    with pytest.raises((IndexError, MirtValidationError)):
        _configured_model(SequentialResponseModel).probability(
            np.array([0.0]), item_idx
        )


@pytest.mark.parametrize("category", [-1, 3, 1.5, True])
def test_invalid_categories_are_rejected(category):
    with pytest.raises((IndexError, MirtValidationError)):
        _configured_model(SequentialResponseModel).category_probability(
            np.array([0.0]), 0, category
        )


def test_invalid_step_is_rejected():
    model = _configured_model(SequentialResponseModel)
    with pytest.raises(IndexError, match="Step"):
        model.step_probability(np.array([0.0]), 0, 2)


@pytest.mark.parametrize("bad_response", [3.0, 1.5, np.inf, -np.inf])
def test_invalid_observed_responses_are_rejected(bad_response):
    responses = np.array([[bad_response, 0.0]])
    with pytest.raises(MirtValidationError, match="responses"):
        _configured_model(SequentialResponseModel).log_likelihood(
            responses, np.array([0.0])
        )


def test_likelihood_validates_shapes_and_person_counts():
    model = _configured_model(SequentialResponseModel)
    with pytest.raises(MirtValidationError, match="responses"):
        model.log_likelihood(np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(MirtValidationError, match="same number"):
        model.log_likelihood(np.zeros((2, 2)), np.zeros(3))


def test_invalid_ability_values_are_rejected(configured_model):
    with pytest.raises(MirtValidationError, match="theta"):
        configured_model.probability(np.array([np.nan]))
    with pytest.raises(MirtValidationError, match="theta"):
        configured_model.probability(np.empty((0, 1)))


def test_extreme_abilities_are_numerically_safe(configured_model):
    theta = np.array([-1e6, 1e6])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = configured_model.probability(theta)
        information = configured_model.information(theta)

    assert np.all(np.isfinite(probabilities))
    assert np.all(np.isfinite(information))
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)


def test_copy_is_independent(configured_model):
    copied = configured_model.copy()
    np.testing.assert_array_equal(
        copied.discrimination, configured_model.discrimination
    )
    np.testing.assert_array_equal(copied.thresholds, configured_model.thresholds)
    copied.thresholds[0, 0] = 99.0
    assert configured_model.thresholds[0, 0] != 99.0
