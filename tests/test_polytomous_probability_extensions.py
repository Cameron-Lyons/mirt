"""Regression coverage for stable, vectorized polytomous probabilities."""

import numpy as np
import pytest

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)


@pytest.mark.parametrize(
    "model_class",
    [
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        RatingScaleModel,
        GradedRatingScaleModel,
        NominalResponseModel,
    ],
)
def test_extreme_category_probabilities_remain_finite(model_class):
    model = model_class(n_items=3, n_categories=5)
    theta = np.array([-1000.0, 0.0, 1000.0])

    with np.errstate(over="raise", invalid="raise"):
        probabilities = model.probability(theta)

    assert probabilities.shape == (3, 3, 5)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)


def test_rating_scale_model_initializes_with_scalar_or_list_categories():
    scalar_model = RatingScaleModel(n_items=3, n_categories=4)
    list_model = RatingScaleModel(n_items=3, n_categories=[4, 4, 4])

    assert scalar_model.thresholds.shape == (3,)
    assert list_model.thresholds.shape == (3,)
    assert scalar_model.n_categories == [4, 4, 4]
    assert list_model.n_categories == [4, 4, 4]


def test_rating_scale_model_is_available_from_top_level_api():
    from mirt import RatingScaleModel as ExportedRatingScaleModel

    assert ExportedRatingScaleModel is RatingScaleModel


@pytest.mark.parametrize(
    "model",
    [
        GradedResponseModel(n_items=2, n_categories=4),
        GeneralizedPartialCredit(n_items=2, n_categories=4),
        RatingScaleModel(n_items=2, n_categories=4),
        GradedRatingScaleModel(n_items=2, n_categories=4),
        NominalResponseModel(n_items=2, n_categories=4),
    ],
)
def test_single_category_probability_matches_full_matrix(model):
    theta = np.linspace(-3.0, 3.0, 11)
    probabilities = model.probability(theta, item_idx=1)

    for category in range(4):
        np.testing.assert_allclose(
            model.category_probability(theta, 1, category),
            probabilities[:, category],
        )


def test_mixed_category_probability_padding_remains_zero():
    model = GeneralizedPartialCredit(n_items=3, n_categories=[2, 4, 3])

    probabilities = model.probability(np.array([-1.0, 0.0, 1.0]))

    assert probabilities.shape == (3, 3, 4)
    np.testing.assert_array_equal(probabilities[:, 0, 2:], 0.0)
    np.testing.assert_array_equal(probabilities[:, 2, 3:], 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)


@pytest.mark.parametrize(
    "model",
    [
        GradedResponseModel(n_items=3, n_categories=5),
        GradedResponseModel(n_items=3, n_categories=[2, 4, 5]),
        GradedResponseModel(n_items=3, n_categories=5, n_factors=2),
        GradedRatingScaleModel(n_items=3, n_categories=5),
    ],
)
def test_graded_information_reuses_category_curves(
    model, monkeypatch: pytest.MonkeyPatch
):
    theta_1d = np.linspace(-4.0, 4.0, 41)
    theta = (
        theta_1d.reshape(-1, 1)
        if model.n_factors == 1
        else np.column_stack([theta_1d, -0.5 * theta_1d])
    )
    expected_items = []
    for item_idx in range(model.n_items):
        probabilities = model.probability(theta, item_idx)
        n_categories = probabilities.shape[1]
        cumulative = np.empty((theta.shape[0], n_categories + 1))
        cumulative[:, 0] = 1.0
        cumulative[:, -1] = 0.0
        for threshold_idx in range(n_categories - 1):
            cumulative[:, threshold_idx + 1] = model.cumulative_probability(
                theta,
                item_idx,
                threshold_idx,
            )

        if isinstance(model, GradedRatingScaleModel):
            discrimination = model.discrimination
        elif model.n_factors == 1:
            discrimination = float(model.discrimination[item_idx])
        else:
            discrimination = float(np.linalg.norm(model.discrimination[item_idx]))
        derivatives = discrimination * (
            cumulative[:, :-1] * (1.0 - cumulative[:, :-1])
            - cumulative[:, 1:] * (1.0 - cumulative[:, 1:])
        )
        expected_items.append(
            np.divide(
                derivatives**2,
                probabilities,
                out=np.zeros_like(derivatives),
                where=probabilities > PROB_EPSILON,
            ).sum(axis=1)
        )

    def reject_recomputation(*args, **kwargs):
        raise AssertionError("information should reuse category probabilities")

    monkeypatch.setattr(model, "cumulative_probability", reject_recomputation)

    for item_idx, expected in enumerate(expected_items):
        np.testing.assert_allclose(
            model.information(theta, item_idx),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )
    np.testing.assert_allclose(
        model.information(theta),
        np.sum(expected_items, axis=0),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "model",
    [
        GeneralizedPartialCredit(n_items=2, n_categories=4),
        RatingScaleModel(n_items=2, n_categories=4),
        NominalResponseModel(n_items=2, n_categories=4),
    ],
)
def test_row_aligned_log_likelihood_supports_single_row_broadcasting(model):
    responses = np.array([[0, 1], [1, 2], [3, -1]])
    theta = np.array([[-1.0], [0.0], [1.0]])

    pairwise = model.log_likelihood(responses, theta)
    shared_theta = model.log_likelihood(responses, theta[:1])
    shared_response = model.log_likelihood(responses[:1], theta)

    assert pairwise.shape == (3,)
    assert shared_theta.shape == (3,)
    assert shared_response.shape == (3,)

    pairwise_probabilities = model.probability(theta)
    expected_pairwise = np.array(
        [
            np.log(pairwise_probabilities[row, 0, responses[row, 0]])
            + (
                0.0
                if responses[row, 1] < 0
                else np.log(pairwise_probabilities[row, 1, responses[row, 1]])
            )
            for row in range(3)
        ]
    )
    np.testing.assert_allclose(pairwise, expected_pairwise)


@pytest.mark.parametrize(
    "model",
    [
        GeneralizedPartialCredit(n_items=2, n_categories=4),
        RatingScaleModel(n_items=2, n_categories=4),
        NominalResponseModel(n_items=2, n_categories=4),
    ],
)
def test_batch_log_likelihood_matches_probability_lookup(model):
    responses = np.array([[0, 1], [2, -1], [3, 2]])
    theta = np.array([[-1.0], [0.0], [1.0]])

    actual = model.log_likelihood_batch(responses, theta)
    expected = np.zeros_like(actual)
    for item_idx in range(model.n_items):
        probabilities = model.probability(theta, item_idx=item_idx)
        for person_idx, response in enumerate(responses[:, item_idx]):
            if response >= 0:
                expected[person_idx] += np.log(probabilities[:, response])

    np.testing.assert_allclose(actual, expected)


def test_log_likelihood_rejects_invalid_categories_and_row_counts():
    model = GeneralizedPartialCredit(n_items=2, n_categories=[2, 4])

    with pytest.raises(MirtDataError, match="item 0 must be below 2"):
        model.log_likelihood(np.array([[2, 0]]), np.array([[0.0]]))

    with pytest.raises(MirtDataError, match="item 0 must be below 2"):
        model.log_likelihood_batch(np.array([[2, 0]]), np.array([[0.0]]))

    with pytest.raises(MirtDataError, match="matching row counts"):
        model.log_likelihood(np.zeros((2, 2), dtype=int), np.zeros((3, 1)))
