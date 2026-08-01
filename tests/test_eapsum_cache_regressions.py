"""Regression coverage for mask-aware EAPsum scoring and caching."""

import numpy as np
import pytest

from mirt import (
    RUST_AVAILABLE,
    GradedResponseModel,
    TwoParameterLogistic,
    get_backend,
    set_backend,
)
from mirt._rust_backend import lord_wingersky_recursion
from mirt.scoring.eapsum import EAPSumScorer


def _fitted_model() -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=4)
    model.set_parameters(
        discrimination=np.array([0.7, 1.0, 1.3, 1.6]),
        difficulty=np.array([-1.2, -0.3, 0.5, 1.4]),
    )
    model._is_fitted = True
    return model


def test_eapsum_all_missing_responses_return_prior_moments():
    model = _fitted_model()
    responses = np.full((2, model.n_items), -1, dtype=int)

    result = EAPSumScorer(n_quadpts=21).score(model, responses)

    np.testing.assert_allclose(result.theta, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.standard_error, 1.0, atol=1e-14)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="requires Rust extension")
def test_eapsum_native_distribution_matches_numpy_distribution():
    model = _fitted_model()
    responses = np.array([[1, 1, 0, 0]])
    previous_backend = get_backend()

    try:
        set_backend("numpy")
        expected = EAPSumScorer(n_quadpts=21).score(model, responses)
        set_backend("rust")
        actual = EAPSumScorer(n_quadpts=21).score(model, responses)
    finally:
        set_backend(previous_backend)

    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-12)
    np.testing.assert_allclose(
        actual.standard_error,
        expected.standard_error,
        atol=1e-12,
    )


def test_native_distribution_wrapper_rejects_mismatched_item_parameters():
    with pytest.raises(ValueError, match="equal lengths"):
        lord_wingersky_recursion(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(2),
            np.zeros(3),
        )


def test_eapsum_missing_items_match_equivalent_short_form():
    model = _fitted_model()
    responses = np.array([[1, -1, 0, -1], [0, -1, 1, -1]])

    actual = EAPSumScorer(n_quadpts=31).score(model, responses)

    short_model = TwoParameterLogistic(n_items=2)
    short_model.set_parameters(
        discrimination=model.discrimination[[0, 2]],
        difficulty=model.difficulty[[0, 2]],
    )
    short_model._is_fitted = True
    expected = EAPSumScorer(n_quadpts=31).score(
        short_model,
        responses[:, [0, 2]],
    )

    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-12)
    np.testing.assert_allclose(
        actual.standard_error,
        expected.standard_error,
        atol=1e-12,
    )


def test_eapsum_supports_polytomous_missing_item_masks():
    model = GradedResponseModel(n_items=3, n_categories=[2, 3, 4])
    model.set_parameters(
        discrimination=np.array([0.8, 1.0, 1.2]),
        thresholds=np.array(
            [
                [-0.5, 0.0, 0.0],
                [-1.0, 0.8, 0.0],
                [-1.2, 0.1, 1.3],
            ]
        ),
    )
    model._is_fitted = True
    responses = np.array([[0, -1, 3], [1, -1, 1]])

    actual = EAPSumScorer(n_quadpts=31).score(model, responses)

    short_model = GradedResponseModel(n_items=2, n_categories=[2, 4])
    short_model.set_parameters(
        discrimination=model.discrimination[[0, 2]],
        thresholds=model.thresholds[[0, 2]],
    )
    short_model._is_fitted = True
    expected = EAPSumScorer(n_quadpts=31).score(
        short_model,
        responses[:, [0, 2]],
    )

    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-12)
    np.testing.assert_allclose(
        actual.standard_error,
        expected.standard_error,
        atol=1e-12,
    )


def test_eapsum_partitions_cache_by_observed_item_mask():
    model = _fitted_model()
    responses = np.array(
        [
            [1, 0, 1, 0],
            [1, -1, 0, -1],
            [-1, -1, -1, -1],
        ]
    )
    scorer = EAPSumScorer(n_quadpts=21)

    scorer.score(model, responses)

    assert set(scorer._lookup_tables) == {(), (0, 2), (0, 1, 2, 3)}


def test_eapsum_reuses_cache_until_model_parameters_change(monkeypatch):
    model = _fitted_model()
    scorer = EAPSumScorer(n_quadpts=21)
    responses = np.array([[1, 0, 1, 0]])
    original = scorer._compute_sum_score_distribution
    call_count = 0

    def counted_distribution(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, "_compute_sum_score_distribution", counted_distribution)

    first = scorer.score(model, responses)
    second = scorer.score(model, responses)
    assert call_count == 1
    np.testing.assert_array_equal(second.theta, first.theta)

    model.set_item_parameter(2, "difficulty", -2.5)
    changed = scorer.score(model, responses)
    expected = EAPSumScorer(n_quadpts=21).score(model, responses)

    assert call_count == 2
    assert changed.theta[0] != pytest.approx(first.theta[0])
    np.testing.assert_allclose(changed.theta, expected.theta, atol=1e-14)
    np.testing.assert_allclose(
        changed.standard_error,
        expected.standard_error,
        atol=1e-14,
    )


def test_eapsum_detects_direct_parameter_array_mutation():
    model = _fitted_model()
    scorer = EAPSumScorer(n_quadpts=21)
    responses = np.array([[1, 0, 1, 0]])

    before = scorer.score(model, responses)
    model._parameters["difficulty"][0] = 2.5
    actual = scorer.score(model, responses)
    expected = EAPSumScorer(n_quadpts=21).score(model, responses)

    assert actual.theta[0] != pytest.approx(before.theta[0])
    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-14)


@pytest.mark.parametrize(
    "responses",
    [
        np.array([[0, 2, 0, 1]]),
        np.array([[0.0, 0.5, 0.0, 1.0]]),
        np.array([[0.0, -0.5, 0.0, 1.0]]),
        np.array([[0.0, np.nan, 0.0, 1.0]]),
    ],
)
def test_eapsum_rejects_invalid_response_codes(responses):
    with pytest.raises(ValueError):
        EAPSumScorer(n_quadpts=21).score(_fitted_model(), responses)
