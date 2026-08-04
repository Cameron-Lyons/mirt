"""Tests for keyed multiple-choice nested-logit models."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, softmax

import mirt
from mirt import FourPLNestedLogit, ThreePLNestedLogit, TwoPLNestedLogit

MODEL_CLASSES = (TwoPLNestedLogit, ThreePLNestedLogit, FourPLNestedLogit)


@pytest.fixture(params=MODEL_CLASSES)
def nested_model(request):
    """Create each nested-logit variant with heterogeneous item parameters."""
    model_class = request.param
    model = model_class(
        n_items=3,
        n_categories=[4, 3, 5],
        correct_response=[2, 0, 4],
        item_names=["A", "B", "C"],
    )
    parameters = {
        "discrimination": np.array([1.3, 0.8, 1.6]),
        "difficulty": np.array([0.2, -0.4, 0.7]),
        "distractor_slopes": np.array(
            [
                [-1.0, 0.5, 0.0, 1.4, 0.0],
                [0.0, -0.8, 0.9, 0.0, 0.0],
                [-1.2, -0.2, 0.6, 1.1, 0.0],
            ]
        ),
        "distractor_intercepts": np.array(
            [
                [0.2, -0.3, 0.0, 0.6, 0.0],
                [0.0, 0.4, -0.5, 0.0, 0.0],
                [-0.7, 0.3, -0.1, 0.8, 0.0],
            ]
        ),
    }
    if issubclass(model_class, ThreePLNestedLogit):
        parameters["guessing"] = np.array([0.12, 0.18, 0.08])
    if issubclass(model_class, FourPLNestedLogit):
        parameters["upper"] = np.array([0.92, 0.88, 0.95])
    model.set_parameters(**parameters)
    return model


class TestNestedLogitSpecification:
    """Specification, parameter-domain, and public-API tests."""

    @pytest.mark.parametrize("model_class", MODEL_CLASSES)
    def test_initialization_and_public_api(self, model_class):
        """Expose every variant from both supported public namespaces."""
        model = model_class(2, [3, 4], correct_response=[1, 3])

        assert model.n_items == 2
        assert model.n_categories == [3, 4]
        assert model.correct_response == [1, 3]
        assert getattr(mirt, model_class.__name__) is model_class

    @pytest.mark.parametrize(
        ("correct_response", "message"),
        [
            ([0], "must match n_items"),
            ([0, 4], "out of range"),
            ([0, -1], "out of range"),
            ([0, 1.5], "must be an integer"),
            ([0, True], "must be an integer"),
        ],
    )
    def test_rejects_invalid_answer_keys(self, correct_response, message):
        """Validate answer-key length, type, and per-item category range."""
        with pytest.raises(ValueError, match=message):
            TwoPLNestedLogit(2, [3, 4], correct_response=correct_response)

    def test_rejects_multidimensional_configuration(self):
        """Nested-logit models are explicitly unidimensional."""
        with pytest.raises(ValueError, match="unidimensional"):
            TwoPLNestedLogit(2, 4, n_factors=2)

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("discrimination", np.array([1.0, 0.0]), "positive"),
            ("difficulty", np.array([0.0, np.nan]), "finite"),
            ("guessing", np.array([0.1, 1.0]), r"\[0, 1\)"),
            ("upper", np.array([0.9, 1.1]), r"\(0, 1\]"),
        ],
    )
    def test_rejects_invalid_parameter_domains(self, name, value, message):
        """Reject invalid curve parameters before mutating stored state."""
        model = FourPLNestedLogit(2, 4)
        before = model.parameters

        with pytest.raises(ValueError, match=message):
            model.set_parameters(**{name: value})

        for parameter, expected in before.items():
            assert_array_equal(model.parameters[parameter], expected)

    def test_joint_asymptote_update_is_atomic(self):
        """Validate lower and upper asymptotes against the full proposed state."""
        model = FourPLNestedLogit(2, 4)
        model.set_parameters(
            guessing=np.array([0.05, 0.1]),
            upper=np.array([0.1, 0.2]),
        )

        assert_allclose(model.guessing, [0.05, 0.1])
        assert_allclose(model.upper, [0.1, 0.2])
        with pytest.raises(ValueError, match="strictly less"):
            model.set_parameters(guessing=np.array([0.1, 0.1]))

    def test_set_item_parameter_preserves_validation(self):
        """Per-item updates use the same domain checks as full arrays."""
        model = ThreePLNestedLogit(2, 4)

        model.set_item_parameter(0, "guessing", 0.35)
        assert model.guessing[0] == pytest.approx(0.35)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            model.set_item_parameter(0, "guessing", 1.0)
        with pytest.raises(ValueError, match="positive"):
            model.set_item_parameter(0, "discrimination", 0.0)

    @pytest.mark.parametrize("item_idx", [-1, 3, 1.5, True])
    def test_rejects_invalid_item_indices(self, nested_model, item_idx):
        """Reject negative and ambiguous item indices."""
        with pytest.raises(IndexError, match="item_idx"):
            nested_model.probability(np.array([0.0]), item_idx=item_idx)

    @pytest.mark.parametrize("category", [-1, 4, 1.5, True])
    def test_rejects_invalid_categories(self, nested_model, category):
        """Reject invalid category selections before array access."""
        with pytest.raises(ValueError, match="category|Category"):
            nested_model.category_probability(np.array([0.0]), 0, category)


class TestNestedLogitCurves:
    """Probability, derivative, and information tests."""

    def test_variable_category_probability_contract(self, nested_model):
        """Return normalized curves and zero-pad unavailable categories."""
        theta = np.array([-1.0, 0.0, 1.0])
        probability = nested_model.probability(theta)

        assert probability.shape == (3, 3, 5)
        assert_allclose(probability.sum(axis=2), 1.0)
        assert_allclose(probability[:, 0, 4], 0.0)
        assert_allclose(probability[:, 1, 3:], 0.0)
        assert np.all(probability >= 0.0)

    def test_keyed_response_matches_logistic_asymptotes(self, nested_model):
        """Use the configured lower and upper asymptotes for the answer key."""
        theta = np.array([-1.0, 0.0, 1.0])
        logistic = expit(1.3 * (theta - 0.2))
        lower = nested_model.guessing[0] if hasattr(nested_model, "guessing") else 0.0
        upper = nested_model.upper[0] if hasattr(nested_model, "upper") else 1.0
        expected = lower + (upper - lower) * logistic

        assert_allclose(nested_model.category_probability(theta, 0, 2), expected)
        assert_allclose(nested_model.probability(theta, 0)[:, 2], expected)

    def test_distractor_distribution_matches_conditional_softmax(self, nested_model):
        """Allocate incorrect-response mass through the nominal submodel."""
        theta = np.array([-0.7, 0.4])
        probability = nested_model.probability(theta, 0)
        distractors = np.array([0, 1, 3])
        slopes = nested_model.distractor_slopes[0, distractors]
        intercepts = nested_model.distractor_intercepts[0, distractors]
        conditional = softmax(theta[:, None] * slopes + intercepts, axis=1)
        incorrect_mass = 1.0 - probability[:, 2]

        assert_allclose(
            probability[:, distractors], incorrect_mass[:, None] * conditional
        )

    def test_category_probability_matches_full_curve(self, nested_model):
        """Single-category evaluation agrees with the vectorized item curve."""
        theta = np.linspace(-2.0, 2.0, 7)
        full = nested_model.probability(theta, 2)

        for category in range(5):
            assert_allclose(
                nested_model.category_probability(theta, 2, category),
                full[:, category],
            )
        assert_allclose(nested_model.category_response_curves(theta, 2), full)

    def test_analytic_category_derivatives_match_finite_difference(self, nested_model):
        """Differentiate both keyed and distractor category probabilities."""
        theta = np.array([-1.2, -0.1, 0.8])
        step = 1e-6
        numerical = (
            nested_model.probability(theta + step, 0)
            - nested_model.probability(theta - step, 0)
        ) / (2.0 * step)

        analytic = nested_model.category_derivative(theta, 0)

        assert_allclose(analytic, numerical, rtol=2e-9, atol=2e-10)
        assert_allclose(analytic.sum(axis=1), 0.0, atol=1e-14)
        assert_allclose(nested_model.category_derivative(theta, 0, 3), analytic[:, 3])

    def test_information_matches_probability_derivatives(self, nested_model):
        """Include information from correctness, distractors, and asymptotes."""
        theta = np.array([-0.7, 0.2, 1.1])
        step = 1e-6
        probability = nested_model.probability(theta, 0)
        derivative = (
            nested_model.probability(theta + step, 0)
            - nested_model.probability(theta - step, 0)
        ) / (2.0 * step)
        expected = np.sum(derivative**2 / probability, axis=1)

        assert_allclose(
            nested_model.information(theta, 0), expected, rtol=3e-9, atol=2e-10
        )
        expected_total = sum(
            nested_model.information(theta, item_idx)
            for item_idx in range(nested_model.n_items)
        )
        assert_allclose(nested_model.information(theta), expected_total)

    def test_distractor_slopes_contribute_information(self):
        """Distinguishable distractors add evidence beyond correct/incorrect status."""
        model = TwoPLNestedLogit(1, 4, correct_response=2)
        theta = np.array([0.0])
        baseline = model.information(theta, 0)[0]
        slopes = model.distractor_slopes.copy()
        slopes[0] = [-2.0, 0.7, 0.0, 1.4]
        model.set_parameters(distractor_slopes=slopes)

        assert model.information(theta, 0)[0] > baseline

    def test_extreme_logits_remain_normalized_and_finite(self):
        """Stable softmax evaluation handles widely separated distractors."""
        model = FourPLNestedLogit(1, 5, correct_response=3)
        model.set_parameters(
            discrimination=np.array([8.0]),
            difficulty=np.array([0.0]),
            guessing=np.array([0.05]),
            upper=np.array([0.97]),
            distractor_slopes=np.array([[-1000.0, -20.0, 700.0, 0.0, 1000.0]]),
            distractor_intercepts=np.array([[900.0, -800.0, 300.0, 0.0, -700.0]]),
        )
        theta = np.array([-100.0, 0.0, 100.0])

        probability = model.probability(theta, 0)
        derivative = model.category_derivative(theta, 0)
        information = model.information(theta, 0)

        assert np.all(np.isfinite(probability))
        assert np.all(np.isfinite(derivative))
        assert np.all(np.isfinite(information))
        assert_allclose(probability.sum(axis=1), 1.0)

    def test_expected_score_matches_probability_weighting(self, nested_model):
        """Vectorized expected scores preserve numeric category semantics."""
        theta = np.array([-0.5, 0.5])
        probability = nested_model.probability(theta)
        expected_items = [
            probability[:, item_idx, :n_categories] @ np.arange(n_categories)
            for item_idx, n_categories in enumerate(nested_model.n_categories)
        ]

        for item_idx, expected in enumerate(expected_items):
            assert_allclose(nested_model.expected_score(theta, item_idx), expected)
        assert_allclose(nested_model.expected_score(theta), sum(expected_items))


class TestNestedLogitLikelihood:
    """Likelihood, response-validation, and copy tests."""

    def test_paired_log_likelihood_matches_manual_selection(self, nested_model):
        """Select each observed category from the vectorized probability tensor."""
        responses = np.array([[2, 0, 4], [1, 2, 0], [-1, 1, 3], [3, -1, -1]])
        theta = np.array([-1.0, -0.2, 0.5, 1.3])
        probability = nested_model.probability(theta)
        expected = np.zeros(responses.shape[0])
        for person_idx in range(responses.shape[0]):
            for item_idx in range(responses.shape[1]):
                category = responses[person_idx, item_idx]
                if category >= 0:
                    expected[person_idx] += np.log(
                        probability[person_idx, item_idx, category]
                    )

        assert_allclose(nested_model.log_likelihood(responses, theta), expected)

    def test_batch_log_likelihood_matches_paired_evaluation(self, nested_model):
        """Evaluate every response pattern and ability combination consistently."""
        responses = np.array([[2, 0, 4], [1, 2, 0], [-1, 1, 3], [3, -1, -1]])
        theta = np.linspace(-2.0, 2.0, 9)
        expected = np.column_stack(
            [
                nested_model.log_likelihood(
                    responses, np.full(responses.shape[0], value)
                )
                for value in theta
            ]
        )

        actual = nested_model.log_likelihood_batch(responses, theta)

        assert actual.shape == (4, 9)
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize(
        "responses",
        [
            np.array([0, 1, 2]),
            np.array([[0, 1]]),
            np.array([[0, 3, 0]]),
            np.array([[0.0, 1.5, 2.0]]),
            np.array([[0.0, np.nan, 2.0]]),
            np.array([[0, "invalid", 2]], dtype=object),
            np.array([[0, -2, 2]]),
        ],
    )
    def test_rejects_invalid_response_matrices(self, nested_model, responses):
        """Require a rectangular matrix with valid category and missing codes."""
        with pytest.raises(ValueError):
            nested_model.log_likelihood_batch(responses, np.array([0.0]))

    def test_rejects_paired_row_mismatch(self, nested_model):
        """Paired likelihoods require one ability row per response pattern."""
        responses = np.array([[0, 1, 2], [1, 0, 3]])
        with pytest.raises(ValueError, match="equal row counts"):
            nested_model.log_likelihood(responses, np.array([0.0]))

    def test_copy_preserves_keyed_state_and_is_independent(self, nested_model):
        """Copy answer keys, names, parameters, and fit state without sharing arrays."""
        nested_model._is_fitted = True
        copied = nested_model.copy()
        copied.distractor_slopes[0, 0] = 99.0

        assert copied.correct_response == nested_model.correct_response
        assert copied.item_names == nested_model.item_names
        assert copied.is_fitted
        assert nested_model.distractor_slopes[0, 0] != 99.0
        theta = np.array([-0.5, 0.5])
        copied.distractor_slopes[0, 0] = nested_model.distractor_slopes[0, 0]
        assert_allclose(copied.probability(theta), nested_model.probability(theta))

    def test_direct_invalid_mutation_is_detected_before_evaluation(self):
        """Guard computations when mutable parameter views were corrupted."""
        model = FourPLNestedLogit(1, 4)
        model.upper[0] = 0.1

        with pytest.raises(ValueError, match="Stored parameters"):
            model.probability(np.array([0.0]))
