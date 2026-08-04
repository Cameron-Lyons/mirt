"""Tests for unfolding and ideal-point item response models."""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from mirt import GeneralizedGradedUnfolding, HyperbolicCosineModel, IdealPointModel
from mirt.exceptions import MirtDataError, MirtValidationError


def _ggum_reference(
    theta: float,
    discrimination: float,
    location: float,
    thresholds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the published GGUM equation directly for moderate inputs."""
    m = len(thresholds)
    c = (m - 1) // 2
    subjective_categories = np.arange(m + 1, dtype=np.float64)
    cumulative_thresholds = np.cumsum(
        np.concatenate((np.zeros(1, dtype=np.float64), thresholds))
    )
    f = np.exp(
        discrimination
        * (subjective_categories * (theta - location) - cumulative_thresholds)
    )
    categories = np.arange(c + 1)
    weights = f[categories] + f[m - categories]
    return weights / np.sum(weights)


class TestGeneralizedGradedUnfolding:
    """GGUM equation, validation, and numerical-stability tests."""

    def test_initialization_uses_full_subjective_threshold_scale(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=4)

        assert model.model_name == "GGUM"
        assert model.thresholds.shape == (2, 7)
        np.testing.assert_allclose(model.thresholds[:, 3], 0.0)
        np.testing.assert_allclose(
            model.thresholds[:, 4:],
            -model.thresholds[
                :,
                :3,
            ][:, ::-1],
        )

    def test_mixed_category_items_use_zero_padding(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 5])

        assert model.thresholds.shape == (2, 9)
        assert model.thresholds_for_item(0).shape == (5,)
        assert model.thresholds_for_item(1).shape == (9,)
        np.testing.assert_array_equal(model.thresholds[0, 5:], 0.0)

        probabilities = model.probability(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(probabilities[:, 0, :3].sum(axis=1), 1.0)
        np.testing.assert_array_equal(probabilities[:, 0, 3:], 0.0)

    def test_probability_matches_published_equation(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=4)
        thresholds = np.array([-1.7, -0.8, -0.2, 0.0, 0.2, 0.8, 1.7])
        model.set_parameters(
            discrimination=np.array([1.3]),
            location=np.array([0.4]),
            thresholds=thresholds[None, :],
        )
        theta = np.array([-1.2, 0.4, 1.8])

        expected = np.vstack(
            [_ggum_reference(value, 1.3, 0.4, thresholds) for value in theta]
        )
        np.testing.assert_allclose(model.probability(theta, 0), expected)

    def test_probabilities_are_symmetric_and_normalized(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=5)
        model.set_parameters(location=np.array([0.7]))
        offsets = np.array([0.0, 0.4, 1.5, 3.0])

        left = model.probability(0.7 - offsets, 0)
        right = model.probability(0.7 + offsets, 0)

        np.testing.assert_allclose(left, right)
        np.testing.assert_allclose(left.sum(axis=1), 1.0)

    def test_extreme_theta_is_finite_without_runtime_warnings(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 5])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probabilities = model.probability(np.array([-1e6, 0.0, 1e6]))
            information = model.information(np.array([-1e6, 0.0, 1e6]))

        assert np.all(np.isfinite(probabilities))
        assert np.all(np.isfinite(information))
        np.testing.assert_allclose(probabilities.sum(axis=2), 1.0)

    def test_category_probability_and_curves_match_probability(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        theta = np.linspace(-2.0, 2.0, 9)

        curves = model.category_response_curves(theta, 1)
        np.testing.assert_allclose(curves, model.probability(theta, 1))
        np.testing.assert_allclose(
            model.category_probability(theta, 1, 2), curves[:, 2]
        )

    @pytest.mark.parametrize("category", [-1, 4, 1.5, True])
    def test_category_probability_rejects_invalid_category(
        self, category: object
    ) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=4)

        with pytest.raises(MirtValidationError, match="category"):
            model.category_probability(np.array([0.0]), 0, category)

    def test_independent_threshold_update_constructs_symmetric_scale(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        independent = np.array([-1.9, -0.7, -0.1])

        model.set_independent_thresholds(1, independent)

        np.testing.assert_array_equal(model.independent_thresholds(1), independent)
        np.testing.assert_array_equal(
            model.thresholds_for_item(1),
            np.array([-1.9, -0.7, -0.1, 0.0, 0.1, 0.7, 1.9]),
        )
        np.testing.assert_array_equal(
            model.thresholds_for_item(1, include_tau_zero=True)[:2],
            np.array([0.0, -1.9]),
        )

    def test_information_matches_numerical_probability_derivative(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=4)
        model.set_parameters(discrimination=np.array([1.4]), location=np.array([-0.2]))
        theta = np.array([-1.1, 0.3, 1.7])
        step = 1e-6

        probabilities = model.probability(theta, 0)
        derivative = (
            model.probability(theta + step, 0) - model.probability(theta - step, 0)
        ) / (2.0 * step)
        expected = np.sum(derivative**2 / probabilities, axis=1)

        np.testing.assert_allclose(model.information(theta, 0), expected, rtol=1e-7)

    def test_information_and_expected_score_sum_over_items(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        theta = np.array([-0.8, 0.0, 1.1])

        expected_information = model.information(theta, 0) + model.information(theta, 1)
        expected_score = model.expected_score(theta, 0) + model.expected_score(theta, 1)

        np.testing.assert_allclose(model.information(theta), expected_information)
        np.testing.assert_allclose(model.expected_score(theta), expected_score)

    def test_log_likelihood_matches_manual_selection(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        responses = np.array([[0, 3], [2, 1], [1, 0]])
        theta = np.array([-0.5, 0.2, 1.3])
        probabilities = model.probability(theta)

        expected = np.array(
            [
                np.log(probabilities[person, 0, responses[person, 0]])
                + np.log(probabilities[person, 1, responses[person, 1]])
                for person in range(len(theta))
            ]
        )
        np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)

    def test_log_likelihood_skips_nan_and_negative_responses(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=3)
        responses = np.array([[np.nan, 1.0], [-1.0, 2.0]])
        theta = np.array([-0.4, 0.6])
        probabilities = model.probability(theta)

        expected = np.log(np.array([probabilities[0, 1, 1], probabilities[1, 1, 2]]))
        np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)

    @pytest.mark.parametrize(
        "responses",
        [
            np.array([0, 1]),
            np.array([[0, 1, 2]]),
            np.array([[0.5, 1.0]]),
            np.array([[0.0, np.inf]]),
            np.array([[0.0, -np.inf]]),
            np.array([[3, 0]]),
        ],
    )
    def test_likelihood_rejects_invalid_responses(
        self, responses: NDArray[np.float64]
    ) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=3)

        with pytest.raises(MirtDataError):
            model.log_likelihood(responses, np.zeros(len(responses)))

    def test_log_likelihood_rejects_person_count_mismatch(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=3)

        with pytest.raises(MirtDataError, match="same number"):
            model.log_likelihood(np.zeros((2, 2), dtype=int), np.zeros(3))

    def test_batch_likelihood_matches_personwise_likelihood(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        responses = np.array([[0, 3], [2, 1], [-1, 0]])
        grid = np.array([-1.5, 0.0, 1.2])

        actual = model.log_likelihood_batch(responses, grid)
        expected = np.column_stack(
            [
                model.log_likelihood(responses, np.full(len(responses), point))
                for point in grid
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_parameter_update_is_validated_and_atomic(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=3)
        before = model.parameters
        invalid_thresholds = model.thresholds.copy()
        invalid_thresholds[0, 2] = 0.2

        with pytest.raises(MirtValidationError, match="center threshold"):
            model.set_parameters(
                discrimination=np.array([2.0]), thresholds=invalid_thresholds
            )

        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda values: values.__setitem__((0, 4), 9.0), "symmetric"),
            (lambda values: values.__setitem__((0, 5), 1.0), "padded"),
        ],
    )
    def test_threshold_validation_rejects_broken_constraints(
        self,
        mutate: Callable[[NDArray[np.float64]], None],
        message: str,
    ) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        values = model.thresholds.copy()
        mutate(values)

        with pytest.raises(MirtValidationError, match=message):
            model.set_parameters(thresholds=values)

    def test_item_updates_validate_scalar_and_positive_discrimination(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=3)

        model.set_item_parameter(1, "location", 0.8)
        assert model.location[1] == pytest.approx(0.8)
        with pytest.raises(MirtValidationError, match="positive"):
            model.set_item_parameter(0, "discrimination", 0.0)
        with pytest.raises(MirtValidationError, match="scalar"):
            model.set_item_parameter(0, "location", np.array([1.0]))

    def test_copy_is_independent(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=2, n_categories=[3, 4])
        copied = model.copy()

        copied.set_item_parameter(0, "location", 2.0)

        assert model.location[0] == 0.0
        assert copied.location[0] == 2.0
        assert copied.n_categories == [3, 4]

    @pytest.mark.parametrize("n_categories", [True, 2.5, [3, 2.5], [3, True]])
    def test_category_configuration_requires_integers(
        self, n_categories: object
    ) -> None:
        with pytest.raises(MirtValidationError):
            GeneralizedGradedUnfolding(2, n_categories=n_categories)

    def test_rejects_multidimensional_theta_and_invalid_items(self) -> None:
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=3)

        with pytest.raises(MirtValidationError):
            model.probability(np.zeros((2, 2)))
        with pytest.raises(IndexError):
            model.probability(np.array([0.0]), item_idx=1)


class TestIdealPointModel:
    """Gaussian ideal-point probability and information tests."""

    def test_probability_matches_equation_and_peaks_at_location(self) -> None:
        model = IdealPointModel(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.5, 0.7]),
            location=np.array([-0.3, 0.8]),
            peak_height=np.array([0.9, 0.6]),
        )
        theta = np.array([-1.0, 0.2, 1.4])

        expected = model.peak_height[None, :] * np.exp(
            -model.discrimination[None, :]
            * (theta[:, None] - model.location[None, :]) ** 2
        )
        np.testing.assert_allclose(model.probability(theta), expected)
        np.testing.assert_array_equal(model.peak_location, model.location)
        np.testing.assert_allclose(model.probability(np.array([-0.3]), 0), 0.9)

    def test_information_matches_numerical_derivative_and_peak_limit(self) -> None:
        model = IdealPointModel(n_items=1)
        model.set_parameters(
            discrimination=np.array([1.7]),
            location=np.array([0.2]),
            peak_height=np.array([1.0]),
        )
        theta = np.array([-0.8, 0.2, 1.1])
        step = 1e-6
        probability = model.probability(theta, 0)
        derivative = (
            model.probability(theta + step, 0) - model.probability(theta - step, 0)
        ) / (2.0 * step)
        expected = np.divide(
            derivative**2,
            probability * (1.0 - probability),
            out=np.zeros_like(probability),
            where=probability < 1.0,
        )
        expected[1] = 4.0 * 1.7

        np.testing.assert_allclose(model.information(theta, 0), expected, rtol=1e-6)

    def test_extreme_theta_probability_and_information_are_finite(self) -> None:
        model = IdealPointModel(n_items=2)
        model.set_parameters(peak_height=np.array([1.0, 0.8]))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probability = model.probability(np.array([-1e200, 1e200]))
            information = model.information(np.array([-1e200, 1e200]))

        assert np.all(np.isfinite(probability))
        assert np.all(np.isfinite(information))
        np.testing.assert_array_equal(probability, 0.0)
        np.testing.assert_array_equal(information, 0.0)

    @pytest.mark.parametrize("value", [-1.0, 0.0, 1.1, np.inf])
    def test_peak_height_validation(self, value: float) -> None:
        model = IdealPointModel(n_items=1)

        with pytest.raises(MirtValidationError):
            model.set_parameters(peak_height=np.array([value]))

    def test_item_updates_are_atomic_and_copy_is_independent(self) -> None:
        model = IdealPointModel(n_items=2)
        copied = model.copy()

        copied.set_item_parameter(1, "peak_height", 0.7)
        assert model.peak_height[1] == 1.0
        assert copied.peak_height[1] == 0.7
        with pytest.raises(MirtValidationError, match="positive"):
            copied.set_item_parameter(0, "discrimination", -1.0)
        assert copied.discrimination[0] == 1.0


class TestHyperbolicCosineModel:
    """Hyperbolic-cosine probability and information tests."""

    def test_probability_matches_equation_and_shifted_peak(self) -> None:
        model = HyperbolicCosineModel(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.4, 0.8]),
            location=np.array([-0.2, 0.5]),
            asymmetry=np.array([0.7, -0.4]),
        )
        theta = np.array([-1.0, 0.0, 1.0])
        predictor = (
            model.discrimination[None, :] * (theta[:, None] - model.location[None, :])
            - model.asymmetry[None, :]
        )

        np.testing.assert_allclose(
            model.probability(theta), 1.0 / (1.0 + np.cosh(predictor))
        )
        np.testing.assert_allclose(
            model.peak_location,
            model.location + model.asymmetry / model.discrimination,
        )
        np.testing.assert_allclose(model.probability(model.peak_location[0:1], 0), 0.5)

    def test_information_matches_numerical_derivative(self) -> None:
        model = HyperbolicCosineModel(n_items=1)
        model.set_parameters(
            discrimination=np.array([1.6]),
            location=np.array([0.2]),
            asymmetry=np.array([-0.3]),
        )
        theta = np.array([-1.0, 0.4, 1.5])
        step = 1e-6
        probability = model.probability(theta, 0)
        derivative = (
            model.probability(theta + step, 0) - model.probability(theta - step, 0)
        ) / (2.0 * step)
        expected = derivative**2 / (probability * (1.0 - probability))

        np.testing.assert_allclose(model.information(theta, 0), expected, rtol=1e-7)

    def test_extreme_theta_probability_and_information_are_finite(self) -> None:
        model = HyperbolicCosineModel(n_items=2)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probability = model.probability(np.array([-1e6, 1e6]))
            information = model.information(np.array([-1e6, 1e6]))

        assert np.all(np.isfinite(probability))
        assert np.all(np.isfinite(information))
        np.testing.assert_array_equal(probability, 0.0)
        np.testing.assert_array_equal(information, 0.0)

    def test_parameter_validation_and_copy(self) -> None:
        model = HyperbolicCosineModel(n_items=2)
        copied = model.copy()

        copied.set_item_parameter(0, "asymmetry", 0.8)
        assert model.asymmetry[0] == 0.0
        assert copied.asymmetry[0] == 0.8
        with pytest.raises(MirtValidationError, match="positive"):
            copied.set_parameters(discrimination=np.array([1.0, 0.0]))


@pytest.mark.parametrize("model_type", [IdealPointModel, HyperbolicCosineModel])
class TestDichotomousUnfoldingBehavior:
    """Shared binary likelihood and response-validation behavior."""

    def test_log_likelihood_matches_manual_selection(
        self,
        model_type: type[IdealPointModel] | type[HyperbolicCosineModel],
    ) -> None:
        model = model_type(n_items=2)
        if isinstance(model, IdealPointModel):
            model.set_parameters(peak_height=np.array([0.8, 0.7]))
        responses = np.array([[0, 1], [1, 0], [-1, 1]])
        theta = np.array([-0.6, 0.3, 1.0])
        probabilities = model.probability(theta)
        expected = np.array(
            [
                np.log(1.0 - probabilities[0, 0]) + np.log(probabilities[0, 1]),
                np.log(probabilities[1, 0]) + np.log(1.0 - probabilities[1, 1]),
                np.log(probabilities[2, 1]),
            ]
        )

        np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)

    def test_batch_likelihood_matches_personwise_likelihood(
        self,
        model_type: type[IdealPointModel] | type[HyperbolicCosineModel],
    ) -> None:
        model = model_type(n_items=2)
        if isinstance(model, IdealPointModel):
            model.set_parameters(peak_height=np.array([0.8, 0.7]))
        responses = np.array([[0, 1], [1, 0], [-1, 1]])
        grid = np.array([-1.2, 0.0, 1.3])

        expected = np.column_stack(
            [
                model.log_likelihood(responses, np.full(len(responses), point))
                for point in grid
            ]
        )
        np.testing.assert_allclose(
            model.log_likelihood_batch(responses, grid), expected
        )

    @pytest.mark.parametrize(
        "responses",
        [
            np.array([0, 1]),
            np.array([[0, 1, 0]]),
            np.array([[0.5, 1.0]]),
            np.array([[0.0, np.inf]]),
            np.array([[0.0, -np.inf]]),
            np.array([[2, 0]]),
        ],
    )
    def test_likelihood_rejects_invalid_responses(
        self,
        model_type: type[IdealPointModel] | type[HyperbolicCosineModel],
        responses: NDArray[np.float64],
    ) -> None:
        model = model_type(n_items=2)

        with pytest.raises(MirtDataError):
            model.log_likelihood(responses, np.zeros(len(responses)))

    def test_probability_rejects_nonfinite_theta_and_invalid_items(
        self,
        model_type: type[IdealPointModel] | type[HyperbolicCosineModel],
    ) -> None:
        model = model_type(n_items=2)

        with pytest.raises(MirtValidationError, match="finite"):
            model.probability(np.array([np.nan]))
        with pytest.raises(IndexError):
            model.probability(np.array([0.0]), item_idx=2)
