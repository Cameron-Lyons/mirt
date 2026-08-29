"""Tests for nonparametric item response models."""

import numpy as np
import pytest

from mirt.constants import PROB_EPSILON
from mirt.models.nonparametric import (
    KernelSmoothingModel,
    MonotonicPolynomialModel,
    MonotonicSplineModel,
)


def _assert_nondecreasing(values, tolerance=1e-12):
    assert np.all(np.diff(values, axis=0) >= -tolerance)


def _finite_difference_information(model, theta, h=1e-6):
    probability = model.probability(theta)
    derivative = (model.probability(theta + h) - model.probability(theta - h)) / (
        2.0 * h
    )
    return derivative**2 / (probability * (1.0 - probability) + PROB_EPSILON)


class TestMonotonicSplineModel:
    def test_probability_shape_bounds_and_item_consistency(self):
        model = MonotonicSplineModel(n_items=3, n_knots=2, degree=2)
        theta = np.linspace(-3, 3, 21)

        probabilities = model.probability(theta)

        assert probabilities.shape == (21, 3)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        for item_idx in range(model.n_items):
            np.testing.assert_allclose(
                probabilities[:, item_idx], model.probability(theta, item_idx)
            )

    def test_curves_remain_monotone_with_unequal_weights(self):
        model = MonotonicSplineModel(n_items=2, n_knots=3, degree=3)
        log_weights = np.array(
            [
                [8.0, -4.0, 2.0, -1.0, 5.0, -8.0, 0.0],
                [-7.0, 3.0, 1.0, 9.0, -2.0, 4.0, -5.0],
            ]
        )
        model.set_parameters(log_weights=log_weights)

        probabilities = model.probability(np.linspace(-6, 6, 401))

        _assert_nondecreasing(probabilities)

    def test_saturated_endpoints_match_curve_bounds(self):
        model = MonotonicSplineModel(n_items=2, n_knots=3, degree=2)
        model.set_parameters(lower=np.array([0.1, 0.2]), upper=np.array([0.8, 0.9]))

        probabilities = model.probability(np.array([-100.0, 100.0]))

        np.testing.assert_allclose(probabilities[0], [0.1, 0.2])
        np.testing.assert_allclose(probabilities[1], [0.8, 0.9])

    def test_interior_knots_are_evenly_spaced(self):
        one_knot = MonotonicSplineModel(n_items=1, n_knots=1)
        four_knots = MonotonicSplineModel(n_items=1, n_knots=4)

        np.testing.assert_allclose(one_knot.knots, [0.0])
        np.testing.assert_allclose(four_knots.knots, [-1.8, -0.6, 0.6, 1.8])

    def test_each_basis_is_normalized_and_monotone(self):
        model = MonotonicSplineModel(n_items=1, n_knots=3, degree=3)
        theta = np.linspace(-5, 5, 201)

        for basis_idx in range(model._n_basis):
            basis = model._ispline_basis(theta, basis_idx)
            assert basis[0] == pytest.approx(0.0)
            assert basis[-1] == pytest.approx(1.0)
            _assert_nondecreasing(basis)

    def test_extreme_log_weights_are_finite(self):
        model = MonotonicSplineModel(n_items=2, n_knots=2, degree=2)
        model.set_parameters(
            log_weights=np.array([[1000.0, -1000.0, 0.0, 500.0, -500.0]] * 2)
        )

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            weights = model.weights
            probabilities = model.probability(np.array([-100.0, 0.0, 100.0]))

        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        assert np.all(np.isfinite(probabilities))

    def test_information_is_finite_nonnegative_and_item_consistent(self):
        model = MonotonicSplineModel(n_items=2, n_knots=3, degree=3)
        theta = np.linspace(-3, 3, 17)

        information = model.information(theta)

        assert information.shape == (17, 2)
        assert np.all(np.isfinite(information))
        assert np.all(information >= 0)
        np.testing.assert_allclose(information[:, 1], model.information(theta, 1))

    def test_information_matches_exact_spline_derivative(self, monkeypatch):
        model = MonotonicSplineModel(n_items=3, n_knots=3, degree=3)
        model.set_parameters(
            log_weights=np.array(
                [
                    [2.0, -1.0, 0.0, 1.0, -2.0, 3.0, -1.0],
                    [-2.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0],
                    [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
                ]
            ),
            lower=np.array([0.05, 0.1, 0.2]),
            upper=np.array([0.9, 0.95, 0.8]),
        )
        theta = np.array([-3.6, -2.2, -0.3, 0.8, 2.2, 3.6])

        expected = _finite_difference_information(model, theta)
        monkeypatch.setattr(
            model,
            "probability",
            lambda *args, **kwargs: pytest.fail("information recomputed probability"),
        )

        np.testing.assert_allclose(
            model.information(theta),
            expected,
            rtol=2e-7,
            atol=5e-10,
        )
        np.testing.assert_array_equal(
            model.information(np.array([-5.0, 5.0])),
            np.zeros((2, model.n_items)),
        )

    def test_log_likelihood_supports_missing_responses(self):
        model = MonotonicSplineModel(n_items=2, n_knots=2, degree=2)
        theta = np.array([[-0.5], [0.5]])
        responses = np.array([[0, 1], [1, -1]])

        result = model.log_likelihood(responses, theta)

        probabilities = model.probability(theta)
        expected = np.array(
            [
                np.log1p(-probabilities[0, 0]) + np.log(probabilities[0, 1]),
                np.log(probabilities[1, 0]),
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_copy_preserves_configuration_and_is_independent(self):
        model = MonotonicSplineModel(
            n_items=2,
            n_knots=2,
            degree=2,
            item_names=["A", "B"],
        ).set_parameters(lower=np.array([0.1, 0.2]))

        copied = model.copy()
        copied.set_parameters(lower=np.array([0.3, 0.4]))

        assert copied.n_knots == 2
        assert copied.degree == 2
        assert copied.item_names == ["A", "B"]
        np.testing.assert_allclose(model.lower, [0.1, 0.2])
        np.testing.assert_allclose(copied.lower, [0.3, 0.4])
        assert copied.probability(np.array([0.0])).shape == (1, 2)

    def test_properties_return_copies(self):
        model = MonotonicSplineModel(n_items=1, n_knots=2, degree=2)
        lower = model.lower
        upper = model.upper
        knots = model.knots
        lower[:] = 0.5
        upper[:] = 0.5
        knots[:] = 99

        np.testing.assert_allclose(model.lower, [0.0])
        np.testing.assert_allclose(model.upper, [1.0])
        assert np.all(model.knots != 99)

    def test_single_item_parameter_updates_use_full_validation(self):
        model = MonotonicSplineModel(n_items=2, n_knots=2, degree=2)
        model.set_item_parameter(1, "lower", 0.2)
        model.set_item_parameter(0, "log_weights", np.arange(5.0))

        np.testing.assert_allclose(model.lower, [0.0, 0.2])
        np.testing.assert_allclose(model.parameters["log_weights"][0], np.arange(5.0))
        with pytest.raises(ValueError):
            model.set_item_parameter(1, "upper", 0.1)
        np.testing.assert_allclose(model.upper, [1.0, 1.0])

    @pytest.mark.parametrize(
        "updates",
        [
            {"unknown": np.zeros(1)},
            {"lower": np.zeros(1)},
            {"log_weights": np.full((2, 5), np.nan)},
            {"lower": np.array([-0.1, 0.0])},
            {"lower": np.array([0.8, 0.0]), "upper": np.array([0.8, 1.0])},
            {"upper": np.array([1.1, 1.0])},
        ],
    )
    def test_parameter_updates_are_validated_atomically(self, updates):
        model = MonotonicSplineModel(n_items=2, n_knots=2, degree=2)
        before = model.parameters

        with pytest.raises(ValueError):
            model.set_parameters(**updates)

        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_knots": 0}, "n_knots"),
            ({"n_knots": True}, "n_knots"),
            ({"degree": 0}, "degree"),
            ({"degree": 1.5}, "degree"),
            ({"n_factors": 2}, "unidimensional"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            MonotonicSplineModel(n_items=2, **kwargs)

    @pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
    def test_rejects_invalid_item_index(self, item_idx):
        model = MonotonicSplineModel(n_items=2)
        with pytest.raises((IndexError, TypeError)):
            model.probability(np.array([0.0]), item_idx)

    @pytest.mark.parametrize("theta", [np.array([np.nan]), np.array([np.inf])])
    def test_rejects_nonfinite_theta(self, theta):
        with pytest.raises(ValueError, match="finite"):
            MonotonicSplineModel(n_items=1).probability(theta)


class TestMonotonicPolynomialModel:
    def test_probability_shape_bounds_and_item_consistency(self):
        model = MonotonicPolynomialModel(n_items=3, degree=4)
        theta = np.linspace(-3, 3, 21)

        probabilities = model.probability(theta)

        assert probabilities.shape == (21, 3)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        for item_idx in range(model.n_items):
            np.testing.assert_allclose(
                probabilities[:, item_idx], model.probability(theta, item_idx)
            )

    def test_adversarial_parameters_still_produce_monotone_curves(self):
        model = MonotonicPolynomialModel(n_items=2, degree=3)
        model.set_parameters(
            log_coefficients=np.array([[8.0, -8.0, 3.0, -2.0], [-5.0, 7.0, -1.0, 4.0]]),
            location=np.array([-1.0, 1.0]),
            scale=np.array([0.25, 4.0]),
        )

        probabilities = model.probability(np.linspace(-20, 20, 1001))

        _assert_nondecreasing(probabilities)

    def test_high_degree_curves_use_stable_evaluation(self):
        rng = np.random.default_rng(7)
        model = MonotonicPolynomialModel(n_items=2, degree=40)
        model.set_parameters(
            log_coefficients=rng.normal(size=(2, 41)),
            location=np.array([-1.0, 1.0]),
            scale=np.array([0.5, 2.0]),
        )

        probabilities = model.probability(np.linspace(-15, 15, 1001))

        assert np.all(np.isfinite(probabilities))
        _assert_nondecreasing(probabilities)

    def test_coefficients_are_ordered_normalized_and_finite(self):
        model = MonotonicPolynomialModel(n_items=2, degree=3)
        model.set_parameters(
            log_coefficients=np.array(
                [[1000.0, -1000.0, 0.0, 500.0], [-800.0, 900.0, 0.0, 1.0]]
            )
        )

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            coefficients = model.coefficients
            probabilities = model.probability(np.array([-1000.0, 1000.0]))

        assert np.all(np.isfinite(coefficients))
        assert np.all(coefficients > 0)
        _assert_nondecreasing(coefficients.T)
        np.testing.assert_allclose(coefficients[:, -1], 1.0)
        assert np.all(np.isfinite(probabilities))

    def test_bernstein_basis_partitions_unity(self):
        model = MonotonicPolynomialModel(n_items=1, degree=5)
        transformed = np.linspace(0, 1, 101)

        basis = model._basis_matrix(transformed)

        np.testing.assert_allclose(np.sum(basis, axis=-1), 1.0, atol=1e-14)
        assert np.all(basis >= 0)

    def test_curve_bounds_are_respected(self):
        model = MonotonicPolynomialModel(n_items=2, degree=4)
        model.set_parameters(lower=np.array([0.1, 0.2]), upper=np.array([0.8, 0.9]))

        probabilities = model.probability(np.linspace(-100, 100, 101))

        assert np.all(probabilities >= np.array([0.1, 0.2]))
        assert np.all(probabilities <= np.array([0.8, 0.9]))
        np.testing.assert_allclose(probabilities[-1], [0.8, 0.9])

    def test_information_is_finite_nonnegative_and_item_consistent(self):
        model = MonotonicPolynomialModel(n_items=2, degree=4)
        theta = np.linspace(-3, 3, 17)

        information = model.information(theta)

        assert information.shape == (17, 2)
        assert np.all(np.isfinite(information))
        assert np.all(information >= 0)
        np.testing.assert_allclose(information[:, 0], model.information(theta, 0))

    @pytest.mark.parametrize("degree", [5, 40])
    def test_information_matches_exact_bernstein_derivative(self, degree, monkeypatch):
        rng = np.random.default_rng(17)
        model = MonotonicPolynomialModel(n_items=3, degree=degree)
        model.set_parameters(
            log_coefficients=rng.normal(size=(3, degree + 1)),
            location=np.array([-0.5, 0.2, 1.0]),
            scale=np.array([0.7, 1.5, 3.0]),
            lower=np.array([0.05, 0.1, 0.2]),
            upper=np.array([0.9, 0.95, 0.8]),
        )
        theta = np.array([-3.6, -2.2, -0.3, 0.8, 2.2, 3.6])

        expected = _finite_difference_information(model, theta)
        monkeypatch.setattr(
            model,
            "probability",
            lambda *args, **kwargs: pytest.fail("information recomputed probability"),
        )

        np.testing.assert_allclose(
            model.information(theta),
            expected,
            rtol=3e-6,
            atol=1e-9,
        )

    def test_log_likelihood_matches_manual_calculation(self):
        model = MonotonicPolynomialModel(n_items=2, degree=3)
        theta = np.array([[-0.5], [0.5]])
        responses = np.array([[0, 1], [1, 0]])
        probabilities = model.probability(theta)

        result = model.log_likelihood(responses, theta)

        expected = np.sum(
            responses * np.log(probabilities)
            + (1 - responses) * np.log1p(-probabilities),
            axis=1,
        )
        np.testing.assert_allclose(result, expected)

    def test_copy_preserves_configuration_and_is_independent(self):
        model = MonotonicPolynomialModel(
            n_items=2, degree=3, item_names=["A", "B"]
        ).set_parameters(location=np.array([-1.0, 1.0]))

        copied = model.copy()
        copied.set_parameters(location=np.array([8.0, 9.0]))

        assert copied.degree == 3
        assert copied.item_names == ["A", "B"]
        np.testing.assert_allclose(model.location, [-1.0, 1.0])
        np.testing.assert_allclose(copied.location, [8.0, 9.0])
        assert copied.probability(np.array([0.0])).shape == (1, 2)

    def test_properties_return_copies(self):
        model = MonotonicPolynomialModel(n_items=1, degree=3)
        for values in [model.location, model.scale, model.lower, model.upper]:
            values[:] = 99

        np.testing.assert_allclose(model.location, [0.0])
        np.testing.assert_allclose(model.scale, [1.0])
        np.testing.assert_allclose(model.lower, [0.0])
        np.testing.assert_allclose(model.upper, [1.0])

    def test_single_item_parameter_updates_use_full_validation(self):
        model = MonotonicPolynomialModel(n_items=2, degree=3)
        model.set_item_parameter(1, "location", 2.0)
        model.set_item_parameter(0, "log_coefficients", np.arange(4.0))

        np.testing.assert_allclose(model.location, [0.0, 2.0])
        np.testing.assert_allclose(
            model.parameters["log_coefficients"][0], np.arange(4.0)
        )
        with pytest.raises(ValueError):
            model.set_item_parameter(1, "scale", -1.0)
        np.testing.assert_allclose(model.scale, [1.0, 1.0])

    @pytest.mark.parametrize(
        "updates",
        [
            {"unknown": np.zeros(1)},
            {"location": np.zeros(1)},
            {"log_coefficients": np.full((2, 4), np.inf)},
            {"scale": np.array([1.0, 0.0])},
            {"scale": np.array([1.0, -1.0])},
            {"lower": np.array([-0.1, 0.0])},
            {"lower": np.array([0.8, 0.0]), "upper": np.array([0.8, 1.0])},
            {"upper": np.array([1.0, 1.1])},
        ],
    )
    def test_parameter_updates_are_validated_atomically(self, updates):
        model = MonotonicPolynomialModel(n_items=2, degree=3)
        before = model.parameters

        with pytest.raises(ValueError):
            model.set_parameters(**updates)

        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"degree": 0}, "degree"),
            ({"degree": True}, "degree"),
            ({"degree": 1.5}, "degree"),
            ({"n_factors": 2}, "unidimensional"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            MonotonicPolynomialModel(n_items=2, **kwargs)

    @pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
    def test_rejects_invalid_item_index(self, item_idx):
        model = MonotonicPolynomialModel(n_items=2)
        with pytest.raises((IndexError, TypeError)):
            model.probability(np.array([0.0]), item_idx)

    @pytest.mark.parametrize("theta", [np.array([np.nan]), np.array([np.inf])])
    def test_rejects_nonfinite_theta(self, theta):
        with pytest.raises(ValueError, match="finite"):
            MonotonicPolynomialModel(n_items=1).probability(theta)


class TestKernelSmoothingModel:
    @staticmethod
    def _fitted_model():
        theta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        responses = np.array([[0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=int)
        return KernelSmoothingModel(
            n_items=2,
            bandwidth=0.6,
            theta_grid=np.linspace(-2.0, 2.0, 9),
        ).calibrate(responses, theta)

    def test_probability_requires_calibration(self):
        model = KernelSmoothingModel(n_items=2)

        with pytest.raises(ValueError, match="calibrated"):
            model.probability(np.array([0.0]))
        with pytest.raises(ValueError, match="calibrated"):
            _ = model.irf_values

    def test_calibrate_returns_self_and_records_counts(self):
        model = KernelSmoothingModel(n_items=2)
        responses = np.array([[0, 1], [1, -1], [1, 0]])

        returned = model.calibrate(responses, np.array([-1.0, 0.0, 1.0]))

        assert returned is model
        assert model.is_fitted
        np.testing.assert_array_equal(model.calibration_counts, [3, 2])
        np.testing.assert_array_equal(model.calibration_weight_sums, [3.0, 2.0])

    def test_probability_shape_bounds_and_item_consistency(self):
        model = self._fitted_model()
        theta = np.linspace(-3, 3, 21)

        probabilities = model.probability(theta)

        assert probabilities.shape == (21, 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        for item_idx in range(model.n_items):
            np.testing.assert_allclose(
                probabilities[:, item_idx], model.probability(theta, item_idx)
            )

    def test_probability_matches_linear_grid_interpolation(self):
        model = self._fitted_model()
        theta = np.array([-1.75, -0.25, 0.25, 1.75])

        probabilities = model.probability(theta)

        expected = np.column_stack(
            [
                np.interp(theta, model.theta_grid, item_irf)
                for item_irf in model.irf_values
            ]
        )
        np.testing.assert_allclose(probabilities, expected)

    def test_calibration_matches_gaussian_kernel_reference(self):
        theta = np.array([-1.5, -0.5, 0.5, 1.5])
        responses = np.array([[0, 1], [0, 0], [1, 1], [1, 1]])
        grid = np.array([-1.0, 0.0, 1.0])
        bandwidth = 0.7
        model = KernelSmoothingModel(n_items=2, bandwidth=bandwidth, theta_grid=grid)

        model.calibrate(responses, theta)

        weights = np.exp(-0.5 * ((theta[:, None] - grid[None, :]) / bandwidth) ** 2)
        expected = responses.T @ weights / np.sum(weights, axis=0)
        np.testing.assert_allclose(model.irf_values, expected)

    def test_weighted_missing_calibration_matches_gaussian_kernel_reference(self):
        theta = np.array([-1.5, -0.5, 0.0, 0.75, 1.5])
        responses = np.array(
            [
                [0, 1, -1],
                [0, -1, 0],
                [1, 1, 1],
                [-1, 0, 1],
                [1, 1, 0],
            ]
        )
        sample_weight = np.array([0.5, 2.0, 0.0, 1.5, 3.0])
        grid = np.array([-1.0, 0.0, 1.0])
        bandwidth = 0.7
        model = KernelSmoothingModel(
            n_items=3,
            bandwidth=bandwidth,
            theta_grid=grid,
        )

        model.calibrate(responses, theta, sample_weight=sample_weight)

        kernel = np.exp(-0.5 * ((theta[:, None] - grid[None, :]) / bandwidth) ** 2)
        weighted_kernel = sample_weight[:, None] * kernel
        observed = responses >= 0
        response_values = np.where(observed, responses, 0.0)
        expected = (response_values.T @ weighted_kernel) / (
            observed.astype(float).T @ weighted_kernel
        )
        np.testing.assert_allclose(model.irf_values, expected)
        np.testing.assert_array_equal(model.calibration_counts, [4, 4, 4])
        np.testing.assert_array_equal(
            model.calibration_weight_sums,
            [5.5, 5.0, 6.5],
        )

    def test_weighted_calibration_is_scale_invariant(self):
        theta = np.array([-1.0, -0.25, 0.5, 1.5])
        responses = np.array([[0, 1], [1, -1], [0, 1], [1, 0]])
        sample_weight = np.array([0.25, 2.0, 1.5, 4.0])

        first = KernelSmoothingModel(n_items=2).calibrate(
            responses,
            theta,
            sample_weight=sample_weight,
        )
        second = KernelSmoothingModel(n_items=2).calibrate(
            responses,
            theta,
            sample_weight=sample_weight * 1e200,
        )

        np.testing.assert_allclose(first.irf_values, second.irf_values)
        np.testing.assert_allclose(
            second.calibration_weight_sums,
            first.calibration_weight_sums * 1e200,
        )

    def test_integer_weights_match_duplicated_persons(self):
        theta = np.array([-1.0, -0.25, 0.5, 1.5])
        responses = np.array([[0, 1], [1, -1], [0, 1], [1, 0]])
        frequency = np.array([1, 3, 2, 4])

        weighted = KernelSmoothingModel(n_items=2).calibrate(
            responses,
            theta,
            sample_weight=frequency,
        )
        duplicated = KernelSmoothingModel(n_items=2).calibrate(
            np.repeat(responses, frequency, axis=0),
            np.repeat(theta, frequency),
        )

        np.testing.assert_allclose(weighted.irf_values, duplicated.irf_values)
        np.testing.assert_array_equal(weighted.calibration_weight_sums, [10.0, 7.0])

    def test_weighted_kernel_combines_distance_and_weight_in_log_space(self):
        model = KernelSmoothingModel(
            n_items=1,
            bandwidth=1.0,
            theta_grid=np.array([0.0, 10.0, 20.0]),
        )

        model.calibrate(
            np.array([[0], [1]]),
            np.array([0.0, 20.0]),
            sample_weight=np.array([1e-300, 1.0]),
        )

        np.testing.assert_allclose(model.irf_values, 1.0, atol=1e-12)

    def test_probability_clamps_outside_grid(self):
        model = self._fitted_model()

        probabilities = model.probability(np.array([-100.0, 100.0]))

        np.testing.assert_allclose(probabilities[0], model.irf_values[:, 0])
        np.testing.assert_allclose(probabilities[1], model.irf_values[:, -1])

    def test_extreme_distances_do_not_underflow(self):
        model = KernelSmoothingModel(n_items=2, bandwidth=0.01)
        theta = np.array([999.0, 1000.0, 1001.0])
        responses = np.ones((3, 2), dtype=int)

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            model.calibrate(responses, theta)
            probabilities = model.probability(np.array([-4.0, 0.0, 4.0]))

        np.testing.assert_array_equal(probabilities, np.ones((3, 2)))

    def test_extreme_finite_inputs_remain_stable(self):
        model = KernelSmoothingModel(
            n_items=1,
            bandwidth=1e-300,
            theta_grid=np.array([-1e308, 0.0, 1e308]),
        )

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            model.calibrate(np.array([[0], [1]]), np.array([-1e308, 1e308]))

        np.testing.assert_allclose(model.irf_values[0], [0.0, 0.5, 1.0], atol=1e-300)

    def test_item_fallback_remains_stable_with_extreme_missing_patterns(self):
        model = KernelSmoothingModel(
            n_items=2,
            bandwidth=1e-300,
            theta_grid=np.array([-1e308, 0.0, 1e308]),
        )
        responses = np.array([[0, -1], [-1, 1]])

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            model.calibrate(responses, np.array([-1e308, 1e308]))

        np.testing.assert_array_equal(model.irf_values[0], 0.0)
        np.testing.assert_array_equal(model.irf_values[1], 1.0)

    def test_single_observation_per_item_produces_constant_curves(self):
        model = KernelSmoothingModel(n_items=2, bandwidth=0.25)
        responses = np.array([[1, 0]])

        model.calibrate(responses, np.array([50.0]))

        np.testing.assert_array_equal(model.irf_values[0], 1.0)
        np.testing.assert_array_equal(model.irf_values[1], 0.0)

    def test_recalibration_replaces_irfs_and_counts(self):
        model = self._fitted_model()
        before = model.irf_values

        model.calibrate(np.ones((3, 2), dtype=int), np.array([-1.0, 0.0, 1.0]))

        assert not np.array_equal(model.irf_values, before)
        np.testing.assert_array_equal(model.irf_values, 1.0)
        np.testing.assert_array_equal(model.calibration_counts, [3, 3])

    def test_failed_recalibration_does_not_mutate_fitted_state(self):
        model = self._fitted_model()
        before_irf = model.irf_values
        before_counts = model.calibration_counts
        before_weight_sums = model.calibration_weight_sums

        with pytest.raises(ValueError):
            model.calibrate(np.array([[0, 2]]), np.array([0.0]))

        np.testing.assert_array_equal(model.irf_values, before_irf)
        np.testing.assert_array_equal(model.calibration_counts, before_counts)
        np.testing.assert_array_equal(
            model.calibration_weight_sums,
            before_weight_sums,
        )
        assert model.is_fitted

    def test_information_is_finite_nonnegative_and_item_consistent(self):
        model = self._fitted_model()
        theta = np.linspace(-1.5, 1.5, 11)

        information = model.information(theta)

        assert information.shape == (11, 2)
        assert np.all(np.isfinite(information))
        assert np.all(information >= 0)
        np.testing.assert_allclose(information[:, 1], model.information(theta, 1))

    def test_information_matches_piecewise_linear_slope(self, monkeypatch):
        model = self._fitted_model()
        theta = np.array([-1.75, -0.75, 0.25, 1.25])

        expected = _finite_difference_information(model, theta)
        monkeypatch.setattr(
            model,
            "probability",
            lambda *args, **kwargs: pytest.fail("information recomputed probability"),
        )

        np.testing.assert_allclose(
            model.information(theta),
            expected,
            rtol=2e-7,
            atol=5e-10,
        )
        np.testing.assert_array_equal(
            model.information(np.array([-3.0, 3.0])),
            np.zeros((2, model.n_items)),
        )

    def test_log_likelihood_uses_calibrated_irfs(self):
        model = self._fitted_model()
        theta = np.array([[-0.75], [0.75]])
        responses = np.array([[0, 1], [1, -1]])
        probabilities = model.probability(theta)

        result = model.log_likelihood(responses, theta)

        expected = np.array(
            [
                np.log1p(-probabilities[0, 0]) + np.log(probabilities[0, 1]),
                np.log(probabilities[1, 0]),
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_copy_preserves_fit_configuration_and_is_independent(self):
        model = self._fitted_model()

        copied = model.copy()
        np.testing.assert_array_equal(
            copied.calibration_weight_sums,
            model.calibration_weight_sums,
        )
        copied.calibrate(np.ones((2, 2), dtype=int), np.array([-1.0, 1.0]))

        assert copied.bandwidth == model.bandwidth
        np.testing.assert_array_equal(copied.theta_grid, model.theta_grid)
        np.testing.assert_array_equal(copied.calibration_weight_sums, [2.0, 2.0])
        np.testing.assert_array_equal(model.calibration_weight_sums, [5.0, 4.0])
        assert copied.is_fitted
        assert not np.array_equal(copied.irf_values, model.irf_values)
        np.testing.assert_allclose(
            model.copy().probability(np.array([0.0])),
            model.probability(np.array([0.0])),
        )

    def test_properties_return_copies(self):
        model = self._fitted_model()
        grid = model.theta_grid
        irf = model.irf_values
        counts = model.calibration_counts
        weight_sums = model.calibration_weight_sums
        grid[:] = 99
        irf[:] = 99
        counts[:] = 99
        weight_sums[:] = 99

        assert np.all(model.theta_grid != 99)
        assert np.all(model.irf_values != 99)
        assert np.all(model.calibration_counts != 99)
        assert np.all(model.calibration_weight_sums != 99)

    @pytest.mark.parametrize("bandwidth", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_invalid_bandwidth(self, bandwidth):
        with pytest.raises(ValueError, match="bandwidth"):
            KernelSmoothingModel(n_items=2, bandwidth=bandwidth)

    @pytest.mark.parametrize(
        "theta_grid",
        [
            np.array([0.0]),
            np.array([[0.0, 1.0]]),
            np.array([0.0, np.nan]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        ],
    )
    def test_rejects_invalid_theta_grid(self, theta_grid):
        with pytest.raises(ValueError, match="theta_grid"):
            KernelSmoothingModel(n_items=2, theta_grid=theta_grid)

    @pytest.mark.parametrize(
        ("responses", "theta"),
        [
            (np.array([0, 1]), np.array([0.0, 1.0])),
            (np.empty((0, 2), dtype=int), np.array([])),
            (np.array([[0], [1]]), np.array([0.0, 1.0])),
            (np.array([[0, 1], [1, 0]], dtype=float) + 0.5, np.array([0.0, 1.0])),
            (np.array([[0.0, np.nan]]), np.array([0.0])),
            (np.array([[0, 2]]), np.array([0.0])),
            (np.array([[0, 1], [1, 0]]), np.array([0.0])),
            (np.array([[0, 1], [1, 0]]), np.array([[0.0, 1.0]])),
            (np.array([[0, 1]]), np.array([np.nan])),
        ],
    )
    def test_calibration_validates_shapes_and_values(self, responses, theta):
        model = KernelSmoothingModel(n_items=2)
        with pytest.raises(ValueError):
            model.calibrate(responses, theta)

    def test_calibration_rejects_items_with_no_observations(self):
        model = KernelSmoothingModel(n_items=2)
        responses = np.array([[0, -1], [1, -1]])

        with pytest.raises(ValueError, match=r"\[1\]"):
            model.calibrate(responses, np.array([-1.0, 1.0]))

    @pytest.mark.parametrize(
        "sample_weight",
        [
            np.array([1.0]),
            np.ones((2, 1)),
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
            np.array([1.0, -0.1]),
            np.zeros(2),
            np.full(2, np.finfo(np.float64).max),
        ],
    )
    def test_calibration_rejects_invalid_sample_weights(self, sample_weight):
        model = KernelSmoothingModel(n_items=2)
        responses = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError, match="sample_weight|positive calibration"):
            model.calibrate(
                responses,
                np.array([-1.0, 1.0]),
                sample_weight=sample_weight,
            )

    def test_calibration_requires_positive_weight_for_each_item(self):
        model = KernelSmoothingModel(n_items=2)
        responses = np.array([[0, -1], [1, 1]])

        with pytest.raises(ValueError, match=r"positive calibration weight: \[1\]"):
            model.calibrate(
                responses,
                np.array([-1.0, 1.0]),
                sample_weight=np.array([1.0, 0.0]),
            )

    @pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
    def test_rejects_invalid_item_index(self, item_idx):
        model = self._fitted_model()
        with pytest.raises((IndexError, TypeError)):
            model.probability(np.array([0.0]), item_idx)

    @pytest.mark.parametrize("theta", [np.array([np.nan]), np.array([np.inf])])
    def test_rejects_nonfinite_prediction_theta(self, theta):
        with pytest.raises(ValueError, match="finite"):
            self._fitted_model().probability(theta)

    def test_rejects_multidimensional_configuration(self):
        with pytest.raises(ValueError, match="unidimensional"):
            KernelSmoothingModel(n_items=2, n_factors=2)
