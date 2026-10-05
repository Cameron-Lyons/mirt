"""Tests for shared numerical kernels."""

import warnings

import numpy as np
import pytest

import mirt.utils.numeric as numeric_module
from mirt.constants import PROB_EPSILON
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.numeric import (
    compute_fit_stats,
    compute_hessian_se,
    compute_probability_moments,
    logsumexp,
    logsumexp_axis1,
    standard_normal_quadrature,
)


class TestStandardNormalQuadrature:
    def test_reproduces_standard_normal_moments(self) -> None:
        nodes, weights = standard_normal_quadrature(21)

        assert np.sum(weights) == pytest.approx(1.0)
        assert np.sum(weights * nodes) == pytest.approx(0.0, abs=1e-15)
        assert np.sum(weights * nodes**2) == pytest.approx(1.0)
        assert np.sum(weights * nodes**4) == pytest.approx(3.0)

    @pytest.mark.parametrize("n_points", [0, -1, True, 1.5])
    def test_requires_positive_integer(self, n_points) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            standard_normal_quadrature(n_points)

    def test_cached_arrays_are_immutable(self) -> None:
        first = standard_normal_quadrature(17)
        second = standard_normal_quadrature(17)

        assert first[0] is second[0]
        assert first[1] is second[1]
        assert not first[0].flags.writeable
        assert not first[1].flags.writeable


class TestLogSumExp:
    def test_large_values_are_stable(self) -> None:
        values = np.array([1000.0, 1001.0, 999.0])
        expected = 1001.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))

        assert float(logsumexp(values)) == pytest.approx(expected)

    def test_axis_and_keepdims(self) -> None:
        values = np.array([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]])

        reduced = logsumexp(values, axis=1)
        kept = logsumexp(values, axis=1, keepdims=True)

        expected = np.log(np.sum(np.exp(values), axis=1))
        np.testing.assert_allclose(reduced, expected)
        np.testing.assert_allclose(kept, expected[:, None])
        assert reduced.shape == (2,)
        assert kept.shape == (2, 1)

    def test_axis1_matches_general_implementation(self) -> None:
        values = np.array([[1000.0, 1001.0], [-1001.0, -1000.0]])

        np.testing.assert_allclose(logsumexp_axis1(values), logsumexp(values, axis=1))

    def test_all_negative_infinity_returns_negative_infinity(self) -> None:
        values = np.full((3, 4), -np.inf)

        result = logsumexp(values, axis=1)

        np.testing.assert_array_equal(result, np.full(3, -np.inf))

    def test_positive_infinity_dominates_finite_values(self) -> None:
        values = np.array([[1.0, np.inf], [-np.inf, 3.0]])

        result = logsumexp(values, axis=1)

        assert result[0] == np.inf
        assert result[1] == pytest.approx(3.0)

    def test_nan_is_propagated(self) -> None:
        assert np.isnan(logsumexp(np.array([0.0, np.nan])))

    def test_translation_identity(self) -> None:
        values = np.linspace(-10.0, 4.0, 50)

        shifted = logsumexp(values + 123.0)

        assert float(shifted) == pytest.approx(float(logsumexp(values)) + 123.0)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            logsumexp(np.array([]))

    def test_axis1_requires_matrix(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            logsumexp_axis1(np.array([0.0, 1.0]))


class TestHessianStandardErrors:
    def test_correlated_quadratic_uses_full_inverse(self) -> None:
        hessian = np.array([[4.0, 1.5], [1.5, 2.0]])

        def objective(x: np.ndarray) -> float:
            return float(0.5 * x @ hessian @ x)

        actual = compute_hessian_se(objective, np.array([0.3, -0.7]))
        expected = np.sqrt(np.diag(np.linalg.inv(hessian)))

        np.testing.assert_allclose(actual, expected, rtol=2e-6)

    def test_three_dimensional_quadratic(self) -> None:
        hessian = np.array(
            [
                [5.0, 0.8, -0.4],
                [0.8, 3.5, 0.6],
                [-0.4, 0.6, 2.5],
            ]
        )

        def objective(x: np.ndarray) -> float:
            return float(0.5 * x @ hessian @ x + np.array([1.0, -2.0, 0.5]) @ x)

        actual = compute_hessian_se(objective, np.array([2.0, -0.25, 1.2]))
        expected = np.sqrt(np.diag(np.linalg.inv(hessian)))

        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    @pytest.mark.parametrize(
        "hessian",
        [
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, -1.0]]),
        ],
    )
    def test_non_positive_definite_hessian_returns_nan(
        self, hessian: np.ndarray
    ) -> None:
        def objective(x: np.ndarray) -> float:
            return float(0.5 * x @ hessian @ x)

        result = compute_hessian_se(objective, np.array([0.1, -0.2]))

        assert np.all(np.isnan(result))

    @pytest.mark.parametrize(
        ("x", "h", "message"),
        [
            (np.array([[0.0, 1.0]]), 1e-5, "one-dimensional"),
            (np.array([]), 1e-5, "non-empty"),
            (np.array([np.nan]), 1e-5, "finite values"),
            (np.array([0.0]), 0.0, "finite and positive"),
            (np.array([0.0]), np.inf, "finite and positive"),
        ],
    )
    def test_invalid_configuration_is_rejected(
        self, x: np.ndarray, h: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            compute_hessian_se(lambda value: float(value @ value), x, h=h)

    def test_nonfinite_objective_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite scalar"):
            compute_hessian_se(lambda _: np.inf, np.array([0.0]))


class TestExpectedVariance:
    def test_probability_moments_expose_validated_probabilities(self) -> None:
        model = ThreeParameterLogistic(3)
        theta = np.linspace(-2.0, 2.0, 21)[:, None]

        probabilities, expected, variance = compute_probability_moments(
            model,
            theta,
            model.n_items,
        )

        np.testing.assert_allclose(probabilities, model.probability(theta))
        np.testing.assert_allclose(expected, probabilities)
        np.testing.assert_allclose(variance, probabilities * (1.0 - probabilities))

    def test_three_pl_matches_model_probabilities(self) -> None:
        model = ThreeParameterLogistic(3)
        model.set_parameters(
            discrimination=np.array([0.8, 1.2, 1.6]),
            difficulty=np.array([-0.5, 0.2, 1.0]),
            guessing=np.array([0.1, 0.2, 0.3]),
        )
        theta = np.linspace(-2.0, 2.0, 21)[:, None]

        _, expected, variance = compute_probability_moments(model, theta, model.n_items)
        probabilities = model.probability(theta)

        np.testing.assert_allclose(expected, probabilities, atol=1e-14)
        np.testing.assert_allclose(
            variance, probabilities * (1.0 - probabilities), atol=1e-14
        )

    def test_polytomous_moments_match_direct_calculation(self) -> None:
        model = GradedResponseModel(3, n_categories=[3, 4, 2])
        theta = np.linspace(-2.0, 2.0, 17)[:, None]

        _, expected, variance = compute_probability_moments(model, theta, model.n_items)
        probabilities = model.probability(theta)
        categories = np.arange(probabilities.shape[2], dtype=float)
        direct_expected = probabilities @ categories
        direct_variance = probabilities @ (categories**2) - direct_expected**2

        np.testing.assert_allclose(expected, direct_expected, atol=1e-14)
        np.testing.assert_allclose(variance, direct_variance, atol=1e-14)

    def test_probability_matrix_is_computed_once(self) -> None:
        model = TwoParameterLogistic(10)
        theta = np.linspace(-2.0, 2.0, 51)[:, None]
        original_probability = model.probability
        calls = 0

        def counted_probability(theta_values, item_idx=None):
            nonlocal calls
            calls += 1
            return original_probability(theta_values, item_idx)

        model.probability = counted_probability  # type: ignore[method-assign]

        compute_probability_moments(model, theta, model.n_items)

        assert calls == 1

    @pytest.mark.parametrize(
        ("theta", "n_items", "message"),
        [
            (np.array([0.0, 1.0]), 2, "two-dimensional"),
            (np.empty((0, 1)), 2, "non-empty"),
            (np.array([[np.inf]]), 2, "finite values"),
            (np.array([[0.0]]), 1, "must match"),
            (np.array([[0.0]]), 2.0, "integer"),
        ],
    )
    def test_invalid_configuration_is_rejected(
        self, theta: np.ndarray, n_items, message: str
    ) -> None:
        model = TwoParameterLogistic(2)

        with pytest.raises(ValueError, match=message):
            compute_probability_moments(model, theta, n_items)

    def test_invalid_model_probabilities_are_rejected(self) -> None:
        model = TwoParameterLogistic(2)
        model.probability = lambda theta, item_idx=None: np.full(  # type: ignore[method-assign]
            (len(theta), 2), 1.2
        )

        with pytest.raises(ValueError, match="outside"):
            compute_probability_moments(model, np.array([[0.0]]), 2)


class TestFitStatistics:
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("block_rows", [1, 3, 100])
    def test_blocks_match_scalar_reference_and_preserve_inputs(
        self, monkeypatch, axis, block_rows
    ) -> None:
        rng = np.random.default_rng(153)
        responses = rng.integers(-1, 5, size=(11, 14))[:, ::2]
        expected = rng.uniform(0, 4, size=(11, 14))[:, ::2]
        variance = rng.uniform(0, 2, size=(11, 14))[:, ::2]
        responses[0] = -9
        responses[:, 0] = -1
        variance[1, 1:5] = [0, -1e-12, PROB_EPSILON / 2, PROB_EPSILON]
        variance[:, 5] = 0
        arrays = (responses, expected, variance)
        originals = [array.copy() for array in arrays]
        for array in arrays:
            array.setflags(write=False)
        monkeypatch.setattr(
            numeric_module, "_FIT_TARGET_CHUNK_ELEMENTS", block_rows * 7
        )

        reference = np.full((2, responses.shape[1 - axis]), np.nan)
        for group in range(reference.shape[1]):
            selector = (slice(None), group) if axis == 0 else (group, slice(None))
            numerator = denominator = standardized = 0.0
            count = 0
            for response, mean, var in zip(
                *(array[selector] for array in arrays), strict=True
            ):
                if response < 0:
                    continue
                var = max(var, 0)
                squared = (response - mean) ** 2
                numerator += squared
                denominator += var
                if var > PROB_EPSILON:
                    standardized += squared / var
                    count += 1
            if denominator > PROB_EPSILON:
                reference[0, group] = numerator / denominator
            if count:
                reference[1, group] = standardized / count

        actual = compute_fit_stats(*arrays, axis=axis)

        np.testing.assert_allclose(actual, reference, rtol=1e-14, equal_nan=True)
        for array, original in zip(arrays, originals, strict=True):
            np.testing.assert_array_equal(array, original)

    def test_infit_threshold_is_applied_after_all_blocks(self, monkeypatch) -> None:
        monkeypatch.setattr(numeric_module, "_FIT_TARGET_CHUNK_ELEMENTS", 1)
        responses = np.ones((3, 1))
        expected = np.full((3, 1), 0.5)
        variance = np.full((3, 1), PROB_EPSILON / 2)

        infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)

        np.testing.assert_allclose(infit, [0.5 / PROB_EPSILON])
        assert np.isnan(outfit[0])

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
    def test_empty_axes_are_defined(self, shape, axis) -> None:
        values = np.empty(shape)
        with np.errstate(all="raise"):
            infit, outfit = compute_fit_stats(values, values, values, axis)
        assert infit.shape == outfit.shape == (shape[1 - axis],)
        assert np.isnan(infit).all() and np.isnan(outfit).all()

    def test_missing_codes_do_not_overflow_residuals(self) -> None:
        with np.errstate(all="raise"):
            infit, outfit = compute_fit_stats(
                [[-1e300, 1]], [[0.5, 0.5]], [[0.25, 0.25]], axis=1
            )
        np.testing.assert_array_equal(infit, [1])
        np.testing.assert_array_equal(outfit, [1])

    @pytest.mark.parametrize("input_index", [0, 1, 2])
    def test_invalid_values_in_later_blocks_are_rejected(
        self, monkeypatch, input_index
    ) -> None:
        monkeypatch.setattr(numeric_module, "_FIT_TARGET_CHUNK_ELEMENTS", 2)
        arrays = [np.ones((7, 2)) for _ in range(3)]
        arrays[input_index][-1, -1] = np.nan
        with pytest.raises(ValueError, match="finite"):
            compute_fit_stats(*arrays, axis=0)

    def test_matches_manual_calculation_with_missing_responses(self) -> None:
        responses = np.array([[1, 0, -1], [0, 1, 1], [1, -9, 0]])
        expected = np.array([[0.7, 0.4, 0.5], [0.6, 0.8, 0.4], [0.9, 0.3, 0.2]])
        variance = expected * (1.0 - expected)

        infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)

        valid = responses >= 0
        residual_squared = (responses - expected) ** 2
        expected_infit = np.sum(
            np.where(valid, residual_squared, 0.0), axis=0
        ) / np.sum(np.where(valid, variance, 0.0), axis=0)
        expected_outfit = np.sum(
            np.where(valid, residual_squared / variance, 0.0), axis=0
        ) / np.sum(valid, axis=0)
        np.testing.assert_allclose(infit, expected_infit)
        np.testing.assert_allclose(outfit, expected_outfit)

    def test_person_axis_has_expected_shape(self) -> None:
        responses = np.array([[1, 0, 1], [0, 1, 0]])
        expected = np.full((2, 3), 0.5)
        variance = np.full((2, 3), 0.25)

        infit, outfit = compute_fit_stats(responses, expected, variance, axis=1)

        assert infit.shape == (2,)
        assert outfit.shape == (2,)
        np.testing.assert_allclose(infit, np.ones(2))
        np.testing.assert_allclose(outfit, np.ones(2))

    def test_all_missing_returns_nan_without_warnings(self) -> None:
        responses = np.full((3, 4), -1)
        expected = np.full((3, 4), 0.5)
        variance = np.full((3, 4), 0.25)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            infit, outfit = compute_fit_stats(responses, expected, variance, axis=0)

        assert not caught
        assert np.all(np.isnan(infit))
        assert np.all(np.isnan(outfit))

    @pytest.mark.parametrize(
        ("responses", "expected", "variance", "axis", "message"),
        [
            (
                np.ones((2, 3)),
                np.ones((2, 1)),
                np.ones((2, 3)),
                0,
                "expected must have the same shape",
            ),
            (
                np.ones((2, 3)),
                np.ones((2, 3)),
                np.ones((2, 1)),
                0,
                "variance must have the same shape",
            ),
            (
                np.ones(3),
                np.ones(3),
                np.ones(3),
                0,
                "two-dimensional",
            ),
            (
                np.array([[1.0, np.nan]]),
                np.ones((1, 2)),
                np.ones((1, 2)),
                0,
                "finite numeric",
            ),
            (
                np.ones((1, 2)),
                np.array([[1.0, np.nan]]),
                np.ones((1, 2)),
                0,
                "expected must contain only finite",
            ),
            (
                np.ones((1, 2)),
                np.ones((1, 2)),
                np.array([[0.2, -0.1]]),
                0,
                "finite non-negative",
            ),
            (
                np.ones((1, 2)),
                np.ones((1, 2)),
                np.ones((1, 2)),
                2,
                "axis must be 0 or 1",
            ),
        ],
    )
    def test_invalid_inputs_are_rejected(
        self,
        responses: np.ndarray,
        expected: np.ndarray,
        variance: np.ndarray,
        axis: int,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            compute_fit_stats(responses, expected, variance, axis)
