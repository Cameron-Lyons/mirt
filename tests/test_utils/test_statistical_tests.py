"""Regression tests for public hypothesis-testing utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from mirt._core import sigmoid
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.statistical_tests import lagrange, likelihood_ratio, wald


def test_wald_uses_model_parameter_order_instead_of_alphabetical_order() -> None:
    model = MultidimensionalModel(2, n_factors=2)
    model.set_parameters(
        slopes=np.array([[1.0, 2.0], [3.0, 4.0]]),
        intercepts=np.array([10.0, 20.0]),
    )

    result = wald(
        model,
        param_indices=[0],
        constraint_values=[0.0],
        vcov=np.eye(6),
    )

    assert_allclose(result.parameter_estimates, [1.0])
    assert_allclose(result.standard_errors, [1.0])
    assert result.statistic == pytest.approx(1.0)
    assert result.df == 1
    assert result.p_value == pytest.approx(stats.chi2.sf(1.0, 1))


def test_wald_supports_general_linear_hypotheses() -> None:
    model = TwoParameterLogistic(2)
    model.set_parameters(
        discrimination=np.array([1.0, 2.0]),
        difficulty=np.array([3.0, 4.0]),
    )
    contrast = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )

    result = wald(
        model,
        constraint_values=[0.0, 0.0],
        vcov=np.eye(4) * 0.25,
        contrast_matrix=contrast,
    )

    assert_allclose(result.parameter_estimates, [-1.0, -1.0])
    assert_allclose(result.standard_errors, np.sqrt([0.5, 0.5]))
    assert result.statistic == pytest.approx(4.0)
    assert result.df == 2
    assert result.p_value == pytest.approx(stats.chi2.sf(4.0, 2))


def test_wald_accepts_a_single_vector_contrast() -> None:
    model = TwoParameterLogistic(2)
    contrast = np.array([1.0, -1.0, 0.0, 0.0])

    result = wald(
        model,
        vcov=np.eye(4),
        contrast_matrix=contrast,
    )

    assert result.df == 1
    assert_allclose(result.parameter_estimates, [0.0])
    assert_allclose(result.standard_errors, [np.sqrt(2.0)])


def test_wald_uses_model_covariance_when_available() -> None:
    model = TwoParameterLogistic(1)
    model.vcov = np.diag([0.25, 0.5])

    result = wald(model, [0], [0.0])

    assert result.statistic == pytest.approx(4.0)
    assert_allclose(result.standard_errors, [0.5])


def test_wald_inverts_model_information_when_available() -> None:
    model = TwoParameterLogistic(1)
    model.information_matrix = lambda: np.diag([4.0, 2.0])

    result = wald(model, [0], [0.0])

    assert result.statistic == pytest.approx(4.0)
    assert_allclose(result.standard_errors, [0.5])


def test_wald_requires_real_covariance_information() -> None:
    model = TwoParameterLogistic(1)

    with pytest.raises(ValueError, match="vcov is required"):
        wald(model, [0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "required"),
        ({"param_indices": []}, "non-empty"),
        ({"param_indices": [0.0]}, "integers"),
        ({"param_indices": [-1]}, "between"),
        ({"param_indices": [0, 0]}, "duplicates"),
        (
            {"param_indices": [0], "constraint_values": [0.0, 1.0]},
            "length 1",
        ),
        (
            {"param_indices": [0], "vcov": np.eye(3)},
            "shape",
        ),
        (
            {
                "param_indices": [0],
                "vcov": np.array([[1.0, 0.2], [0.0, 1.0]]),
            },
            "symmetric",
        ),
        (
            {
                "param_indices": [0],
                "vcov": np.array([[1.0, 2.0], [2.0, 1.0]]),
            },
            "positive semidefinite",
        ),
        (
            {
                "param_indices": [0],
                "contrast_matrix": np.array([1.0, 0.0]),
            },
            "mutually exclusive",
        ),
        (
            {"contrast_matrix": np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)},
            "linearly independent",
        ),
    ],
)
def test_wald_validates_hypothesis_inputs(kwargs: dict, message: str) -> None:
    model = TwoParameterLogistic(1)
    kwargs.setdefault("vcov", np.eye(2))

    with pytest.raises(ValueError, match=message):
        wald(model, **kwargs)


def test_lagrange_matches_analytic_two_parameter_scores_with_missing_data() -> None:
    model = TwoParameterLogistic(1)
    model.set_parameters(
        discrimination=np.array([1.2]),
        difficulty=np.array([0.3]),
    )
    original_parameters = model.parameters
    responses = np.array([[0.0], [1.0], [np.nan]])
    theta = np.array([-1.0, 0.5, 1.0])
    covariance = np.diag([0.2, 0.1])

    result = lagrange(
        model,
        responses,
        theta,
        param_indices=[0, 1],
        vcov=covariance,
    )

    observed_theta = theta[:2]
    probabilities = sigmoid(1.2 * (observed_theta - 0.3))
    residuals = responses[:2, 0] - probabilities
    expected_scores = np.array(
        [
            np.sum(residuals * (observed_theta - 0.3)),
            np.sum(residuals * -1.2),
        ]
    )
    expected_statistic = float(expected_scores @ covariance @ expected_scores)

    assert_allclose(result.scores, expected_scores, rtol=1e-8, atol=1e-9)
    assert result.statistic == pytest.approx(expected_statistic)
    assert result.p_value == pytest.approx(stats.chi2.sf(expected_statistic, 2))
    for name, values in original_parameters.items():
        assert_allclose(model.parameters[name], values)


def test_lagrange_accepts_one_person_multidimensional_theta() -> None:
    model = MultidimensionalModel(2, n_factors=2)
    original_parameters = model.parameters

    result = lagrange(
        model,
        responses=np.array([[1, 0]]),
        theta=np.array([0.1, 0.2]),
        param_indices=[0, 1],
        vcov=np.eye(6),
    )

    assert result.df == 2
    assert np.isfinite(result.statistic)
    assert np.all(np.isfinite(result.scores))
    for name, values in original_parameters.items():
        assert_allclose(model.parameters[name], values)


def test_lagrange_supports_polytomous_native_log_likelihood() -> None:
    model = GradedResponseModel(1, n_categories=3)

    result = lagrange(
        model,
        responses=np.array([[0], [1], [2]]),
        theta=np.array([-1.0, 0.0, 1.0]),
        param_indices=[0, 1, 2],
        vcov=np.eye(3),
    )

    assert result.df == 3
    assert np.isfinite(result.statistic)
    assert np.all(np.isfinite(result.scores))


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        (
            ThreeParameterLogistic(2),
            {
                "discrimination": np.array([1.2, 0.9]),
                "difficulty": np.array([0.3, -0.4]),
                "guessing": np.array([0.15, 0.2]),
            },
        ),
        (
            FourParameterLogistic(2),
            {
                "discrimination": np.array([1.2, 0.9]),
                "difficulty": np.array([0.3, -0.4]),
                "guessing": np.array([0.15, 0.2]),
                "upper": np.array([0.95, 0.9]),
            },
        ),
    ],
)
def test_lagrange_logistic_scores_match_finite_differences(
    model: ThreeParameterLogistic | FourParameterLogistic,
    parameters: dict[str, np.ndarray],
) -> None:
    model.set_parameters(**parameters)
    responses = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    theta = np.array([-1.5, -0.25, 0.5, 1.25])
    flattened = np.concatenate([values.ravel() for values in model.parameters.values()])
    indices = list(range(flattened.size))

    result = lagrange(
        model,
        responses,
        theta,
        param_indices=indices,
        vcov=np.eye(flattened.size),
    )

    expected = np.empty_like(flattened)
    step = 1e-6
    offset = 0
    original = model.parameters
    for name, values in original.items():
        for local_index in range(values.size):
            plus = {key: value.copy() for key, value in original.items()}
            minus = {key: value.copy() for key, value in original.items()}
            plus[name].flat[local_index] += step
            minus[name].flat[local_index] -= step
            model.set_parameters(**plus)
            likelihood_plus = float(np.sum(model.log_likelihood(responses, theta)))
            model.set_parameters(**minus)
            likelihood_minus = float(np.sum(model.log_likelihood(responses, theta)))
            expected[offset + local_index] = (likelihood_plus - likelihood_minus) / (
                2 * step
            )
        offset += values.size
    model.set_parameters(**original)

    assert_allclose(result.scores, expected, rtol=1e-7, atol=1e-8)


def test_lagrange_uses_compatible_model_score_function() -> None:
    model = TwoParameterLogistic(1)
    model.score_function = lambda responses, theta: np.array([[1.0, 2.0], [3.0, 4.0]])

    result = lagrange(
        model,
        responses=np.array([[0], [1]]),
        theta=np.array([-0.5, 0.5]),
        param_indices=[1],
        vcov=np.eye(2),
    )

    assert_allclose(result.scores, [6.0])
    assert result.statistic == pytest.approx(36.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"responses": np.array([0, 1])}, "2D matrix"),
        ({"responses": np.ones((2, 2))}, "1 items"),
        ({"responses": np.array([[0.5], [1.0]])}, "integer-valued"),
        ({"responses": np.array([[2], [1]])}, "dichotomous"),
        ({"theta": np.array([0.0])}, "theta must have shape"),
        ({"param_indices": [2]}, "between"),
        ({"step": 0.0}, "positive"),
        ({"vcov": np.zeros((2, 2))}, "positive definite"),
    ],
)
def test_lagrange_validates_inputs(kwargs: dict, message: str) -> None:
    model = TwoParameterLogistic(1)
    inputs = {
        "responses": np.array([[0], [1]]),
        "theta": np.array([-0.5, 0.5]),
        "param_indices": [0],
        "vcov": np.eye(2),
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        lagrange(model, **inputs)


def test_likelihood_ratio_uses_stable_survival_probability() -> None:
    statistic, p_value = likelihood_ratio(0.0, -50.0, 1)

    assert statistic == 100.0
    assert p_value == pytest.approx(stats.chi2.sf(100.0, 1))
    assert p_value > 0.0


def test_likelihood_ratio_tolerates_roundoff_at_zero() -> None:
    full = -100.0
    reduced = np.nextafter(full, np.inf)
    statistic, p_value = likelihood_ratio(full, reduced, 1)

    assert statistic == 0.0
    assert p_value == 1.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((np.nan, -2.0, 1), "finite"),
        ((-2.0, np.inf, 1), "finite"),
        ((-2.0, -3.0, 0), "positive"),
        ((-2.0, -3.0, 1.5), "integer"),
        ((-10.0, -9.0, 1), "at least as large"),
    ],
)
def test_likelihood_ratio_validates_nested_models(
    args: tuple[float, float, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        likelihood_ratio(*args)
