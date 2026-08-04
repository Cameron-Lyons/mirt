"""Tests for zero-inflated IRT models."""

import numpy as np
import pytest

from mirt import HurdleIRT, ZeroInflated2PL, ZeroInflated3PL
from mirt.exceptions import MirtValidationError


def _configured_models():
    zi2 = ZeroInflated2PL(n_items=2)
    zi2.set_parameters(
        discrimination=np.array([1.4, 0.8]),
        difficulty=np.array([0.2, -0.5]),
        zero_inflation=np.array([0.35, 0.1]),
    )

    zi3 = ZeroInflated3PL(n_items=2)
    zi3.set_parameters(
        discrimination=np.array([1.4, 0.8]),
        difficulty=np.array([0.2, -0.5]),
        guessing=np.array([0.18, 0.25]),
        zero_inflation=np.array([0.35, 0.1]),
    )

    hurdle = HurdleIRT(n_items=2)
    hurdle.set_parameters(
        discrimination=np.array([1.4, 0.8]),
        difficulty=np.array([0.2, -0.5]),
        engagement_intercept=np.array([-0.3, 1.0]),
        engagement_slope=np.array([1.1, -0.2]),
    )
    return zi2, zi3, hurdle


@pytest.fixture(params=range(3), ids=["zi2", "zi3", "hurdle"])
def configured_model(request):
    return _configured_models()[request.param]


def test_initialization_and_public_parameters():
    zi2, zi3, hurdle = _configured_models()

    assert zi2.model_name == "ZI-2PL"
    assert zi3.model_name == "ZI-3PL"
    assert hurdle.model_name == "Hurdle"
    np.testing.assert_array_equal(zi2.zero_inflation, [0.35, 0.1])
    np.testing.assert_array_equal(zi3.guessing, [0.18, 0.25])
    np.testing.assert_array_equal(hurdle.engagement_intercept, [-0.3, 1.0])
    np.testing.assert_array_equal(hurdle.engagement_slope, [1.1, -0.2])


def test_probability_information_and_diagnostic_shapes(configured_model):
    theta = np.array([-1.0, 0.0, 1.0])

    for method_name in (
        "probability",
        "probability_zero",
        "information",
        "structural_zero_probability",
        "structural_zero_posterior",
    ):
        method = getattr(configured_model, method_name)
        assert method(theta).shape == (3, 2)
        assert method(theta, item_idx=1).shape == (3,)


def test_zero_process_probabilities_are_coherent(configured_model):
    theta = np.array([-1.0, 0.0, 1.0])
    probability = configured_model.probability(theta)
    probability_zero = configured_model.probability_zero(theta)
    structural = configured_model.structural_zero_probability(theta)
    posterior = configured_model.structural_zero_posterior(theta)

    np.testing.assert_allclose(probability + probability_zero, 1.0)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.all(structural <= probability_zero)
    np.testing.assert_allclose(posterior, structural / probability_zero)
    assert np.all((posterior >= 0.0) & (posterior <= 1.0))


def test_information_matches_probability_derivative(configured_model):
    theta = np.array([-1.3, -0.1, 0.8, 1.7])
    step = 1e-6
    probability = configured_model.probability(theta)
    derivative = (
        configured_model.probability(theta + step)
        - configured_model.probability(theta - step)
    ) / (2.0 * step)
    expected = derivative**2 / (probability * (1.0 - probability))

    np.testing.assert_allclose(
        configured_model.information(theta), expected, rtol=2e-6, atol=1e-10
    )


def test_information_matches_item_specific_probability_derivative(configured_model):
    theta = np.array([-1.3, -0.1, 0.8, 1.7])
    step = 1e-6
    probability = configured_model.probability(theta, item_idx=1)
    derivative = (
        configured_model.probability(theta + step, item_idx=1)
        - configured_model.probability(theta - step, item_idx=1)
    ) / (2.0 * step)
    expected = derivative**2 / (probability * (1.0 - probability))

    np.testing.assert_allclose(
        configured_model.information(theta, item_idx=1),
        expected,
        rtol=2e-6,
        atol=1e-10,
    )


def test_zi2_probability_decomposition():
    model = _configured_models()[0]
    theta = np.array([-0.5, 0.5])

    np.testing.assert_allclose(
        model.probability(theta),
        (1.0 - model.zero_inflation[None, :]) * model.probability_2pl(theta),
    )


def test_zi3_probability_decomposition():
    model = _configured_models()[1]
    theta = np.array([-0.5, 0.5])

    np.testing.assert_allclose(
        model.probability(theta),
        (1.0 - model.zero_inflation[None, :]) * model.probability_3pl(theta),
    )


def test_hurdle_probability_decomposition():
    model = _configured_models()[2]
    theta = np.array([-0.5, 0.5])

    np.testing.assert_allclose(
        model.probability(theta),
        model.engagement_probability(theta) * model.probability_2pl(theta),
    )
    np.testing.assert_allclose(
        model.structural_zero_probability(theta),
        1.0 - model.engagement_probability(theta),
    )


def test_empirical_structural_zero_fraction(configured_model):
    theta = np.array([-1.5, -0.2, 0.4, 1.8])
    structural = configured_model.structural_zero_probability(theta)
    probability_zero = configured_model.probability_zero(theta)
    expected = structural.mean(axis=0) / probability_zero.mean(axis=0)

    np.testing.assert_allclose(
        configured_model.expected_structural_zero_fraction(theta), expected
    )
    np.testing.assert_allclose(
        configured_model.expected_proportion_zeros_from_inflation(theta), expected
    )
    assert np.isscalar(
        configured_model.expected_structural_zero_fraction(theta, item_idx=1)
    )


def test_default_structural_zero_fraction_uses_normal_distribution():
    model = ZeroInflated2PL(n_items=1)
    model.set_parameters(
        discrimination=np.array([1.7]),
        difficulty=np.array([1.1]),
        zero_inflation=np.array([0.3]),
    )
    nodes, weights = np.polynomial.hermite.hermgauss(101)
    theta = np.sqrt(2.0) * nodes
    probability_zero = model.probability_zero(theta, item_idx=0)
    expected = 0.3 / np.sum(weights * probability_zero / np.sqrt(np.pi))

    np.testing.assert_allclose(
        model.expected_structural_zero_fraction(item_idx=0),
        expected,
        rtol=2e-10,
    )


@pytest.mark.parametrize(
    ("model", "parameter", "bad_value"),
    [
        (ZeroInflated2PL(2), "discrimination", 0.0),
        (ZeroInflated2PL(2), "discrimination", np.nan),
        (ZeroInflated2PL(2), "difficulty", np.inf),
        (ZeroInflated2PL(2), "zero_inflation", -0.01),
        (ZeroInflated2PL(2), "zero_inflation", 1.0),
        (ZeroInflated3PL(2), "guessing", -0.01),
        (ZeroInflated3PL(2), "guessing", 1.0),
        (ZeroInflated3PL(2), "zero_inflation", 1.0),
        (HurdleIRT(2), "engagement_intercept", np.inf),
        (HurdleIRT(2), "engagement_slope", np.nan),
    ],
)
def test_parameter_domains_are_validated(model, parameter, bad_value):
    values = model.parameters[parameter]
    values[0] = bad_value

    with pytest.raises(MirtValidationError, match=parameter):
        model.set_parameters(**{parameter: values})


def test_parameter_updates_are_atomic_and_detached():
    model = ZeroInflated2PL(2)
    difficulty = np.array([0.2, 0.4])

    with pytest.raises(MirtValidationError, match="zero_inflation"):
        model.set_parameters(
            difficulty=difficulty,
            zero_inflation=np.array([0.2, 1.2]),
        )
    np.testing.assert_array_equal(model.difficulty, [0.0, 0.0])

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 99.0
    np.testing.assert_array_equal(model.difficulty, [0.2, 0.4])


def test_item_parameter_update_is_validated():
    model = ZeroInflated3PL(2)
    model.set_item_parameter(1, "guessing", 0.4)
    assert model.guessing[1] == 0.4

    with pytest.raises(MirtValidationError, match="guessing"):
        model.set_item_parameter(1, "guessing", 1.0)
    assert model.guessing[1] == 0.4

    with pytest.raises(MirtValidationError, match="scalar"):
        model.set_item_parameter(1, "guessing", np.array([0.2]))


@pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
def test_invalid_item_indices_are_rejected(item_idx):
    model = ZeroInflated2PL(2)
    with pytest.raises((IndexError, MirtValidationError)):
        model.probability(np.array([0.0]), item_idx=item_idx)


@pytest.mark.parametrize(
    "theta",
    [np.array([np.nan]), np.array([np.inf]), np.empty((0, 1))],
)
def test_invalid_ability_values_are_rejected(theta):
    with pytest.raises(MirtValidationError, match="theta"):
        ZeroInflated2PL(1).probability(theta)


def test_mutated_invalid_state_is_rejected_at_evaluation():
    model = ZeroInflated2PL(1)
    model.zero_inflation[0] = 1.2

    with pytest.raises(MirtValidationError, match="zero_inflation"):
        model.probability(np.array([0.0]))


def test_extreme_finite_abilities_are_numerically_safe(configured_model):
    theta = np.array([-1e6, 1e6])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probability = configured_model.probability(theta)
        information = configured_model.information(theta)
        posterior = configured_model.structural_zero_posterior(theta)

    assert np.all(np.isfinite(probability))
    assert np.all(np.isfinite(information))
    assert np.all(np.isfinite(posterior))


def test_log_likelihood_and_copy(dichotomous_responses, configured_model):
    model = configured_model
    responses = dichotomous_responses["responses"][:, :2]
    theta = dichotomous_responses["theta"].reshape(-1, 1)

    log_likelihood = model.log_likelihood(responses, theta)
    copied = model.copy()

    assert log_likelihood.shape == (len(responses),)
    assert np.all(log_likelihood <= 0.0)
    for name, values in model.parameters.items():
        np.testing.assert_array_equal(copied.parameters[name], values)
        assert not np.shares_memory(copied.parameters[name], values)
