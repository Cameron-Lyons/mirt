"""Tests for joint response-time models."""

import numpy as np
import pytest

from mirt.estimation.rt_gibbs import ResponseTimeGibbsSampler
from mirt.exceptions import MirtValidationError
from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult


def _configured_model(accuracy_model="2PL"):
    parameters = {
        "n_items": 2,
        "accuracy_model": accuracy_model,
        "discrimination": np.array([1.4, 0.8]),
        "difficulty": np.array([0.2, -0.5]),
        "time_intensity": np.array([1.0, 1.5]),
        "time_discrimination": np.array([1.2, 0.7]),
        "ability_speed_mean": np.array([0.1, -0.2]),
        "ability_speed_cov": np.array([[1.2, 0.3], [0.3, 0.8]]),
    }
    if accuracy_model == "3PL":
        parameters["guessing"] = np.array([0.18, 0.25])
    return ResponseTimeModel(**parameters)


def test_default_initialization():
    model = ResponseTimeModel(n_items=3)

    assert model.accuracy_model == "2PL"
    assert model.item_names == ["Item_0", "Item_1", "Item_2"]
    np.testing.assert_array_equal(model.discrimination, np.ones(3))
    np.testing.assert_array_equal(model.difficulty, np.zeros(3))
    np.testing.assert_array_equal(model.time_intensity, np.zeros(3))
    np.testing.assert_array_equal(model.time_discrimination, np.ones(3))
    np.testing.assert_array_equal(model.ability_speed_mean, np.zeros(2))
    np.testing.assert_array_equal(model.ability_speed_cov, np.eye(2))
    assert model.guessing is None


def test_three_parameter_defaults():
    model = ResponseTimeModel(n_items=2, accuracy_model="3PL")
    np.testing.assert_array_equal(model.guessing, np.full(2, 0.2))


def test_constructor_detaches_mutable_inputs():
    discrimination = np.array([1.0, 1.5])
    covariance = np.array([[1.0, 0.2], [0.2, 1.0]])
    names = ["A", "B"]
    model = ResponseTimeModel(
        n_items=2,
        discrimination=discrimination,
        ability_speed_cov=covariance,
        item_names=names,
    )

    discrimination[0] = 99.0
    covariance[0, 0] = 99.0
    names[0] = "changed"
    np.testing.assert_array_equal(model.discrimination, [1.0, 1.5])
    np.testing.assert_array_equal(model.ability_speed_cov, [[1.0, 0.2], [0.2, 1.0]])
    assert model.item_names == ["A", "B"]


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"n_items": 0}, "n_items"),
        ({"n_items": True}, "n_items"),
        ({"n_items": 1, "accuracy_model": "1PL"}, "accuracy_model"),
        ({"n_items": 2, "item_names": ["A"]}, "item_names"),
        ({"n_items": 1, "item_names": [""]}, "item_names"),
        ({"n_items": 2, "difficulty": np.array([0.0])}, "difficulty"),
        ({"n_items": 1, "difficulty": np.array([np.inf])}, "difficulty"),
        ({"n_items": 1, "discrimination": np.array([0.0])}, "discrimination"),
        (
            {"n_items": 1, "time_discrimination": np.array([-1.0])},
            "time_discrimination",
        ),
        ({"n_items": 1, "guessing": np.array([0.2])}, "guessing"),
        (
            {
                "n_items": 1,
                "accuracy_model": "3PL",
                "guessing": np.array([-0.1]),
            },
            "guessing",
        ),
        (
            {
                "n_items": 1,
                "accuracy_model": "3PL",
                "guessing": np.array([1.0]),
            },
            "guessing",
        ),
        ({"n_items": 1, "ability_speed_mean": np.zeros(3)}, "ability_speed_mean"),
        (
            {"n_items": 1, "ability_speed_cov": np.array([[1.0, 0.2], [0.1, 1.0]])},
            "symmetric",
        ),
        (
            {"n_items": 1, "ability_speed_cov": np.array([[1.0, 2.0], [2.0, 1.0]])},
            "positive definite",
        ),
    ],
)
def test_invalid_initialization_is_rejected(parameters, match):
    with pytest.raises(MirtValidationError, match=match):
        ResponseTimeModel(**parameters)


def test_parameter_updates_are_atomic_and_detached():
    model = _configured_model("3PL")
    original_difficulty = model.difficulty.copy()
    difficulty = np.array([0.8, 0.9])

    with pytest.raises(MirtValidationError, match="guessing"):
        model.set_parameters(
            difficulty=difficulty,
            guessing=np.array([0.2, 1.1]),
        )
    np.testing.assert_array_equal(model.difficulty, original_difficulty)

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 99.0
    np.testing.assert_array_equal(model.difficulty, [0.8, 0.9])

    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        model.set_parameters(unknown=np.ones(2))


def test_mutated_invalid_state_is_rejected_during_evaluation():
    model = _configured_model()
    model.time_discrimination[0] = 0.0

    with pytest.raises(MirtValidationError, match="time_discrimination"):
        model.accuracy_probability(np.array([0.0]))

    model = _configured_model()
    model.accuracy_model = "invalid"
    with pytest.raises(MirtValidationError, match="accuracy_model"):
        model.accuracy_probability(np.array([0.0]))


def test_ability_speed_correlation():
    model = ResponseTimeModel(
        n_items=1,
        ability_speed_cov=np.array([[4.0, 1.0], [1.0, 1.0]]),
    )
    assert model.ability_speed_corr == pytest.approx(0.5)


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_accuracy_probability_matches_formula(accuracy_model):
    model = _configured_model(accuracy_model)
    theta = np.array([-1.0, 0.0, 1.0])
    logistic = 1.0 / (
        1.0
        + np.exp(
            -model.discrimination[None, :]
            * (theta[:, None] - model.difficulty[None, :])
        )
    )
    expected = logistic
    if accuracy_model == "3PL":
        expected = model.guessing[None, :] + (1.0 - model.guessing[None, :]) * logistic

    np.testing.assert_allclose(model.accuracy_probability(theta), expected)
    np.testing.assert_allclose(model.accuracy_probability(theta, 1), expected[:, 1])
    assert model.accuracy_probability(0.0).shape == (1, 2)


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_accuracy_information_matches_probability_derivative(accuracy_model):
    model = _configured_model(accuracy_model)
    theta = np.array([-1.2, -0.1, 0.8, 1.5])
    step = 1e-6
    probability = model.accuracy_probability(theta)
    derivative = (
        model.accuracy_probability(theta + step)
        - model.accuracy_probability(theta - step)
    ) / (2.0 * step)
    expected = derivative**2 / (probability * (1.0 - probability))

    np.testing.assert_allclose(
        model.accuracy_information(theta), expected, rtol=2e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        model.accuracy_information(theta, 1), expected[:, 1], rtol=2e-6, atol=1e-10
    )


@pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
def test_invalid_item_indices_are_rejected(item_idx):
    with pytest.raises((IndexError, MirtValidationError)):
        _configured_model().accuracy_probability(np.array([0.0]), item_idx)


def test_extreme_abilities_are_numerically_safe():
    model = _configured_model("3PL")
    theta = np.array([-1e6, 1e6])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probability = model.accuracy_probability(theta)
        information = model.accuracy_information(theta)

    assert np.all(np.isfinite(probability))
    assert np.all(np.isfinite(information))


def test_response_time_log_density_matches_normal_formula():
    model = _configured_model()
    tau = np.array([-0.4, 0.2, 0.8])
    log_rt = np.array([[0.8, 1.1], [1.0, 1.7], [1.4, 0.9]])
    alpha = model.time_discrimination[None, :]
    residual = log_rt - (model.time_intensity[None, :] - tau[:, None])
    expected = np.log(alpha) - 0.5 * np.log(2.0 * np.pi) - 0.5 * (alpha * residual) ** 2

    np.testing.assert_allclose(model.rt_log_density(log_rt, tau), expected)
    np.testing.assert_allclose(
        model.rt_log_density(log_rt[:, 1], tau, 1), expected[:, 1]
    )
    assert model.rt_log_density(1.0, np.array([0.0, 0.5]), 0).shape == (2,)


def test_timing_information_and_expected_response_time():
    model = _configured_model()
    tau = np.array([-0.2, 0.6])
    expected_time = np.exp(
        model.time_intensity[None, :]
        - tau[:, None]
        + 0.5 / model.time_discrimination[None, :] ** 2
    )

    np.testing.assert_allclose(
        model.response_time_information(), model.time_discrimination**2
    )
    assert model.response_time_information(1) == pytest.approx(
        model.time_discrimination[1] ** 2
    )
    np.testing.assert_allclose(model.expected_response_time(tau), expected_time)
    np.testing.assert_allclose(
        model.expected_response_time(tau, 1), expected_time[:, 1]
    )


def test_joint_log_likelihood_matches_manual_sum():
    model = _configured_model("3PL")
    responses = np.array([[1, 0], [0, 1]])
    log_rt = np.array([[1.0, 1.2], [0.7, 1.8]])
    theta = np.array([-0.3, 0.8])
    tau = np.array([0.2, -0.4])
    probability = model.accuracy_probability(theta)
    accuracy = responses * np.log(probability) + (1 - responses) * np.log1p(
        -probability
    )
    timing = model.rt_log_density(log_rt, tau)

    np.testing.assert_allclose(
        model.joint_log_likelihood(responses, log_rt, theta, tau),
        np.sum(accuracy + timing, axis=1),
    )


def test_accuracy_and_timing_missingness_are_independent():
    model = ResponseTimeModel(n_items=2)
    responses = np.array([[-1.0, 1.0], [0.0, np.nan], [np.nan, -1.0]])
    log_rt = np.array([[0.0, np.nan], [np.nan, 0.2], [np.nan, np.nan]])
    theta = np.zeros(3)
    tau = np.zeros(3)
    result = model.joint_log_likelihood(responses, log_rt, theta, tau)

    timing_only = model.rt_log_density(np.array([0.0]), np.array([0.0]), 0)[0]
    accuracy_only = np.log(0.5)
    second_timing = model.rt_log_density(np.array([0.2]), np.array([0.0]), 1)[0]
    np.testing.assert_allclose(
        result,
        [timing_only + accuracy_only, accuracy_only + second_timing, 0.0],
    )


@pytest.mark.parametrize("bad_response", [2.0, 0.5, np.inf, -np.inf])
def test_invalid_observed_responses_are_rejected(bad_response):
    with pytest.raises(MirtValidationError, match="responses"):
        ResponseTimeModel(1).joint_log_likelihood(
            np.array([[bad_response]]),
            np.array([[0.0]]),
            np.array([0.0]),
            np.array([0.0]),
        )


def test_joint_likelihood_validates_shapes_and_values():
    model = ResponseTimeModel(2)
    responses = np.zeros((2, 2))
    log_rt = np.zeros((2, 2))

    with pytest.raises(MirtValidationError, match="responses"):
        model.joint_log_likelihood(
            responses[:, :1], log_rt[:, :1], np.zeros(2), np.zeros(2)
        )
    with pytest.raises(MirtValidationError, match="log_rt"):
        model.joint_log_likelihood(responses, log_rt[:, :1], np.zeros(2), np.zeros(2))
    with pytest.raises(MirtValidationError, match="theta"):
        model.joint_log_likelihood(responses, log_rt, np.zeros(1), np.zeros(2))
    with pytest.raises(MirtValidationError, match="tau"):
        model.joint_log_likelihood(
            responses, log_rt, np.zeros(2), np.array([0, np.nan])
        )
    with pytest.raises(MirtValidationError, match="log_rt"):
        model.joint_log_likelihood(
            responses, np.array([[0.0, np.inf], [0.0, 0.0]]), np.zeros(2), np.zeros(2)
        )


def test_simulation_is_reproducible_and_preserves_supplied_values():
    model = _configured_model("3PL")
    theta = np.linspace(-1.0, 1.0, 20)
    tau = np.linspace(0.5, -0.5, 20)
    first = model.simulate(20, theta=theta, tau=tau, seed=42)
    second = model.simulate(20, theta=theta, tau=tau, seed=42)

    for first_value, second_value in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_value, second_value)
    responses, response_times, sampled_theta, sampled_tau = first
    assert responses.shape == (20, 2)
    assert response_times.shape == (20, 2)
    assert np.all(response_times > 0.0)
    np.testing.assert_array_equal(sampled_theta, theta)
    np.testing.assert_array_equal(sampled_tau, tau)
    assert not np.shares_memory(sampled_theta, theta)
    assert not np.shares_memory(sampled_tau, tau)


def test_partial_simulation_uses_conditional_population_distribution():
    model = ResponseTimeModel(
        n_items=1,
        ability_speed_mean=np.array([0.0, 0.0]),
        ability_speed_cov=np.array([[1.0, 0.8], [0.8, 1.0]]),
    )
    n_persons = 50_000
    theta = np.full(n_persons, 2.0)
    _, _, _, sampled_tau = model.simulate(n_persons, theta=theta, seed=1)
    assert sampled_tau.mean() == pytest.approx(1.6, abs=0.015)
    assert sampled_tau.var() == pytest.approx(0.36, abs=0.015)

    tau = np.full(n_persons, -1.5)
    _, _, sampled_theta, _ = model.simulate(n_persons, tau=tau, seed=2)
    assert sampled_theta.mean() == pytest.approx(-1.2, abs=0.015)
    assert sampled_theta.var() == pytest.approx(0.36, abs=0.015)


def test_joint_simulation_recovers_population_moments():
    model = ResponseTimeModel(
        n_items=1,
        ability_speed_mean=np.array([0.4, -0.3]),
        ability_speed_cov=np.array([[1.2, 0.5], [0.5, 0.8]]),
    )
    _, _, theta, tau = model.simulate(50_000, seed=4)

    np.testing.assert_allclose([theta.mean(), tau.mean()], [0.4, -0.3], atol=0.015)
    np.testing.assert_allclose(np.cov(theta, tau), model.ability_speed_cov, atol=0.025)


@pytest.mark.parametrize("n_persons", [0, -1, 1.5, True])
def test_simulation_rejects_invalid_person_count(n_persons):
    with pytest.raises(MirtValidationError, match="n_persons"):
        ResponseTimeModel(1).simulate(n_persons)


def test_simulation_rejects_wrong_person_parameter_length():
    with pytest.raises(MirtValidationError, match="theta"):
        ResponseTimeModel(1).simulate(3, theta=np.zeros(2))


def test_model_and_result_summaries():
    model = _configured_model("3PL")
    result = ResponseTimeResult(
        model=model,
        theta_estimates=np.array([0.1, 0.2]),
        tau_estimates=np.array([-0.1, -0.2]),
        theta_se=np.array([0.3, 0.4]),
        tau_se=np.array([0.2, 0.3]),
        chains=None,
        log_likelihood=-10.0,
        dic=22.0,
        waic=24.0,
        rhat={"difficulty": 1.01},
        ess={"difficulty": 200.0},
        n_iterations=1000,
        n_chains=2,
        converged=True,
    )

    assert "Speed-Ability Corr" in model.summary()
    assert "Log-Likelihood" in result.summary()
    assert "first 1 persons" in result.person_summary(1)
    with pytest.raises(MirtValidationError, match="n_show"):
        result.person_summary(-1)


def test_gibbs_sampler_integration():
    generating_model = ResponseTimeModel(3)
    responses, response_times, _, _ = generating_model.simulate(8, seed=7)
    result = ResponseTimeGibbsSampler(
        n_iter=12,
        burnin=6,
        thin=1,
        proposal_sd=0.1,
        adapt_interval=3,
        seed=7,
    ).fit(responses, response_times)

    assert result.model.n_items == 3
    assert result.theta_estimates.shape == (8,)
    assert result.tau_estimates.shape == (8,)
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.dic)
    assert np.isfinite(result.waic)
