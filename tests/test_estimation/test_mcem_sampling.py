"""Regression tests for Monte Carlo EM sampling and objectives."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest

import mirt.estimation.mcem as mcem_module
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.numeric import logsumexp


@pytest.mark.parametrize("value", [True, 49, 50.0, 0])
def test_mcem_rejects_invalid_sample_counts(value):
    with pytest.raises(ValueError, match="n_samples must be an integer"):
        MCEMEstimator(n_samples=value)


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_mcem_rejects_invalid_seeds(value):
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        MCEMEstimator(seed=value)


def test_mcem_rejects_non_boolean_importance_sampling():
    with pytest.raises(TypeError, match="importance_sampling must be a boolean"):
        MCEMEstimator(importance_sampling=1)


@pytest.mark.parametrize("value", [True, 0, 1.5])
def test_stochastic_em_rejects_invalid_chain_counts(value):
    with pytest.raises(ValueError, match="n_samples must be an integer"):
        StochasticEMEstimator(n_chains=value)


def test_stochastic_em_documented_default_is_usable():
    estimator = StochasticEMEstimator()

    assert estimator.n_chains == 5
    assert estimator.n_samples == 5
    assert not estimator.importance_sampling


@pytest.mark.parametrize(
    ("prior_mean", "prior_cov", "message"),
    [
        (np.zeros(1), np.eye(2), "prior_mean must have shape"),
        (np.array([0.0, np.nan]), np.eye(2), "prior_mean must contain"),
        (np.zeros(2), np.eye(1), "prior_cov must have shape"),
        (
            np.zeros(2),
            np.array([[1.0, np.inf], [np.inf, 1.0]]),
            "prior_cov must contain",
        ),
        (
            np.zeros(2),
            np.array([[1.0, 0.2], [0.0, 1.0]]),
            "prior_cov must be symmetric",
        ),
        (
            np.zeros(2),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            "prior_cov must be positive definite",
        ),
    ],
)
def test_fit_rejects_invalid_gaussian_priors(prior_mean, prior_cov, message):
    estimator = MCEMEstimator(n_samples=50, max_iter=1, seed=1)
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.array([[0], [1]], dtype=np.int_)

    with pytest.raises(ValueError, match=message):
        estimator.fit(model, responses, prior_mean=prior_mean, prior_cov=prior_cov)


def test_fit_does_not_mutate_prior_inputs(monkeypatch):
    estimator = MCEMEstimator(n_samples=50, max_iter=1, seed=2)
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.array([[0], [1]], dtype=np.int_)
    prior_mean = np.array([0.5, -0.25])
    prior_cov = np.array([[1.5, 0.2], [0.2, 0.75]])
    expected_mean = prior_mean.copy()
    expected_cov = prior_cov.copy()
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *_args: None)
    monkeypatch.setattr(estimator, "_compute_standard_errors_mc", lambda *_args: {})

    estimator.fit(model, responses, prior_mean=prior_mean, prior_cov=prior_cov)

    np.testing.assert_array_equal(prior_mean, expected_mean)
    np.testing.assert_array_equal(prior_cov, expected_cov)


def _direct_sample_log_likelihoods(model, responses, theta_samples):
    return np.column_stack(
        [
            model.log_likelihood(responses, theta_samples[:, sample_idx, :])
            for sample_idx in range(theta_samples.shape[1])
        ]
    )


def test_batched_sample_likelihood_matches_direct_multidimensional_evaluation():
    estimator = MCEMEstimator(n_samples=50)
    model = TwoParameterLogistic(2, n_factors=2).set_parameters(
        discrimination=np.array([[1.3, 0.4], [0.2, 1.1]]),
        difficulty=np.array([-0.3, 0.7]),
    )
    responses = np.array([[1, 0], [0, -1], [1, 1]], dtype=np.int_)
    samples = np.random.default_rng(3).normal(size=(3, 50, 2))

    actual = estimator._sample_log_likelihoods(model, responses, samples)
    expected = _direct_sample_log_likelihoods(model, responses, samples)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_batched_sample_likelihood_matches_polytomous_evaluation():
    estimator = MCEMEstimator(n_samples=50)
    model = GradedResponseModel(2, n_categories=[3, 4])
    responses = np.array([[0, 3], [2, -1], [1, 0]], dtype=np.int_)
    samples = np.random.default_rng(4).normal(size=(3, 50, 1))

    actual = estimator._sample_log_likelihoods(model, responses, samples)
    expected = _direct_sample_log_likelihoods(model, responses, samples)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_batched_sample_likelihood_uses_one_model_call(monkeypatch):
    estimator = MCEMEstimator(n_samples=50)
    model = OneParameterLogistic(2)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int_)
    samples = np.zeros((2, 50, 1))
    original = model.log_likelihood
    calls = 0

    def counted(response_values, theta_values):
        nonlocal calls
        calls += 1
        return original(response_values, theta_values)

    monkeypatch.setattr(model, "log_likelihood", counted)

    estimator._sample_log_likelihoods(model, responses, samples)

    assert calls == 1


@pytest.mark.parametrize(
    "samples",
    [np.zeros((2, 49, 1)), np.full((2, 50, 1), np.nan)],
)
def test_batched_sample_likelihood_rejects_invalid_samples(samples):
    estimator = MCEMEstimator(n_samples=50)
    model = OneParameterLogistic(1)
    responses = np.array([[0], [1]], dtype=np.int_)

    with pytest.raises(ValueError, match="theta_samples"):
        estimator._sample_log_likelihoods(model, responses, samples)


def test_mcem_e_step_uses_seeded_prior_draws_and_normalized_weights():
    estimator = MCEMEstimator(n_samples=50, seed=8)
    estimator._rng = np.random.default_rng(estimator.seed)
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.array([[0], [1], [1]], dtype=np.int_)
    prior_mean = np.array([0.5, -0.25])
    cholesky = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.8]]))
    expected_rng = np.random.default_rng(8)
    z = expected_rng.standard_normal((3, 50, 2))
    expected_samples = prior_mean + np.einsum("ij,...j->...i", cholesky, z)

    samples, weights = estimator._e_step_mc(model, responses, prior_mean, cholesky, 2)
    log_likelihoods = _direct_sample_log_likelihoods(model, responses, samples)
    expected_weights = np.exp(
        log_likelihoods - logsumexp(log_likelihoods, axis=1, keepdims=True)
    )

    np.testing.assert_array_equal(samples, expected_samples)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize("sequence", ["sobol", "halton"])
def test_qmcem_supports_arbitrary_finite_sample_counts(sequence):
    estimator = QMCEMEstimator(n_samples=70, seed=9, sequence=sequence)
    model = TwoParameterLogistic(2, n_factors=2)
    responses = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int_)
    prior_mean = np.array([0.3, -0.4])
    cholesky = np.linalg.cholesky(np.array([[1.2, 0.2], [0.2, 0.7]]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples, weights = estimator._e_step_mc(
            model, responses, prior_mean, cholesky, 2
        )

    assert samples.shape == (3, 70, 2)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(samples[0], samples[1])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(samples[0].mean(axis=0), prior_mean, atol=0.12)


def test_qmcem_is_reproducible_and_uses_one_batch_call(monkeypatch):
    first = QMCEMEstimator(n_samples=70, seed=10)
    second = QMCEMEstimator(n_samples=70, seed=10)
    model = OneParameterLogistic(2)
    responses = np.array([[0, 1], [1, 0]], dtype=np.int_)
    original = model.log_likelihood_batch
    calls = 0

    def counted(response_values, theta_values):
        nonlocal calls
        calls += 1
        return original(response_values, theta_values)

    monkeypatch.setattr(model, "log_likelihood_batch", counted)
    first_samples, first_weights = first._e_step_mc(
        model, responses, np.zeros(1), np.eye(1), 1
    )
    second_samples, second_weights = second._e_step_mc(
        model, responses, np.zeros(1), np.eye(1), 1
    )

    assert calls == 2
    np.testing.assert_array_equal(first_samples, second_samples)
    np.testing.assert_array_equal(first_weights, second_weights)


def test_stochastic_sampler_respects_nonstandard_prior_and_batches_calls(monkeypatch):
    estimator = StochasticEMEstimator(n_chains=5, seed=11)
    estimator._rng = np.random.default_rng(estimator.seed)
    model = OneParameterLogistic(1)
    responses = np.full((400, 1), -1, dtype=np.int_)
    original = model.log_likelihood
    calls = 0

    def counted(response_values, theta_values):
        nonlocal calls
        calls += 1
        return original(response_values, theta_values)

    monkeypatch.setattr(model, "log_likelihood", counted)
    samples, weights = estimator._e_step_mc(
        model, responses, np.array([5.0]), np.array([[0.1]]), 1
    )

    assert calls == 21
    assert samples.mean() == pytest.approx(5.0, abs=0.01)
    assert samples.std() == pytest.approx(0.1, abs=0.01)
    np.testing.assert_array_equal(weights, np.full((400, 5), 0.2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MCEMEstimator(
            n_samples=50, seed=12, importance_sampling=False, max_iter=1
        ),
        lambda: StochasticEMEstimator(n_chains=50, seed=12, max_iter=1),
    ],
)
def test_posterior_sampling_tracks_response_direction(
    factory: Callable[[], MCEMEstimator],
):
    estimator = factory()
    estimator._rng = np.random.default_rng(estimator.seed)
    model = OneParameterLogistic(1)
    responses = np.vstack(
        [np.zeros((100, 1), dtype=np.int_), np.ones((100, 1), dtype=np.int_)]
    )

    samples, weights = estimator._e_step_mc(model, responses, np.zeros(1), np.eye(1), 1)

    assert samples[:100].mean() < -0.2
    assert samples[100:].mean() > 0.2
    np.testing.assert_allclose(weights, 1.0 / estimator.n_samples)


def test_refresh_uses_arithmetic_identity_for_prior_samples():
    estimator = MCEMEstimator(n_samples=50)
    model = OneParameterLogistic(1)
    responses = np.array([[0], [1]], dtype=np.int_)
    samples = np.linspace(-2.0, 2.0, 100).reshape(2, 50, 1)
    initial_weights = np.full((2, 50), 0.02)
    log_likelihoods = _direct_sample_log_likelihoods(model, responses, samples)
    normalizer = logsumexp(log_likelihoods, axis=1, keepdims=True)

    likelihood, weights = estimator._refresh_mc_state(
        model, responses, samples, initial_weights
    )

    expected = float(np.sum(normalizer.ravel() - np.log(50)))
    assert likelihood == pytest.approx(expected)
    np.testing.assert_allclose(weights, np.exp(log_likelihoods - normalizer))


def test_refresh_uses_harmonic_identity_for_posterior_samples():
    estimator = StochasticEMEstimator(n_chains=5)
    model = OneParameterLogistic(1)
    responses = np.array([[0], [1]], dtype=np.int_)
    samples = np.linspace(-2.0, 2.0, 10).reshape(2, 5, 1)
    weights = np.full((2, 5), 0.2)
    log_likelihoods = _direct_sample_log_likelihoods(model, responses, samples)

    likelihood, refreshed_weights = estimator._refresh_mc_state(
        model, responses, samples, weights
    )

    expected = float(np.sum(-(logsumexp(-log_likelihoods, axis=1) - np.log(5))))
    assert likelihood == pytest.approx(expected)
    np.testing.assert_array_equal(refreshed_weights, weights)


def _direct_item_objective(model, item_idx, responses, samples, weights):
    total = 0.0
    for sample_idx in range(samples.shape[1]):
        probabilities = model.probability(samples[:, sample_idx, :], item_idx)
        if model.is_polytomous:
            selected = probabilities[np.arange(len(responses)), responses]
            log_probability = np.log(np.clip(selected, 1e-300, 1.0))
        else:
            probabilities = np.clip(probabilities, 1e-300, 1.0 - 1e-300)
            log_probability = responses * np.log(probabilities) + (
                1 - responses
            ) * np.log1p(-probabilities)
        total += float(weights[:, sample_idx] @ log_probability)
    return total


@pytest.mark.parametrize(
    ("model", "responses"),
    [
        (OneParameterLogistic(1), np.array([0, 1, 1], dtype=np.int_)),
        (GradedResponseModel(1, n_categories=4), np.array([0, 2, 3], dtype=np.int_)),
    ],
)
def test_vectorized_item_objective_matches_direct_evaluation(
    monkeypatch, model, responses
):
    estimator = MCEMEstimator(n_samples=50)
    samples = np.random.default_rng(13).normal(size=(3, 50, 1))
    weights = np.random.default_rng(14).uniform(size=(3, 50))
    weights /= weights.sum(axis=1, keepdims=True)
    original = model.probability
    calls = 0

    def counted(theta, item_idx=None):
        nonlocal calls
        calls += 1
        return original(theta, item_idx)

    monkeypatch.setattr(model, "probability", counted)
    actual = estimator._item_expected_log_likelihood(
        model, 0, responses, samples, weights
    )
    assert calls == 1
    monkeypatch.setattr(model, "probability", original)
    expected = _direct_item_objective(model, 0, responses, samples, weights)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_item_objective_rejects_invalid_weights():
    estimator = MCEMEstimator(n_samples=50)
    model = OneParameterLogistic(1)
    responses = np.array([0, 1], dtype=np.int_)
    samples = np.zeros((2, 50, 1))
    weights = np.full((2, 50), 0.02)
    weights[0, 0] = -1.0

    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        estimator._item_expected_log_likelihood(model, 0, responses, samples, weights)


def test_failed_item_optimization_restores_original_parameters(monkeypatch):
    estimator = MCEMEstimator(n_samples=50)
    model = OneParameterLogistic(1)
    responses = np.array([[0], [1]], dtype=np.int_)
    samples = np.zeros((2, 50, 1))
    weights = np.full((2, 50), 0.02)
    original = model.parameters

    def fail_after_objective(objective, **_kwargs):
        objective(np.array([2.0]))
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(mcem_module, "minimize", fail_after_objective)

    with pytest.raises(RuntimeError, match="optimizer failed"):
        estimator._optimize_item_mc(model, 0, responses, samples, weights)

    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(
    "estimator",
    [
        MCEMEstimator(n_samples=50, max_iter=1, seed=15),
        QMCEMEstimator(n_samples=50, max_iter=1, seed=15),
        StochasticEMEstimator(n_chains=5, max_iter=1, seed=15),
    ],
)
def test_monte_carlo_estimators_complete_small_fit(estimator):
    model = OneParameterLogistic(1)
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)

    result = estimator.fit(model, responses)

    assert model.is_fitted
    assert result.n_iterations == 1
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert len(estimator.convergence_history) == 2
