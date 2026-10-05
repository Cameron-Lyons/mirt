"""Ascent-based MCEM stopping rule with Monte Carlo sample-size growth."""

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt
from mirt.estimation._acceleration import FreeItemParameters
from mirt.estimation.em import EMEstimator
from mirt.estimation.mcem import (
    MCEMEstimator,
    QMCEMEstimator,
    StochasticEMEstimator,
    _log_likelihood_change,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


@pytest.fixture(scope="module")
def responses():
    return mirt.simdata("2PL", n_persons=300, n_items=8, seed=0)


@pytest.mark.parametrize("importance_sampling", [True, False])
def test_mcem_converges_near_the_em_estimate(responses, importance_sampling):
    reference = EMEstimator(tol=1e-8, compute_standard_errors=False).fit(
        TwoParameterLogistic(8), responses
    )
    estimator = MCEMEstimator(
        n_samples=100,
        max_iter=100,
        tol=1e-2,
        seed=1,
        importance_sampling=importance_sampling,
    )
    result = estimator.fit(TwoParameterLogistic(8), responses)

    assert result.converged
    assert result.n_iterations < 100
    # Short Metropolis chains do not reach the posteriors, so the posterior-draw
    # variant has a fixed point of its own; only importance sampling targets
    # the maximum likelihood estimate.
    if importance_sampling:
        for name, values in reference.model.parameters.items():
            np.testing.assert_allclose(result.model.parameters[name], values, atol=0.1)
    sizes = estimator.sample_size_history
    assert len(sizes) == result.n_iterations
    assert sizes[0] == 100 and np.all(np.diff(sizes) >= 0)
    assert max(sizes) <= estimator.max_samples == 1000
    assert estimator.n_samples == 100


def test_mcem_stops_when_monte_carlo_precision_is_exhausted(responses):
    estimator = MCEMEstimator(
        n_samples=50, max_samples=50, max_iter=200, tol=1e-9, seed=2
    )
    with pytest.warns(RuntimeWarning, match="within Monte Carlo error"):
        result = estimator.fit(TwoParameterLogistic(8), responses)
    assert not result.converged
    assert result.n_iterations < 200
    assert set(estimator.sample_size_history) == {50}


def test_deterministic_and_stochastic_variants_keep_the_plain_rule(responses):
    qmcem = QMCEMEstimator(n_samples=64, max_iter=100, seed=3)
    result = qmcem.fit(TwoParameterLogistic(8), responses)
    history = qmcem.convergence_history
    assert result.converged
    assert abs(history[-1] - history[-2]) < qmcem.tol
    assert set(qmcem.sample_size_history) == {64}

    sem = StochasticEMEstimator(max_iter=5, seed=3)
    sem.fit(TwoParameterLogistic(8), responses)
    assert set(sem.sample_size_history) == {5}


def test_iterate_change_restores_the_current_parameters():
    data = mirt.simdata("GRM", n_persons=80, n_items=3, n_categories=4, seed=5)
    model = GradedResponseModel(3, n_categories=4)
    model.set_parameters(discrimination=np.array([1.2, 0.8, 1.5]))
    before = model.parameters
    free = FreeItemParameters(model)
    previous = free.get(model)
    previous[:3] = 1.0

    estimator = MCEMEstimator(n_samples=50, seed=6)
    estimator._rng = np.random.default_rng(6)
    prior_mean, cholesky = np.zeros(1), np.eye(1)
    samples, weights = estimator._e_step_mc(model, data, prior_mean, cholesky, 1)
    change, error = estimator._iterate_change(
        model, data, samples, weights, free, previous
    )
    for name, values in before.items():
        np.testing.assert_array_equal(model.parameters[name], values)
    assert np.isfinite(change) and error > 0.0


def test_max_samples_must_cover_the_initial_sample():
    with pytest.raises(ValueError, match="max_samples"):
        MCEMEstimator(n_samples=100, max_samples=99)
    assert MCEMEstimator(n_samples=60).max_samples == 600


def _marginal_log_likelihood(model, responses):
    quadrature = GaussHermiteQuadrature(61)
    joint = model.log_likelihood_batch(responses, quadrature.nodes)
    return float(np.sum(logsumexp(joint + np.log(quadrature.weights), axis=1)))


def test_common_draw_change_estimate_and_standard_error_are_calibrated(responses):
    responses = responses[:150]
    base = TwoParameterLogistic(8)
    other = TwoParameterLogistic(8).set_parameters(
        discrimination=np.full(8, 1.1), difficulty=np.full(8, 0.05)
    )
    expected = _marginal_log_likelihood(other, responses) - _marginal_log_likelihood(
        base, responses
    )
    rng = np.random.default_rng(4)
    n_persons, n_draws = len(responses), 100
    repeated = np.repeat(responses, n_draws, axis=0)
    estimates, errors = [], []
    for _ in range(100):
        draws = rng.standard_normal((n_persons * n_draws, 1))
        base_values = base.log_likelihood(repeated, draws).reshape(n_persons, -1)
        other_values = other.log_likelihood(repeated, draws).reshape(n_persons, -1)
        weights = np.exp(base_values - logsumexp(base_values, axis=1, keepdims=True))
        change, error = _log_likelihood_change(weights, base_values, other_values)
        estimates.append(change)
        errors.append(error)
    estimates = np.asarray(estimates)
    spread = estimates.std(ddof=1)
    assert abs(estimates.mean() - expected) < 4 * spread / np.sqrt(len(estimates))
    assert np.median(errors) == pytest.approx(spread, rel=0.3)
