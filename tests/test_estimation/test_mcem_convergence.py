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
    _batch_means_errors,
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
    # Posterior chains continue across iterations, so short Metropolis runs
    # track the posteriors and both variants target the maximum likelihood
    # estimate; restarted chains used to leave slopes about 0.18 off.
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
    with pytest.warns(RuntimeWarning, match="within the burn-in"):
        sem.fit(TwoParameterLogistic(8), responses)
    assert set(sem.sample_size_history) == {5}


def test_qmcem_without_seed_scrambles_one_grid_per_fit(responses):
    # A fresh scramble in every E-step kept the log-likelihood estimate moving
    # by more than tol, so unseeded fits never converged.
    estimator = QMCEMEstimator(n_samples=64, max_iter=100)
    grids = []
    original = QMCEMEstimator._e_step_mc_state

    def recorded(self, *args):
        state = original(self, *args)
        grids.append(np.array(state[0][0]))
        return state

    estimator._e_step_mc_state = recorded.__get__(estimator)
    result = estimator.fit(TwoParameterLogistic(8), responses)
    assert result.converged
    assert result.n_iterations < 100
    for grid in grids[1:]:
        np.testing.assert_array_equal(grid, grids[0])

    first = grids[0]
    grids.clear()
    estimator.fit(TwoParameterLogistic(8), responses)
    assert not np.array_equal(grids[0], first)


def _em_reference(responses):
    return EMEstimator(tol=1e-8, compute_standard_errors=False).fit(
        TwoParameterLogistic(8), responses
    )


def test_stochastic_em_returns_the_average_of_post_burn_in_iterates(responses):
    estimator = StochasticEMEstimator(seed=2, tol=0.03, burnin=20)
    free = None
    iterates = []
    m_step = estimator._m_step_mc

    def recorded(model, *args):
        nonlocal free
        m_step(model, *args)
        free = free or FreeItemParameters(model)
        iterates.append(free.get(model))

    estimator._m_step_mc = recorded
    result = estimator.fit(TwoParameterLogistic(8), responses)

    assert result.converged
    assert result.n_iterations == len(iterates) < estimator.max_iter
    np.testing.assert_allclose(
        free.get(result.model), np.mean(iterates[20:], axis=0), rtol=1e-12
    )
    # The last iterate of the noisy chain used to be returned instead, about
    # 0.25 from the EM slopes after 1000 iterations.
    reference = _em_reference(responses)
    for name, values in reference.model.parameters.items():
        np.testing.assert_allclose(result.model.parameters[name], values, atol=0.1)


def test_stochastic_em_warns_when_the_average_is_not_precise(responses):
    estimator = StochasticEMEstimator(seed=2, max_iter=60, tol=1e-6, burnin=5)
    with pytest.warns(RuntimeWarning, match="Monte Carlo standard error"):
        result = estimator.fit(TwoParameterLogistic(8), responses)
    assert not result.converged
    assert result.n_iterations == 60

    estimator = StochasticEMEstimator(seed=2, max_iter=30, burnin=5)
    with pytest.warns(RuntimeWarning, match="fewer than 50 iterates"):
        assert not estimator.fit(TwoParameterLogistic(8), responses).converged


def test_batch_means_errors_match_the_error_of_an_autoregressive_mean():
    rng = np.random.default_rng(6)
    rho, n_iterates, n_chains = 0.6, 400, 300
    chains = np.empty((n_iterates, n_chains))
    chains[0] = rng.standard_normal(n_chains) / np.sqrt(1 - rho**2)
    for index in range(1, n_iterates):
        chains[index] = rho * chains[index - 1] + rng.standard_normal(n_chains)
    errors = _batch_means_errors(chains)
    # Var(mean) of a stationary AR(1) chain with unit innovations.
    expected = np.sqrt((1 + rho) / (1 - rho) / (1 - rho**2) / n_iterates)
    assert np.median(errors) == pytest.approx(expected, rel=0.15)
    assert chains.mean(axis=0).std() == pytest.approx(expected, rel=0.15)


def test_posterior_chains_continue_from_the_previous_draws():
    model = TwoParameterLogistic(4)
    data = mirt.simdata("2PL", n_persons=30, n_items=4, seed=8)
    estimator = MCEMEstimator(n_samples=50, seed=9, importance_sampling=False)
    estimator._rng = np.random.default_rng(9)
    prior = (np.zeros(1), np.eye(1), 1)

    first, _ = estimator._e_step_mc(model, data, *prior)
    start = first.copy()
    second, _ = estimator._e_step_mc(model, data, *prior)
    # The chains move on in place from the previous draw.
    assert second is first
    assert not np.array_equal(second, start)

    # Sample-size growth adds chains that start at copies of existing ones.
    estimator.n_samples = 75
    grown, _ = estimator._e_step_mc(model, data, *prior)
    assert grown.shape == (30, 75, 1)
    assert estimator._chain_state is grown

    # A new fit restarts the chains at the prior.
    estimator._start_draws()
    fresh, _ = estimator._e_step_mc(model, data, *prior)
    assert fresh is not grown


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


@pytest.mark.parametrize("importance_sampling", [True, False])
def test_change_estimate_reuses_the_e_step_likelihoods(
    responses, importance_sampling, monkeypatch
):
    original = MCEMEstimator._iterate_change
    checked = []

    def iterate_change(self, model, data, samples, *args, current_values=None):
        # The E-step's likelihoods equal a fresh evaluation at the current
        # parameters on the same draws.
        expected = self._sample_log_likelihoods(model, data, samples)
        np.testing.assert_array_equal(current_values, expected)
        checked.append(current_values.shape)
        return original(
            self, model, data, samples, *args, current_values=current_values
        )

    monkeypatch.setattr(MCEMEstimator, "_iterate_change", iterate_change)
    estimator = MCEMEstimator(
        n_samples=60,
        max_iter=15,
        tol=1e-2,
        seed=4,
        importance_sampling=importance_sampling,
    )
    result = estimator.fit(TwoParameterLogistic(8), responses)
    assert len(checked) == result.n_iterations - 1 > 0
    assert checked[-1] == (len(responses), estimator.sample_size_history[-1])


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
