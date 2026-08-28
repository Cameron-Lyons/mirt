"""Regression coverage for EM-family final-state reporting."""

from __future__ import annotations

import numpy as np
import pytest

from mirt.estimation.em import EMEstimator
from mirt.estimation.gvem import GVEMEstimator
from mirt.estimation.irtree_em import IRTreeEMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.mcem import MCEMEstimator, StochasticEMEstimator
from mirt.estimation.regularized import RegularizedMIRTEstimator
from mirt.estimation.sparse_bayesian import SparseBayesianEstimator
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.irtree import IRTreeModel
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.model import MultigroupModel
from mirt.utils.numeric import logsumexp


def test_em_refreshes_likelihood_and_posterior_after_last_m_step(monkeypatch):
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)
    model = OneParameterLogistic(1)
    estimator = EMEstimator(n_quadpts=5, max_iter=1, tol=1e9, use_gpu=False)
    captured: dict[str, np.ndarray] = {}

    def force_m_step(*_args):
        model._parameters["difficulty"][:] = 2.0

    def capture_standard_errors(_model, _responses, posterior):
        captured["posterior"] = posterior.copy()
        return {}

    monkeypatch.setattr(estimator, "_m_step", force_m_step)
    monkeypatch.setattr(estimator, "_compute_standard_errors", capture_standard_errors)

    result = estimator.fit(model, responses)
    expected_posterior, marginal = estimator._e_step(model, responses)
    expected_ll = float(np.sum(np.log(marginal + 1e-300)))

    assert result.log_likelihood == pytest.approx(expected_ll)
    assert result.converged
    assert len(estimator.convergence_history) == 2
    np.testing.assert_allclose(captured["posterior"], expected_posterior)


def test_mcem_refreshes_existing_draw_after_last_m_step(monkeypatch):
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)
    model = OneParameterLogistic(1)
    estimator = MCEMEstimator(n_samples=50, max_iter=1, tol=1e9, seed=17)
    captured: dict[str, np.ndarray] = {}

    def force_m_step(*_args):
        model._parameters["difficulty"][:] = 2.0

    def capture_standard_errors(_model, _responses, _samples, weights):
        captured["weights"] = weights.copy()
        return {}

    monkeypatch.setattr(estimator, "_m_step_mc", force_m_step)
    monkeypatch.setattr(
        estimator, "_compute_standard_errors_mc", capture_standard_errors
    )

    result = estimator.fit(model, responses)
    samples = np.random.default_rng(17).standard_normal((4, 50, 1))
    log_likes = np.column_stack(
        [
            model.log_likelihood(responses, samples[:, sample, :])
            for sample in range(estimator.n_samples)
        ]
    )
    normalizer = logsumexp(log_likes, axis=1, keepdims=True)
    expected_ll = float(np.sum(normalizer.ravel() - np.log(estimator.n_samples)))

    assert result.log_likelihood == pytest.approx(expected_ll)
    assert result.converged
    assert len(estimator.convergence_history) == 2
    np.testing.assert_allclose(captured["weights"], np.exp(log_likes - normalizer))


def test_stochastic_em_reuses_sampled_data_weights(monkeypatch):
    responses = np.array([[0], [1]], dtype=np.int_)
    model = OneParameterLogistic(1)
    estimator = StochasticEMEstimator(n_chains=50, max_iter=1, tol=1e9, seed=11)
    samples = np.linspace(-2.0, 2.0, 100).reshape(2, 50, 1)
    weights = np.full((2, 50), 1.0 / 50)
    captured: dict[str, object] = {"e_steps": 0}

    def fixed_e_step(*_args):
        captured["e_steps"] = int(captured["e_steps"]) + 1
        return samples, weights

    def force_m_step(*_args):
        model._parameters["difficulty"][:] = 1.5

    def capture_standard_errors(_model, _responses, _samples, final_weights):
        captured["weights"] = final_weights.copy()
        return {}

    monkeypatch.setattr(estimator, "_e_step_mc", fixed_e_step)
    monkeypatch.setattr(estimator, "_m_step_mc", force_m_step)
    monkeypatch.setattr(
        estimator, "_compute_standard_errors_mc", capture_standard_errors
    )

    result = estimator.fit(model, responses)

    assert result.log_likelihood == pytest.approx(
        estimator._estimate_marginal_ll(model, responses, samples, weights)
    )
    assert captured["e_steps"] == 1
    np.testing.assert_array_equal(captured["weights"], weights)


def test_gvem_refreshes_variational_state_after_last_m_step(monkeypatch):
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)
    model = TwoParameterLogistic(1)
    estimator = GVEMEstimator(max_iter=1, tol=1e9, n_inner_iter=1, use_gpu=False)
    original_e_step = estimator._e_step
    calls = 0

    def counted_e_step(*args):
        nonlocal calls
        calls += 1
        original_e_step(*args)

    def force_m_step(*_args):
        estimator._intercepts[:] = 2.0

    monkeypatch.setattr(estimator, "_e_step", counted_e_step)
    monkeypatch.setattr(estimator, "_m_step", force_m_step)
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *_args: {})

    result = estimator.fit(model, responses)
    expected = estimator._compute_elbo(model, responses, np.zeros(1), np.eye(1))

    assert result.log_likelihood == pytest.approx(expected)
    assert result.converged
    assert calls == 2
    assert estimator.elbo_history == pytest.approx(estimator.convergence_history)


def test_irtree_refreshes_final_posterior_and_likelihood(monkeypatch):
    responses = np.array([[0], [1], [3], [4]], dtype=np.int_)
    model = IRTreeModel(n_items=1, tree_spec="direction_intensity")
    estimator = IRTreeEMEstimator(
        n_quadpts=3, max_iter=1, tol=1e9, estimate_correlations=False
    )

    def force_m_step(*_args):
        model._parameters["difficulty"] += 1.5

    monkeypatch.setattr(estimator, "_m_step", force_m_step)
    monkeypatch.setattr(estimator, "_compute_standard_errors", lambda *_args: {})
    monkeypatch.setattr(
        estimator,
        "_compute_eap_scores",
        lambda *_args: (
            np.zeros((len(responses), model.n_traits)),
            np.ones((len(responses), model.n_traits)),
        ),
    )

    result = estimator.fit(model, responses)
    pseudo, assignments, valid = model.expand_to_pseudo_items(responses)
    _posterior, marginal = estimator._e_step(
        model,
        pseudo,
        assignments,
        valid,
        np.zeros(model.n_traits),
        np.eye(model.n_traits),
    )

    assert result.log_likelihood == pytest.approx(
        float(np.sum(np.log(marginal + 1e-300)))
    )
    assert result.converged
    assert len(estimator.convergence_history) == 2


def test_regularized_mirt_refreshes_final_fit_statistics(monkeypatch):
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)
    estimator = RegularizedMIRTEstimator(n_factors=2, n_quadpts=3, max_iter=1, tol=1e9)

    def force_m_step(_responses, _posterior, loadings, intercepts, _valid_masks):
        return loadings.copy(), intercepts + 2.0

    monkeypatch.setattr(estimator, "_m_step_penalized", force_m_step)
    monkeypatch.setattr(GaussianDensity, "update", lambda *_args: None)

    result = estimator.fit(responses)
    density = GaussianDensity(n_dimensions=2)
    _posterior, marginal = estimator._e_step(
        responses, result.loadings, result.intercepts, density
    )
    expected_ll = float(np.sum(np.log(marginal + 1e-300)))
    expected_penalized = expected_ll - estimator._compute_penalty(result.loadings)

    assert result.log_likelihood == pytest.approx(expected_ll)
    assert result.penalized_ll == pytest.approx(expected_penalized)
    assert result.aic == pytest.approx(-2 * expected_ll + 2 * result.n_parameters)
    assert result.bic == pytest.approx(
        -2 * expected_ll + result.n_parameters * np.log(len(responses))
    )
    assert result.converged
    assert len(estimator.convergence_history) == 2


def test_sparse_bayesian_refreshes_final_variational_state(monkeypatch):
    responses = np.array([[0], [0], [1], [1]], dtype=np.int_)
    model = TwoParameterLogistic(1)
    estimator = SparseBayesianEstimator(
        k_max=1, adaptive_theta=False, max_iter=1, tol=1e9
    )

    def force_m_step(_responses):
        estimator._intercepts[:] += 2.0

    monkeypatch.setattr(estimator, "_m_step_ssl", force_m_step)

    result = estimator.fit(model, responses)
    expected = estimator._compute_elbo(responses, np.zeros(1), np.eye(1))
    expected_log_likelihood = estimator._compute_log_likelihood(
        responses,
        result.sparse_loadings,
    )

    assert result.elbo == pytest.approx(expected)
    assert result.log_likelihood == pytest.approx(expected_log_likelihood)
    assert result.elbo_history[-1] == pytest.approx(expected)
    assert result.converged
    assert len(estimator.convergence_history) == 2


def test_multigroup_reports_convergence_on_last_allowed_iteration(monkeypatch):
    model = MultigroupModel(OneParameterLogistic(1), n_groups=2)
    responses = [
        np.array([[0], [1]], dtype=np.int_),
        np.array([[0], [1]], dtype=np.int_),
    ]
    estimator = MultigroupEMEstimator(n_quadpts=5, max_iter=2, tol=1e9)
    monkeypatch.setattr(estimator, "_m_step", lambda *_args: None)

    result = estimator.fit(model, responses)

    assert result.n_iterations == estimator.max_iter
    assert result.converged
