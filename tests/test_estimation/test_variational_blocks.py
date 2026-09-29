"""Scalar references and memory contracts for Gaussian variational updates."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.constants import PROB_EPSILON
from mirt.estimation import _variational
from mirt.estimation.gvem import GVEMEstimator
from mirt.estimation.sparse_bayesian import SparseBayesianEstimator


def _scalar_reference(
    responses, loadings, intercepts, prior_mean, prior_cov, xi, iterations, floor
):
    """Update each person with explicit sums of observed item contributions."""
    means = np.empty((len(responses), len(prior_mean)))
    covariances = np.empty((len(responses), len(prior_mean), len(prior_mean)))
    updated = xi.copy()
    prior_precision = np.linalg.inv(prior_cov)
    for person, row in enumerate(responses):
        observed = np.flatnonzero(row >= 0)
        for _ in range(iterations):
            precision = prior_precision.copy()
            natural = prior_precision @ prior_mean
            for item in observed:
                x = abs(updated[person, item])
                lam = 0.125 if x < 1e-6 else np.tanh(x / 2) / (4 * x)
                a = loadings[item]
                precision += 2 * lam * np.outer(a, a)
                natural += (row[item] - 0.5 - 2 * lam * intercepts[item]) * a
            covariances[person] = np.linalg.inv(precision)
            means[person] = np.linalg.solve(precision, natural)
            for item in observed:
                a = loadings[item]
                eta = a @ means[person] + intercepts[item]
                variance = a @ covariances[person] @ a
                updated[person, item] = np.sqrt(max(variance + eta**2, floor))
    return means, covariances, updated


def _estimator(kind, loadings, intercepts, xi, iterations):
    n_persons, n_items = xi.shape
    n_factors = loadings.shape[1]
    if kind == "gvem":
        estimator = GVEMEstimator(n_inner_iter=iterations, use_gpu=False)
        estimator._slopes = loadings
        update = estimator._e_step_python
    else:
        estimator = SparseBayesianEstimator(k_max=n_factors, n_inner_iter=iterations)
        estimator._loadings = loadings
        update = estimator._e_step
    estimator._intercepts = intercepts
    estimator._mu = np.zeros((n_persons, n_factors))
    estimator._sigma = np.broadcast_to(
        np.eye(n_factors), (n_persons, n_factors, n_factors)
    ).copy()
    estimator._xi = xi
    return estimator, update


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("factors", [1, 3, 6])
@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("budget", [1, 1024, 1_000_000])
def test_blocked_updates_match_scalar_reference(
    monkeypatch, kind, factors, iterations, budget
):
    monkeypatch.setattr(_variational, "_MAX_WORKING_ELEMENTS", budget)
    rng = np.random.default_rng(41)
    responses = rng.integers(-2, 2, (38, 12))[::2, ::2]
    responses[0] = -99
    responses[:, -1] = -8
    loadings = rng.normal(scale=0.5, size=(12, factors))[::2]
    intercepts = rng.normal(size=12)[::2]
    xi = rng.uniform(0.0, 2.0, (38, 12))[::2, ::2]
    xi[1, :3] = [0.0, -1e-8, 1e-5]
    prior_mean = np.linspace(-0.4, 0.8, factors)
    prior_cov = np.eye(factors) * 0.9 + np.full((factors, factors), 0.15)
    inputs = (responses, loadings, intercepts, xi, prior_mean, prior_cov)
    originals = [array.copy() for array in inputs]
    for array in inputs:
        array.setflags(write=False)
    expected = _scalar_reference(
        responses,
        loadings,
        intercepts,
        prior_mean,
        prior_cov,
        xi,
        iterations,
        0.0 if kind == "gvem" else PROB_EPSILON,
    )
    estimator, update = _estimator(kind, loadings, intercepts, xi, iterations)
    old_mu, old_sigma = estimator._mu, estimator._sigma
    old_mu.setflags(write=False)
    old_sigma.setflags(write=False)
    update(responses, prior_mean, np.linalg.inv(prior_cov))
    for actual, reference in zip(
        (estimator._mu, estimator._sigma, estimator._xi), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(estimator._mu[0], prior_mean, atol=1e-14)
    np.testing.assert_allclose(estimator._sigma[0], prior_cov, atol=1e-14)
    np.testing.assert_array_equal(estimator._xi[responses < 0], xi[responses < 0])
    for array, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(array, original)
    np.testing.assert_array_equal(old_mu, np.zeros_like(old_mu))
    np.testing.assert_array_equal(
        old_sigma, np.broadcast_to(np.eye(factors), old_sigma.shape)
    )


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
def test_zero_loading_keeps_each_estimators_bound_floor(kind):
    xi = np.ones((3, 2))
    data = np.array([[0, 1], [1, -1], [-1, -1]])
    estimator, update = _estimator(kind, np.zeros((2, 1)), np.zeros(2), xi, 2)
    update(data, np.array([0.3]), np.eye(1))
    expected = np.where(
        data >= 0, 0.0 if kind == "gvem" else np.sqrt(PROB_EPSILON), 1.0
    )
    np.testing.assert_array_equal(estimator._xi, expected)
    np.testing.assert_allclose(estimator._mu, 0.3)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
def test_working_arrays_are_bounded_by_item_and_factor_counts(monkeypatch, kind):
    monkeypatch.setattr(_variational, "_MAX_WORKING_ELEMENTS", 1000)
    seen = []
    estimator, update = _estimator(
        kind, np.ones((10, 3)), np.zeros(10), np.ones((37, 10)), 3
    )
    original = estimator._lambda if kind == "gvem" else estimator._lambda_jj

    def record(xi):
        seen.append(xi.shape)
        return original(xi)

    monkeypatch.setattr(
        estimator, "_lambda" if kind == "gvem" else "_lambda_jj", record
    )
    update(np.zeros((37, 10), dtype=int), np.zeros(3), np.eye(3))
    assert len(seen) > 3
    assert all(rows * (10 * 13 + 4 * 9) <= 1000 for rows, _ in seen)
    assert sum(rows for rows, _ in seen) == 37 * 3


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
def test_empty_sample_returns_empty_variational_arrays(kind):
    estimator, update = _estimator(
        kind, np.ones((2, 3)), np.zeros(2), np.empty((0, 2)), 1
    )
    update(np.empty((0, 2), dtype=int), np.zeros(3), np.eye(3))
    assert estimator._mu.shape == (0, 3)
    assert estimator._sigma.shape == (0, 3, 3)
    assert estimator._xi.shape == (0, 2)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
def test_full_fit_preserves_results_across_block_sizes(monkeypatch, kind):
    previous = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        data = mirt.simdata(n_persons=120, n_items=5, n_factors=2, seed=15)
        data[::7, 0] = -1
        results = []
        for budget in (1_000_000, 600):
            monkeypatch.setattr(_variational, "_MAX_WORKING_ELEMENTS", budget)
            estimator = (
                GVEMEstimator(max_iter=4, tol=1e-10, use_gpu=False)
                if kind == "gvem"
                else SparseBayesianEstimator(k_max=2, max_iter=4, tol=1e-10)
            )
            result = estimator.fit(
                mirt.TwoParameterLogistic(5, n_factors=2),
                data,
                prior_mean=np.array([0.2, -0.1]),
                prior_cov=np.array([[1.2, 0.1], [0.1, 0.8]]),
            )
            results.append((estimator, result))
        (first_est, first), (second_est, second) = results
        for name in ("_mu", "_sigma", "_xi"):
            np.testing.assert_allclose(
                getattr(first_est, name),
                getattr(second_est, name),
                rtol=1e-11,
                atol=1e-11,
            )
        np.testing.assert_allclose(
            first_est._elbo_history, second_est._elbo_history, rtol=1e-12, atol=1e-12
        )
        for name, value in first.model.parameters.items():
            np.testing.assert_allclose(
                value, second.model.parameters[name], atol=1e-11, rtol=1e-11
            )
        assert first.log_likelihood == pytest.approx(second.log_likelihood, abs=1e-10)
        assert first.converged == second.converged
        assert first.n_iterations == second.n_iterations
        if kind == "gvem":
            for name, values in first.standard_errors.items():
                np.testing.assert_allclose(
                    values, second.standard_errors[name], rtol=2e-3, atol=1e-5
                )
        else:
            np.testing.assert_array_equal(
                first.sparsity_pattern, second.sparsity_pattern
            )
            np.testing.assert_allclose(
                first.inclusion_probabilities,
                second.inclusion_probabilities,
                rtol=1e-12,
                atol=1e-12,
            )
    finally:
        mirt.set_backend(previous)
