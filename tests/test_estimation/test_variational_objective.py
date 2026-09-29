"""Scalar and end-to-end checks for the shared variational objective."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
import mirt.estimation._variational_objective as objective
from mirt.estimation.gvem import GVEMEstimator
from mirt.estimation.sparse_bayesian import SparseBayesianEstimator
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic


def _state(n_persons: int, n_items: int, n_factors: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(482)
    responses = rng.integers(0, 2, (n_persons, n_items))
    loadings = rng.normal(scale=0.6, size=(n_items, n_factors))
    intercepts = rng.normal(size=n_items)
    mu = rng.normal(size=(n_persons, n_factors))
    root = rng.normal(scale=0.2, size=(n_persons, n_factors, n_factors))
    sigma = root @ root.swapaxes(1, 2) + 0.4 * np.eye(n_factors)
    xi = rng.uniform(-3.0, 3.0, size=responses.shape)
    prior_mean = rng.normal(size=n_factors)
    root = rng.normal(scale=0.4, size=(n_factors, n_factors))
    prior_cov = root @ root.T + 0.8 * np.eye(n_factors)
    return responses, loadings, intercepts, mu, sigma, xi, prior_mean, prior_cov


def _scalar_elbo(*state: np.ndarray, lambda_value: float | None = None) -> float:
    responses, loadings, intercepts, mu, sigma, xi, prior_mean, prior_cov = state
    value = 0.0
    for person, item in np.argwhere(responses >= 0):
        eta = float(loadings[item] @ mu[person] + intercepts[item])
        variance = float(loadings[item] @ sigma[person] @ loadings[item])
        bound = float(xi[person, item])
        lam = lambda_value
        if lam is None:
            lam = 0.125 if abs(bound) < 1e-6 else math.tanh(bound / 2) / (4 * bound)
        value += (
            -float(np.logaddexp(0.0, -bound))
            + (responses[person, item] - 0.5) * eta
            - 0.5 * bound
            - lam * (variance + eta**2 - bound**2)
        )
    precision = np.linalg.inv(prior_cov)
    log_det_prior = np.linalg.slogdet(prior_cov)[1]
    for mean, cov in zip(mu, sigma, strict=True):
        diff = mean - prior_mean
        value -= 0.5 * (
            diff @ precision @ diff
            + np.trace(precision @ cov)
            + log_det_prior
            - np.linalg.slogdet(cov)[1]
            - len(prior_mean)
        )
    return float(value)


def _readonly_strided(array: np.ndarray) -> np.ndarray:
    storage = np.empty((array.shape[0] * 2, *array.shape[1:]), dtype=array.dtype)
    view = storage[::2]
    view[...] = array
    view.setflags(write=False)
    return view


@pytest.mark.parametrize("n_factors", [1, 3, 8])
@pytest.mark.parametrize("budget", [1, 700, 1_000_000])
@pytest.mark.parametrize("missing", [False, True])
def test_objective_matches_scalar_bound_without_changing_state(
    n_factors: int, budget: int, missing: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(19, 7, n_factors)
    responses, _, _, _, _, xi, _, _ = state
    xi[0, :3] = [0.0, 1e-12, -1e-7]
    if missing:
        responses[::3, 1::2] = -9
        responses[4] = -1
        responses[:, 2] = -1
        xi[responses < 0] = np.nan
    expected = _scalar_elbo(*state)
    original = [array.copy() for array in state]
    state = tuple(_readonly_strided(array) for array in state)
    monkeypatch.setattr(objective, "_MAX_WORKING_ELEMENTS", budget)

    with np.errstate(invalid="ignore"):
        actual = objective.variational_elbo(*state, GVEMEstimator._lambda)

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
    for array, saved in zip(state, original, strict=True):
        assert_array_equal(array, saved)


@pytest.mark.parametrize("kind", ["gvem", "sparse", "fixed_sparse"])
def test_estimators_share_bounded_likelihood_and_kl_with_custom_lambda(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(11, 5, 3)
    responses, loadings, intercepts, mu, sigma, xi, prior_mean, prior_cov = state
    responses[2] = -1
    expected = _scalar_elbo(*state, lambda_value=0.07)
    block_lengths = []
    covariance_lengths = []

    def custom_lambda(block_xi: np.ndarray) -> np.ndarray:
        block_lengths.append(len(block_xi))
        return np.full_like(block_xi, 0.07)

    slogdet = np.linalg.slogdet

    def record_slogdet(matrix: np.ndarray):
        if matrix.ndim == 3:
            covariance_lengths.append(len(matrix))
        return slogdet(matrix)

    monkeypatch.setattr(objective, "_MAX_WORKING_ELEMENTS", 400)
    monkeypatch.setattr(np.linalg, "slogdet", record_slogdet)
    if kind == "gvem":
        estimator = GVEMEstimator(use_gpu=False)
        estimator._slopes = loadings
        estimator._lambda = custom_lambda
    else:
        estimator = SparseBayesianEstimator(k_max=3)
        estimator._loadings = loadings
        estimator._fixed_loadings = kind == "fixed_sparse"
        estimator._lambda_jj = custom_lambda
        if not estimator._fixed_loadings:
            expected += np.sum(estimator._ssl_prior.log_pdf(loadings))
    estimator._intercepts = intercepts
    estimator._mu, estimator._sigma, estimator._xi = mu, sigma, xi

    if kind == "gvem":
        actual = estimator._compute_elbo_python(
            TwoParameterLogistic(5, 3), responses, prior_mean, prior_cov
        )
    else:
        actual = estimator._compute_elbo(responses, prior_mean, prior_cov)

    assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
    assert block_lengths == [4, 4, 3]
    assert covariance_lengths == block_lengths


@pytest.mark.parametrize("n_persons,n_items", [(0, 4), (5, 0), (0, 0)])
def test_empty_axes_preserve_the_gaussian_prior(n_persons: int, n_items: int) -> None:
    state = _state(n_persons, n_items, 3)
    assert_allclose(
        objective.variational_elbo(*state, GVEMEstimator._lambda),
        _scalar_elbo(*state),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("budget", [1, 1_000_000])
def test_large_negative_bound_preserves_floating_point_overflow(
    budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(100, 1, 1)
    state[1][:] = 1e154
    state[3][:] = 0.0
    state[4][:] = 1.0
    state[5][:] = 1.0
    monkeypatch.setattr(objective, "_MAX_WORKING_ELEMENTS", budget)
    with np.errstate(over="ignore", invalid="ignore"):
        assert objective.variational_elbo(*state, GVEMEstimator._lambda) == -np.inf


def test_large_gaussian_kl_scales_terms_before_adding() -> None:
    state = _state(1, 1, 1)
    state[0][:] = -1
    state[1][:] = 0.0
    state[3][:] = 1e154
    state[4][:] = 1e308
    state[6][:] = 0.0
    state[7][:] = 1.0
    with np.errstate(over="raise", invalid="raise"):
        actual = objective.variational_elbo(*state, GVEMEstimator._lambda)
    assert_allclose(actual, -1e308, rtol=1e-15)


@pytest.mark.parametrize(
    "kind,se_step_size", [("gvem", 1e-5), ("gvem", 1e-3), ("sparse", 1e-5)]
)
@pytest.mark.parametrize("n_factors", [1, 3])
def test_complete_fits_agree_with_scalar_objective(
    kind: str, n_factors: int, se_step_size: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = mirt.simdata(
        model="2PL", n_persons=25, n_items=5, n_factors=n_factors, seed=483
    )
    responses[1, 2:] = -1
    responses[4] = -1
    prior_mean = np.linspace(-0.3, 0.5, n_factors)
    prior_cov = np.eye(n_factors) * 1.2 + 0.1
    monkeypatch.setattr(objective, "_MAX_WORKING_ELEMENTS", 1)

    def scalar_objective(*args):
        return _scalar_elbo(*args[:-1])

    results, histories = [], []
    previous_backend = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        for scalar in (False, True):
            if scalar:
                monkeypatch.setattr(
                    f"mirt.estimation.{kind if kind == 'gvem' else 'sparse_bayesian'}.variational_elbo",
                    scalar_objective,
                )
            if kind == "gvem":
                estimator = GVEMEstimator(
                    max_iter=4, tol=1e-10, use_gpu=False, se_step_size=se_step_size
                )
            else:
                estimator = SparseBayesianEstimator(
                    k_max=n_factors, max_iter=4, tol=1e-10
                )
            model_class = (
                OneParameterLogistic if n_factors == 1 else TwoParameterLogistic
            )
            results.append(
                estimator.fit(
                    model_class(5, n_factors),
                    responses,
                    prior_mean=prior_mean,
                    prior_cov=prior_cov,
                )
            )
            histories.append(estimator.elbo_history)

    finally:
        mirt.set_backend(previous_backend)

    actual, expected = results
    assert actual.n_iterations == expected.n_iterations
    assert actual.converged == expected.converged
    assert_allclose(histories[0], histories[1], rtol=1e-13, atol=1e-12)
    assert_allclose(actual.log_likelihood, expected.log_likelihood, rtol=1e-13)
    for name in actual.model.parameters:
        assert_array_equal(
            actual.model.parameters[name], expected.model.parameters[name]
        )
    if kind == "gvem":
        for name in actual.standard_errors:
            if se_step_size == 1e-5:
                assert_array_equal(
                    actual.standard_errors[name], expected.standard_errors[name]
                )
            assert_allclose(
                actual.standard_errors[name], expected.standard_errors[name], rtol=1e-6
            )
    else:
        assert_array_equal(actual.sparsity_pattern, expected.sparsity_pattern)
        assert_array_equal(
            actual.inclusion_probabilities, expected.inclusion_probabilities
        )
