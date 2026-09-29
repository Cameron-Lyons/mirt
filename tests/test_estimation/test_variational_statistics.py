"""Independent itemwise checks for the shared variational M-step reductions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
import mirt.estimation._variational_statistics as statistics
from mirt.constants import PROB_EPSILON, REGULARIZATION_EPSILON
from mirt.estimation.gvem import GVEMEstimator
from mirt.estimation.sparse_bayesian import SparseBayesianEstimator
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic


def _strided_readonly(array):
    storage = np.empty((2 * len(array), *array.shape[1:]), dtype=array.dtype)
    view = storage[::2]
    view[...] = array
    view.setflags(write=False)
    return view


def _estimator(kind, n_factors, fixed, missing=True):
    rng = np.random.default_rng(942)
    responses = rng.integers(0, 2, (17, 6))
    if missing:
        responses[rng.random(responses.shape) < 0.2] = -9
        responses[3] = -1
        responses[:, 5] = -1
    model = (OneParameterLogistic if fixed else TwoParameterLogistic)(6, n_factors)
    estimator = (
        GVEMEstimator(use_gpu=False)
        if kind == "gvem"
        else SparseBayesianEstimator(k_max=n_factors)
    )
    loadings = rng.normal(scale=0.6, size=(6, n_factors))
    if kind == "gvem":
        estimator._slopes = loadings
    else:
        estimator._loadings = loadings
        estimator._gamma = np.full_like(loadings, 0.5)
        estimator._fixed_loadings = fixed
    estimator._intercepts = rng.normal(scale=0.5, size=6)
    estimator._mu = _strided_readonly(rng.normal(size=(17, n_factors)))
    root = rng.normal(scale=0.4, size=(17, n_factors, n_factors))
    estimator._sigma = _strided_readonly(
        root @ root.swapaxes(1, 2) + np.eye(n_factors) * 0.2
    )
    xi = rng.uniform(-3.0, 3.0, responses.shape)
    xi[0, :3] = [0.0, 1e-12, -1e-7]
    xi[responses < 0] = np.nan
    estimator._xi = _strided_readonly(xi)
    return estimator, model, _strided_readonly(responses)


def _reference_update(estimator, model, responses):
    """Accumulate each scalar observation and update each item independently."""
    is_gvem = isinstance(estimator, GVEMEstimator)
    fixed = model.model_name == "1PL"
    loadings = (estimator._slopes if is_gvem else estimator._loadings).copy()
    intercepts = estimator._intercepts.copy()
    gamma = None if is_gvem else estimator._gamma.copy()
    lambda_function = estimator._lambda if is_gvem else estimator._lambda_jj
    for item in range(model.n_items):
        observed = np.flatnonzero(responses[:, item] >= 0)
        if is_gvem and not len(observed):
            continue
        lambdas = lambda_function(estimator._xi[observed, item])
        if fixed:
            loadings[item] = 1.0
            if gamma is not None:
                gamma[item] = 1.0
        else:
            curvature = REGULARIZATION_EPSILON * np.eye(model.n_factors)
            score = np.zeros(model.n_factors)
            for person, lam in zip(observed, lambdas, strict=True):
                mu = estimator._mu[person]
                curvature += 2 * lam * (estimator._sigma[person] + np.outer(mu, mu))
                score += (
                    responses[person, item] - 0.5 - 2 * lam * intercepts[item]
                ) * mu
            try:
                candidate = np.linalg.solve(curvature, score)
            except np.linalg.LinAlgError:
                if not is_gvem:
                    raise
                candidate = np.linalg.lstsq(curvature, score, rcond=None)[0]
            if is_gvem:
                loadings[item] = candidate
            else:
                gamma[item] = estimator._ssl_prior.compute_posterior_inclusion(
                    candidate
                )
                penalty = estimator._ssl_prior.compute_effective_penalty(gamma[item])
                threshold = penalty / (np.diag(curvature) + PROB_EPSILON)
                loadings[item] = estimator._ssl_prior.soft_threshold(
                    candidate, threshold
                )
        numerator, denominator = 0.0, 0.0
        for person, lam in zip(observed, lambdas, strict=True):
            numerator += (
                responses[person, item]
                - 0.5
                - 2 * lam * (estimator._mu[person] @ loadings[item])
            )
            denominator += 2 * lam
        if denominator > PROB_EPSILON:
            intercepts[item] = numerator / denominator
        elif is_gvem:
            intercepts[item] = 0.0
        intercepts[item] = np.clip(intercepts[item], -10.0, 10.0)
    return loadings, intercepts, gamma


def _update(estimator, model, responses):
    if isinstance(estimator, GVEMEstimator):
        estimator._m_step_python(model, responses)
    else:
        estimator._m_step_ssl(responses)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("n_factors,fixed", [(1, False), (3, False), (1, True)])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("budget", [1, 500, 1_000_000])
def test_m_step_matches_scalar_updates_and_preserves_variational_inputs(
    kind, n_factors, fixed, missing, budget, monkeypatch
):
    estimator, model, responses = _estimator(kind, n_factors, fixed, missing)
    expected_loadings, expected_intercepts, expected_gamma = _reference_update(
        estimator, model, responses
    )
    saved = [
        array.copy()
        for array in (responses, estimator._mu, estimator._sigma, estimator._xi)
    ]
    slopes = getattr(estimator, "_slopes", None)
    intercepts = estimator._intercepts
    monkeypatch.setattr(statistics, "_MAX_WORKING_ELEMENTS", budget)

    _update(estimator, model, responses)

    actual_loadings = estimator._slopes if kind == "gvem" else estimator._loadings
    assert_allclose(actual_loadings, expected_loadings, rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._intercepts, expected_intercepts, rtol=1e-12, atol=1e-12)
    if kind == "gvem":
        assert estimator._slopes is slopes
        assert estimator._intercepts is intercepts
    else:
        assert_allclose(estimator._gamma, expected_gamma, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(
        (responses, estimator._mu, estimator._sigma, estimator._xi), saved, strict=True
    ):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("fixed", [False, True])
def test_all_missing_data_retains_each_estimators_item_policy(kind, fixed):
    estimator, model, original_responses = _estimator(kind, 1, fixed)
    responses = np.full_like(original_responses, -1)
    expected = _reference_update(estimator, model, responses)

    _update(estimator, model, responses)

    actual_loadings = estimator._slopes if kind == "gvem" else estimator._loadings
    assert_array_equal(actual_loadings, expected[0])
    assert_array_equal(estimator._intercepts, expected[1])
    if kind == "sparse":
        assert_array_equal(estimator._gamma, expected[2])


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("fixed", [False, True])
def test_small_curvature_retains_each_estimators_intercept_policy(kind, fixed):
    estimator, model, responses = _estimator(kind, 1, fixed)
    estimator._intercepts[:] = 0.75
    lambda_name = "_lambda" if kind == "gvem" else "_lambda_jj"
    setattr(estimator, lambda_name, lambda xi: np.full_like(xi, 1e-100))

    _update(estimator, model, responses)

    expected = np.full(6, 0.75)
    if kind == "gvem":
        expected[:5] = 0.0
    assert_array_equal(estimator._intercepts, expected)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("n_factors,fixed", [(1, True), (3, False)])
def test_updates_bound_lambda_calls_and_fixed_updates_skip_covariance(
    kind, n_factors, fixed, monkeypatch
):
    estimator, model, responses = _estimator(kind, n_factors, fixed)
    calls = []

    def constant_lambda(xi):
        calls.append(len(xi))
        return np.full_like(xi, 0.1)

    setattr(estimator, "_lambda" if kind == "gvem" else "_lambda_jj", constant_lambda)
    expected = _reference_update(estimator, model, responses)
    calls.clear()
    # Fixed updates do not need access to posterior covariance matrices.
    if fixed:
        estimator._sigma = None
    monkeypatch.setattr(statistics, "_MAX_WORKING_ELEMENTS", 300 if fixed else 400)

    _update(estimator, model, responses)

    assert sum(calls) == len(responses)
    assert max(calls) <= 5
    actual_loadings = estimator._slopes if kind == "gvem" else estimator._loadings
    assert_allclose(actual_loadings, expected[0], rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._intercepts, expected[1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("mean", [-100.0, 100.0])
def test_intercept_clipping_preserves_missing_item_policy(kind, mean):
    estimator, model, responses = _estimator(kind, 1, True)
    estimator._mu = np.full_like(estimator._mu, mean)
    estimator._intercepts[:] = 12.0

    _update(estimator, model, responses)

    expected = np.full(6, -np.sign(mean) * 10.0)
    expected[5] = 12.0 if kind == "gvem" else 10.0
    assert_array_equal(estimator._intercepts, expected)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
def test_missing_item_intercept_does_not_contaminate_loading_statistics(kind):
    estimator, model, responses = _estimator(kind, 3, False)
    estimator._intercepts[5] = np.inf
    estimator._xi = np.ones_like(estimator._xi)
    expected = _reference_update(estimator, model, responses)

    with np.errstate(invalid="raise"):
        _update(estimator, model, responses)

    actual_loadings = estimator._slopes if kind == "gvem" else estimator._loadings
    assert_allclose(actual_loadings, expected[0], rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._intercepts, expected[1], rtol=1e-12, atol=1e-12)


def test_gvem_retains_itemwise_least_squares_for_singular_curvature(monkeypatch):
    estimator = GVEMEstimator(use_gpu=False)
    estimator._slopes = np.ones((2, 2))
    estimator._intercepts = np.zeros(2)
    estimator._mu = np.array([[0.0, 0.0], [1.0, 0.0]])
    estimator._sigma = np.array([-4 * REGULARIZATION_EPSILON * np.eye(2), np.eye(2)])
    estimator._xi = np.zeros((2, 2))
    responses = np.array([[1, -1], [-1, 1]])
    model = TwoParameterLogistic(2, 2)
    expected = _reference_update(estimator, model, responses)
    original_lstsq = np.linalg.lstsq
    calls = []

    def record_lstsq(matrix, rhs, rcond):
        calls.append(matrix.shape)
        return original_lstsq(matrix, rhs, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", record_lstsq)
    estimator._m_step_python(model, responses)

    assert calls == [(2, 2)]
    assert_allclose(estimator._slopes, expected[0], rtol=1e-12, atol=1e-12)
    assert_allclose(estimator._intercepts, expected[1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["gvem", "sparse"])
@pytest.mark.parametrize("n_factors,fixed", [(1, True), (3, False)])
def test_complete_fits_match_itemwise_m_step(kind, n_factors, fixed, monkeypatch):
    responses = mirt.simdata(n_persons=40, n_items=6, n_factors=n_factors, seed=947)
    responses[2] = -1
    responses[::3, 1] = -1
    model_class = OneParameterLogistic if fixed else TwoParameterLogistic
    monkeypatch.setattr(statistics, "_MAX_WORKING_ELEMENTS", 1)
    previous_backend = mirt.get_backend()
    mirt.set_backend("numpy")
    fitted = []
    try:
        for reference in (False, True):
            model = model_class(6, n_factors)
            if kind == "gvem":
                estimator = GVEMEstimator(max_iter=5, tol=1e-10, use_gpu=False)
                if reference:

                    def scalar_step(model, responses):
                        loadings, intercepts, _ = _reference_update(
                            estimator, model, responses
                        )
                        estimator._slopes[:] = loadings
                        estimator._intercepts[:] = intercepts

                    estimator._m_step_python = scalar_step
            else:
                estimator = SparseBayesianEstimator(
                    k_max=n_factors, max_iter=5, tol=1e-10
                )
                if reference:

                    def scalar_step(responses):
                        estimator._loadings, estimator._intercepts, estimator._gamma = (
                            _reference_update(estimator, model, responses)
                        )

                    estimator._m_step_ssl = scalar_step
            result = estimator.fit(model, responses)
            fitted.append((estimator, result))
    finally:
        mirt.set_backend(previous_backend)

    (actual_estimator, actual), (expected_estimator, expected) = fitted
    assert actual.n_iterations == expected.n_iterations
    assert actual.converged == expected.converged
    assert_allclose(
        actual_estimator.elbo_history, expected_estimator.elbo_history, rtol=1e-12
    )
    for name in actual.model.parameters:
        assert_allclose(
            actual.model.parameters[name],
            expected.model.parameters[name],
            rtol=1e-10,
            atol=1e-12,
        )
    if kind == "gvem":
        for name in actual.standard_errors:
            assert_allclose(
                actual.standard_errors[name],
                expected.standard_errors[name],
                rtol=1e-10,
                atol=1e-12,
            )
    else:
        assert_array_equal(actual.sparsity_pattern, expected.sparsity_pattern)
        assert_allclose(
            actual.inclusion_probabilities,
            expected.inclusion_probabilities,
            rtol=1e-12,
            atol=1e-12,
        )
