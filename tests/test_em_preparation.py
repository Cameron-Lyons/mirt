"""Numerical and resource contracts for prepared EM fits."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.estimation._em_context import EMFitContext
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.utils.numeric import logsumexp

native = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native backend unavailable"
)


@pytest.fixture(autouse=True)
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


@pytest.mark.parametrize("chunked", [False, True])
def test_expected_counts_match_missing_and_weighted_reference(monkeypatch, chunked):
    import mirt.estimation._em_context as preparation

    rng = np.random.default_rng(34)
    responses = rng.integers(-2, 2, (73, 8))[:, ::2]
    responses[:, -1] = -9
    responses[0] = -1
    weights = rng.random((73, 11)) * rng.integers(1, 10, (73, 1))
    if chunked:
        monkeypatch.setattr(preparation, "_MAX_COUNT_ENTRIES", 40)
    with EMFitContext(responses) as context:
        for posterior in (weights, weights[::-1]):
            actual_r, actual_n = context.expected_counts(posterior)
            expected_r = np.array(
                [posterior[responses[:, j] == 1].sum(axis=0) for j in range(4)]
            )
            expected_n = np.array(
                [posterior[responses[:, j] >= 0].sum(axis=0) for j in range(4)]
            )
            np.testing.assert_allclose(actual_r, expected_r, rtol=1e-13, atol=1e-12)
            np.testing.assert_allclose(actual_n, expected_n, rtol=1e-13, atol=1e-12)
        if chunked:
            assert context._components is None


def test_preparation_preserves_full_width_response_values():
    responses = np.array([[2**40, -1]], dtype=np.int64)
    with EMFitContext(responses, native=True) as context:
        np.testing.assert_array_equal(context.responses, responses)


@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM"])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("rust", marks=native)])
def test_long_test_likelihood_remains_in_log_space(kind, backend):
    mirt.set_backend(backend)
    responses = np.tile([0, 1], (4, 600))
    responses[1::2] = 1 - responses[1::2]
    options = dict(model=kind, n_quadpts=5, max_iter=2, tol=1e-12)
    if kind == "GRM":
        options["n_categories"] = 2
    result = mirt.fit_mirt(responses, compute_standard_errors=False, **options)
    quad = GaussHermiteQuadrature(n_points=5)
    # Evaluate the final parameters independently, without exponentiating the
    # marginal likelihoods. Balanced long tests underflow in probability space.
    log_likes = np.column_stack(
        [result.model.log_likelihood(responses, theta[None, :]) for theta in quad.nodes]
    )
    expected = float(logsumexp(log_likes + np.log(quad.weights), axis=1).sum())
    assert expected < -750 * len(responses)
    assert result.log_likelihood == pytest.approx(expected, abs=1e-8)
    assert result.aic == pytest.approx(-2 * expected + 2 * result.n_parameters)


def test_custom_likelihood_buffer_is_not_mutated():
    model = TwoParameterLogistic(2)
    responses = np.array([[0, 1], [1, 0]])
    cached = np.arange(10, dtype=float).reshape(2, 5) * -0.2
    before = cached.copy()
    model.log_likelihood_batch = lambda *_args: cached
    estimator = EMEstimator(n_quadpts=5, use_gpu=False)
    estimator._quadrature = GaussHermiteQuadrature(n_points=5)
    estimator._latent_density = GaussianDensity()
    posterior, log_marginal = estimator._e_step(model, responses)
    expected = logsumexp(before + np.log(estimator._quadrature.weights), axis=1)
    np.testing.assert_array_equal(cached, before)
    np.testing.assert_allclose(log_marginal, expected)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


@native
def test_native_standard_errors_retain_pattern_compression(monkeypatch):
    import mirt.estimation.standard_errors as standard_errors
    from mirt.estimation.base import _parameter_bounds

    mirt.set_backend("rust")
    pool = np.array([[0, 1, -1], [1, 0, 1], [-1, 0, 0], [0, 1, 1]])
    responses = np.repeat(pool, [100, 200, 300, 400], axis=0)
    before = responses.copy()
    original = standard_errors.estimate_covariance
    calls = []

    def capture(model, data, *args, **kwargs):
        calls.append((data.shape, kwargs["frequencies"].sum()))
        return original(model, data, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(standard_errors, "estimate_covariance", capture)
        result = mirt.fit_mirt(responses, n_quadpts=7, max_iter=3)
    assert calls == [((4, 3), 1000.0)]
    np.testing.assert_array_equal(responses, before)
    quad = GaussHermiteQuadrature(n_points=7)
    expected = original(
        result.model,
        responses,
        quad,
        quad.weights,
        "oakes",
        bounds=lambda name: _parameter_bounds(result.model, name),
    )
    assert result.se_method == "oakes"
    np.testing.assert_allclose(result.vcov, expected.covariance, rtol=1e-9)
    for name, values in expected.standard_errors.items():
        np.testing.assert_allclose(result.standard_errors[name], values, rtol=1e-9)
    assert result.n_observations == 1000


@native
@pytest.mark.parametrize("fail", [False, True])
def test_native_fit_reuses_prepared_responses_and_pool(monkeypatch, fail):
    from mirt.backends.rust._helpers import mirt_rs

    mirt.set_backend("rust")
    original = mirt_rs.m_step_polytomous
    seen = []
    contexts = []

    def capture(*args):
        seen.append((args[0], args[-1]))
        contexts.append(estimator._fit_context)
        result = original(*args)
        if fail:
            raise RuntimeError("fit interrupted")
        return result

    monkeypatch.setattr(mirt_rs, "m_step_polytomous", capture)
    rng = np.random.default_rng(81)
    responses = rng.integers(0, 3, (80, 6))[:, ::2]
    estimator = EMEstimator(
        n_quadpts=7,
        max_iter=3,
        tol=1e-14,
        n_jobs=2,
        use_gpu=False,
        compute_standard_errors=False,
    )
    if fail:
        with pytest.raises(RuntimeError, match="fit interrupted"):
            estimator.fit(GradedResponseModel(3, n_categories=3), responses)
    else:
        result = estimator.fit(GradedResponseModel(3, n_categories=3), responses)
        assert np.isfinite(result.log_likelihood)
        assert len(seen) == 3
    prepared, workers = seen[0]
    assert prepared.dtype == np.int32 and prepared.flags.c_contiguous
    assert workers is not None
    assert all(data is prepared and pool is workers for data, pool in seen)
    assert all(context._native_pool is None for context in contexts)
    assert estimator._fit_context is None

    if not fail:
        # Reusing the estimator for different data must create fresh resources.
        estimator.fit(GradedResponseModel(3, n_categories=3), responses[::-1])
        assert seen[3][0] is not prepared
        assert seen[3][1] is not workers


@pytest.mark.parametrize("fail", [False, True])
def test_python_fit_reuses_and_closes_executor(monkeypatch, fail):
    mirt.set_backend("numpy")
    rng = np.random.default_rng(92)
    responses = rng.integers(0, 2, (80, 3))
    estimator = EMEstimator(
        n_quadpts=5,
        max_iter=3,
        tol=1e-14,
        n_jobs=2,
        use_gpu=False,
        compute_standard_errors=False,
    )
    original = estimator._m_step
    executors = []

    def capture(*args):
        original(*args)
        executors.append(estimator._fit_context.executor(2))
        if fail:
            raise RuntimeError("fit interrupted")

    monkeypatch.setattr(estimator, "_m_step", capture)
    if fail:
        with pytest.raises(RuntimeError, match="fit interrupted"):
            estimator.fit(TwoParameterLogistic(3), responses)
    else:
        estimator.fit(TwoParameterLogistic(3), responses)
        assert len(executors) == 3
    assert all(executor is executors[0] for executor in executors)
    with pytest.raises(RuntimeError, match="shutdown"):
        executors[0].submit(lambda: None)
    assert estimator._fit_context is None
