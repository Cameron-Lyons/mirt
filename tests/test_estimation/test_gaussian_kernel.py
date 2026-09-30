"""Gaussian prior arithmetic, buffer ownership, and posterior sampling."""

import tracemalloc
from types import MethodType

import numpy as np
import pytest

from mirt.estimation import _gaussian_kernel as kernel_module
from mirt.estimation.mcem import (
    MCEMEstimator,
    QMCEMEstimator,
    StochasticEMEstimator,
    _validated_prior,
)
from mirt.models.dichotomous import TwoParameterLogistic


def _reference(theta, mean, factor):
    centered = np.asarray(theta, dtype=np.float64) - mean
    standardized = np.linalg.solve(factor, centered.reshape(-1, len(mean)).T).T
    return (-0.5 * np.sum(standardized**2, axis=1)).reshape(centered.shape[:-1])


def _factor(kind, dimensions=3):
    if kind == "identity":
        return np.eye(dimensions)
    if kind == "diagonal":
        return np.diag(np.linspace(0.7, 1.5, dimensions))
    matrix = np.eye(dimensions) + 0.1 * np.ones((dimensions, dimensions))
    factor = np.linalg.cholesky(matrix)
    if kind == "general":
        factor[0, -1] = 0.2
    return factor


@pytest.mark.parametrize("kind", ["identity", "diagonal", "correlated", "general"])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        "vector",
        "strided_vector",
        "matrix",
        "tensor",
        "strided",
        "reversed",
        "four_dims",
        "float32",
    ],
)
def test_gaussian_kernel_matches_full_solve_and_preserves_inputs(
    kind, blocked, layout, monkeypatch
):
    rng = np.random.default_rng(901)
    mean, factor = np.array([0.3, -0.2, 0.5]), _factor(kind)
    theta = rng.normal(size=(11, 7, 3))
    if layout == "vector":
        theta = theta[0, 0].copy()
    elif layout == "strided_vector":
        theta = theta[0, 0, ::-1]
    elif layout == "matrix":
        theta = theta.reshape(-1, 3)
    elif layout == "strided":
        storage = np.zeros((11, 14, 3))
        storage[:, ::2] = theta
        theta = storage[:, ::2]
    elif layout == "reversed":
        theta = theta[::-1, ::-1, ::-1]
    elif layout == "four_dims":
        theta = rng.normal(size=(3, 5, 7, 3))
    elif layout == "float32":
        theta = theta.astype(np.float32)
    expected = _reference(theta, mean, factor)
    originals = [value.copy() for value in (theta, mean, factor)]
    for value in (theta, mean, factor):
        value.setflags(write=False)
    if blocked:
        monkeypatch.setattr(kernel_module, "_MAX_GAUSSIAN_KERNEL_ELEMENTS", 17)
    actual = MCEMEstimator._gaussian_log_kernel(theta, mean, factor)
    assert actual.shape == theta.shape[:-1] and actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    for value, original in zip((theta, mean, factor), originals, strict=True):
        np.testing.assert_array_equal(value, original)
        assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("kind", ["identity", "diagonal", "correlated"])
def test_representable_tail_kernel_does_not_overflow_square_or_sum(kind):
    factor = _factor(kind, 4)
    standardized = np.array([[1.5e154, 0, 0, 0], [7.5e153] * 4])
    theta = standardized @ factor.T
    actual = MCEMEstimator._gaussian_log_kernel(theta, np.zeros(4), factor)
    expected = np.asarray(
        -0.5 * np.sum(standardized.astype(np.longdouble) ** 2, axis=1), dtype=np.float64
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("kind", ["identity", "correlated"])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_kernel_peak_excludes_full_sample_conversion_and_square(kind, layout):
    rng = np.random.default_rng(905)
    theta = rng.normal(size=(4000, 64, 12)).astype(np.float32)
    if layout == "strided":
        storage = np.empty((4000, 128, 12), dtype=np.float32)
        storage[:, ::2] = theta
        theta = storage[:, ::2]
    theta.setflags(write=False)
    mean, factor = np.zeros(12), _factor(kind, 12)
    tracemalloc.start()
    try:
        actual = MCEMEstimator._gaussian_log_kernel(theta, mean, factor)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.isfinite(actual).all()
    assert peak < actual.nbytes + 6 * kernel_module._MAX_GAUSSIAN_KERNEL_ELEMENTS * 8


def test_tiny_prior_terms_retain_the_standard_reduction_precision():
    theta = np.array([[3e-162], [5e-162], [0.0]])
    expected = -0.5 * np.sum(theta**2, axis=-1)
    actual = MCEMEstimator._gaussian_log_kernel(theta, np.zeros(1), np.eye(1))
    assert actual[0] != 0.0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["diagonal", "correlated"])
def test_large_covariance_symmetrization_keeps_a_finite_factor(kind):
    covariance = np.array([[1e308, 0.25e308], [0.25e308, 1e308]])
    if kind == "diagonal":
        covariance[0, 1] = covariance[1, 0] = 0
    covariance.setflags(write=False)
    original = covariance.copy()
    _, factor = _validated_prior(None, covariance, 2)
    assert np.isfinite(factor).all()
    np.testing.assert_allclose(factor @ factor.T, original, rtol=1e-14)
    np.testing.assert_array_equal(covariance, original)


def test_subnormal_covariance_symmetrization_retains_small_variances():
    tiny = np.nextafter(0.0, 1.0)
    covariance = np.array([[4 * tiny, tiny], [tiny, 4 * tiny]])
    _, factor = _validated_prior(None, covariance, 2)
    assert np.isfinite(factor).all() and np.all(np.diag(factor) > 0)
    np.testing.assert_array_equal(factor @ factor.T, covariance)


@pytest.mark.parametrize("method", ["importance", "posterior", "stochastic", "quasi"])
def test_complete_fit_accepts_a_finite_large_prior_covariance(method, monkeypatch):
    model = TwoParameterLogistic(1)
    responses = np.full((17, 1), -1)
    if method == "stochastic":
        estimator = StochasticEMEstimator(n_chains=5, max_iter=1, seed=907)
    elif method == "quasi":
        estimator = QMCEMEstimator(n_samples=64, max_iter=1, seed=907)
    else:
        estimator = MCEMEstimator(
            n_samples=50,
            max_iter=1,
            seed=907,
            importance_sampling=method == "importance",
        )
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *args: None)
    result = estimator.fit(model, responses, prior_cov=np.array([[1e308]]))
    assert result.log_likelihood == 0.0 and np.isfinite(result.aic)


@pytest.mark.parametrize("method", ["posterior", "stochastic"])
@pytest.mark.parametrize("offset", [-1e300, 1e300, -1e16, 1e16])
def test_large_common_likelihood_offsets_do_not_change_posterior_draws(method, offset):
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.full((97, 1), -1)
    cls = StochasticEMEstimator if method == "stochastic" else MCEMEstimator
    options = (
        {"n_chains": 5}
        if method == "stochastic"
        else {"n_samples": 50, "importance_sampling": False}
    )
    shifted, zero = cls(**options, seed=911), cls(**options, seed=911)
    cached = np.full((97, shifted.n_samples), offset)
    cached.setflags(write=False)
    shifted._sample_log_likelihoods = lambda *args: cached
    zero._sample_log_likelihoods = lambda *args: np.zeros(cached.shape)
    shifted._rng = np.random.default_rng(911)
    zero._rng = np.random.default_rng(911)
    arguments = (model, responses, np.array([0.3, -0.4]), _factor("correlated", 2), 2)
    samples, weights = shifted._e_step_mc(*arguments)
    expected_samples, expected_weights = zero._e_step_mc(*arguments)
    np.testing.assert_array_equal(samples, expected_samples)
    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(cached, np.full(cached.shape, offset))


@pytest.mark.parametrize("kind", ["identity", "diagonal", "correlated"])
@pytest.mark.parametrize("method", ["posterior", "stochastic"])
def test_seeded_sampler_matches_public_likelihood_and_full_prior_solve(kind, method):
    model = TwoParameterLogistic(3, n_factors=3)
    responses = np.random.default_rng(912).integers(-1, 2, (17, 3))
    if method == "posterior":
        estimator = MCEMEstimator(n_samples=50, importance_sampling=False, seed=912)
    else:
        estimator = StochasticEMEstimator(n_chains=5, seed=912)
    mean, factor = np.array([0.3, -0.4, 0.2]), _factor(kind)
    rng = np.random.default_rng(912)
    shape = (17, estimator.n_samples, 3)
    expected = mean + np.einsum("ij,...j->...i", factor, rng.standard_normal(shape))

    def likelihood(points):
        expanded = np.repeat(responses, estimator.n_samples, axis=0)
        return model.log_likelihood(expanded, points.reshape(-1, 3)).reshape(shape[:2])

    current_likelihood = likelihood(expected)
    current_prior = _reference(expected, mean, factor)
    for _ in range(20):
        proposal = expected + 0.5 * np.einsum(
            "ij,...j->...i", factor, rng.standard_normal(shape)
        )
        proposed_likelihood = likelihood(proposal)
        proposed_prior = _reference(proposal, mean, factor)
        ratio = (proposed_likelihood + proposed_prior) - (
            current_likelihood + current_prior
        )
        accepted = (
            np.log(np.maximum(rng.random(shape[:2]), np.nextafter(0.0, 1.0))) < ratio
        )
        expected[accepted] = proposal[accepted]
        current_likelihood[accepted] = proposed_likelihood[accepted]
        current_prior[accepted] = proposed_prior[accepted]
    estimator._rng = np.random.default_rng(912)
    actual, weights = estimator._e_step_mc(model, responses, mean, factor, 3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(weights, 1.0 / estimator.n_samples)


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
@pytest.mark.parametrize("callback", ["likelihood", "prior", "both"])
def test_sampler_preserves_cached_read_only_callback_buffers(
    override, callback, monkeypatch
):
    estimator = StochasticEMEstimator(n_chains=5, seed=913)
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.zeros((17, 1), dtype=np.int64)
    cached, originals, calls = [], [], {"likelihood": 0, "prior": 0}

    def likelihood(self, model, responses, theta):
        calls["likelihood"] += 1
        value = -0.25 * np.sum(theta**2, axis=-1)
        originals.append(value.copy())
        value.setflags(write=False)
        cached.append(value)
        return value

    def prior(theta, mean, factor):
        calls["prior"] += 1
        value = _reference(theta, mean, factor)
        originals.append(value.copy())
        value.setflags(write=False)
        cached.append(value)
        return value

    methods = {}
    if callback in ("likelihood", "both"):
        methods["_sample_log_likelihoods"] = likelihood
    if callback in ("prior", "both"):
        methods["_gaussian_log_kernel"] = staticmethod(prior)
    if override == "subclass":
        estimator = type("CustomSampler", (StochasticEMEstimator,), methods)(
            n_chains=5, seed=913
        )
    else:
        for name, method in methods.items():
            if override == "class":
                monkeypatch.setattr(MCEMEstimator, name, method)
            elif name == "_gaussian_log_kernel":
                setattr(estimator, name, prior)
            else:
                setattr(estimator, name, MethodType(likelihood, estimator))
    estimator._rng = np.random.default_rng(913)
    samples, weights = estimator._e_step_mc(model, responses, np.zeros(2), np.eye(2), 2)
    assert np.isfinite(samples).all() and samples.flags.writeable
    np.testing.assert_array_equal(weights, np.full((17, 5), 0.2))
    for name in ("likelihood", "prior"):
        assert calls[name] == (21 if callback in (name, "both") else 0)
    for value, original in zip(cached, originals, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("method", ["posterior", "stochastic"])
@pytest.mark.parametrize("read_only", [False, True])
def test_sampler_snapshots_likelihood_before_prior_reuses_callback_buffer(
    method, read_only
):
    model = TwoParameterLogistic(1, n_factors=2)
    responses = np.zeros((97, 1), dtype=np.int64)

    def draw(shared):
        estimator = (
            StochasticEMEstimator(n_chains=5, seed=925)
            if method == "stochastic"
            else MCEMEstimator(n_samples=50, importance_sampling=False, seed=925)
        )
        scratch = np.empty((97, estimator.n_samples))
        calls = {"likelihood": 0, "prior": 0}

        def publish(value):
            if shared:
                scratch.setflags(write=True)
                np.copyto(scratch, value)
                value = scratch
            value.setflags(write=not read_only)
            return value

        def likelihood(model, responses, theta):
            calls["likelihood"] += 1
            return publish(-0.25 * np.sum(theta**2, axis=-1))

        def prior(theta, mean, factor):
            calls["prior"] += 1
            return publish(-0.5 * np.sum(theta**2, axis=-1))

        estimator._sample_log_likelihoods = likelihood
        estimator._gaussian_log_kernel = prior
        estimator._rng = np.random.default_rng(925)
        result = estimator._e_step_mc(model, responses, np.zeros(2), np.eye(2), 2)
        assert calls == {"likelihood": 21, "prior": 21}
        return result

    actual_samples, actual_weights = draw(shared=True)
    expected_samples, expected_weights = draw(shared=False)
    np.testing.assert_array_equal(actual_samples, expected_samples)
    np.testing.assert_array_equal(actual_weights, expected_weights)


def test_sampler_peak_excludes_retained_normals_and_accepted_draw_copies():
    estimator = MCEMEstimator(n_samples=64, importance_sampling=False, seed=917)
    model = TwoParameterLogistic(1, n_factors=8)
    responses = np.full((1000, 1), -1)
    cached = np.zeros((1000, 64))
    cached.setflags(write=False)
    estimator._sample_log_likelihoods = lambda *args: cached
    estimator._rng = np.random.default_rng(917)
    tracemalloc.start()
    try:
        samples, weights = estimator._e_step_mc(
            model, responses, np.zeros(8), np.eye(8), 8
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.isfinite(samples).all()
    assert peak < 4 * samples.nbytes
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize("callback", ["likelihood", "prior"])
@pytest.mark.parametrize("phase", ["initial", "proposal"])
def test_sampler_validates_callback_shapes_before_updating_chain(callback, phase):
    estimator = StochasticEMEstimator(n_chains=5, seed=919)
    model = TwoParameterLogistic(1)
    responses = np.zeros((17, 1), dtype=np.int64)
    calls = 0
    bad = np.zeros((17, 4))
    bad.setflags(write=False)

    def values(*args):
        nonlocal calls
        calls += 1
        return bad if calls == (1 if phase == "initial" else 2) else np.zeros((17, 5))

    setattr(
        estimator,
        "_sample_log_likelihoods"
        if callback == "likelihood"
        else "_gaussian_log_kernel",
        values,
    )
    estimator._rng = np.random.default_rng(919)
    with pytest.raises(ValueError, match="shape"):
        estimator._e_step_mc(model, responses, np.zeros(1), np.eye(1), 1)
    np.testing.assert_array_equal(bad, np.zeros((17, 4)))


@pytest.mark.parametrize("invalid", ["mean_shape", "factor_shape", "singular"])
def test_invalid_prior_kernel_dimensions_and_singular_factors_are_rejected(invalid):
    theta, mean, factor = np.zeros((2, 5, 2)), np.zeros(2), np.eye(2)
    if invalid == "mean_shape":
        mean = np.zeros(3)
    elif invalid == "factor_shape":
        factor = np.eye(3)
    else:
        factor[0, 0] = 0
    error = np.linalg.LinAlgError if invalid == "singular" else ValueError
    with pytest.raises(error):
        MCEMEstimator._gaussian_log_kernel(theta, mean, factor)
