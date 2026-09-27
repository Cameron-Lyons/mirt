"""Numerical and bounded-memory contracts for posterior information criteria."""

import concurrent.futures
import tracemalloc

import numpy as np
import pytest
from scipy.special import logsumexp

from mirt.diagnostics import bayesian


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("block_columns", [1, 4, 100])
@pytest.mark.parametrize("shift", [0.0, 1000.0, 1e6])
def test_waic_matches_direct_reference(monkeypatch, layout, block_columns, shift):
    rng = np.random.default_rng(451)
    values = rng.normal(-2, 0.7, (127, 17)) - np.linspace(shift, shift * 2, 17)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[::-2, ::-2]
    values.setflags(write=False)
    original = values.copy()
    n_samples, n_obs = values.shape
    monkeypatch.setattr(
        bayesian, "_LOG_LIKELIHOOD_CHUNK_ELEMENTS", n_samples * block_columns
    )
    expected_log_mean = logsumexp(values, axis=0) - np.log(n_samples)
    expected_variance = np.var(values, axis=0, ddof=1)
    expected_pointwise = -2 * (expected_log_mean - expected_variance)
    result = bayesian.waic(values)
    np.testing.assert_allclose(
        result.pointwise, expected_pointwise, rtol=1e-13, atol=1e-9
    )
    assert result.p_waic == pytest.approx(expected_variance.sum(), rel=1e-12)
    assert result.elpd_waic == pytest.approx(
        expected_log_mean.sum() - expected_variance.sum()
    )
    assert result.waic == pytest.approx(-2 * result.elpd_waic)
    assert result.se_waic == pytest.approx(
        np.sqrt(n_obs * np.var(expected_pointwise, ddof=1))
    )
    np.testing.assert_array_equal(values, original)


def test_variance_preserves_tiny_changes_around_large_offsets(monkeypatch):
    values = np.array([0, 1, -1, 2, -2], dtype=np.float64)[:, None] * 2**-10
    shifted = values + 2**40
    monkeypatch.setattr(bayesian, "_LOG_LIKELIHOOD_CHUNK_ELEMENTS", 3)
    assert bayesian.waic(shifted).p_waic == pytest.approx(
        bayesian.waic(values).p_waic, rel=1e-14
    )


def test_constant_large_log_likelihood_has_zero_variance():
    values = np.full((31, 3), -1e200)
    result = bayesian.waic(values)
    assert result.p_waic == 0.0
    np.testing.assert_array_equal(result.pointwise, np.full(3, 2e200))


@pytest.mark.parametrize("n_samples", [2, 5, 31, 1000])
@pytest.mark.parametrize("kind", ["normal", "ties", "constant", "heavy_tail"])
@pytest.mark.parametrize("efficiency", [0.001, 1.0, 10000.0])
def test_partition_cutoff_matches_full_sort(monkeypatch, n_samples, kind, efficiency):
    rng = np.random.default_rng(8912)
    values = rng.normal(size=n_samples)
    if kind == "ties":
        values = np.round(values)
    elif kind == "constant":
        values.fill(13.0)
    elif kind == "heavy_tail":
        values = np.log1p(rng.pareto(1.3, n_samples))
    values = values[::-1]
    values.setflags(write=False)
    original = values.copy()
    actual, actual_shape = bayesian._pareto_smooth_log_weights(values, efficiency)
    # Full sorting is an independent order-statistic reference, including ties
    # at the cutoff; only the selected upper tail should require ordering.
    monkeypatch.setattr(np, "partition", lambda data, kth: np.sort(data))
    expected, expected_shape = bayesian._pareto_smooth_log_weights(values, efficiency)
    np.testing.assert_array_equal(actual, expected)
    assert actual_shape == expected_shape
    assert np.exp(actual).sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(values, original)


def test_extreme_log_span_remains_stable_without_variance():
    values = np.array([[-1e308, 1e308], [1e308, -1e308]])
    actual, variance = bayesian._log_predictive_moments(values)
    with np.errstate(over="ignore"):
        expected = logsumexp(values, axis=0) - np.log(2)
    np.testing.assert_array_equal(actual, expected)
    assert variance is None


def test_extreme_finite_span_reports_infinite_rather_than_nan_variance():
    values = np.array([[-1e308], [1e308]])
    with np.errstate(over="ignore"):
        result = bayesian.waic(values)
    assert result.p_waic == np.inf
    assert result.se_waic == 0.0


@pytest.mark.parametrize("function", [bayesian.waic, bayesian.psis_loo])
def test_invalid_final_block_is_rejected(monkeypatch, function):
    values = np.zeros((11, 7))
    values[-1, -1] = np.nan
    monkeypatch.setattr(bayesian, "_LOG_LIKELIHOOD_CHUNK_ELEMENTS", 11)
    with pytest.raises(ValueError, match="only finite"):
        function(values)


def test_parallel_submission_is_bounded_and_matches_serial(monkeypatch):
    values = np.random.default_rng(21).normal(-2, 0.7, (100, 137))
    expected = bayesian.psis_loo(values)
    submitted = []

    class ImmediateExecutor:
        def __init__(self, max_workers):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, observations):
            observations = list(observations)
            assert len(observations) <= 12
            submitted.extend(observations)
            return map(function, observations)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor)
    actual = bayesian.psis_loo(values, n_jobs=3)
    assert submitted == list(range(137))
    np.testing.assert_array_equal(actual.pointwise, expected.pointwise)
    np.testing.assert_array_equal(actual.pareto_k, expected.pareto_k)
    assert actual.p_loo == expected.p_loo


@pytest.mark.parametrize("function", [bayesian.waic, bayesian.psis_loo])
def test_temporary_memory_is_bounded_across_observations(monkeypatch, function):
    monkeypatch.setattr(bayesian, "_LOG_LIKELIHOOD_CHUNK_ELEMENTS", 4096)
    peaks = []
    for n_obs in (16, 1024):
        values = np.full((200, n_obs), -1.0)
        function(values)
        tracemalloc.start()
        try:
            function(values)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] < peaks[0] + 512 * 1024
