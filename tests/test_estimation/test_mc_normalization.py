"""Stable Monte Carlo importance weights preserve borrowed likelihood arrays."""

import tracemalloc

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt.estimation._posterior as posterior_module
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator
from mirt.models.dichotomous import TwoParameterLogistic


def _input(kind, *, offset=None):
    values = (
        np.random.default_rng(825).normal(-500, 150, (7, 50))
        if offset is None
        else np.full((7, 50), offset)
    )
    if kind == "float32":
        values = values.astype(np.float32)
    elif kind == "view":
        values = np.broadcast_to(values[:1], values.shape)
    elif kind == "list":
        return values.tolist()
    if kind != "ordinary":
        values.setflags(write=False)
    return values


@pytest.mark.parametrize("kind", ["ordinary", "readonly", "float32", "view", "list"])
@pytest.mark.parametrize("blocked", [False, True])
def test_normalized_weights_match_independent_reference_and_protect_input(
    kind, blocked, monkeypatch
):
    values = _input(kind)
    original = np.array(values, dtype=np.float64)
    expected_log = logsumexp(original, axis=1, keepdims=True)
    expected = np.exp(original - expected_log)
    if blocked:
        monkeypatch.setattr(posterior_module, "_MAX_NORMALIZATION_ELEMENTS", 13)
    weights, normalizer = MCEMEstimator._normalized_importance_weights(values)
    assert weights.dtype == normalizer.dtype == np.float64
    assert normalizer.shape == (len(weights), 1)
    np.testing.assert_allclose(weights, expected, rtol=8e-14, atol=1e-15)
    np.testing.assert_allclose(normalizer, expected_log, atol=1e-13)
    np.testing.assert_array_equal(values, original)
    if isinstance(values, np.ndarray):
        assert not np.shares_memory(weights, values)


@pytest.mark.parametrize("offset", [-1e16, 1e16, -1e300, 1e300])
@pytest.mark.parametrize("kind", ["ordinary", "readonly", "view", "list"])
def test_huge_common_offsets_still_produce_unit_posterior_rows(offset, kind):
    values = _input(kind, offset=offset)
    weights, normalizer = MCEMEstimator._normalized_importance_weights(values)
    np.testing.assert_allclose(weights, 1.0 / 50.0, atol=0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(normalizer, np.full((7, 1), offset + np.log(50)))
    np.testing.assert_array_equal(values, np.full((7, 50), offset))


def test_nonfinite_normalization_keeps_legacy_undefined_rows():
    values = np.array([[-np.inf, -np.inf], [np.inf, 0.0], [np.nan, 0.0]])
    original = values.copy()
    with np.errstate(invalid="ignore"):
        expected_log = logsumexp(original, axis=1, keepdims=True)
        expected = np.exp(original - expected_log)
    weights, normalizer = MCEMEstimator._normalized_importance_weights(values)
    np.testing.assert_array_equal(normalizer, expected_log)
    np.testing.assert_array_equal(weights, expected)
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("cls", [MCEMEstimator, QMCEMEstimator])
def test_importance_e_step_and_refresh_preserve_borrowed_extreme_likelihoods(
    cls, monkeypatch
):
    estimator = cls(n_samples=50, seed=27)
    estimator._rng = np.random.default_rng(27)
    model = TwoParameterLogistic(1)
    responses = np.array([[0], [1], [-1]])
    borrowed = np.full((3, 50), -1e300)
    borrowed.setflags(write=False)
    monkeypatch.setattr(estimator, "_sample_log_likelihoods", lambda *args: borrowed)
    monkeypatch.setattr(model, "log_likelihood_batch", lambda *args: borrowed)
    samples, weights = estimator._e_step_mc(model, responses, np.zeros(1), np.eye(1), 1)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)
    likelihood, refreshed = estimator._refresh_mc_state(
        model, responses, samples, weights
    )
    np.testing.assert_allclose(refreshed.sum(axis=1), 1.0, atol=1e-15)
    assert likelihood == -3e300
    np.testing.assert_array_equal(borrowed, -1e300)


def test_normalization_peak_needs_only_one_owned_sample_buffer():
    borrowed = np.full((5000, 256), -500.0)
    borrowed.setflags(write=False)
    tracemalloc.start()
    try:
        weights, _ = MCEMEstimator._normalized_importance_weights(borrowed)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * weights.nbytes
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_array_equal(borrowed, -500.0)
