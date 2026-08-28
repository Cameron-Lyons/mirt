"""Regression coverage for stable Pareto-smoothed model diagnostics."""

import concurrent.futures
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.bayesian import (
    _pareto_smooth_weights,
    psis_loo,
    waic,
)


def _reference_log_likelihood() -> np.ndarray:
    """Return the deterministic fixture used by the published PSIS code."""
    rng = np.random.default_rng(1701)
    log_likelihood = rng.normal(-2.0, 0.7, size=(400, 6))
    log_likelihood[:, 0] -= rng.pareto(1.8, 400)
    return log_likelihood


def test_psis_loo_matches_published_reference_implementation() -> None:
    result = psis_loo(_reference_log_likelihood())

    assert result.elpd_loo == pytest.approx(-20.773024882317852)
    assert result.p_loo == pytest.approx(9.743591283053611)
    assert_allclose(
        result.pointwise,
        np.array(
            [
                -9.480103239991280,
                -2.233673912506187,
                -2.230579377558221,
                -2.293863003992652,
                -2.258341388555968,
                -2.276463959713542,
            ]
        ),
        rtol=1e-13,
    )
    assert_allclose(
        result.pareto_k,
        np.array(
            [
                2.561691903923107,
                0.309899089400143,
                0.208355141322391,
                0.428480798843824,
                0.149818389891060,
                0.350511232122710,
            ]
        ),
        rtol=1e-13,
    )


def test_pareto_tail_is_smoothed_and_shape_is_recovered() -> None:
    rng = np.random.default_rng(20260827)
    target_shape = 0.7
    raw_ratios = 1.0 + rng.pareto(1.0 / target_shape, 10_000)
    log_ratios = np.log(raw_ratios)
    raw_weights = raw_ratios / np.sum(raw_ratios)

    smoothed_weights, estimated_shape = _pareto_smooth_weights(log_ratios)

    assert np.max(np.abs(smoothed_weights - raw_weights)) > 1e-3
    assert estimated_shape == pytest.approx(target_shape, abs=0.1)
    assert np.sum(smoothed_weights) == pytest.approx(1.0)


def test_psis_loo_is_stable_under_large_finite_log_shifts() -> None:
    log_likelihood = _reference_log_likelihood()
    shifts = np.array([1_000.0, 1_200.0, 900.0, 1_100.0, 800.0, 1_300.0])

    baseline = psis_loo(log_likelihood)
    shifted = psis_loo(log_likelihood - shifts)

    assert np.isfinite(shifted.elpd_loo)
    assert np.isfinite(shifted.p_loo)
    assert_allclose(shifted.pointwise, baseline.pointwise - shifts)
    assert shifted.p_loo == pytest.approx(baseline.p_loo)
    assert_allclose(shifted.pareto_k, baseline.pareto_k)


def test_relative_efficiency_accepts_per_observation_values() -> None:
    log_likelihood = _reference_log_likelihood()

    scalar = psis_loo(log_likelihood, relative_eff=1.0)
    vector = psis_loo(log_likelihood, relative_eff=np.ones(6))

    assert_allclose(vector.pointwise, scalar.pointwise)
    assert_allclose(vector.pareto_k, scalar.pareto_k)


def test_parallel_psis_matches_serial_results() -> None:
    """Threaded observation smoothing preserves deterministic outputs."""
    log_likelihood = _reference_log_likelihood()

    serial = psis_loo(log_likelihood, n_jobs=1)
    parallel = psis_loo(log_likelihood, n_jobs=3)

    assert parallel.elpd_loo == serial.elpd_loo
    assert parallel.p_loo == serial.p_loo
    assert parallel.looic == serial.looic
    assert parallel.se_elpd == serial.se_elpd
    assert parallel.n_high_k == serial.n_high_k
    np.testing.assert_array_equal(parallel.pointwise, serial.pointwise)
    np.testing.assert_array_equal(parallel.pareto_k, serial.pareto_k)


def test_parallel_psis_caps_threads_at_observation_count(monkeypatch) -> None:
    """Parallel execution never creates idle observation workers."""
    requested_workers = []

    class ImmediateExecutor:
        def __init__(self, max_workers):
            requested_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, values):
            return map(function, values)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor)

    parallel = psis_loo(_reference_log_likelihood(), n_jobs=20)

    assert requested_workers == [6]
    assert parallel.pointwise.shape == (6,)


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5, "2"])
def test_psis_loo_validates_worker_count(n_jobs: object) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        psis_loo(_reference_log_likelihood(), n_jobs=n_jobs)


@pytest.mark.parametrize("function", [psis_loo, waic])
@pytest.mark.parametrize(
    ("log_likelihood", "message"),
    [
        (np.array([-1.0]), "at least two posterior samples"),
        (np.empty((2, 0)), "at least one observation"),
        (np.zeros((2, 1, 1)), "one- or two-dimensional"),
        (np.array([[-1.0], [np.nan]]), "only finite"),
    ],
)
def test_information_criteria_validate_log_likelihood(
    function: Callable[..., object],
    log_likelihood: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(log_likelihood)


@pytest.mark.parametrize("k_threshold", [0.0, -0.1, np.nan, True, "bad"])
def test_psis_loo_validates_k_threshold(k_threshold: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        psis_loo(_reference_log_likelihood(), k_threshold=k_threshold)


@pytest.mark.parametrize("relative_eff", [0.0, np.nan, True, [1.0, 1.0]])
def test_psis_loo_validates_relative_efficiency(relative_eff: object) -> None:
    with pytest.raises(ValueError, match="relative_eff"):
        psis_loo(_reference_log_likelihood(), relative_eff=relative_eff)


def test_single_observation_standard_errors_are_zero() -> None:
    log_likelihood = _reference_log_likelihood()[:, 0]

    assert psis_loo(log_likelihood).se_elpd == 0.0
    assert waic(log_likelihood).se_waic == 0.0
