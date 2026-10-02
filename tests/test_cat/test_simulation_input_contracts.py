"""CAT diagnostic requests are validated before any session is reset or run."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.cat import CATEngine, MCATEngine
from mirt.models import MultidimensionalModel, TwoParameterLogistic


def _engine(n_factors: int):
    if n_factors == 1:
        model = TwoParameterLogistic(n_items=3)
        model.set_parameters(
            discrimination=np.array([0.9, 1.2, 1.5]),
            difficulty=np.array([-0.5, 0.0, 0.5]),
        )
    else:
        model = MultidimensionalModel(n_items=3, n_factors=2)
        model.set_parameters(
            slopes=np.array([[0.9, 0.4], [1.2, 0.8], [0.3, 1.5]]),
            intercepts=np.array([-0.5, 0.0, 0.5]),
        )
    model._is_fitted = True
    engine_class = CATEngine if n_factors == 1 else MCATEngine
    return engine_class(model, n_quadpts=7, min_items=3, max_items=3, seed=42)


def _assert_preserved(engine, before):
    after = engine.get_current_state()
    assert after.responses == before.responses
    assert after.items_administered == before.items_administered
    assert after.next_item == before.next_item
    assert_allclose(after.theta, before.theta, rtol=0.0, atol=0.0)
    assert_allclose(after.standard_error, before.standard_error, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "theta", [np.nan, np.inf, -np.inf, "invalid", [0.0, 0.1], np.complex128(1 + 2j)]
)
def test_cat_rejects_invalid_simulation_ability_without_resetting_session(theta):
    engine = _engine(1)
    engine.administer_item(1)
    before = engine.get_current_state()

    with pytest.raises(ValueError, match="true_theta"):
        engine.run_simulation(theta)

    _assert_preserved(engine, before)


@pytest.mark.parametrize(
    "theta",
    [
        [0.0, np.nan],
        [np.inf, 0.0],
        [0.0],
        [[0.0, 0.0]],
        ["invalid", 0.0],
        np.array([1 + 2j, 0.0]),
    ],
)
def test_mcat_rejects_invalid_simulation_ability_without_resetting_session(theta):
    engine = _engine(2)
    engine.administer_item(1)
    before = engine.get_current_state()

    with pytest.raises(ValueError, match="true_theta"):
        engine.run_simulation(theta)

    _assert_preserved(engine, before)


@pytest.mark.parametrize(
    "method", ["run_batch_simulation", "compute_conditional_metrics"]
)
@pytest.mark.parametrize(
    "thetas",
    [
        0.0,
        [],
        np.empty((0, 2)),
        [0.0],
        np.zeros((2, 3)),
        np.zeros((1, 2, 1)),
        [[0.0, 0.0], [np.nan, 0.0]],
        [[0.0, 0.0], [0.0, np.inf]],
        [["invalid", 0.0]],
        np.array([[0.0, 0.0], [1 + 2j, 0.0]]),
    ],
)
def test_mcat_batch_request_validates_all_abilities_before_any_session(method, thetas):
    engine = _engine(2)
    engine.administer_item(1)
    before = engine.get_current_state()

    with pytest.raises(ValueError, match="true_thetas"):
        getattr(engine, method)(thetas, n_replications=1)

    _assert_preserved(engine, before)


@pytest.mark.parametrize(
    "method", ["run_batch_simulation", "compute_conditional_metrics"]
)
@pytest.mark.parametrize("replications", [0, -1, True, 1.5, "2", np.nan])
def test_mcat_batch_rejects_invalid_replications_without_resetting_session(
    method, replications
):
    engine = _engine(2)
    engine.administer_item(1)
    before = engine.get_current_state()

    with pytest.raises(ValueError, match="n_replications"):
        getattr(engine, method)([[0.0, 0.0]], n_replications=replications)

    _assert_preserved(engine, before)


def test_mcat_accepts_one_vector_and_numpy_replication_count():
    result = _engine(2).run_batch_simulation([0.5, -0.5], n_replications=np.int64(2))
    assert len(result) == 2
    assert all(item.n_items_administered == 3 for item in result)
    assert all(np.all(np.isfinite(item.theta)) for item in result)


def test_conditional_metrics_equal_independently_aggregated_sessions():
    thetas = np.array([[-0.5, 0.3], [1.2, -0.8]])
    replications = 3
    results = _engine(2).run_batch_simulation(thetas, n_replications=replications)
    expected_bias, expected_mse, expected_length = [], [], []
    for index, theta in enumerate(thetas):
        group = results[index * replications : (index + 1) * replications]
        errors = np.array([result.theta - theta for result in group])
        expected_bias.append(errors.mean(axis=0))
        expected_mse.append((errors**2).mean(axis=0))
        expected_length.append(
            np.mean([result.n_items_administered for result in group])
        )

    actual = _engine(2).compute_conditional_metrics(thetas, n_replications=replications)
    assert_array_equal(actual["true_thetas"], thetas)
    assert_allclose(actual["bias"], expected_bias, atol=1e-15)
    assert_allclose(actual["mse"], expected_mse, atol=1e-15)
    assert_allclose(actual["avg_items"], expected_length, atol=0.0)


def test_cat_conditional_mse_equals_independently_aggregated_sessions():
    thetas = np.array([-0.5, 1.2])
    replications = 3
    results = _engine(1).run_batch_simulation(
        thetas, n_replications=replications, use_rust=False
    )
    expected_bias, expected_mse, expected_length = [], [], []
    for index, theta in enumerate(thetas):
        group = results[index * replications : (index + 1) * replications]
        errors = np.array([result.theta - theta for result in group])
        expected_bias.append(errors.mean())
        expected_mse.append((errors**2).mean())
        expected_length.append(
            np.mean([result.n_items_administered for result in group])
        )

    points, bias, mse, length = _engine(1).compute_conditional_mse(
        thetas, n_replications=replications, use_rust=False
    )
    assert_array_equal(points, thetas)
    assert_allclose(bias, expected_bias, atol=1e-15)
    assert_allclose(mse, expected_mse, atol=1e-15)
    assert_allclose(length, expected_length, atol=0.0)


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_conditional_diagnostics_release_estimates_as_replications_finish(
    n_factors, monkeypatch
):
    engine = _engine(n_factors)
    original = engine.run_simulation
    estimates = []
    live_counts = []

    def capture(theta):
        live_counts.append(sum(reference() is not None for reference in estimates))
        result = original(theta)
        # Track a distinct estimate array without retaining the result itself.
        # CAT's scalar and MCAT's vector follow the same diagnostic contract.
        estimate = np.array(result.theta, dtype=np.float64, copy=True)
        estimates.append(weakref.ref(estimate))
        return SimpleNamespace(
            theta=estimate, n_items_administered=result.n_items_administered
        )

    monkeypatch.setattr(engine, "run_simulation", capture)
    if n_factors == 1:
        engine.compute_conditional_mse([0.0], n_replications=12, use_rust=False)
    else:
        engine.compute_conditional_metrics([[0.0, 0.0]], n_replications=12)

    assert len(estimates) == 12
    assert max(live_counts) <= 1
    assert all(reference() is None for reference in estimates)
