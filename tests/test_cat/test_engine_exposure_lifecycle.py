"""Exposure denominators follow actual CAT sessions without unused-reset drift."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.cat import CATEngine, MCATEngine, ProgressiveRestricted, SympsonHetter
from mirt.models import MultidimensionalModel, TwoParameterLogistic


def _engine(n_factors: int, exposure=None):
    if n_factors == 1:
        model = TwoParameterLogistic(n_items=3)
        model.set_parameters(
            discrimination=np.array([0.8, 1.2, 1.5]),
            difficulty=np.array([-0.6, 0.2, 0.9]),
        )
    else:
        model = MultidimensionalModel(n_items=3, n_factors=2)
        model.set_parameters(
            slopes=np.array([[0.8, 1.1], [1.2, 0.7], [1.5, 1.3]]),
            intercepts=np.array([-0.6, 0.2, 0.9]),
        )
    model._is_fitted = True
    if exposure is None:
        exposure = SympsonHetter(np.ones(model.n_items), seed=17)
    engine_class = CATEngine if n_factors == 1 else MCATEngine
    engine = engine_class(
        model,
        min_items=3,
        max_items=3,
        n_quadpts=7,
        exposure_control=exposure,
        seed=11,
    )
    return engine, exposure


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_first_interactive_session_is_counted_without_manual_reset(n_factors):
    engine, exposure = _engine(n_factors)
    assert exposure.n_examinees == 1
    item = engine.get_current_state().next_item
    assert engine.select_next_item() == item
    assert engine.get_current_state().next_item == item
    state = engine.administer_item(1)

    report = exposure.exposure_report(n_items=3)
    assert report.n_examinees == 1
    assert report.selection_counts.sum() == 1
    assert report.selection_counts[item] == 1
    assert report.exposure_rates[item] == 1.0
    assert state.items_administered == [item]


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_unused_resets_do_not_add_examinees_or_change_simulation_denominator(n_factors):
    engine, exposure = _engine(n_factors)
    for _ in range(3):
        engine.reset()
    assert exposure.n_examinees == 1

    theta = 0.0 if n_factors == 1 else np.zeros(2)
    result = engine.run_simulation(theta)
    report = exposure.exposure_report(n_items=3)
    assert result.n_items_administered == 3
    assert report.n_examinees == 1
    assert_array_equal(report.selection_counts, [1, 1, 1])
    assert_allclose(report.exposure_rates, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_completed_and_restarted_sessions_are_counted_once_each(n_factors):
    engine, exposure = _engine(n_factors)
    for _ in range(3):
        engine.administer_item(1)
    assert engine.get_result().n_items_administered == 3

    engine.reset()
    engine.reset()
    assert exposure.n_examinees == 2
    for _ in range(3):
        engine.administer_item(0)

    theta = 0.5 if n_factors == 1 else np.array([0.5, -0.3])
    engine.run_simulation(theta)
    report = exposure.exposure_report(n_items=3)
    assert report.n_examinees == 3
    assert_array_equal(report.selection_counts, [3, 3, 3])
    assert_allclose(report.exposure_rates, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_batch_simulation_exposure_denominator_matches_returned_sessions(n_factors):
    engine, exposure = _engine(n_factors)
    if n_factors == 1:
        results = engine.run_batch_simulation([-0.5, 0.5], n_replications=3)
    else:
        results = engine.run_batch_simulation(
            [[-0.5, 0.2], [0.5, -0.2]], n_replications=3
        )

    report = exposure.exposure_report(n_items=3)
    assert report.n_examinees == len(results) == 6
    assert_array_equal(report.selection_counts, [6, 6, 6])
    assert_allclose(report.exposure_rates, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_abandoned_disclosed_session_is_distinct_from_replacement_session(n_factors):
    engine, exposure = _engine(n_factors)
    engine.get_current_state()
    engine.reset()
    engine.reset()
    assert exposure.n_examinees == 2
    engine.administer_item(1)
    report = exposure.exposure_report(n_items=3)
    assert report.n_examinees == 2
    assert report.selection_counts.sum() == 1
    assert report.exposure_rates.sum() == 0.5


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_reset_clears_other_exposure_state_immediately(n_factors):
    exposure = ProgressiveRestricted(seed=19)
    engine, _ = _engine(n_factors, exposure)
    engine.get_current_state()
    assert exposure.max_information_seen

    engine.reset()
    assert exposure.max_information_seen == {}
    engine.reset()
    assert exposure.max_information_seen == {}


@pytest.mark.parametrize("n_factors", [1, 2], ids=["CAT", "MCAT"])
def test_rejected_simulation_request_keeps_existing_session_count(n_factors):
    engine, exposure = _engine(n_factors)
    before = engine.get_current_state()
    invalid_theta = np.nan if n_factors == 1 else np.array([np.nan, 0.0])
    with pytest.raises(ValueError):
        engine.run_simulation(invalid_theta)
    assert exposure.n_examinees == 1
    assert engine.get_current_state().next_item == before.next_item
    engine.administer_item(1)
    assert exposure.exposure_report(n_items=3).n_examinees == 1
