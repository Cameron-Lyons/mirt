"""Native summaries retain authored stop priority and item-pool exhaustion."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
from mirt.cat import CATEngine
from mirt.cat.stopping import CombinedStop, MaxItemsStop, StandardErrorStop
from mirt.models import TwoParameterLogistic

pytestmark = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native extension unavailable"
)


@pytest.fixture
def native_batches(monkeypatch):
    from mirt.cat import engine as engine_module

    previous = mirt.get_backend()
    mirt.set_backend("rust")
    original = engine_module.rust_cat_simulate_batch_full
    batches = []

    def capture(*args):
        payload = original(*args)
        assert payload is not None
        batches.append(payload)
        return payload

    monkeypatch.setattr(engine_module, "rust_cat_simulate_batch_full", capture)
    yield batches
    mirt.set_backend(previous)


def _model():
    model = TwoParameterLogistic(3)
    model.set_parameters(difficulty=np.array([-1.7, -0.3, 1.1]))
    model._is_fitted = True
    return model


def _engine(model, configuration):
    if configuration in ("maximum_first", "precision_first"):
        rules = [MaxItemsStop(1), StandardErrorStop(100.0)]
        if configuration == "precision_first":
            rules.reverse()
        stopping = CombinedStop(rules, min_items=1)
        return CATEngine(model, stopping_rule=stopping, n_quadpts=9, seed=42)
    options = {"max_items": 8} if configuration == "cap_beyond_pool" else {}
    return CATEngine(model, se_threshold=1e-8, n_quadpts=9, seed=42, **options)


@pytest.mark.parametrize(
    ("configuration", "expected_reason"),
    [
        ("maximum_first", "Maximum items reached (1)"),
        ("precision_first", "SE threshold reached (SE <= 100.0)"),
        ("cap_beyond_pool", "Item pool exhausted"),
        ("bare_precision", "Item pool exhausted"),
    ],
)
def test_native_stopping_reason_matches_replayed_actual_paths(
    native_batches, configuration, expected_reason
):
    model = _model()
    engine = _engine(model, configuration)
    results = engine.run_batch_simulation([0.0], n_replications=4, use_rust=True)
    assert len(native_batches) == 1

    for result in results:
        replay = _engine(model, configuration)
        for item, response in zip(
            result.items_administered, result.responses, strict=True
        ):
            assert replay.select_next_item() == item
            replay.administer_item(int(response))
        reference = replay.get_result()

        assert result.stopping_reason == reference.stopping_reason == expected_reason
        assert result.n_items_administered == reference.n_items_administered
        assert_array_equal(result.responses, reference.responses)
        assert_allclose(result.theta, reference.theta, atol=1e-13)
        assert_allclose(result.standard_error, reference.standard_error, atol=1e-13)


def test_native_stopping_reconstruction_preserves_previous_session_state(
    native_batches,
):
    stopping = CombinedStop([MaxItemsStop(2), StandardErrorStop(100.0)], min_items=1)
    engine = CATEngine(_model(), stopping_rule=stopping, n_quadpts=9, seed=42)
    engine.run_simulation(0.0)
    previous_state = engine.get_current_state()
    previous_result = engine.get_result().to_dict()
    previous_trigger = engine._stopping._triggered_rule
    assert previous_trigger is engine._stopping.rules[1]
    assert previous_result["stopping_reason"] == "SE threshold reached (SE <= 100.0)"

    # Both conditions will hold in the next batch, whose authored first rule
    # must report the maximum without replacing the previous session's trigger.
    engine._stopping.rules[0].max_items = 1
    results = engine.run_batch_simulation([0.0], n_replications=4, use_rust=True)

    assert len(native_batches) == 1
    assert engine.get_current_state() == previous_state
    assert engine.get_result().to_dict() == previous_result
    assert engine._stopping._triggered_rule is previous_trigger
    assert all(
        result.stopping_reason == "Maximum items reached (1)" for result in results
    )
