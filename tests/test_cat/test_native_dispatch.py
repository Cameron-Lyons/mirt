"""Native CAT simulation must preserve authored Python extension hooks."""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.cat import CATEngine
from mirt.cat.content import NoContentConstraint
from mirt.cat.exposure import NoExposureControl
from mirt.cat.selection import MaxFisherInformation
from mirt.cat.stopping import CombinedStop, MaxItemsStop, StandardErrorStop
from mirt.models import OneParameterLogistic, TwoParameterLogistic
from mirt.scoring import fscores


def _model(cls=TwoParameterLogistic):
    model = cls(4)
    model.set_parameters(difficulty=np.array([-1.5, -0.5, 0.5, 1.5]))
    model._is_fitted = True
    return model


@pytest.mark.parametrize("model_class", [OneParameterLogistic, TwoParameterLogistic])
def test_native_dispatch_accepts_original_models_with_custom_reporting_names(
    model_class,
):
    model = _model(model_class)
    model.model_name = "My calibrated item bank"
    assert CATEngine(model, max_items=3)._can_use_rust_simulation()


@pytest.mark.parametrize("location", ["instance", "class"])
@pytest.mark.parametrize(
    ("component", "name"),
    [
        ("model", "probability"),
        ("model", "log_likelihood"),
        ("model", "log_likelihood_batch"),
        ("model", "information"),
        ("selection", "select_item"),
        ("selection", "get_item_criteria"),
        ("selection", "_compute_criterion"),
        ("exposure", "filter_items"),
        ("exposure", "update"),
        ("exposure", "reset"),
        ("content", "filter_items"),
        ("content", "reset"),
        ("stopping", "should_stop"),
        ("stopping", "get_reason"),
        ("stopping", "reset"),
        ("se_rule", "should_stop"),
        ("max_rule", "should_stop"),
        ("engine", "_generate_response"),
        ("engine", "_update_theta"),
        ("engine", "run_simulation"),
    ],
)
def test_modified_methods_require_python_simulation(
    monkeypatch, location, component, name
):
    engine = CATEngine(_model(), max_items=3)
    components = {
        "model": engine.model,
        "selection": engine._selection,
        "exposure": engine._exposure,
        "content": engine._content,
        "stopping": engine._stopping,
        "se_rule": engine._stopping.rules[0],
        "max_rule": engine._stopping.rules[1],
        "engine": engine,
    }
    target = components[component]
    if location == "class":
        target = type(target)
    monkeypatch.setattr(target, name, lambda *args, **kwargs: None)

    assert not engine._can_use_rust_simulation()


@pytest.mark.parametrize(
    "cls",
    [NoExposureControl, NoContentConstraint, MaxFisherInformation, StandardErrorStop],
)
def test_unregistered_strategy_and_control_subclasses_require_python_simulation(cls):
    class Customized(cls):
        pass

    keyword = {
        NoExposureControl: "exposure_control",
        NoContentConstraint: "content_constraint",
        MaxFisherInformation: "item_selection",
        StandardErrorStop: "stopping_rule",
    }[cls]
    engine = CATEngine(_model(), max_items=3, **{keyword: Customized()})
    assert not engine._can_use_rust_simulation()


def test_custom_model_subclass_requires_python_simulation():
    class Customized(TwoParameterLogistic):
        pass

    assert not CATEngine(_model(Customized), max_items=3)._can_use_rust_simulation()


def test_custom_combined_stop_and_max_items_subclasses_require_python_simulation():
    class CustomizedCombined(CombinedStop):
        pass

    class CustomizedMaximum(MaxItemsStop):
        pass

    for rule in (
        CustomizedCombined([StandardErrorStop()], min_items=1),
        CombinedStop([StandardErrorStop(), CustomizedMaximum(3)], min_items=1),
    ):
        assert not CATEngine(_model(), stopping_rule=rule)._can_use_rust_simulation()


def test_custom_simulation_executes_response_selection_and_stopping_hooks(monkeypatch):
    calls = []

    class CustomizedEngine(CATEngine):
        def _generate_response(self, item_idx, true_theta):
            calls.append(("response", item_idx))
            return 0

    class CustomizedSelection(MaxFisherInformation):
        def select_item(self, model, theta, available_items, *args, **kwargs):
            calls.append(("selection", len(available_items)))
            return max(available_items)

    class CustomizedStop(StandardErrorStop):
        def should_stop(self, state):
            calls.append(("stopping", state.n_items))
            return state.n_items >= 2

    monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda requested: True)

    def fail_native(*args):
        pytest.fail("custom simulation dispatched to a native kernel")

    monkeypatch.setattr("mirt.cat.engine.rust_cat_simulate_batch_full", fail_native)
    model = _model()
    engine = CustomizedEngine(
        model,
        item_selection=CustomizedSelection(),
        stopping_rule=CustomizedStop(),
        seed=8,
    )
    result = engine.run_batch_simulation([0.0], use_rust=True)[0]

    assert result.items_administered == [3, 2]
    assert_array_equal(result.responses, [0, 0])
    assert [entry for entry in calls if entry[0] == "stopping"] == [
        ("stopping", 1),
        ("stopping", 2),
    ]
    responses = np.array([[-1, -1, 0, 0]])
    reference = fscores(model, responses, n_quadpts=engine.n_quadpts)
    assert_allclose(result.theta, reference.theta[0], atol=1e-14)
    assert_allclose(result.standard_error, reference.standard_error[0], atol=1e-14)


@pytest.mark.parametrize("operation", ["batch", "mse"])
def test_custom_model_probability_controls_actual_simulation_and_diagnostics(
    monkeypatch, operation
):
    model = _model()
    calls = []

    def constant_probability(theta, item_idx=None):
        calls.append(item_idx)
        return np.full((len(theta), model.n_items if item_idx is None else 1), 0.5)

    monkeypatch.setattr(model, "probability", constant_probability)
    monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda requested: True)

    def fail_native(*args):
        pytest.fail("custom model dispatched to a native kernel")

    monkeypatch.setattr("mirt.cat.engine.rust_cat_simulate_batch_full", fail_native)
    monkeypatch.setattr("mirt.cat.engine.rust_cat_conditional_mse", fail_native)
    engine = CATEngine(
        model, n_quadpts=9, min_items=2, max_items=2, se_threshold=0.01, seed=8
    )
    if operation == "batch":
        result = engine.run_batch_simulation([0.75], use_rust=True)[0]
        # Constant item probabilities carry no information about ability.
        assert_allclose(result.theta, 0.0, atol=1e-14)
        assert_allclose(result.standard_error, 1.0, atol=1e-14)
    else:
        thetas, bias, mse, average_items = engine.compute_conditional_mse(
            [0.75], n_replications=1, use_rust=True
        )
        assert_allclose(thetas, [0.75])
        assert_allclose(bias, [-0.75], atol=1e-14)
        assert_allclose(mse, [0.75**2], atol=1e-14)
        assert_allclose(average_items, [2.0])
    assert calls


@pytest.mark.parametrize(
    "n_quadpts", [1, 4, True, 5.0, np.float64(21), np.array(21), None]
)
def test_invalid_eap_controls_preserve_python_fallback(monkeypatch, n_quadpts):
    monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda requested: True)

    def fail_native(*args):
        pytest.fail("invalid EAP controls dispatched to a native kernel")

    monkeypatch.setattr("mirt.cat.engine.rust_cat_simulate_batch_full", fail_native)
    model = _model()
    options = dict(n_quadpts=n_quadpts, min_items=2, max_items=2, seed=3)
    native_requested = CATEngine(model, **options)
    python_requested = CATEngine(model, **options)

    actual = native_requested.run_batch_simulation([0.75], use_rust=True)[0]
    expected = python_requested.run_batch_simulation([0.75], use_rust=False)[0]

    assert_allclose(actual.theta, expected.theta)
    assert_allclose(actual.standard_error, expected.standard_error)
    assert actual.items_administered == expected.items_administered
    assert_array_equal(actual.responses, expected.responses)


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("min_items", 1.5),
        ("min_items", True),
        ("max_items", 1.5),
        ("max_items", True),
        ("max_items", 0),
        ("threshold", np.nan),
        ("threshold", np.inf),
        ("threshold", True),
        ("threshold", 0.0),
        ("threshold", -0.1),
    ],
)
def test_mutated_stopping_controls_preserve_python_semantics(
    monkeypatch, control, value
):
    monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda requested: True)

    def fail_native(*args):
        pytest.fail("mutated stopping controls dispatched to a native kernel")

    monkeypatch.setattr("mirt.cat.engine.rust_cat_simulate_batch_full", fail_native)

    def configured_engine():
        engine = CATEngine(
            _model(), min_items=1, max_items=3, se_threshold=0.01, seed=3
        )
        target = (
            engine._stopping
            if control == "min_items"
            else engine._stopping.rules[1 if control == "max_items" else 0]
        )
        setattr(target, control, value)
        return engine

    native_requested = configured_engine()
    python_requested = configured_engine()
    assert not native_requested._can_use_rust_simulation()
    actual = native_requested.run_batch_simulation([0.75], use_rust=True)[0]
    expected = python_requested.run_batch_simulation([0.75], use_rust=False)[0]

    assert_allclose(actual.theta, expected.theta)
    assert_allclose(actual.standard_error, expected.standard_error)
    assert actual.items_administered == expected.items_administered
    assert_array_equal(actual.responses, expected.responses)
    if control == "max_items" and value == 1.5:
        assert actual.n_items_administered == 2


def test_valid_updated_numpy_controls_still_allow_native_dispatch():
    engine = CATEngine(_model(), min_items=1, max_items=4)
    engine.n_quadpts = np.int64(9)
    engine._stopping.min_items = np.int64(2)
    engine._stopping.rules[0].threshold = np.float64(0.4)
    engine._stopping.rules[1].max_items = np.int64(3)

    assert engine._can_use_rust_simulation()
    assert engine._get_stopping_parameters() == (0.4, 3, 2)


@pytest.mark.parametrize(
    ("module", "cls", "method"),
    [
        ("mirt.cat.selection", "MaxFisherInformation", "select_item"),
        ("mirt.cat.exposure", "ExposureControl", "update"),
        ("mirt.cat.content", "ContentConstraint", "reset"),
        ("mirt.cat.stopping", "StoppingRule", "reset"),
        ("mirt.models.dichotomous", "TwoParameterLogistic", "information"),
        ("mirt.models.base", "BaseItemModel", "information"),
    ],
)
def test_hooks_changed_before_engine_import_do_not_become_native_defaults(
    module, cls, method
):
    script = f"""
import sys
from {module} import {cls}
assert 'mirt.cat.engine' not in sys.modules
{cls}.{method} = lambda *args, **kwargs: None
from mirt.cat import CATEngine
from mirt.models import TwoParameterLogistic
model = TwoParameterLogistic(4)
model._is_fitted = True
assert not CATEngine(model, max_items=3)._can_use_rust_simulation()
"""
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stdout + process.stderr
