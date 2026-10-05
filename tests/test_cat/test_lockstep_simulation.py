"""Lock-step batch simulation must reproduce independent CAT sessions."""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.cat._lockstep as lockstep
from mirt.cat import CATEngine
from mirt.cat.content import ContentArea, ContentBlueprint
from mirt.cat.selection import MaxFisherInformation
from mirt.cat.stopping import MaxItemsStop, StandardErrorStop, ThetaChangeStop
from mirt.models import (
    FourParameterLogistic,
    GeneralizedPartialCredit,
    GradedResponseModel,
    MultidimensionalModel,
    NominalResponseModel,
    OneParameterLogistic,
    PartialCreditModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.scoring import fscores


def _model(kind: str, n_items: int = 24, seed: int = 4):
    rng = np.random.default_rng(seed)
    discrimination = rng.lognormal(0.0, 0.3, n_items)
    if kind == "GRM":
        model = GradedResponseModel(
            n_items=n_items, n_categories=[2 + item % 4 for item in range(n_items)]
        )
        model.set_parameters(
            discrimination=discrimination,
            thresholds=np.sort(rng.normal(0.0, 1.0, (n_items, 4)), axis=1),
        )
    elif kind == "GPCM":
        model = GeneralizedPartialCredit(
            n_items=n_items, n_categories=[2 + item % 3 for item in range(n_items)]
        )
        model.set_parameters(
            discrimination=discrimination, steps=rng.normal(0.0, 1.0, (n_items, 3))
        )
    elif kind == "PCM":
        model = PartialCreditModel(
            n_items=n_items, n_categories=[2 + item % 3 for item in range(n_items)]
        )
        model.set_parameters(steps=rng.normal(0.0, 1.0, (n_items, 3)))
    else:
        model_class = {
            "1PL": OneParameterLogistic,
            "2PL": TwoParameterLogistic,
            "3PL": ThreeParameterLogistic,
            "4PL": FourParameterLogistic,
        }[kind]
        model = model_class(n_items=n_items)
        parameters = {"difficulty": rng.normal(0.0, 1.0, n_items)}
        if kind != "1PL":
            parameters["discrimination"] = discrimination
        if kind in ("3PL", "4PL"):
            parameters["guessing"] = np.full(n_items, 0.2)
        if kind == "4PL":
            parameters["upper"] = np.full(n_items, 0.95)
        model.set_parameters(**parameters)
    model._is_fitted = True
    return model


def _response_matrix(model, true_thetas, seed: int = 9):
    """Draw one fixed response per examinee and item."""
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(model.probability(true_thetas[:, None]))
    if model.is_polytomous:
        cumulative = np.cumsum(probabilities, axis=2)
        uniforms = rng.random(probabilities.shape[:2])
        return np.sum(cumulative <= uniforms[:, :, None], axis=2)
    return (rng.random(probabilities.shape) < probabilities).astype(np.int_)


def _inject_responses(monkeypatch, true_thetas, responses):
    """Answer lock-step items from a fixed matrix keyed by distinct abilities."""
    rows = {float(theta): row for row, theta in enumerate(true_thetas)}

    def answer(model, true_theta, items, uniforms):
        index = np.array([rows[float(theta)] for theta in true_theta])
        return responses[index, items]

    monkeypatch.setattr(lockstep, "_simulated_responses", answer)


def _assert_same_sessions(actual, expected, *, atol=1e-12):
    assert len(actual) == len(expected)
    for fast, reference in zip(actual, expected, strict=True):
        assert fast.items_administered == reference.items_administered
        assert_array_equal(fast.responses, reference.responses)
        assert fast.n_items_administered == reference.n_items_administered
        assert fast.stopping_reason == reference.stopping_reason
        assert_allclose(fast.theta, reference.theta, rtol=0.0, atol=atol)
        assert_allclose(
            fast.standard_error, reference.standard_error, rtol=0.0, atol=atol
        )
        for name in ("theta_history", "se_history", "item_info_history"):
            values = getattr(fast, name)
            assert len(values) == fast.n_items_administered
            assert_allclose(values, getattr(reference, name), rtol=0.0, atol=atol)


@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM", "GPCM"])
@pytest.mark.parametrize(
    "controls",
    [
        {"se_threshold": 0.35, "max_items": 15},
        {"se_threshold": 0.3, "min_items": 5, "max_items": 9},
        {"se_threshold": 0.45, "initial_theta": 0.75},
        {"se_threshold": 1e-6, "max_items": 6, "min_items": 6},
        {"stopping_rule": MaxItemsStop(7), "min_items": 2},
    ],
    ids=["se-max", "se-min-max", "se-pool", "fixed", "max-items-rule"],
)
@pytest.mark.parametrize("block_rows", [4096, 3])
def test_lockstep_matches_independent_sessions_for_identical_responses(
    monkeypatch, kind, controls, block_rows
):
    model = _model(kind)
    true_thetas = np.linspace(-2.0, 2.0, 11) + 1e-9 * np.arange(11)
    responses = _response_matrix(model, true_thetas)
    engine = CATEngine(model, n_quadpts=15, seed=5, **deepcopy(controls))
    assert lockstep.supports_lockstep(engine)

    expected = [
        engine.run_simulation(
            float(theta),
            response_generator=lambda item, _, row=row: responses[row, item],
        )
        for row, theta in enumerate(true_thetas)
    ]
    _inject_responses(monkeypatch, true_thetas, responses)
    monkeypatch.setattr(lockstep, "_BLOCK_ROWS", block_rows)
    actual = engine.run_batch_simulation(true_thetas, use_rust=False)

    _assert_same_sessions(actual, expected)


@pytest.mark.parametrize("kind", ["1PL", "4PL", "PCM"])
def test_lockstep_matches_sessions_for_remaining_model_families(monkeypatch, kind):
    model = _model(kind)
    true_thetas = np.linspace(-2.0, 2.0, 7) + 1e-9 * np.arange(7)
    responses = _response_matrix(model, true_thetas)
    engine = CATEngine(model, n_quadpts=15, se_threshold=0.45, max_items=12)
    assert lockstep.supports_lockstep(engine)

    expected = [
        engine.run_simulation(
            float(theta),
            response_generator=lambda item, _, row=row: responses[row, item],
        )
        for row, theta in enumerate(true_thetas)
    ]
    _inject_responses(monkeypatch, true_thetas, responses)
    actual = engine.run_batch_simulation(true_thetas, use_rust=False)

    _assert_same_sessions(actual, expected)


def test_lockstep_reaches_pool_exhaustion_like_the_sequential_engine(monkeypatch):
    model = _model("3PL", n_items=5)
    true_thetas = np.array([-1.0, 0.0, 1.0])
    responses = _response_matrix(model, true_thetas)
    engine = CATEngine(model, se_threshold=1e-6, n_quadpts=9)
    expected = [
        engine.run_simulation(
            float(theta),
            response_generator=lambda item, _, row=row: responses[row, item],
        )
        for row, theta in enumerate(true_thetas)
    ]
    _inject_responses(monkeypatch, true_thetas, responses)

    actual = engine.run_batch_simulation(true_thetas, use_rust=False)

    _assert_same_sessions(actual, expected)
    assert {result.stopping_reason for result in actual} == {"Item pool exhausted"}


@pytest.mark.parametrize("kind", ["3PL", "GRM"])
def test_lockstep_results_rescore_to_reported_estimates(kind):
    model = _model(kind)
    engine = CATEngine(model, n_quadpts=17, se_threshold=0.35, max_items=10, seed=12)

    results = engine.run_batch_simulation([-1.0, 0.5], n_replications=6)

    responses = np.full((len(results), model.n_items), -1)
    for row, result in enumerate(results):
        responses[row, result.items_administered] = result.responses
    scored = fscores(model, responses, n_quadpts=17)
    assert_allclose([r.theta for r in results], scored.theta, atol=1e-12)
    assert_allclose(
        [r.standard_error for r in results], scored.standard_error, atol=1e-12
    )
    for result in results:
        information = [
            float(model.information(np.array([[theta]]), item_idx=item).sum())
            for theta, item in zip(
                [0.0, *result.theta_history[:-1]],
                result.items_administered,
                strict=True,
            )
        ]
        assert_allclose(result.item_info_history, information, rtol=1e-12)


@pytest.mark.parametrize("kind", ["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM"])
@pytest.mark.parametrize(
    "controls",
    [
        {"se_threshold": 1e-6, "max_items": 7},
        {"stopping_rule": MaxItemsStop(6), "min_items": 3},
        {"se_threshold": 1e-6},
    ],
    ids=["tiny-se", "max-items-rule", "whole-pool"],
)
def test_seeded_fixed_length_batches_equal_sequential_sessions(kind, controls):
    model = _model(kind, n_items=12)
    true_thetas = np.array([-1.5, -0.2, 0.4, 1.1, 2.0])
    lockstep_engine = CATEngine(model, n_quadpts=15, seed=21, **deepcopy(controls))
    loop_engine = CATEngine(model, n_quadpts=15, seed=21, **deepcopy(controls))
    assert lockstep.supports_lockstep(lockstep_engine)

    actual = lockstep_engine.run_batch_simulation(
        true_thetas, n_replications=2, use_rust=False
    )
    expected = loop_engine.run_batch_simulation(
        true_thetas, n_replications=2, use_rust=False, vectorized=False
    )

    assert len({result.n_items_administered for result in expected}) == 1
    _assert_same_sessions(actual, expected)
    assert (
        lockstep_engine.rng.bit_generator.state == loop_engine.rng.bit_generator.state
    )


def test_lockstep_reserves_max_items_draws_per_examinee():
    model = _model("2PL")
    true_thetas = np.array([-1.5, -0.2, 0.4, 1.1])
    engine = CATEngine(model, se_threshold=0.6, max_items=12, seed=21)

    results = engine.run_batch_simulation(true_thetas, use_rust=False)

    assert len({result.n_items_administered for result in results}) > 1
    uniforms = np.random.default_rng(21).random((len(true_thetas), 12))
    for row, result in enumerate(results):
        count = result.n_items_administered
        probabilities = model.probability_pairs(
            np.full((count, 1), true_thetas[row]), result.items_administered
        )
        expected = (uniforms[row, :count] < probabilities).astype(np.int_)
        assert_array_equal(result.responses, expected)


def test_lockstep_is_seeded_and_leaves_interactive_state_untouched():
    model = _model("GRM")
    first = CATEngine(model, se_threshold=0.4, max_items=8, seed=3)
    second = CATEngine(model, se_threshold=0.4, max_items=8, seed=3)

    left = first.run_batch_simulation([-0.5, 0.5], n_replications=3)
    right = second.run_batch_simulation([-0.5, 0.5], n_replications=3)

    _assert_same_sessions(left, right, atol=0.0)
    assert first._items_administered == []
    assert first._theta_history == []
    assert not first._is_complete
    assert first.get_current_state().theta == 0.0


def test_sequential_reference_remains_available():
    model = _model("3PL")
    options = dict(se_threshold=0.35, max_items=10, n_quadpts=11, seed=17)
    batch_engine = CATEngine(model, **options)
    loop_engine = CATEngine(model, **options)

    actual = batch_engine.run_batch_simulation(
        [-1.0, 1.0], n_replications=2, vectorized=False
    )
    expected = [loop_engine.run_simulation(theta) for theta in (-1.0, -1.0, 1.0, 1.0)]

    _assert_same_sessions(actual, expected, atol=0.0)


def test_vectorized_flag_must_be_boolean():
    engine = CATEngine(_model("2PL"), max_items=3)
    with pytest.raises(ValueError, match="vectorized must be boolean"):
        engine.run_batch_simulation([0.0], vectorized="yes")
    with pytest.raises(ValueError, match="vectorized must be boolean"):
        engine.compute_conditional_mse([0.0], vectorized=1)


@pytest.mark.parametrize("block_rows", [4096, 5])
def test_lockstep_conditional_mse_matches_aggregated_batch(monkeypatch, block_rows):
    monkeypatch.setattr(lockstep, "_BLOCK_ROWS", block_rows)
    model = _model("3PL")
    thetas = np.array([-1.0, 0.25, 1.5])
    replications = 7
    options = dict(se_threshold=0.35, max_items=10, n_quadpts=11, seed=29)

    results = CATEngine(model, **options).run_batch_simulation(
        thetas, n_replications=replications
    )
    points, bias, mse, length = CATEngine(model, **options).compute_conditional_mse(
        thetas, n_replications=replications
    )

    estimates = np.array([r.theta for r in results]).reshape(len(thetas), replications)
    lengths = np.array([r.n_items_administered for r in results]).reshape(
        len(thetas), replications
    )
    errors = estimates - thetas[:, None]
    assert_array_equal(points, thetas)
    assert_allclose(bias, errors.mean(axis=1), rtol=0.0, atol=1e-14)
    assert_allclose(mse, (errors**2).mean(axis=1), rtol=0.0, atol=1e-14)
    assert_allclose(length, lengths.mean(axis=1), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("operation", ["batch", "mse"])
def test_nonfinite_posterior_restores_generator_and_uses_sequential_path(
    monkeypatch, operation
):
    model = _model("3PL")
    options = dict(se_threshold=0.35, max_items=6, n_quadpts=11, seed=31)
    original = lockstep._posterior_moments
    calls = []

    def fail_later(plan, log_posterior):
        calls.append(len(log_posterior))
        if len(calls) == 3:
            raise lockstep._Fallback
        return original(plan, log_posterior)

    monkeypatch.setattr(lockstep, "_posterior_moments", fail_later)
    engine = CATEngine(model, **options)
    reference = CATEngine(model, **options)
    if operation == "batch":
        actual = engine.run_batch_simulation([-0.5, 0.5], n_replications=2)
        expected = reference.run_batch_simulation(
            [-0.5, 0.5], n_replications=2, vectorized=False
        )
        _assert_same_sessions(actual, expected, atol=0.0)
    else:
        actual = engine.compute_conditional_mse([-0.5, 0.5], n_replications=2)
        expected = reference.compute_conditional_mse(
            [-0.5, 0.5], n_replications=2, vectorized=False
        )
        for left, right in zip(actual, expected, strict=True):
            assert_array_equal(left, right)
    assert len(calls) == 3


def test_nonfinite_information_restores_generator_and_uses_sequential_path(
    monkeypatch,
):
    model = _model("2PL")
    options = dict(se_threshold=0.35, max_items=6, n_quadpts=11, seed=37)
    original = lockstep._item_information
    calls = []

    def poison_later(model, theta):
        calls.append(len(theta))
        values = original(model, theta)
        if len(calls) == 2:
            values[0, 0] = np.nan
        return values

    monkeypatch.setattr(lockstep, "_item_information", poison_later)
    actual = CATEngine(model, **options).run_batch_simulation(
        [-0.5, 0.5], use_rust=False
    )
    expected = CATEngine(model, **options).run_batch_simulation(
        [-0.5, 0.5], use_rust=False, vectorized=False
    )

    _assert_same_sessions(actual, expected, atol=0.0)
    assert len(calls) == 2


def test_maximum_length_rule_without_se_uses_lockstep_but_not_rust(monkeypatch):
    model = _model("2PL")
    engine = CATEngine(model, stopping_rule=MaxItemsStop(5), seed=3)
    assert engine._native_stopping_parameters() is None
    assert engine._native_stopping_parameters(require_standard_error=False) == (
        -np.inf,
        5,
        1,
    )
    assert not engine._can_use_rust_simulation()
    assert lockstep.supports_lockstep(engine)

    calls = []
    original = lockstep.simulate

    def capture(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr("mirt.cat.engine.simulate_lockstep", capture)
    results = engine.run_batch_simulation([0.0, 1.0])

    assert len(calls) == 1
    assert [result.n_items_administered for result in results] == [5, 5]
    assert {result.stopping_reason for result in results} == {
        "Maximum items reached (5)"
    }


def _custom_stop_engine(model):
    return CATEngine(model, stopping_rule=ThetaChangeStop(), max_items=4)


def _instance_hook_engine(model):
    engine = CATEngine(model, max_items=4)
    engine.model.information = lambda theta, item_idx=None: np.ones(
        (len(theta), model.n_items)
    )
    return engine


class _Engine(CATEngine):
    pass


class _Selection(MaxFisherInformation):
    pass


@pytest.mark.parametrize(
    "factory",
    [
        lambda model: _Engine(model, max_items=4),
        lambda model: CATEngine(model, max_items=4, item_selection=_Selection()),
        lambda model: CATEngine(model, max_items=4, item_selection="MEI"),
        lambda model: CATEngine(model, max_items=4, scoring_method="MAP"),
        lambda model: CATEngine(model, max_items=4, exposure_control="randomesque"),
        lambda model: CATEngine(
            model,
            max_items=4,
            content_constraint=ContentBlueprint(
                [ContentArea("all", items=set(range(model.n_items)))]
            ),
        ),
        _custom_stop_engine,
        lambda model: CATEngine(
            model, stopping_rule=StandardErrorStop(0.3), min_items=5, max_items=4
        ),
        lambda model: CATEngine(model, max_items=4, n_quadpts=4),
        lambda model: CATEngine(model, max_items=4, n_quadpts=21.0),
        lambda model: CATEngine(model, max_items=4, initial_theta=np.nan),
        _instance_hook_engine,
    ],
    ids=[
        "engine-subclass",
        "selection-subclass",
        "mei",
        "map",
        "exposure",
        "content",
        "custom-stop",
        "min-above-max",
        "few-quadpts",
        "float-quadpts",
        "nan-start",
        "instance-hook",
    ],
)
def test_customized_configurations_keep_the_sequential_path(monkeypatch, factory):
    engine = factory(_model("2PL"))
    assert not lockstep.supports_lockstep(engine)

    def fail(*args, **kwargs):
        pytest.fail("customized configuration used lock-step simulation")

    monkeypatch.setattr("mirt.cat.engine.simulate_lockstep", fail)
    monkeypatch.setattr("mirt.cat.engine.lockstep_error_moments", fail)
    monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda requested: False)
    if lockstep.uses_native_defaults(engine, CATEngine) and np.isfinite(
        engine.initial_theta
    ):
        engine.run_batch_simulation([0.0])


@pytest.mark.parametrize(
    "model",
    [
        NominalResponseModel(n_items=4, n_categories=3),
        MultidimensionalModel(n_items=4, n_factors=2),
    ],
    ids=["unsupported-family", "multidimensional"],
)
def test_unsupported_models_keep_the_sequential_path(model):
    model._is_fitted = True
    assert not lockstep.supports_lockstep(CATEngine(model, max_items=3))


def test_customized_model_subclass_keeps_the_sequential_path():
    class Customized(TwoParameterLogistic):
        pass

    model = Customized(n_items=4)
    model._is_fitted = True
    assert not lockstep.supports_lockstep(CATEngine(model, max_items=3))


@pytest.mark.performance
def test_lockstep_batch_is_much_faster_than_sequential_sessions():
    model = _model("3PL", n_items=150)
    true_thetas = np.random.default_rng(2).normal(size=400)
    engine = CATEngine(model, se_threshold=0.3, max_items=20, seed=1)

    start = time.perf_counter()
    engine.run_batch_simulation(true_thetas[:40], vectorized=False)
    sequential = (time.perf_counter() - start) * 10.0
    start = time.perf_counter()
    engine.run_batch_simulation(true_thetas)
    vectorized = time.perf_counter() - start

    assert vectorized * 5.0 < sequential
