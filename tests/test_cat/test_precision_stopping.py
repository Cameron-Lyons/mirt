"""Precision-change, minimum-information, and predicted-SE-reduction stopping."""

import math

import numpy as np
import pytest

from mirt.cat import (
    CATEngine,
    MinInformationStop,
    PredictedSEReductionStop,
    SEChangeStop,
)
from mirt.cat._lockstep import supports_lockstep
from mirt.cat.results import CATState
from mirt.cat.stopping import CombinedStop, StandardErrorStop, create_stopping_rule
from mirt.models import GradedResponseModel, TwoParameterLogistic


def _binary_model(n_items: int = 200, seed: int = 0) -> TwoParameterLogistic:
    rng = np.random.default_rng(seed)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    return model


def _graded_model(n_items: int = 12, seed: int = 1) -> GradedResponseModel:
    rng = np.random.default_rng(seed)
    model = GradedResponseModel(n_items=n_items, n_categories=4)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, n_items),
        thresholds=np.sort(rng.normal(0.0, 1.0, (n_items, 3)), axis=1),
    )
    model._is_fitted = True
    return model


def _state(theta: float, se: float, administered: list[int]) -> CATState:
    return CATState(
        theta=theta,
        standard_error=se,
        items_administered=list(administered),
        responses=[1] * len(administered),
        n_items=len(administered),
    )


def _best_remaining(model, theta: float, administered: list[int]) -> float:
    theta_arr = np.array([[theta]])
    return max(
        float(np.sum(model.information(theta_arr, item_idx=item)))
        for item in range(model.n_items)
        if item not in administered
    )


class TestSEChangeStop:
    def test_stops_once_consecutive_changes_are_small(self):
        rule = SEChangeStop(threshold=0.01)
        sequence = [np.inf, 0.8, 0.5, 0.45, 0.445]

        decisions = [rule.should_stop(_state(0.0, se, [])) for se in sequence]

        assert decisions == [False, False, False, False, True]
        assert rule.get_reason() == "SE stabilized (change <= 0.01 for 1 items)"

    def test_threshold_is_inclusive_and_runs_must_be_consecutive(self):
        rule = SEChangeStop(threshold=0.25, n_stable=2)
        sequence = [1.5, 1.25, 0.75, 0.5, 0.25]

        decisions = [rule.should_stop(_state(0.0, se, [])) for se in sequence]

        assert decisions == [False, False, False, False, True]

    def test_non_finite_standard_errors_never_count_as_stable(self):
        rule = SEChangeStop(threshold=0.5)

        decisions = [
            rule.should_stop(_state(0.0, se, [])) for se in (np.inf, np.inf, np.nan)
        ]

        assert decisions == [False, False, False]

    def test_reset_forgets_previous_session(self):
        rule = SEChangeStop(threshold=0.01)
        rule.should_stop(_state(0.0, 0.5, []))

        rule.reset()

        assert rule.should_stop(_state(0.0, 0.5, [])) is False
        assert rule.should_stop(_state(0.0, 0.5, [])) is True

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"threshold": 0.0}, "positive"),
            ({"threshold": np.nan}, "finite"),
            ({"threshold": True}, "finite"),
            ({"n_stable": 0}, "at least 1"),
            ({"n_stable": 1.5}, "integer"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SEChangeStop(**kwargs)

    def test_factory_and_engine_name(self):
        rule = create_stopping_rule("se_change", threshold=0.02, n_stable=2)
        assert isinstance(rule, SEChangeStop)
        assert (rule.threshold, rule.n_stable) == (0.02, 2)

        engine = CATEngine(_binary_model(30), stopping_rule="se_change", seed=0)
        result = engine.run_simulation(0.0)
        assert result.stopping_reason.startswith("SE stabilized")


class TestMinInformationStop:
    def test_threshold_edge_uses_best_unadministered_item(self):
        model = _binary_model(40)
        administered = [0, 1, 2, 3, 4]
        best = _best_remaining(model, 0.3, administered)
        state = _state(0.3, 0.4, administered)

        assert MinInformationStop(model, best).should_stop(state) is False
        assert MinInformationStop(model, best * (1 + 1e-9)).should_stop(state) is True

    def test_administered_items_are_not_remaining(self):
        model = _binary_model(40)
        info = np.asarray(model.information(np.array([[0.0]]))).ravel()
        top = int(np.argmax(info))
        threshold = float(np.sort(info)[-1] + np.sort(info)[-2]) / 2.0

        rule = MinInformationStop(model, threshold)

        assert rule.should_stop(_state(0.0, 0.5, [])) is False
        assert rule.should_stop(_state(0.0, 0.5, [top])) is True

    def test_exhausted_pool_defers_to_engine(self):
        model = _binary_model(3)

        rule = MinInformationStop(model, 100.0)

        assert rule.should_stop(_state(0.0, 0.5, [0, 1, 2])) is False

    def test_polytomous_information_is_itemwise(self):
        model = _graded_model()
        administered = [2, 5]
        best = _best_remaining(model, 1.0, administered)
        state = _state(1.0, 0.5, administered)

        assert MinInformationStop(model, best).should_stop(state) is False
        assert MinInformationStop(model, best * 1.000001).should_stop(state) is True

    @pytest.mark.parametrize(
        ("threshold", "message"),
        [(0.0, "positive"), (np.inf, "finite"), ("0.1", "finite")],
    )
    def test_rejects_invalid_threshold(self, threshold, message):
        with pytest.raises(ValueError, match=message):
            MinInformationStop(_binary_model(5), threshold)

    def test_rejects_multidimensional_models(self):
        model = _binary_model(5)
        model.n_factors = 2

        with pytest.raises(ValueError, match="unidimensional"):
            MinInformationStop(model, 0.1)


class TestPredictedSEReductionStop:
    def test_reduction_edge_matches_information_update(self):
        model = _binary_model(40)
        administered = [7, 8, 9]
        se = 0.45
        best = _best_remaining(model, -0.2, administered)
        reduction = se - 1.0 / math.sqrt(se**-2 + best)
        state = _state(-0.2, se, administered)

        assert PredictedSEReductionStop(model, reduction).should_stop(state) is False
        assert (
            PredictedSEReductionStop(model, reduction * (1 + 1e-9)).should_stop(state)
            is True
        )

    @pytest.mark.parametrize(("se", "expected"), [(np.inf, False), (np.nan, False)])
    def test_non_finite_standard_error_never_stops(self, se, expected):
        rule = PredictedSEReductionStop(_binary_model(10), 10.0)

        assert rule.should_stop(_state(0.0, se, [])) is expected

    def test_zero_standard_error_cannot_be_reduced(self):
        rule = PredictedSEReductionStop(_binary_model(10), 1e-6)

        assert rule.should_stop(_state(0.0, 0.0, [])) is True

    def test_exhausted_pool_defers_to_engine(self):
        rule = PredictedSEReductionStop(_binary_model(2), 10.0)

        assert rule.should_stop(_state(0.0, 0.5, [0, 1])) is False

    def test_polytomous_model(self):
        model = _graded_model()
        best = _best_remaining(model, 0.5, [0])
        reduction = 0.6 - 1.0 / math.sqrt(0.6**-2 + best)

        rule = PredictedSEReductionStop(model, reduction * 1.000001)

        assert rule.should_stop(_state(0.5, 0.6, [0])) is True

    def test_rejects_invalid_reduction(self):
        with pytest.raises(ValueError, match="min_reduction must be positive"):
            PredictedSEReductionStop(_binary_model(5), -0.1)


@pytest.mark.parametrize(
    ("rule_factory", "reason"),
    [
        (
            lambda model: PredictedSEReductionStop(model, 0.01),
            "Predicted SE reduction below threshold (0.01)",
        ),
        (
            lambda model: MinInformationStop(model, 0.3),
            "Remaining item information below threshold (0.3)",
        ),
    ],
)
def test_extreme_examinees_stop_before_the_length_cap(rule_factory, reason):
    model = _binary_model()
    precision_only = CATEngine(model, se_threshold=0.3, max_items=40, seed=1)
    combined = CATEngine(
        model,
        stopping_rule=CombinedStop([StandardErrorStop(0.3), rule_factory(model)]),
        max_items=40,
        seed=1,
    )

    for theta in (3.5, -3.5):
        baseline = precision_only.run_simulation(theta)
        result = combined.run_simulation(theta)

        assert baseline.n_items_administered == 40
        assert result.n_items_administered < 30
        assert result.stopping_reason == reason


def test_engine_with_graded_model_and_reduction_rule():
    model = _graded_model(30)
    stopping = CombinedStop(
        [StandardErrorStop(0.2), PredictedSEReductionStop(model, 0.02)]
    )
    engine = CATEngine(model, stopping_rule=stopping, max_items=20, seed=1)

    result = engine.run_simulation(3.5)

    assert result.n_items_administered < 20
    assert result.stopping_reason.startswith("Predicted SE reduction")


def test_information_rules_use_independent_batch_sessions():
    model = _binary_model(60)
    stopping = CombinedStop([StandardErrorStop(0.3), MinInformationStop(model, 0.3)])
    engine = CATEngine(model, stopping_rule=stopping, max_items=15, seed=2)

    assert not supports_lockstep(engine)
    assert not engine._can_use_rust_simulation()
    results = engine.run_batch_simulation([-3.5, 0.0], use_rust=True)

    assert [len(result.theta_history) for result in results] == [
        result.n_items_administered for result in results
    ]
