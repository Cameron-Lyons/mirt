"""Tests for CATEngine."""

from unittest.mock import patch

import numpy as np
import pytest

from mirt.backends.rust import is_rust_available
from mirt.cat.engine import CATEngine
from mirt.cat.selection import MaxFisherInformation, RandomSelection
from mirt.cat.stopping import ThetaChangeStop


class TestCATEngineInitialization:
    """Tests for CATEngine initialization."""

    def test_basic_initialization(self, fitted_2pl_model):
        """Test basic engine initialization."""
        model = fitted_2pl_model.model
        cat = CATEngine(model)

        assert cat.model is model
        assert cat.scoring_method == "EAP"
        assert cat.initial_theta == 0.0
        assert not cat._is_complete

    def test_initialization_with_string_selection(self, fitted_2pl_model):
        """Test initialization with string item selection."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, item_selection="MFI")
        assert isinstance(cat._selection, MaxFisherInformation)

        cat = CATEngine(model, item_selection="random")
        assert isinstance(cat._selection, RandomSelection)

    def test_initialization_with_strategy_object(self, fitted_2pl_model):
        """Test initialization with strategy object."""
        model = fitted_2pl_model.model
        strategy = RandomSelection(seed=42)
        cat = CATEngine(model, item_selection=strategy)
        assert cat._selection is strategy

    def test_initialization_with_string_stopping(self, fitted_2pl_model):
        """Test initialization with string stopping rule."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, stopping_rule="SE", se_threshold=0.4)
        assert cat._stopping is not None

    def test_initialization_with_max_items(self, fitted_2pl_model):
        """Test initialization with max_items."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=5)
        state = cat.get_current_state()
        assert not state.is_complete

    def test_initialization_with_custom_parameters(self, fitted_2pl_model):
        """Test initialization with custom parameters."""
        model = fitted_2pl_model.model
        cat = CATEngine(
            model,
            scoring_method="MAP",
            initial_theta=0.5,
            se_threshold=0.25,
            max_items=10,
            min_items=3,
            n_quadpts=15,
            theta_bounds=(-3.0, 3.0),
            seed=123,
        )

        assert cat.scoring_method == "MAP"
        assert cat.initial_theta == 0.5
        assert cat.n_quadpts == 15
        assert cat.theta_bounds == (-3.0, 3.0)
        assert cat.seed == 123

    def test_unfitted_model_raises_error(self):
        """Test that unfitted model raises ValueError."""
        from mirt.models.dichotomous import TwoParameterLogistic

        unfitted_model = TwoParameterLogistic(n_items=5)

        with pytest.raises(ValueError, match="Model must be fitted"):
            CATEngine(unfitted_model)


class TestCATEngineStateManagement:
    """Tests for CATEngine state management."""

    def test_get_current_state(self, fitted_2pl_model):
        """Test getting current state."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, seed=42)
        state = cat.get_current_state()

        assert state.theta == 0.0
        assert state.n_items == 0
        assert state.items_administered == []
        assert state.responses == []
        assert not state.is_complete
        assert state.next_item is not None

    def test_reset(self, fitted_2pl_model):
        """Test reset functionality."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=3, seed=42)

        for _ in range(3):
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(1)

        cat.reset()
        state = cat.get_current_state()

        assert state.theta == 0.0
        assert state.n_items == 0
        assert state.items_administered == []
        assert not state.is_complete

    def test_theta_history_tracking(self, fitted_2pl_model):
        """Test theta history is tracked."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=5, seed=42)

        for _ in range(5):
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(1)

        assert len(cat._theta_history) > 0
        assert len(cat._se_history) > 0


class TestCATEngineItemSelection:
    """Tests for item selection workflow."""

    def test_select_next_item(self, fitted_2pl_model):
        """Test selecting next item."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, seed=42)
        item = cat.select_next_item()

        assert isinstance(item, int)
        assert 0 <= item < model.n_items

    def test_disclosed_item_remains_pending_until_response(self, fitted_2pl_model):
        """The response must be recorded against the item shown to the user."""
        model = fitted_2pl_model.model
        cat = CATEngine(
            model,
            item_selection=RandomSelection(seed=42),
            max_items=5,
        )

        with patch.object(cat, "_select_next_item", wraps=cat._select_next_item) as spy:
            disclosed_item = cat.get_current_state().next_item
            assert cat.select_next_item() == disclosed_item
            assert spy.call_count == 1
            state = cat.administer_item(1)

        assert state.items_administered == [disclosed_item]
        assert spy.call_count == 2

    def test_completion_does_not_select_an_unused_item(self, fitted_2pl_model):
        """Stopping should occur before the engine computes another item."""
        cat = CATEngine(fitted_2pl_model.model, max_items=1)

        with patch.object(cat, "_select_next_item", wraps=cat._select_next_item) as spy:
            cat.select_next_item()
            state = cat.administer_item(1)

        assert state.is_complete
        assert state.next_item is None
        assert spy.call_count == 1

    def test_reset_discards_pending_item(self, fitted_2pl_model):
        """A new session must not inherit an unadministered item."""
        cat = CATEngine(fitted_2pl_model.model, item_selection="random", seed=42)
        cat.select_next_item()

        cat.reset()

        assert not hasattr(cat, "_pending_item")

    def test_select_item_complete_raises_error(self, fitted_2pl_model):
        """Test that selecting when complete raises error."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=1, seed=42)
        cat.administer_item(1)

        with pytest.raises(RuntimeError, match="complete"):
            cat.select_next_item()

    def test_items_not_reselected(self, fitted_2pl_model):
        """Test that administered items are not reselected."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=5, seed=42)
        selected = set()

        for _ in range(5):
            state = cat.get_current_state()
            if state.is_complete:
                break
            item = state.next_item
            assert item not in selected
            selected.add(item)
            cat.administer_item(1)


class TestCATEngineResponseAdministration:
    """Tests for response administration and theta updates."""

    def test_administer_item(self, fitted_2pl_model):
        """Test administering an item."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, seed=42)
        initial_state = cat.get_current_state()
        initial_item = initial_state.next_item

        new_state = cat.administer_item(1)

        assert new_state.n_items == 1
        assert initial_item in new_state.items_administered
        assert new_state.responses == [1]

    def test_theta_updates_after_response(self, fitted_2pl_model):
        """Test that theta updates after response."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, seed=42)
        initial_state = cat.get_current_state()
        initial_theta = initial_state.theta

        cat.administer_item(1)
        new_state = cat.get_current_state()

        assert new_state.theta != initial_theta or new_state.n_items == 1

    def test_administer_complete_raises_error(self, fitted_2pl_model):
        """Test that administering when complete raises error."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=1, seed=42)
        cat.administer_item(1)

        with pytest.raises(RuntimeError, match="complete"):
            cat.administer_item(1)

    def test_responses_recorded(self, fitted_2pl_model):
        """Test that responses are recorded correctly."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=3, seed=42)

        responses = [1, 0, 1]
        for resp in responses:
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(resp)

        state = cat.get_current_state()
        assert state.responses[: len(responses)] == responses[: state.n_items]


class TestCATEngineStoppingRules:
    """Tests for stopping rule integration."""

    def test_max_items_stopping(self, fitted_2pl_model):
        """Test max items stopping rule."""
        model = fitted_2pl_model.model
        max_items = 3
        cat = CATEngine(model, max_items=max_items, seed=42)

        for i in range(max_items + 5):
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(1)

        assert cat._is_complete
        assert len(cat._items_administered) <= max_items

    def test_se_threshold_stopping(self, fitted_2pl_model):
        """Test SE threshold stopping rule."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, se_threshold=1.0, max_items=50, seed=42)

        for _ in range(50):
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(1)

        assert cat._is_complete
        result = cat.get_result()
        assert "SE" in result.stopping_reason or "Max" in result.stopping_reason

    def test_min_items_respected(self, fitted_2pl_model):
        """Test that minimum items is respected."""
        model = fitted_2pl_model.model
        min_items = 3
        cat = CATEngine(
            model, se_threshold=10.0, min_items=min_items, max_items=10, seed=42
        )

        for _ in range(min_items):
            state = cat.get_current_state()
            if state.is_complete:
                break
            cat.administer_item(1)

        assert len(cat._items_administered) >= min_items or cat._is_complete


class TestCATEngineSimulation:
    """Tests for CAT simulation."""

    def test_run_simulation(self, fitted_2pl_model):
        """Test running a single simulation."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=5, seed=42)
        result = cat.run_simulation(true_theta=0.0)

        assert result.n_items_administered <= 5
        assert len(result.items_administered) == result.n_items_administered
        assert len(result.responses) == result.n_items_administered

    def test_simulation_recovers_theta(self, fitted_2pl_model):
        """Test that simulation recovers true theta reasonably."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, se_threshold=0.5, max_items=20, seed=42)

        true_theta = 1.0
        result = cat.run_simulation(true_theta=true_theta)

        assert abs(result.theta - true_theta) < 2.0

    def test_batch_simulation(self, fitted_2pl_model):
        """Test batch simulation."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=5, seed=42)
        true_thetas = np.array([-1.0, 0.0, 1.0])

        results = cat.run_batch_simulation(
            true_thetas, n_replications=1, use_rust=False
        )

        assert len(results) == 3
        for result in results:
            assert result.n_items_administered <= 5

    def test_native_batch_reconstructs_complete_results(
        self,
        fitted_2pl_model,
        monkeypatch,
    ):
        """Native summaries retain exact administered item and response paths."""
        model = fitted_2pl_model.model
        cat = CATEngine(
            model,
            se_threshold=0.45,
            min_items=2,
            max_items=3,
            seed=42,
        )
        payload = (
            np.array([0.1, 0.2]),
            np.array([0.5, 0.3]),
            np.array([3, 2], dtype=np.int32),
            np.array([-1.0, 1.0]),
            np.array([[4, 1, 3], [2, 0, -1]], dtype=np.int32),
            np.array([[1, 0, 1], [0, 1, -1]], dtype=np.int32),
        )
        monkeypatch.setattr("mirt.cat.engine.should_use_rust", lambda value: True)
        monkeypatch.setattr(
            "mirt.cat.engine.rust_cat_simulate_batch_full",
            lambda *args: payload,
        )

        results = cat.run_batch_simulation([-1.0, 1.0], use_rust=True)

        assert cat._get_stopping_parameters() == (0.45, 3, 2)
        assert [result.items_administered for result in results] == [[4, 1, 3], [2, 0]]
        assert [result.responses.tolist() for result in results] == [[1, 0, 1], [0, 1]]
        assert [result.n_items_administered for result in results] == [3, 2]
        assert results[0].stopping_reason == "Maximum items reached (3)"
        assert results[1].stopping_reason == "SE threshold reached (SE <= 0.45)"

    def test_native_batch_requires_matching_engine_semantics(self, fitted_2pl_model):
        """Unsupported selection, scoring, start, and stopping rules use Python."""
        model = fitted_2pl_model.model

        assert not CATEngine(model, scoring_method="MAP")._can_use_rust_simulation()
        assert not CATEngine(model, initial_theta=0.5)._can_use_rust_simulation()
        assert not CATEngine(model, item_selection="random")._can_use_rust_simulation()
        assert not CATEngine(
            model,
            stopping_rule=ThetaChangeStop(),
        )._can_use_rust_simulation()

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"true_thetas": []}, "non-empty"),
            ({"true_thetas": [np.nan]}, "finite"),
            ({"true_thetas": ["bad"]}, "numeric"),
            ({"n_replications": 0}, "positive integer"),
            ({"n_replications": True}, "positive integer"),
            ({"use_rust": "yes"}, "boolean"),
        ],
    )
    def test_batch_simulation_validates_controls(
        self,
        fitted_2pl_model,
        arguments,
        message,
    ):
        """Batch controls fail consistently before backend selection."""
        cat = CATEngine(fitted_2pl_model.model, max_items=3)
        defaults = {"true_thetas": [0.0], "n_replications": 1, "use_rust": False}
        defaults.update(arguments)

        with pytest.raises(ValueError, match=message):
            cat.run_batch_simulation(**defaults)

    def test_installed_native_batch_returns_complete_results(self, fitted_2pl_model):
        """The installed native extension satisfies portable result invariants."""
        if not is_rust_available():
            pytest.skip("native backend is not installed")
        cat = CATEngine(
            fitted_2pl_model.model,
            se_threshold=0.4,
            min_items=2,
            max_items=5,
            seed=71,
        )

        results = cat.run_batch_simulation([-1.0, 0.0, 1.0], use_rust=True)

        assert len(results) == 3
        for result in results:
            assert 2 <= result.n_items_administered <= 5
            assert len(result.items_administered) == result.n_items_administered
            assert len(result.responses) == result.n_items_administered
            assert len(set(result.items_administered)) == result.n_items_administered
            assert np.isin(result.responses, [0, 1]).all()

    @pytest.mark.slow
    def test_batch_simulation_many_examinees(self, fitted_2pl_model):
        """Larger CAT batch for the weekly slow suite."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=8, se_threshold=0.4, seed=7)
        true_thetas = np.linspace(-2.0, 2.0, 40)
        results = cat.run_batch_simulation(true_thetas, n_replications=2, use_rust=True)
        assert len(results) == 80


class TestCATEngineGetResult:
    """Tests for getting CAT results."""

    def test_get_result_incomplete_raises_error(self, fitted_2pl_model):
        """Test that get_result raises error when incomplete."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=10, seed=42)

        with pytest.raises(RuntimeError, match="not complete"):
            cat.get_result()

    def test_get_result_complete(self, fitted_2pl_model):
        """Test getting result when complete."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, max_items=2, seed=42)

        cat.administer_item(1)
        cat.administer_item(0)

        result = cat.get_result()

        assert result.theta == cat._current_theta
        assert result.standard_error == cat._current_se
        assert result.n_items_administered == 2
        assert len(result.stopping_reason) > 0


class TestCATEngineRepr:
    """Tests for string representation."""

    def test_repr(self, fitted_2pl_model):
        """Test __repr__ method."""
        model = fitted_2pl_model.model
        cat = CATEngine(model, seed=42)
        repr_str = repr(cat)

        assert "CATEngine" in repr_str
        assert "n_items" in repr_str


def _stratified_pool(n_items: int = 60):
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(0)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=np.linspace(0.5, 2.5, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    return model


class TestCATEngineSelectionHorizon:
    """Length-aware selection plans over the length the rules enforce."""

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({}, 60),
            ({"max_items": 12}, 12),
            ({"stopping_rule": "max_items", "max_items": 12}, 12),
            ({"max_items": 12, "min_items": 12}, 12),
            ({"max_items": 90}, 60),
        ],
    )
    def test_horizon_from_engine_options(self, options, expected):
        engine = CATEngine(_stratified_pool(), **options)

        assert engine._selection_horizon == expected

    def test_horizon_from_stopping_rule_objects(self):
        from mirt.cat.stopping import CombinedStop, MaxItemsStop, StandardErrorStop

        model = _stratified_pool()
        assert CATEngine(model, stopping_rule=MaxItemsStop(12))._selection_horizon == 12
        nested = CombinedStop([StandardErrorStop(0.2), MaxItemsStop(9)])
        assert CATEngine(model, stopping_rule=nested)._selection_horizon == 9
        assert (
            CATEngine(model, stopping_rule=MaxItemsStop(20), max_items=12)
        )._selection_horizon == 12

    def test_modified_length_rules_do_not_shorten_the_horizon(self):
        """Only an unmodified MaxItemsStop is trusted to stop by its limit."""
        from mirt.cat.shadow import ShadowTestSelection
        from mirt.cat.stopping import MaxItemsStop

        model = _stratified_pool(40)
        rule = MaxItemsStop(5)
        rule.should_stop = lambda state: state.n_items >= 15
        engine = CATEngine(
            model, item_selection=ShadowTestSelection(), stopping_rule=rule, seed=3
        )

        assert engine._selection_horizon == 40
        assert len(engine.run_simulation(0.0).items_administered) == 15

    def test_horizon_follows_a_rule_changed_after_construction(self):
        from mirt.cat.shadow import ShadowTestSelection
        from mirt.cat.stopping import MaxItemsStop

        rule = MaxItemsStop(5)
        engine = CATEngine(
            _stratified_pool(40),
            item_selection=ShadowTestSelection(),
            stopping_rule=rule,
            seed=3,
        )
        rule.max_items = 15

        assert engine._selection_horizon == 15
        assert len(engine.run_simulation(0.0).items_administered) == 15

    def test_a_stratified_selection_spans_strata_with_a_length_rule(self):
        """A MaxItemsStop rule schedules strata like max_items does."""
        from mirt.cat.selection import AStratified
        from mirt.cat.stopping import MaxItemsStop

        model = _stratified_pool()
        by_rule = CATEngine(
            model,
            item_selection=AStratified(n_strata=3),
            stopping_rule=MaxItemsStop(12),
            seed=1,
        ).run_simulation(0.0)
        by_option = CATEngine(
            model,
            item_selection=AStratified(n_strata=3),
            se_threshold=1e-6,
            max_items=12,
            min_items=12,
            seed=1,
        ).run_simulation(0.0)

        assert by_rule.items_administered == by_option.items_administered
        strata = np.asarray(by_rule.items_administered) // 20
        assert sorted(set(strata.tolist())) == [0, 1, 2]

    def test_mcat_horizon_from_stopping_rule(self):
        from mirt.cat import MCATEngine
        from mirt.cat.mcat_stopping import MaxItemsMCATStop
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=20, n_factors=2)
        model._is_fitted = True

        assert MCATEngine(model)._selection_horizon == 20
        engine = MCATEngine(model, stopping_rule=MaxItemsMCATStop(7))
        assert engine._selection_horizon == 7
        named = MCATEngine(model, stopping_rule="max_items", max_items=5)
        assert named._selection_horizon == 5
        assert isinstance(named._stopping, MaxItemsMCATStop)


class TestCATEngineNamedStoppingRules:
    """Every documented stopping-rule name builds an engine."""

    @pytest.mark.parametrize("name", ["SE", "max_items", "theta_change", "se_change"])
    def test_documented_names_construct(self, name):
        engine = CATEngine(_stratified_pool(20), stopping_rule=name, max_items=10)

        result = engine.run_simulation(0.0)
        assert 1 <= result.n_items_administered <= 10

    def test_max_items_name_uses_max_items(self):
        from mirt.cat.stopping import MaxItemsStop

        engine = CATEngine(_stratified_pool(20), stopping_rule="max_items", max_items=4)

        assert isinstance(engine._stopping, MaxItemsStop)
        assert engine.run_simulation(0.0).n_items_administered == 4

    def test_max_items_name_requires_max_items(self):
        with pytest.raises(ValueError, match="requires max_items"):
            CATEngine(_stratified_pool(20), stopping_rule="max_items")

    def test_classification_name_asks_for_a_rule_object(self):
        with pytest.raises(ValueError, match="ClassificationStop"):
            CATEngine(_stratified_pool(20), stopping_rule="classification")

    @pytest.mark.parametrize("name", ["min_items", "combined"])
    def test_names_needing_arguments_are_rejected(self, name):
        with pytest.raises(ValueError, match="needs arguments"):
            CATEngine(_stratified_pool(20), stopping_rule=name)

    def test_unknown_names_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown stopping rule"):
            CATEngine(_stratified_pool(20), stopping_rule="sometimes")


def _location_pool(n_items: int = 40, seed: int = 0):
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(seed)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    return model


class TestCATEngineScoringConfiguration:
    """Scoring settings are validated when the engine is built."""

    @pytest.mark.parametrize("name", ["eap", "Eap", " EAP "])
    def test_scoring_method_names_ignore_case(self, name):
        model = _location_pool()

        def answers(item, theta):
            return int(item % 3 != 0)

        expected = CATEngine(model, max_items=6).run_simulation(0.0, answers)
        engine = CATEngine(model, scoring_method=name, max_items=6)
        actual = engine.run_simulation(0.0, answers)

        # A lower-case name used to fall back to logit(proportion correct).
        assert engine.scoring_method == "EAP"
        np.testing.assert_allclose(actual.theta_history, expected.theta_history)
        np.testing.assert_allclose(actual.se_history, expected.se_history)

    @pytest.mark.parametrize("name", ["bogus", "MLE", "PV", "EAPsum"])
    def test_unknown_scoring_methods_are_rejected(self, name):
        with pytest.raises(ValueError, match="Unknown CATEngine scoring_method"):
            CATEngine(_location_pool(), scoring_method=name)

    def test_invalid_theta_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="theta_bounds"):
            CATEngine(_location_pool(), scoring_method="ML", theta_bounds=(2.0, -2.0))

    def test_ml_extreme_pattern_reports_bound_with_infinite_se(self):
        engine = CATEngine(
            _location_pool(), scoring_method="ML", theta_bounds=(-3.0, 3.0), max_items=4
        )

        result = engine.run_simulation(0.0, lambda item, theta: 1)

        assert result.theta_history == [3.0] * 4
        assert result.se_history == [np.inf] * 4

    def test_numerical_scoring_failure_warns_and_uses_score_proportion(
        self, monkeypatch
    ):
        engine = CATEngine(_location_pool(), max_items=3)

        def fail(engine):
            raise ValueError("model likelihoods must produce finite posterior mass")

        monkeypatch.setattr("mirt.cat._eap.score_binary_eap", fail)
        engine.select_next_item()
        with pytest.warns(RuntimeWarning, match="proportion of the maximum score"):
            state = engine.administer_item(1)

        from mirt.constants import PROB_CLIP_MAX

        assert state.theta == pytest.approx(np.log(PROB_CLIP_MAX / (1 - PROB_CLIP_MAX)))

    def test_configuration_errors_during_scoring_propagate(self, monkeypatch):
        engine = CATEngine(_location_pool(), max_items=3)

        def fail(engine):
            raise ValueError("Unknown scoring method: eap")

        monkeypatch.setattr("mirt.cat._eap.score_binary_eap", fail)
        engine.select_next_item()
        with pytest.raises(ValueError, match="Unknown scoring method"):
            engine.administer_item(1)


class TestCATEngineTestLength:
    """Length settings that cannot be honoured are rejected."""

    @pytest.mark.parametrize(
        "options",
        [
            {"max_items": 10, "min_items": 20},
            {"max_items": 10, "min_items": 20, "stopping_rule": "max_items"},
            {"min_items": 20, "stopping_rule": "max-items", "max_items": 10},
        ],
    )
    def test_min_items_above_max_items_is_rejected(self, options):
        # Such tests used to run 20 items while reporting "Maximum items (10)".
        with pytest.raises(ValueError, match=r"min_items \(20\) exceeds"):
            CATEngine(_location_pool(), **options)

    def test_min_items_above_a_length_rule_is_rejected(self):
        from mirt.cat.stopping import MaxItemsStop

        with pytest.raises(ValueError, match=r"min_items \(12\) exceeds"):
            CATEngine(_location_pool(), stopping_rule=MaxItemsStop(8), min_items=12)

    @pytest.mark.parametrize("min_items", [-3, 1.5, True])
    def test_invalid_min_items_are_rejected(self, min_items):
        with pytest.raises(ValueError, match="min_items"):
            CATEngine(_location_pool(), min_items=min_items)

    def test_min_items_may_equal_max_items(self):
        engine = CATEngine(_location_pool(), max_items=5, min_items=5)
        assert engine.run_simulation(0.0).n_items_administered == 5

    def test_multidimensional_models_are_rejected(self):
        from mirt.models import MultidimensionalModel, TwoParameterLogistic

        for model in (
            TwoParameterLogistic(n_items=6, n_factors=2),
            MultidimensionalModel(n_items=6, n_factors=2),
        ):
            model._is_fitted = True
            with pytest.raises(ValueError, match="Use MCATEngine"):
                CATEngine(model)


class TestCATEngineNameNormalization:
    """Engine string options resolve names like every factory."""

    @pytest.mark.parametrize("name", ["se", " Se ", "SE"])
    def test_se_rule_names(self, name):
        from mirt.cat.stopping import CombinedStop, StandardErrorStop

        engine = CATEngine(_location_pool(), stopping_rule=name, se_threshold=0.4)
        assert isinstance(engine._stopping, StandardErrorStop)
        assert engine._stopping.threshold == 0.4
        engine = CATEngine(_location_pool(), stopping_rule=name, max_items=4)
        assert isinstance(engine._stopping, CombinedStop)

    def test_max_items_name_variants_use_one_rule(self):
        from mirt.cat.stopping import MaxItemsStop

        engine = CATEngine(_location_pool(), stopping_rule="Max-Items", max_items=4)
        assert isinstance(engine._stopping, MaxItemsStop)

    def test_exposure_control_name_variants(self):
        from mirt.cat.exposure import SympsonHetter

        engine = CATEngine(_location_pool(), exposure_control="sympson_hetter")
        assert isinstance(engine._exposure, SympsonHetter)


class TestCATEngineScoringPrior:
    """EAP and MAP updates use the configured or fitted latent population."""

    @pytest.mark.parametrize("method", ["EAP", "MAP"])
    def test_prior_is_used_in_every_update(self, method):
        from mirt.scoring import fscores

        model = _location_pool()
        engine = CATEngine(
            model,
            scoring_method=method,
            prior_mean=0.5,
            prior_sd=0.8,
            max_items=5,
            min_items=5,
        )
        result = engine.run_simulation(0.0, lambda item, theta: item % 2)

        assert engine.initial_theta == 0.5
        responses = np.full((1, model.n_items), -1)
        responses[0, result.items_administered] = result.responses
        reference = fscores(
            model,
            responses,
            method=method,
            n_quadpts=21,
            prior_mean=np.array([0.5]),
            # The engine's variance is prior_sd**2, one ulp above 0.64. MAP
            # errors are second differences of the log posterior, whose
            # rounding can turn that last bit into a 1e-6 change in the error.
            prior_cov=np.array([[0.8**2]]),
            bounds=(-4.0, 4.0),
        )
        np.testing.assert_allclose(result.theta, reference.theta.ravel()[0], atol=1e-8)
        np.testing.assert_allclose(
            result.standard_error, reference.standard_error.ravel()[0], atol=1e-8
        )

    def test_fit_result_supplies_the_prior(self):
        from mirt.results.fit_result import FitResult

        model = _location_pool()
        fit = FitResult(
            model=model,
            log_likelihood=0.0,
            n_iterations=1,
            converged=True,
            standard_errors={},
            aic=0.0,
            bic=0.0,
            latent_mean=np.array([0.5]),
            latent_covariance=np.array([[0.64]]),
        )
        from_fit = CATEngine(fit, max_items=4)
        explicit = CATEngine(model, prior_mean=0.5, prior_sd=0.8, max_items=4)

        assert from_fit.model is model
        assert (from_fit.prior_mean, from_fit.prior_sd) == (0.5, pytest.approx(0.8))
        answers = lambda item, theta: item % 2  # noqa: E731
        assert from_fit.run_simulation(0.0, answers).theta_history == pytest.approx(
            explicit.run_simulation(0.0, answers).theta_history
        )

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"prior_sd": 0.0}, "prior_sd"),
            ({"prior_sd": np.nan}, "prior_sd"),
            ({"prior_mean": np.inf}, "prior_mean"),
        ],
    )
    def test_invalid_priors_are_rejected(self, options, message):
        with pytest.raises(ValueError, match=message):
            CATEngine(_location_pool(), **options)

    @pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
    def test_native_simulation_uses_the_prior(self):
        from mirt.scoring import fscores

        model = _location_pool()
        engine = CATEngine(
            model, prior_mean=0.3, prior_sd=0.6, initial_theta=0.0, max_items=6, seed=4
        )
        assert engine._can_use_rust_simulation()
        results = engine.run_batch_simulation([0.0, 1.0], n_replications=3)

        responses = np.full((len(results), model.n_items), -1)
        for row, result in enumerate(results):
            responses[row, result.items_administered] = result.responses
        reference = fscores(
            model,
            responses,
            n_quadpts=21,
            prior_mean=np.array([0.3]),
            prior_cov=np.array([[0.36]]),
        )
        np.testing.assert_allclose(
            [result.theta for result in results], reference.theta.ravel(), atol=1e-10
        )


@pytest.mark.skipif(not is_rust_available(), reason="Rust backend not available")
class TestCATEngineNativeSeeds:
    """Native batch sessions draw independent, advancing random streams."""

    @staticmethod
    def _engine(seed):
        return CATEngine(_location_pool(200), se_threshold=0.3, max_items=30, seed=seed)

    def test_neighbouring_seeds_do_not_replay_sessions(self):
        first = self._engine(10).run_batch_simulation([0.0], n_replications=6)
        second = self._engine(11).run_batch_simulation([0.0], n_replications=6)

        # Seed 11 replication r used to equal seed 10 replication r + 1.
        assert not any(
            first[r + 1].items_administered == second[r].items_administered
            and first[r + 1].theta == second[r].theta
            for r in range(5)
        )

    def test_many_replications_do_not_share_streams_across_thetas(self):
        results = self._engine(5).run_batch_simulation([0.5, 0.5], n_replications=1001)

        assert (
            results[1000].items_administered != results[1001].items_administered
            or results[1000].theta != results[1001].theta
        )

    def test_successive_calls_advance_and_seeds_reproduce(self):
        engine = self._engine(7)
        first = engine.run_batch_simulation([0.0] * 5)
        second = engine.run_batch_simulation([0.0] * 5)
        replay = self._engine(7).run_batch_simulation([0.0] * 5)

        def paths(results):
            return [result.items_administered for result in results]

        assert paths(first) != paths(second)
        assert paths(first) == paths(replay)

    def test_conditional_mse_calls_advance_and_seeds_reproduce(self):
        engine = self._engine(3)
        first = engine.compute_conditional_mse([0.0, 1.0], n_replications=50)
        second = engine.compute_conditional_mse([0.0, 1.0], n_replications=50)
        replay = self._engine(3).compute_conditional_mse([0.0, 1.0], n_replications=50)

        # A seeded engine used to return the same "new" estimate every call.
        assert not np.array_equal(first[2], second[2])
        for expected, actual in zip(first, replay, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_conditional_mse_seeds_are_independent_of_neighbours(self):
        def errors(seed):
            engine = self._engine(seed)
            return engine.compute_conditional_mse([0.0], n_replications=2000)[1:]

        # Seeds 10 and 11 used to share 1999 of their 2000 sessions, which
        # bounds the differences by about 0.002 (bias) and 0.015 (length).
        bias_10, _, length_10 = errors(10)
        bias_11, _, length_11 = errors(11)
        assert (
            abs(length_10[0] - length_11[0]) > 0.015
            or abs(bias_10[0] - bias_11[0]) > 0.004
        )


@pytest.mark.skipif(not is_rust_available(), reason="Rust extension unavailable")
class TestNativeCATKernelSeeds:
    """Every native session draws from a stream mixed from (seed, theta, rep)."""

    @staticmethod
    def _batch(thetas, n_replications, seed):
        from mirt import mirt_rs

        model = _location_pool(200)
        points = np.linspace(-4.0, 4.0, 41)
        weights = np.exp(-0.5 * points**2)
        return mirt_rs.cat_simulate_batch(
            np.asarray(thetas, dtype=np.float64),
            model.parameters["discrimination"],
            model.parameters["difficulty"],
            points,
            weights / weights.sum(),
            0.3,
            30,
            1,
            n_replications,
            seed,
        )

    @staticmethod
    def _paths(payload):
        return [tuple(path) for path in np.asarray(payload[4])]

    def test_neighbouring_seeds_and_thetas_do_not_share_streams(self):
        # Additive seeds replayed replication r + 1 of seed s as replication r
        # of seed s + 1, and replication 1000 of one theta as replication 0 of
        # the next.
        first = self._paths(self._batch([0.0], 6, 10))
        second = self._paths(self._batch([0.0], 6, 11))
        assert not set(first[1:]) & set(second[:-1])
        many = self._paths(self._batch([0.5, 0.5], 1001, 5))
        assert many[1000] != many[1001]
        assert self._paths(self._batch([0.0], 6, 10)) == first


class TestNativeCATCalls:
    """Native calls are split into blocks that cover every session once."""

    @pytest.mark.parametrize(
        ("n_replications", "chunk", "expected"),
        [
            (3, 7, [(0, 2, 3), (2, 2, 3), (4, 1, 3)]),
            (20, 7, [(t, 1, r) for t in range(5) for r in (7, 7, 6)]),
        ],
    )
    def test_blocks_cover_sessions_and_draw_fresh_seeds(
        self, n_replications, chunk, expected
    ):
        engine = CATEngine(_location_pool(), seed=1)
        thetas = np.linspace(-1.0, 1.0, 5)
        seen = []

        def kernel(block, replications, seed):
            seen.append((tuple(block), replications, seed))
            return block.size

        blocks = list(engine._native_calls(thetas, n_replications, kernel, chunk))

        assert [block[:3] for block in blocks] == expected
        assert [block[3] for block in blocks] == [count for _, count, _ in expected]
        for (start, count, _), (block, _, _) in zip(expected, seen, strict=True):
            assert block == tuple(thetas[start : start + count])
        assert len({seed for *_, seed in seen}) == len(seen)

    def test_conditional_mse_pools_split_replications(self, monkeypatch):
        import mirt.cat.engine as engine_module

        def fake_mse(thetas, *args):
            replications = args[-2]
            # Each call reports moments equal to its replication count.
            moments = np.full(len(thetas), float(replications))
            return thetas, moments, 2 * moments, 3 * moments

        engine = CATEngine(_location_pool(), seed=2)
        monkeypatch.setattr(engine_module, "_NATIVE_MSE_SESSIONS", 7)
        monkeypatch.setattr(engine_module, "rust_cat_conditional_mse", fake_mse)
        monkeypatch.setattr(engine, "_can_use_rust_simulation", lambda: True)
        monkeypatch.setattr(engine_module, "should_use_rust", lambda use_rust: True)

        _, bias, mse, lengths = engine.compute_conditional_mse([0.0, 1.0], 20)

        # Calls of 7, 7 and 6 replications pool to (49 + 49 + 36) / 20.
        np.testing.assert_allclose(bias, [6.7, 6.7])
        np.testing.assert_allclose(mse, [13.4, 13.4])
        np.testing.assert_allclose(lengths, [20.1, 20.1])
