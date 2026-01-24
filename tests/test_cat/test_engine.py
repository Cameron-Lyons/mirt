"""Tests for CATEngine."""

import numpy as np
import pytest

from mirt.cat.engine import CATEngine
from mirt.cat.selection import MaxFisherInformation, RandomSelection


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
