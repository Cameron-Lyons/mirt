"""Tests for exposure control methods."""

import numpy as np
import pytest

from mirt.cat.exposure import (
    NoExposureControl,
    ProgressiveRestricted,
    Randomesque,
    SympsonHetter,
    create_exposure_control,
)


class TestNoExposureControl:
    """Tests for NoExposureControl."""

    def test_filter_items_returns_all(self, fitted_2pl_model):
        """Test that all items are returned."""
        model = fitted_2pl_model.model
        no_control = NoExposureControl()
        available = set(range(model.n_items))

        filtered = no_control.filter_items(available, model, theta=0.0)

        assert filtered == available

    def test_filter_items_subset(self, fitted_2pl_model):
        """Test filtering with subset of items."""
        model = fitted_2pl_model.model
        no_control = NoExposureControl()
        available = {0, 2, 4}

        filtered = no_control.filter_items(available, model, theta=0.0)

        assert filtered == available

    def test_update_does_nothing(self, fitted_2pl_model):
        """Test that update does nothing."""
        no_control = NoExposureControl()
        no_control.update(0)

    def test_reset_does_nothing(self, fitted_2pl_model):
        """Test that reset does nothing."""
        no_control = NoExposureControl()
        no_control.reset()


class TestSympsonHetter:
    """Tests for Sympson-Hetter exposure control."""

    def test_initialization_default(self):
        """Test default initialization."""
        sh = SympsonHetter()
        assert sh.target_rate == 0.25
        assert sh._params == {}

    def test_initialization_with_params_array(self):
        """Test initialization with array parameters."""
        params = np.array([0.5, 0.8, 1.0])
        sh = SympsonHetter(exposure_params=params)

        assert sh._params[0] == 0.5
        assert sh._params[1] == 0.8
        assert sh._params[2] == 1.0

    def test_initialization_with_params_dict(self):
        """Test initialization with dict parameters."""
        params = {0: 0.5, 2: 0.8}
        sh = SympsonHetter(exposure_params=params)

        assert sh._params[0] == 0.5
        assert sh._params[2] == 0.8

    def test_filter_items_probabilistic(self, fitted_2pl_model):
        """Test that filtering is probabilistic."""
        model = fitted_2pl_model.model
        params = {i: 0.5 for i in range(model.n_items)}
        sh = SympsonHetter(exposure_params=params, seed=42)
        available = set(range(model.n_items))

        filtered_sets = []
        for _ in range(20):
            filtered = sh.filter_items(available, model, theta=0.0)
            filtered_sets.append(tuple(sorted(filtered)))

        unique_sets = set(filtered_sets)
        assert len(unique_sets) > 1

    def test_filter_items_guarantees_at_least_one(self, fitted_2pl_model):
        """Test that at least one item is always returned."""
        model = fitted_2pl_model.model
        params = {i: 0.001 for i in range(model.n_items)}
        sh = SympsonHetter(exposure_params=params, seed=42)
        available = set(range(model.n_items))

        for _ in range(10):
            filtered = sh.filter_items(available, model, theta=0.0)
            assert len(filtered) >= 1

    def test_filter_items_empty_returns_empty(self, fitted_2pl_model):
        """Test that empty available returns empty."""
        model = fitted_2pl_model.model
        sh = SympsonHetter()
        filtered = sh.filter_items(set(), model, theta=0.0)
        assert filtered == set()

    def test_update_tracks_selection(self):
        """Test that update tracks selections."""
        sh = SympsonHetter()

        sh.update(0)
        sh.update(0)
        sh.update(1)

        assert sh._selection_counts[0] == 2
        assert sh._selection_counts[1] == 1

    def test_reset_increments_examinees(self):
        """Test that reset increments examinee count."""
        sh = SympsonHetter()

        assert sh._n_examinees == 0

        sh.reset()
        assert sh._n_examinees == 1

        sh.reset()
        assert sh._n_examinees == 2

    def test_calibrate_adjusts_params(self, fitted_2pl_model):
        """Test that calibration adjusts parameters."""
        model = fitted_2pl_model.model
        sh = SympsonHetter(target_rate=0.25, seed=42)

        for _ in range(4):
            sh.update(0)
            sh.reset()

        sh.calibrate(n_items=model.n_items)

        assert 0 in sh._params
        assert sh._params[0] <= 1.0

    def test_get_exposure_rates(self):
        """Test getting exposure rates."""
        sh = SympsonHetter()

        sh.update(0)
        sh.reset()
        sh.update(0)
        sh.update(1)
        sh.reset()

        rates = sh.get_exposure_rates()

        assert rates[0] == 1.0
        assert rates[1] == 0.5


class TestRandomesque:
    """Tests for Randomesque exposure control."""

    def test_initialization(self):
        """Test initialization with k."""
        rand = Randomesque(k=5, seed=42)
        assert rand.k == 5

    def test_initialization_invalid_k(self):
        """Test that k < 1 raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            Randomesque(k=0)

    def test_filter_items_returns_all(self, fitted_2pl_model):
        """Test that filter_items returns all available."""
        model = fitted_2pl_model.model
        rand = Randomesque(k=3)
        available = set(range(model.n_items))

        filtered = rand.filter_items(available, model, theta=0.0)

        assert filtered == available

    def test_select_from_ranked_top_k(self):
        """Test selection from top-k ranked items."""
        rand = Randomesque(k=3, seed=42)
        ranked = [(0, 10.0), (1, 8.0), (2, 6.0), (3, 4.0), (4, 2.0)]

        selections = [rand.select_from_ranked(ranked) for _ in range(100)]

        unique_selected = set(selections)
        assert unique_selected.issubset({0, 1, 2})

    def test_select_from_ranked_small_pool(self):
        """Test selection when pool smaller than k."""
        rand = Randomesque(k=5, seed=42)
        ranked = [(0, 10.0), (1, 8.0)]

        item = rand.select_from_ranked(ranked)
        assert item in {0, 1}

    def test_select_from_ranked_empty_raises_error(self):
        """Test that empty ranked list raises error."""
        rand = Randomesque(k=3)

        with pytest.raises(ValueError, match="No items"):
            rand.select_from_ranked([])

    def test_reproducibility_with_seed(self):
        """Test reproducibility with same seed."""
        ranked = [(0, 10.0), (1, 8.0), (2, 6.0)]

        rand1 = Randomesque(k=3, seed=42)
        rand2 = Randomesque(k=3, seed=42)

        item1 = rand1.select_from_ranked(ranked)
        item2 = rand2.select_from_ranked(ranked)

        assert item1 == item2


class TestProgressiveRestricted:
    """Tests for Progressive-restricted exposure control."""

    def test_initialization(self):
        """Test initialization with parameters."""
        prog = ProgressiveRestricted(window_size=0.5, seed=42)
        assert prog.window_size == 0.5

    def test_filter_items_returns_informative(self, fitted_2pl_model):
        """Test that filtering returns informative items."""
        model = fitted_2pl_model.model
        prog = ProgressiveRestricted(window_size=0.5)
        available = set(range(model.n_items))

        filtered = prog.filter_items(available, model, theta=0.0)

        assert len(filtered) >= 1
        assert filtered.issubset(available)

    def test_filter_items_empty_returns_empty(self, fitted_2pl_model):
        """Test that empty available returns empty."""
        model = fitted_2pl_model.model
        prog = ProgressiveRestricted()
        filtered = prog.filter_items(set(), model, theta=0.0)
        assert filtered == set()

    def test_filter_items_tight_window(self, fitted_2pl_model):
        """Test filtering with tight window returns fewer items."""
        model = fitted_2pl_model.model
        prog_tight = ProgressiveRestricted(window_size=0.1)
        prog_wide = ProgressiveRestricted(window_size=10.0)

        available = set(range(model.n_items))

        filtered_tight = prog_tight.filter_items(available, model, theta=0.0)
        filtered_wide = prog_wide.filter_items(available, model, theta=0.0)

        assert len(filtered_tight) <= len(filtered_wide)

    def test_reset_clears_state(self):
        """Test that reset clears internal state."""
        prog = ProgressiveRestricted()
        prog._max_info_seen = {0: 1.0, 1: 2.0}

        prog.reset()

        assert prog._max_info_seen == {}


class TestCreateExposureControl:
    """Tests for create_exposure_control factory."""

    @pytest.mark.parametrize(
        "method,expected_class",
        [
            ("sympson-hetter", SympsonHetter),
            ("randomesque", Randomesque),
            ("progressive", ProgressiveRestricted),
            ("none", NoExposureControl),
            (None, NoExposureControl),
        ],
    )
    def test_create_valid_methods(self, method, expected_class):
        """Test creating valid exposure control methods."""
        control = create_exposure_control(method)
        assert isinstance(control, expected_class)

    def test_create_with_kwargs(self):
        """Test creating with kwargs."""
        control = create_exposure_control("randomesque", k=5, seed=42)
        assert isinstance(control, Randomesque)
        assert control.k == 5

        control = create_exposure_control("sympson-hetter", target_rate=0.3, seed=42)
        assert isinstance(control, SympsonHetter)
        assert control.target_rate == 0.3

    def test_create_invalid_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown exposure control"):
            create_exposure_control("invalid_method")

    def test_case_insensitivity(self):
        """Test case insensitivity."""
        control1 = create_exposure_control("Sympson-Hetter")
        control2 = create_exposure_control("SYMPSON-HETTER")

        assert type(control1) is type(control2)


class TestExposureControlInterface:
    """Tests for ExposureControl interface."""

    def test_all_controls_have_filter_items(self, fitted_2pl_model):
        """Test that all controls implement filter_items."""
        model = fitted_2pl_model.model
        controls = [
            NoExposureControl(),
            SympsonHetter(),
            Randomesque(),
            ProgressiveRestricted(),
        ]

        available = set(range(model.n_items))

        for control in controls:
            assert hasattr(control, "filter_items")
            filtered = control.filter_items(available, model, theta=0.0)
            assert isinstance(filtered, set)

    def test_all_controls_have_update(self):
        """Test that all controls implement update."""
        controls = [
            NoExposureControl(),
            SympsonHetter(),
            Randomesque(),
            ProgressiveRestricted(),
        ]

        for control in controls:
            assert hasattr(control, "update")
            control.update(0)

    def test_all_controls_have_reset(self):
        """Test that all controls implement reset."""
        controls = [
            NoExposureControl(),
            SympsonHetter(),
            Randomesque(),
            ProgressiveRestricted(),
        ]

        for control in controls:
            assert hasattr(control, "reset")
            control.reset()
