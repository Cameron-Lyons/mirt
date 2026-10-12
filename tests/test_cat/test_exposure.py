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
        """Test that reset clears the selection window and opens a session."""
        prog = ProgressiveRestricted()
        prog._latest_item_indices = np.array([0, 1])
        prog._latest_information = np.array([1.0, 2.0])

        prog.reset()

        assert prog._latest_item_indices.size == 0
        assert prog._latest_information.size == 0
        assert prog.n_examinees == 1


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


def _exposure_pool(n_items: int = 60, seed: int = 0):
    from mirt.models import TwoParameterLogistic

    rng = np.random.default_rng(seed)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.lognormal(0.2, 0.35, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    return model


class TestProgressiveRestrictedMethod:
    """Revuelta and Ponsoda's restricted maximum and progressive weight."""

    def test_defaults_restrict_exposure_without_a_window(self):
        control = ProgressiveRestricted()
        assert control.window_size is None
        assert control.max_exposure == 0.25
        assert control.acceleration == 1.0

    def test_maximum_exposure_rate_is_enforced(self):
        from mirt.cat import CATEngine

        model = _exposure_pool()
        control = ProgressiveRestricted(max_exposure=0.3, seed=1)
        engine = CATEngine(
            model, exposure_control=control, max_items=10, min_items=10, seed=2
        )
        thetas = np.random.default_rng(3).normal(size=150)
        engine.run_batch_simulation(thetas)

        rates = control.get_exposure_rates()
        assert control.n_examinees == 150
        # The old 0.5-information window let the best items reach 1.0.
        assert max(rates.values()) < 0.3 + 1.0 / 150 + 1e-12
        assert sum(rates.values()) == pytest.approx(10.0)

    def test_restriction_is_relaxed_when_every_item_is_exhausted(
        self, fitted_2pl_model
    ):
        model = fitted_2pl_model.model
        control = ProgressiveRestricted(max_exposure=0.25, seed=0)
        control.reset()
        for item in range(model.n_items):
            control.update(item)

        eligible = control.filter_items(set(range(model.n_items)), model, 0.0)
        assert eligible == set(range(model.n_items))

    def test_restriction_excludes_items_at_the_maximum(self, fitted_2pl_model):
        model = fitted_2pl_model.model
        control = ProgressiveRestricted(max_exposure=0.5, seed=0)
        control.reset()
        control.reset()
        control.update(0)
        control.update(1)
        control.update(1)

        eligible = control.filter_items(set(range(model.n_items)), model, 0.0)
        assert eligible == set(range(model.n_items)) - {0, 1}

    def test_small_acceleration_moves_to_information_after_one_item(self):
        from mirt.cat.exposure import _candidate_information

        model = _exposure_pool()
        information = _candidate_information(
            model, np.array([[0.0]]), np.arange(model.n_items)
        )
        for seed in range(5):
            control = ProgressiveRestricted(
                max_exposure=None, acceleration=1e-9, seed=seed
            )
            eligible = control.filter_items(set(range(model.n_items)), model, 0.0)
            chosen = control.select_from_eligible(
                eligible, n_administered=1, max_items=10
            )
            assert chosen == int(np.argmax(information))

    @pytest.mark.parametrize(
        ("options", "error"),
        [
            ({"max_exposure": 0.0}, ValueError),
            ({"max_exposure": 1.5}, ValueError),
            ({"acceleration": 0.0}, ValueError),
            ({"acceleration": np.inf}, ValueError),
            ({"acceleration": "fast"}, TypeError),
        ],
    )
    def test_invalid_settings_are_rejected(self, options, error):
        with pytest.raises(error):
            ProgressiveRestricted(**options)


class TestSympsonHetterCalibration:
    """Each calibration round measures only the current parameters."""

    def test_calibration_uses_sessions_since_the_previous_round(self):
        control = SympsonHetter(target_rate=0.25)
        control.record_sessions(np.array([[0, 1]] * 10))
        control.calibrate(4)
        assert control.exposure_parameters[0] == pytest.approx(0.25)
        assert control.n_examinees == 0

        # Two of ten sessions under the new parameters: below the target.
        control.record_sessions(np.array([[0, 1]] * 2 + [[2, 3]] * 8))
        control.calibrate(4)
        # Cumulative counts (12 of 20) used to shrink it to 0.104 instead.
        assert control.exposure_parameters[0] == pytest.approx(0.275)

    def test_cumulative_calibration_remains_available(self):
        control = SympsonHetter(target_rate=0.25)
        control.record_sessions(np.array([[0, 1]] * 10))
        control.calibrate(4, reset_counts=False)
        control.record_sessions(np.array([[0, 1]] * 2 + [[2, 3]] * 8))
        control.calibrate(4, reset_counts=False)
        assert control.exposure_parameters[0] == pytest.approx(0.25 * 0.25 / 0.6)

    def test_reset_counts_keeps_parameters(self):
        control = SympsonHetter({0: 0.4, 1: 0.9}, target_rate=0.3)
        control.record_sessions(np.array([[0, 1]]))
        control.reset_counts()
        assert control.exposure_parameters == {0: 0.4, 1: 0.9}
        assert control.n_examinees == 0
        assert control.get_exposure_rates() == {0: 0.0, 1: 0.0}

    @pytest.mark.parametrize("open_new_engine", [False, True])
    def test_engine_sessions_after_a_reset_are_counted(self, open_new_engine):
        from mirt.cat import CATEngine

        model = _exposure_pool(20)
        control = SympsonHetter(target_rate=0.5, seed=1)
        engine = CATEngine(model, exposure_control=control, max_items=4, seed=2)
        engine.run_batch_simulation([0.0, 0.5, 1.0])
        if open_new_engine:
            # The new engine's open session predates the reset.
            engine = CATEngine(model, exposure_control=control, max_items=4, seed=3)
        control.calibrate(model.n_items)
        engine.run_batch_simulation([-0.5, 0.5])

        report = control.exposure_report(n_items=model.n_items)
        assert report.n_examinees == 2
        assert report.selection_counts.sum() == 8

    def test_reset_counts_flag_must_be_boolean(self):
        with pytest.raises(TypeError, match="reset_counts"):
            SympsonHetter().calibrate(4, reset_counts=1)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("sympson_hetter", SympsonHetter),
        ("Sympson_Hetter", SympsonHetter),
        (" RANDOMESQUE ", Randomesque),
        ("None", NoExposureControl),
    ],
)
def test_exposure_factory_normalizes_names(name, expected_class):
    assert isinstance(create_exposure_control(name), expected_class)
