"""Tests for item selection strategies."""

from unittest.mock import patch

import numpy as np
import pytest

from mirt.cat.engine import CATEngine
from mirt.cat.exposure import Randomesque
from mirt.cat.selection import (
    AStratified,
    KullbackLeibler,
    MaxExpectedInformation,
    MaxFisherInformation,
    RandomSelection,
    UrryRule,
    create_selection_strategy,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


class TestMaxFisherInformation:
    """Tests for MFI item selection."""

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that MFI returns a valid item index."""
        model = fitted_2pl_model.model
        mfi = MaxFisherInformation()
        available = set(range(model.n_items))

        item = mfi.select_item(model, theta=0.0, available_items=available)

        assert isinstance(item, int)
        assert item in available

    def test_select_item_maximizes_information(self, fitted_2pl_model):
        """Test that MFI selects maximum information item."""
        model = fitted_2pl_model.model
        mfi = MaxFisherInformation()
        available = set(range(model.n_items))
        theta = 0.0

        selected = mfi.select_item(model, theta=theta, available_items=available)
        criteria = mfi.get_item_criteria(model, theta, available)

        max_info_item = max(criteria, key=criteria.get)
        assert selected == max_info_item

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        mfi = MaxFisherInformation()

        with pytest.raises(ValueError, match="No available items"):
            mfi.select_item(model, theta=0.0, available_items=set())

    def test_get_item_criteria(self, fitted_2pl_model):
        """Test get_item_criteria returns dict for all items."""
        model = fitted_2pl_model.model
        mfi = MaxFisherInformation()
        available = {0, 1, 2}

        criteria = mfi.get_item_criteria(model, theta=0.0, available_items=available)

        assert isinstance(criteria, dict)
        assert set(criteria.keys()) == available
        assert all(isinstance(v, float) for v in criteria.values())

    def test_vectorizes_information_across_item_bank(self, fitted_2pl_model):
        """MFI should evaluate a dichotomous bank in one model call."""
        model = fitted_2pl_model.model
        mfi = MaxFisherInformation()
        available = {0, 2, 4}
        expected = model.information(np.array([[0.25]])).ravel()

        with patch.object(model, "information", wraps=model.information) as spy:
            criteria = mfi.get_item_criteria(model, 0.25, available)

        assert spy.call_count == 1
        assert spy.call_args.kwargs == {}
        assert criteria == {item: float(expected[item]) for item in available}

    def test_preserves_itemwise_fallback_for_polytomous_models(self):
        """Polytomous total information must not be treated as itemwise."""
        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        mfi = MaxFisherInformation()
        available = {0, 2}

        expected = {
            item: float(model.information(np.array([[0.25]]), item_idx=item).sum())
            for item in available
        }
        with patch.object(model, "information", wraps=model.information) as spy:
            criteria = mfi.get_item_criteria(model, 0.25, available)

        assert spy.call_count == len(available)
        assert criteria == expected


class TestMaxExpectedInformation:
    """Tests for MEI item selection."""

    def test_initialization(self):
        """Test MEI initialization."""
        mei = MaxExpectedInformation(n_quadpts=31)
        assert mei.n_quadpts == 31

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that MEI returns a valid item index."""
        model = fitted_2pl_model.model
        mei = MaxExpectedInformation()
        available = set(range(model.n_items))

        item = mei.select_item(model, theta=0.0, available_items=available)

        assert isinstance(item, int)
        assert item in available

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        mei = MaxExpectedInformation()

        with pytest.raises(ValueError, match="No available items"):
            mei.select_item(model, theta=0.0, available_items=set())


class TestKullbackLeibler:
    """Tests for KL divergence item selection."""

    def test_initialization(self):
        """Test KL initialization with parameters."""
        kl = KullbackLeibler(delta=0.2, n_points=10)
        assert kl.delta == 0.2
        assert kl.n_points == 10

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that KL returns a valid item index."""
        model = fitted_2pl_model.model
        kl = KullbackLeibler()
        available = set(range(model.n_items))

        item = kl.select_item(model, theta=0.0, available_items=available)

        assert isinstance(item, int)
        assert item in available

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        kl = KullbackLeibler()

        with pytest.raises(ValueError, match="No available items"):
            kl.select_item(model, theta=0.0, available_items=set())

    def test_kl_divergence_computation(self, fitted_2pl_model):
        """Test that KL divergence is computed correctly."""
        model = fitted_2pl_model.model
        kl = KullbackLeibler(delta=0.1, n_points=5)
        available = set(range(model.n_items))

        criteria = kl.get_item_criteria(model, theta=0.0, available_items=available)

        assert all(v >= 0 for v in criteria.values())

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"delta": 0.0}, "finite positive"),
            ({"delta": -0.1}, "finite positive"),
            ({"delta": np.nan}, "finite positive"),
            ({"delta": np.inf}, "finite positive"),
            ({"delta": True}, "finite positive"),
            ({"n_points": 1}, "at least 2"),
            ({"n_points": 2.5}, "integer"),
            ({"n_points": True}, "integer"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            KullbackLeibler(**kwargs)

    @pytest.mark.parametrize("theta", [np.nan, np.inf, -np.inf, True])
    def test_rejects_invalid_theta(self, fitted_2pl_model, theta):
        model = fitted_2pl_model.model

        with pytest.raises(ValueError, match="theta must be finite"):
            KullbackLeibler().get_item_criteria(model, theta, {0})

    def test_even_grid_averages_every_neighbor(self, fitted_2pl_model):
        model = fitted_2pl_model.model
        rule = KullbackLeibler(delta=0.5, n_points=4)
        theta = 0.2
        current = model.probability(np.array([[theta]]), item_idx=0)
        neighbors = np.linspace(theta - rule.delta, theta + rule.delta, 4)
        expected = np.mean(
            [
                rule._kl_divergence(
                    current,
                    model.probability(np.array([[neighbor]]), item_idx=0),
                )
                for neighbor in neighbors
            ]
        )

        assert rule._compute_kl_info(model, theta, 0) == pytest.approx(expected)

    def test_batches_dichotomous_bank_in_one_probability_call(
        self,
        fitted_2pl_model,
    ):
        model = fitted_2pl_model.model
        rule = KullbackLeibler(delta=0.2, n_points=9)
        available = {0, 2, 4}

        with patch.object(model, "probability", wraps=model.probability) as spy:
            criteria = rule.get_item_criteria(model, 0.25, available)

        assert spy.call_count == 1
        assert set(criteria) == available

    def test_batches_each_polytomous_item_across_grid(self):
        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        rule = KullbackLeibler(delta=0.2, n_points=4)
        available = {0, 2}

        with patch.object(model, "probability", wraps=model.probability) as spy:
            criteria = rule.get_item_criteria(model, 0.25, available)

        neighbors = np.linspace(0.25 - rule.delta, 0.25 + rule.delta, 4)
        expected = {}
        for item_idx in available:
            current = model.probability(np.array([[0.25]]), item_idx=item_idx)
            expected[item_idx] = np.mean(
                [
                    rule._kl_divergence(
                        current,
                        model.probability(
                            np.array([[neighbor]]),
                            item_idx=item_idx,
                        ),
                    )
                    for neighbor in neighbors
                ]
            )

        assert spy.call_count == len(available)
        assert set(criteria) == available
        assert criteria == pytest.approx(expected)

    def test_ties_select_lowest_item_index(self):
        model = GradedResponseModel(n_items=4, n_categories=3)
        rule = KullbackLeibler()

        assert rule.select_item(model, 0.0, {3, 1, 2}) == 1


class TestUrryRule:
    """Tests for Urry's rule item selection."""

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that Urry returns a valid item index."""
        model = fitted_2pl_model.model
        urry = UrryRule()
        available = set(range(model.n_items))

        item = urry.select_item(model, theta=0.0, available_items=available)

        assert isinstance(item, int)
        assert item in available

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        urry = UrryRule()

        with pytest.raises(ValueError, match="No available items"):
            urry.select_item(model, theta=0.0, available_items=set())

    def test_selects_closest_difficulty(self, fitted_2pl_model):
        """Test that Urry selects item with closest difficulty to theta."""
        model = fitted_2pl_model.model
        urry = UrryRule()
        available = set(range(model.n_items))
        theta = 0.0

        selected = urry.select_item(model, theta=theta, available_items=available)

        params = model.get_item_parameters(selected)
        selected_diff = params.get("difficulty", 0.0)

        for item in available:
            if item == selected:
                continue
            other_params = model.get_item_parameters(item)
            other_diff = other_params.get("difficulty", 0.0)
            assert abs(theta - selected_diff) <= abs(theta - other_diff)


class TestRandomSelection:
    """Tests for random item selection."""

    def test_initialization_with_seed(self):
        """Test random selection initialization with seed."""
        rand = RandomSelection(seed=42)
        assert rand.rng is not None

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that random selection returns a valid item index."""
        model = fitted_2pl_model.model
        rand = RandomSelection(seed=42)
        available = set(range(model.n_items))

        item = rand.select_item(model, theta=0.0, available_items=available)

        assert isinstance(item, int)
        assert item in available

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        rand = RandomSelection()

        with pytest.raises(ValueError, match="No available items"):
            rand.select_item(model, theta=0.0, available_items=set())

    def test_reproducibility_with_seed(self, fitted_2pl_model):
        """Test that same seed produces same selection."""
        model = fitted_2pl_model.model
        available = set(range(model.n_items))

        rand1 = RandomSelection(seed=42)
        rand2 = RandomSelection(seed=42)

        item1 = rand1.select_item(model, theta=0.0, available_items=available)
        item2 = rand2.select_item(model, theta=0.0, available_items=available)

        assert item1 == item2

    def test_variability_without_seed(self, fitted_2pl_model):
        """Test that different seeds produce different selections (most of the time)."""
        model = fitted_2pl_model.model
        available = set(range(model.n_items))
        items = []

        for seed in range(100):
            rand = RandomSelection(seed=seed)
            item = rand.select_item(model, theta=0.0, available_items=available)
            items.append(item)

        unique_items = set(items)
        assert len(unique_items) > 1


class TestAStratified:
    """Tests for a-stratified item selection."""

    def test_initialization(self):
        """Test a-stratified initialization."""
        astrat = AStratified(n_strata=5)
        assert astrat.n_strata == 5
        assert astrat._strata is None

    def test_select_item_returns_valid_index(self, fitted_2pl_model):
        """Test that a-stratified returns a valid item index."""
        model = fitted_2pl_model.model
        astrat = AStratified(n_strata=3)
        available = set(range(model.n_items))

        item = astrat.select_item(
            model,
            theta=0.0,
            available_items=available,
            administered_items=[],
        )

        assert isinstance(item, int)
        assert item in available

    def test_select_item_empty_raises_error(self, fitted_2pl_model):
        """Test that empty available items raises error."""
        model = fitted_2pl_model.model
        astrat = AStratified()

        with pytest.raises(ValueError, match="No available items"):
            astrat.select_item(model, theta=0.0, available_items=set())

    def test_strata_initialization(self, fitted_2pl_model):
        """Test that strata are initialized on first call."""
        model = fitted_2pl_model.model
        astrat = AStratified(n_strata=2)
        available = set(range(model.n_items))

        assert astrat._strata is None

        astrat.select_item(
            model,
            theta=0.0,
            available_items=available,
            administered_items=[],
        )

        assert astrat._strata is not None
        assert len(astrat._strata) == 2


class TestCreateSelectionStrategy:
    """Tests for create_selection_strategy factory."""

    @pytest.mark.parametrize(
        "method,expected_class",
        [
            ("MFI", MaxFisherInformation),
            ("MEI", MaxExpectedInformation),
            ("KL", KullbackLeibler),
            ("Urry", UrryRule),
            ("random", RandomSelection),
            ("a-stratified", AStratified),
        ],
    )
    def test_create_valid_strategies(self, method, expected_class):
        """Test creating valid strategies."""
        strategy = create_selection_strategy(method)
        assert isinstance(strategy, expected_class)

    def test_create_with_kwargs(self):
        """Test creating strategy with kwargs."""
        strategy = create_selection_strategy("random", seed=42)
        assert isinstance(strategy, RandomSelection)

        strategy = create_selection_strategy("a-stratified", n_strata=5)
        assert isinstance(strategy, AStratified)
        assert strategy.n_strata == 5

    def test_create_invalid_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown selection method"):
            create_selection_strategy("invalid_method")

    def test_case_insensitivity(self):
        """Test case insensitivity for standard methods."""
        mfi1 = create_selection_strategy("MFI")
        mfi2 = create_selection_strategy("mfi")

        assert type(mfi1) is type(mfi2)

    @pytest.mark.parametrize(
        "method,expected_class",
        [
            ("a_stratified", AStratified),
            ("A-Stratified", AStratified),
            ("urry", UrryRule),
            ("URRY", UrryRule),
            ("Random", RandomSelection),
            ("RANDOM", RandomSelection),
            (" mei ", MaxExpectedInformation),
            ("kl", KullbackLeibler),
        ],
    )
    def test_documented_aliases_resolve(self, method, expected_class):
        assert type(create_selection_strategy(method)) is expected_class

    def test_error_lists_canonical_names(self):
        with pytest.raises(
            ValueError,
            match="MFI, MEI, MPWI, MLWI, KL, KLP, MEPV, Urry, random, a-stratified",
        ):
            create_selection_strategy("b-stratified")

    def test_engine_configures_mei_from_any_spelling(self, fitted_2pl_model):
        engine = CATEngine(
            fitted_2pl_model.model,
            item_selection="mei",
            n_quadpts=9,
            theta_bounds=(-3.0, 3.0),
        )
        assert isinstance(engine._selection, MaxExpectedInformation)
        assert engine._selection.n_quadpts == 9
        assert engine._selection.theta_bounds == (-3.0, 3.0)


def _stratified_pool(n_items: int = 300) -> TwoParameterLogistic:
    """2PL pool with distinct discriminations in shuffled item order."""
    rng = np.random.default_rng(13)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.permutation(np.linspace(0.4, 2.4, n_items)),
        difficulty=rng.normal(size=n_items),
    )
    model._is_fitted = True
    return model


def _stratum_of(strategy: AStratified, item: int) -> int:
    assert strategy._strata is not None
    return next(
        index for index, stratum in enumerate(strategy._strata) if item in stratum
    )


class TestAStratifiedSchedule:
    """a-stratified selection spreads the test length across strata."""

    @pytest.mark.parametrize("from_string", [True, False])
    def test_engine_spends_equal_test_shares_in_increasing_strata(self, from_string):
        model = _stratified_pool()
        strategy = AStratified(3)
        engine = CATEngine(
            model,
            item_selection="a-stratified" if from_string else strategy,
            max_items=30,
            se_threshold=1e-6,
        )

        result = engine.run_simulation(0.4)

        stages = [
            _stratum_of(engine._selection, item) for item in result.items_administered
        ]
        assert stages == [0] * 10 + [1] * 10 + [2] * 10
        assert strategy.test_length is None
        discrimination = model.discrimination[result.items_administered]
        assert discrimination[:10].max() < discrimination[10:20].min()
        assert discrimination[10:20].max() < discrimination[20:].min()

    def test_configured_test_length_overrides_engine_horizon(self):
        model = _stratified_pool()
        strategy = AStratified(3, test_length=9)
        engine = CATEngine(
            model, item_selection=strategy, max_items=12, se_threshold=1e-6
        )

        result = engine.run_simulation(-0.2)

        stages = [_stratum_of(strategy, item) for item in result.items_administered]
        assert stages == [0] * 3 + [1] * 3 + [2] * 6

    def test_standalone_schedule_uses_pool_size(self):
        model = _stratified_pool(30)
        strategy = AStratified(3)

        assert strategy.current_stratum(model, 9) == 0
        assert strategy.current_stratum(model, 10) == 1
        assert strategy.current_stratum(model, 25) == 2
        assert strategy.current_stratum(model, 3, test_length=6) == 1
        assert AStratified(3, test_length=12).current_stratum(model, 4) == 1

    def test_exhausted_stratum_moves_to_next_available_stratum(self):
        model = _stratified_pool(30)
        strategy = AStratified(3)
        strategy._initialize_strata(model)
        middle_and_top = strategy._strata[1] | strategy._strata[2]

        item = strategy.select_item(model, 0.0, middle_and_top, administered_items=[])
        assert item in strategy._strata[1]
        item = strategy.select_item(
            model, 0.0, strategy._strata[0], administered_items=list(range(25))
        )
        assert item in strategy._strata[0]

    def test_criteria_cover_only_the_scheduled_stratum(self):
        model = _stratified_pool(60)
        strategy = AStratified(3, test_length=30)
        available = set(range(model.n_items))

        low = strategy.get_item_criteria(model, -2.0, available, [])
        high = strategy.get_item_criteria(model, 2.0, available, [])
        later = strategy.get_item_criteria(model, 0.0, available, list(range(12)))

        assert set(low) == set(high) == strategy._strata[0]
        assert set(later) == strategy._strata[1]
        assert max(low, key=low.__getitem__) != max(high, key=high.__getitem__)

    def test_randomesque_draws_within_the_current_stratum(self):
        model = _stratified_pool(60)
        engine = CATEngine(
            model,
            item_selection="a-stratified",
            exposure_control=Randomesque(k=4, seed=1),
            max_items=6,
            se_threshold=1e-6,
            seed=2,
        )

        results = [engine.run_simulation(theta) for theta in (-1.5, 0.0, 1.5)]

        strategy = engine._selection
        for result in results:
            stages = [_stratum_of(strategy, item) for item in result.items_administered]
            assert stages == [0, 0, 1, 1, 2, 2]
        assert max(max(result.items_administered) for result in results) > 13

    def test_b_matching_ranks_by_difficulty_distance(self):
        model = _stratified_pool(30)
        strategy = AStratified(3, within="b_matching")
        available = set(range(model.n_items))

        criteria = strategy.get_item_criteria(model, 0.7, available, [])

        for item, value in criteria.items():
            assert value == pytest.approx(-abs(0.7 - model.difficulty[item]))
        assert strategy.within == "b-matching"

    def test_strata_are_rebuilt_for_a_different_pool(self):
        strategy = AStratified(2)
        small, large = _stratified_pool(10), _stratified_pool(20)

        strategy.select_item(small, 0.0, set(range(10)), [])
        strategy.select_item(large, 0.0, set(range(20)), [])

        assert sum(len(stratum) for stratum in strategy._strata) == 20

    def test_strata_follow_model_identity_not_object_ids(self, monkeypatch):
        # A collected model's id() can be reused by a new pool of equal size.
        import mirt.cat.selection as selection

        monkeypatch.setattr(selection, "id", lambda value: 0, raising=False)
        strategy = AStratified(2)
        first = _stratified_pool(20)
        second = _stratified_pool(20)
        second.set_parameters(discrimination=first.discrimination[::-1].copy())

        strategy.select_item(first, 0.0, set(range(20)), [])
        strategy.select_item(second, 0.0, set(range(20)), [])

        lowest = set(np.argsort(second.discrimination)[:10].tolist())
        assert strategy._strata is not None
        assert strategy._strata[0] == lowest

    def test_planned_length_is_capped_at_the_pool_size(self):
        model = _stratified_pool(30)

        assert AStratified(3, test_length=90).current_stratum(model, 20) == 2
        assert AStratified(3).current_stratum(model, 10, test_length=60) == 1

    def test_instance_hooks_with_original_signatures_still_run(self):
        strategy = AStratified(3)
        original = strategy.select_item

        def select_item(model, theta, available_items, administered=None, resp=None):
            return original(model, theta, available_items, administered, resp)

        strategy.select_item = select_item
        engine = CATEngine(
            _stratified_pool(30),
            item_selection=strategy,
            max_items=4,
            se_threshold=1e-6,
        )

        assert engine.run_simulation(0.0).n_items_administered == 4

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"n_strata": 0}, "n_strata"),
            ({"n_strata": 2.0}, "n_strata"),
            ({"test_length": 0}, "test_length"),
            ({"test_length": True}, "test_length"),
            ({"within": "KL"}, "within"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            AStratified(**kwargs)

    @pytest.mark.parametrize("exposure", [None, "randomesque"])
    @pytest.mark.parametrize("hook", ["select_item", "get_item_criteria"])
    def test_subclasses_with_original_signatures_still_run(self, hook, exposure):
        class LegacySelect(AStratified):
            def select_item(
                self,
                model,
                theta,
                available_items,
                administered_items=None,
                responses=None,
            ):
                return super().select_item(
                    model, theta, available_items, administered_items, responses
                )

        class LegacyCriteria(AStratified):
            def get_item_criteria(
                self,
                model,
                theta,
                available_items,
                administered_items=None,
                responses=None,
            ):
                return super().get_item_criteria(
                    model, theta, available_items, administered_items, responses
                )

        strategy_class = LegacySelect if hook == "select_item" else LegacyCriteria
        engine = CATEngine(
            _stratified_pool(30),
            item_selection=strategy_class(3),
            exposure_control=exposure,
            max_items=6,
            se_threshold=1e-6,
            seed=1,
        )
        assert engine.run_simulation(0.0).n_items_administered == 6


class TestItemSelectionStrategyInterface:
    """Tests for ItemSelectionStrategy interface."""

    def test_all_strategies_have_select_item(self, fitted_2pl_model):
        """Test that all strategies implement select_item."""
        model = fitted_2pl_model.model
        strategies = [
            MaxFisherInformation(),
            MaxExpectedInformation(),
            KullbackLeibler(),
            UrryRule(),
            RandomSelection(),
            AStratified(),
        ]

        available = set(range(model.n_items))

        for strategy in strategies:
            assert hasattr(strategy, "select_item")
            item = strategy.select_item(
                model,
                theta=0.0,
                available_items=available,
                administered_items=[],
            )
            assert isinstance(item, int)

    def test_all_strategies_have_get_item_criteria(self, fitted_2pl_model):
        """Test that all strategies implement get_item_criteria."""
        model = fitted_2pl_model.model
        strategies = [
            MaxFisherInformation(),
            MaxExpectedInformation(),
            KullbackLeibler(),
            UrryRule(),
            RandomSelection(),
        ]

        available = set(range(model.n_items))

        for strategy in strategies:
            assert hasattr(strategy, "get_item_criteria")
            criteria = strategy.get_item_criteria(
                model, theta=0.0, available_items=available
            )
            assert isinstance(criteria, dict)


def _located_pool(model_class, locations):
    """Polytomous or unfolding pool whose item locations are ``locations``."""
    from mirt.models import (
        GeneralizedGradedUnfolding,
        GeneralizedPartialCredit,
        IdealPointModel,
        PartialCreditModel,
    )

    n_items = len(locations)
    offsets = locations[:, None] + np.array([-0.8, 0.0, 0.8])
    if model_class in (GeneralizedPartialCredit, PartialCreditModel):
        model = model_class(n_items=n_items, n_categories=4)
        model.set_parameters(steps=offsets)
    elif model_class is GradedResponseModel:
        model = model_class(n_items=n_items, n_categories=4)
        model.set_parameters(thresholds=offsets)
    elif model_class is GeneralizedGradedUnfolding:
        model = model_class(n_items=n_items, n_categories=4)
        model.set_parameters(location=locations)
    else:
        assert model_class is IdealPointModel
        model = model_class(n_items=n_items)
        model.set_parameters(location=locations)
    model._is_fitted = True
    return model


class TestItemLocations:
    """Urry and b-matching rank items by each family's location parameters."""

    @pytest.fixture(
        params=[
            "GeneralizedPartialCredit",
            "PartialCreditModel",
            "GradedResponseModel",
            "GeneralizedGradedUnfolding",
            "IdealPointModel",
        ]
    )
    def model_class(self, request):
        import mirt.models

        return getattr(mirt.models, request.param)

    def test_urry_selects_the_nearest_location(self, model_class):
        locations = np.linspace(-2.5, 2.5, 30)
        model = _located_pool(model_class, locations)

        # GPCM, PCM, GGUM and ideal-point items used to sit at 0, giving item 0.
        for theta in (-2.0, 2.0):
            expected = int(np.argmin(np.abs(locations - theta)))
            assert UrryRule().select_item(model, theta, set(range(30))) == expected

    def test_b_matching_uses_item_locations(self, model_class):
        locations = np.linspace(-2.5, 2.5, 30)
        model = _located_pool(model_class, locations)
        strategy = AStratified(n_strata=1, within="b-matching")

        criteria = strategy.get_item_criteria(model, 2.0, set(range(30)), [])

        assert criteria == pytest.approx(
            {item: -abs(2.0 - location) for item, location in enumerate(locations)}
        )
        engine = CATEngine(model, item_selection=strategy, max_items=1, initial_theta=2)
        assert engine.select_next_item() == 26

    def test_padded_steps_are_ignored(self):
        from mirt.cat.selection import _item_locations
        from mirt.models import GeneralizedPartialCredit

        model = GeneralizedPartialCredit(n_items=2, n_categories=[3, 4])
        model.set_parameters(steps=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]))

        assert_locations = np.testing.assert_allclose
        assert_locations(_item_locations(model), [-0.5, 1.0])

    def test_models_without_locations_are_rejected(self):
        from mirt.models import NominalResponseModel

        model = NominalResponseModel(n_items=4, n_categories=3)
        model._is_fitted = True
        with pytest.raises(ValueError, match="no item locations"):
            UrryRule().select_item(model, 0.0, {0, 1, 2, 3})

    def test_urry_criteria_rank_like_selection(self):
        model = _stratified_pool(50)
        available = set(range(0, 50, 3))
        criteria = UrryRule().get_item_criteria(model, 0.4, available)

        assert list(criteria) == sorted(available)
        assert max(criteria, key=criteria.__getitem__) == UrryRule().select_item(
            model, 0.4, available
        )
        assert criteria == pytest.approx(
            {item: -abs(0.4 - model.difficulty[item]) for item in available}
        )


class TestPolytomousInformationKernel:
    """Polytomous item information comes from the all-item kernels."""

    @staticmethod
    def _pool(model_class, n_items=30):
        rng = np.random.default_rng(3)
        model = model_class(
            n_items=n_items, n_categories=[3 + i % 2 for i in range(n_items)]
        )
        key = "thresholds" if model_class is GradedResponseModel else "steps"
        model.set_parameters(
            discrimination=rng.lognormal(0.0, 0.3, n_items),
            **{key: np.sort(rng.normal(0.0, 1.0, (n_items, 3)), axis=1)},
        )
        model._is_fitted = True
        return model

    @staticmethod
    def _itemwise(model, theta, items):
        return np.array(
            [
                [
                    float(model.information(row[None, :], item_idx=item).sum())
                    for item in items
                ]
                for row in theta
            ]
        )

    @pytest.fixture(params=["GradedResponseModel", "GeneralizedPartialCredit"])
    def model(self, request):
        import mirt.models

        return self._pool(getattr(mirt.models, request.param))

    def test_mfi_criteria_use_the_kernel(self, model, monkeypatch):
        import mirt.cat.selection as selection

        calls = []
        original = selection.polytomous_item_information

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(selection, "polytomous_item_information", spy)
        available = set(range(0, 30, 2))
        criteria = MaxFisherInformation().get_item_criteria(model, 0.3, available)

        assert calls == [1]
        expected = self._itemwise(model, np.array([[0.3]]), sorted(available))[0]
        np.testing.assert_allclose(list(criteria.values()), expected, atol=1e-12)
        assert list(criteria) == sorted(available)

    def test_exposure_stopping_and_assembly_match_itemwise_information(self, model):
        from mirt.cat.assembly import _information_matrix
        from mirt.cat.exposure import _candidate_information
        from mirt.cat.results import CATState
        from mirt.cat.stopping import MinInformationStop

        items = np.arange(5, 25)
        theta = np.array([[-0.4]])
        np.testing.assert_allclose(
            _candidate_information(model, theta, items),
            self._itemwise(model, theta, items)[0],
            atol=1e-12,
        )
        grid = np.linspace(-2.0, 2.0, 7)
        np.testing.assert_allclose(
            _information_matrix(model, grid, model.n_items, items),
            self._itemwise(model, grid[:, None], items),
            atol=1e-12,
        )
        state = CATState(theta=-0.4, standard_error=0.5, items_administered=[0, 1])
        best = self._itemwise(model, theta, range(2, 30)).max()
        assert MinInformationStop(model, threshold=best + 1e-9).should_stop(state)
        assert not MinInformationStop(model, threshold=best - 1e-9).should_stop(state)

    def test_randomesque_paths_match_itemwise_evaluation(self, model):
        def run(model):
            engine = CATEngine(
                model, exposure_control=Randomesque(4, seed=2), max_items=8, seed=5
            )
            return [
                r.items_administered
                for r in engine.run_batch_simulation([-1.0, 0.0, 1.0])
            ]

        kernel_paths = run(model)
        # An instance-level hook keeps the item-by-item reference evaluation.
        with patch.object(model, "information", wraps=model.information):
            itemwise_paths = run(model)
        assert kernel_paths == itemwise_paths


class TestPosteriorWeightedCriteria:
    """MPWI, MLWI, KLP and MEPV integrate over the history posterior."""

    HISTORY = ([3, 11, 20], [1, 0, 1])

    @staticmethod
    def _grid(flat=False, n=21, bounds=(-4.0, 4.0)):
        nodes, weights = np.polynomial.legendre.leggauss(n)
        half = (bounds[1] - bounds[0]) / 2
        nodes = (bounds[1] + bounds[0]) / 2 + half * nodes
        weights = half * weights
        if not flat:
            weights = weights * np.exp(-0.5 * nodes**2)
        return nodes, weights

    def _posterior(self, model, flat=False):
        nodes, weights = self._grid(flat)
        items, responses = self.HISTORY
        likelihood = np.ones_like(nodes)
        for item, response in zip(items, responses, strict=True):
            likelihood *= self._curves(model, nodes[:, None], item)[:, response]
        posterior = weights * likelihood
        return nodes, posterior / posterior.sum()

    @staticmethod
    def _curves(model, theta, item):
        p = np.asarray(model.probability(theta, item_idx=item)).reshape(len(theta), -1)
        return p if model.is_polytomous else np.column_stack((1 - p[:, 0], p[:, 0]))

    def _reference(self, name, model, theta, items):
        flat = name == "MLWI"
        nodes, posterior = self._posterior(model, flat)
        values = []
        for item in items:
            curves = self._curves(model, nodes[:, None], item)
            current = self._curves(model, np.array([[theta]]), item)[0]
            if name in ("MPWI", "MLWI"):
                info = model.information(nodes[:, None], item_idx=item)
                values.append(posterior @ np.asarray(info).reshape(len(nodes)))
            elif name == "KLP":
                kl = np.sum(current * np.log(current / curves), axis=1)
                values.append(posterior @ kl)
            else:
                expected = 0.0
                for k, probability in enumerate(current):
                    mass = posterior * curves[:, k]
                    mean = mass @ nodes / mass.sum()
                    expected += probability * (mass @ (nodes - mean) ** 2) / mass.sum()
                values.append(-expected)
        return np.array(values)

    @pytest.fixture(params=["2PL", "GRM"])
    def model(self, request):
        rng = np.random.default_rng(8)
        if request.param == "2PL":
            pool = TwoParameterLogistic(n_items=30)
            pool.set_parameters(
                discrimination=rng.lognormal(0.0, 0.3, 30),
                difficulty=rng.normal(0.0, 1.0, 30),
            )
        else:
            pool = GradedResponseModel(n_items=30, n_categories=4)
            pool.set_parameters(
                discrimination=rng.lognormal(0.0, 0.3, 30),
                thresholds=np.sort(rng.normal(0.0, 1.0, (30, 3)), axis=1),
            )
        pool._is_fitted = True
        return pool

    @pytest.mark.parametrize("name", ["MPWI", "MLWI", "KLP", "MEPV"])
    def test_criteria_match_quadrature_references(self, model, name):
        strategy = create_selection_strategy(name)
        items, responses = self.HISTORY
        available = set(range(30)) - set(items)

        criteria = strategy.get_item_criteria(model, 0.4, available, items, responses)

        assert list(criteria) == sorted(available)
        expected = self._reference(name, model, 0.4, sorted(available))
        np.testing.assert_allclose(list(criteria.values()), expected, rtol=1e-9)
        best = max(criteria, key=criteria.__getitem__)
        assert strategy.select_item(model, 0.4, available, items, responses) == best

    @pytest.mark.parametrize("name", ["mpwi", "MLWI", "klp", "MEPV"])
    def test_engine_runs_named_criteria_with_its_quadrature(self, model, name):
        engine = CATEngine(model, item_selection=name, n_quadpts=15, max_items=5)

        assert engine._selection.n_quadpts == 15
        result = engine.run_simulation(0.5, lambda item, theta: item % 2)
        assert result.n_items_administered == 5
        assert len(set(result.items_administered)) == 5

    def test_criteria_reject_multidimensional_models(self):
        from mirt.models import MultidimensionalModel

        model = MultidimensionalModel(n_items=4, n_factors=2)
        model._is_fitted = True
        with pytest.raises(ValueError, match="unidimensional"):
            create_selection_strategy("MPWI").get_item_criteria(model, 0.0, {0, 1})

    def test_shrinking_kl_interval(self):
        model = _stratified_pool(40)
        items = list(range(4))
        available = set(range(4, 40))
        shrinking = KullbackLeibler(delta=2.0, delta_scaling="sqrt_n")

        criteria = shrinking.get_item_criteria(model, 0.3, available, items, [1] * 4)

        # After four items, delta / sqrt(4) = 1.
        expected = KullbackLeibler(delta=1.0).get_item_criteria(model, 0.3, available)
        assert criteria == pytest.approx(expected, rel=1e-12)
        first = shrinking.get_item_criteria(model, 0.3, available)
        assert first == pytest.approx(
            KullbackLeibler(delta=2.0).get_item_criteria(model, 0.3, available)
        )

    def test_kl_rejects_unknown_scaling(self):
        with pytest.raises(ValueError, match="delta_scaling"):
            KullbackLeibler(delta_scaling="log_n")
