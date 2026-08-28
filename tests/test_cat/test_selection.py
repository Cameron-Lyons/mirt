"""Tests for item selection strategies."""

from unittest.mock import patch

import numpy as np
import pytest

from mirt.cat.selection import (
    AStratified,
    KullbackLeibler,
    MaxExpectedInformation,
    MaxFisherInformation,
    RandomSelection,
    UrryRule,
    create_selection_strategy,
)
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
