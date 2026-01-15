"""Tests for multidimensional computerized adaptive testing (MCAT)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat import (
    AOptimality,
    AvgSEStop,
    BayesianMCAT,
    CombinedMCATStop,
    COptimality,
    CovarianceDeterminantStop,
    CovarianceTraceStop,
    DOptimality,
    KullbackLeiblerMCAT,
    MaxItemsMCATStop,
    MaxSEStop,
    MCATEngine,
    MCATResult,
    MCATState,
    RandomMCATSelection,
    ThetaChangeMCATStop,
    create_mcat_selection_strategy,
    create_mcat_stopping_rule,
)
from mirt.models.multidimensional import MultidimensionalModel


@pytest.fixture
def fitted_mirt_model():
    """Create a fitted 2-factor MIRT model."""
    n_items = 20
    n_factors = 2
    model = MultidimensionalModel(n_items=n_items, n_factors=n_factors)

    rng = np.random.default_rng(42)
    slopes = rng.uniform(0.5, 2.0, size=(n_items, n_factors))
    slopes[:10, 1] = 0.0
    slopes[10:, 0] = 0.0

    intercepts = rng.uniform(-1.5, 1.5, size=n_items)

    model.set_parameters(slopes=slopes, intercepts=intercepts)
    model._is_fitted = True

    return model


@pytest.fixture
def fitted_3factor_model():
    """Create a fitted 3-factor MIRT model."""
    n_items = 30
    n_factors = 3
    model = MultidimensionalModel(n_items=n_items, n_factors=n_factors)

    rng = np.random.default_rng(123)
    slopes = rng.uniform(0.5, 2.0, size=(n_items, n_factors))

    intercepts = rng.uniform(-1.5, 1.5, size=n_items)

    model.set_parameters(slopes=slopes, intercepts=intercepts)
    model._is_fitted = True

    return model


class TestMCATState:
    """Tests for MCATState dataclass."""

    def test_create_state(self):
        theta = np.array([0.5, -0.3])
        cov = np.array([[0.1, 0.02], [0.02, 0.15]])
        se = np.sqrt(np.diag(cov))

        state = MCATState(
            theta=theta,
            covariance=cov,
            standard_error=se,
            items_administered=[0, 1, 2],
            responses=[1, 0, 1],
            n_items=3,
            is_complete=False,
            next_item=5,
        )

        assert state.n_factors == 2
        assert_allclose(state.theta, theta)
        assert_allclose(state.covariance, cov)
        assert state.n_items == 3
        assert not state.is_complete
        assert state.next_item == 5

    def test_trace_covariance(self):
        theta = np.array([0.0, 0.0])
        cov = np.array([[0.25, 0.0], [0.0, 0.36]])
        se = np.sqrt(np.diag(cov))

        state = MCATState(theta=theta, covariance=cov, standard_error=se)

        assert_allclose(state.trace_covariance, 0.61)

    def test_det_covariance(self):
        theta = np.array([0.0, 0.0])
        cov = np.array([[0.25, 0.0], [0.0, 0.36]])
        se = np.sqrt(np.diag(cov))

        state = MCATState(theta=theta, covariance=cov, standard_error=se)

        assert_allclose(state.det_covariance, 0.09)


class TestMCATResult:
    """Tests for MCATResult dataclass."""

    def test_create_result(self):
        theta = np.array([1.2, -0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.05]])
        se = np.sqrt(np.diag(cov))

        result = MCATResult(
            theta=theta,
            covariance=cov,
            standard_error=se,
            items_administered=[0, 3, 7, 12],
            responses=np.array([1, 1, 0, 1]),
            n_items_administered=4,
            stopping_reason="Covariance trace threshold reached",
        )

        assert result.n_factors == 2
        assert result.n_items_administered == 4
        assert "trace" in result.stopping_reason.lower()

    def test_summary(self):
        theta = np.array([1.0, -0.5])
        cov = np.array([[0.04, 0.0], [0.0, 0.05]])
        se = np.sqrt(np.diag(cov))

        result = MCATResult(
            theta=theta,
            covariance=cov,
            standard_error=se,
            items_administered=[0, 1, 2],
            responses=np.array([1, 1, 0]),
            n_items_administered=3,
            stopping_reason="Test stop",
        )

        summary = result.summary()
        assert "MCAT Result Summary" in summary
        assert "Dimensions" in summary
        assert "2" in summary


class TestMCATSelectionStrategies:
    """Tests for MCAT item selection strategies."""

    def test_d_optimality(self, fitted_mirt_model):
        strategy = DOptimality()
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = set(range(fitted_mirt_model.n_items))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available
        assert 0 <= item < fitted_mirt_model.n_items

    def test_a_optimality(self, fitted_mirt_model):
        strategy = AOptimality()
        theta = np.array([0.5, -0.5])
        cov = np.eye(2) * 0.5
        available = set(range(10, 20))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available

    def test_c_optimality(self, fitted_mirt_model):
        weights = np.array([1.0, 0.0])
        strategy = COptimality(weights=weights)
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = set(range(fitted_mirt_model.n_items))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available

    def test_kl_mcat(self, fitted_mirt_model):
        strategy = KullbackLeiblerMCAT()
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = set(range(fitted_mirt_model.n_items))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available

    def test_bayesian_mcat(self, fitted_mirt_model):
        strategy = BayesianMCAT()
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = set(range(fitted_mirt_model.n_items))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available

    def test_random_mcat_selection(self, fitted_mirt_model):
        strategy = RandomMCATSelection(seed=42)
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = set(range(5))

        item = strategy.select_item(fitted_mirt_model, theta, cov, available)

        assert item in available

    def test_create_mcat_selection_strategy(self):
        strategy = create_mcat_selection_strategy("D-optimality")
        assert isinstance(strategy, DOptimality)

        strategy = create_mcat_selection_strategy("A-optimality")
        assert isinstance(strategy, AOptimality)

        with pytest.raises(ValueError):
            create_mcat_selection_strategy("invalid")

    def test_get_item_criteria(self, fitted_mirt_model):
        strategy = DOptimality()
        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        available = {0, 1, 2, 3, 4}

        criteria = strategy.get_item_criteria(fitted_mirt_model, theta, cov, available)

        assert len(criteria) == 5
        assert all(isinstance(v, float) for v in criteria.values())


class TestMCATStoppingRules:
    """Tests for MCAT stopping rules."""

    def test_covariance_trace_stop(self):
        rule = CovarianceTraceStop(threshold=0.5)

        theta = np.array([0.0, 0.0])
        cov_high = np.eye(2) * 0.5
        se = np.sqrt(np.diag(cov_high))
        state_high = MCATState(
            theta=theta, covariance=cov_high, standard_error=se, n_items=5
        )

        assert not rule.should_stop(state_high)

        cov_low = np.eye(2) * 0.2
        se_low = np.sqrt(np.diag(cov_low))
        state_low = MCATState(
            theta=theta, covariance=cov_low, standard_error=se_low, n_items=10
        )

        assert rule.should_stop(state_low)
        assert "trace" in rule.get_reason().lower()

    def test_covariance_determinant_stop(self):
        rule = CovarianceDeterminantStop(threshold=0.01)

        theta = np.array([0.0, 0.0])
        cov_high = np.eye(2) * 0.5
        se = np.sqrt(np.diag(cov_high))
        state_high = MCATState(
            theta=theta, covariance=cov_high, standard_error=se, n_items=5
        )

        assert not rule.should_stop(state_high)

        cov_low = np.eye(2) * 0.05
        se_low = np.sqrt(np.diag(cov_low))
        state_low = MCATState(
            theta=theta, covariance=cov_low, standard_error=se_low, n_items=10
        )

        assert rule.should_stop(state_low)

    def test_max_se_stop(self):
        rule = MaxSEStop(threshold=0.3)

        theta = np.array([0.0, 0.0])
        se_high = np.array([0.5, 0.4])
        cov = np.diag(se_high**2)
        state_high = MCATState(
            theta=theta, covariance=cov, standard_error=se_high, n_items=5
        )

        assert not rule.should_stop(state_high)

        se_low = np.array([0.2, 0.25])
        cov_low = np.diag(se_low**2)
        state_low = MCATState(
            theta=theta, covariance=cov_low, standard_error=se_low, n_items=10
        )

        assert rule.should_stop(state_low)

    def test_avg_se_stop(self):
        rule = AvgSEStop(threshold=0.3)

        theta = np.array([0.0, 0.0])
        se_high = np.array([0.5, 0.4])
        cov = np.diag(se_high**2)
        state_high = MCATState(
            theta=theta, covariance=cov, standard_error=se_high, n_items=5
        )

        assert not rule.should_stop(state_high)

        se_low = np.array([0.2, 0.3])
        cov_low = np.diag(se_low**2)
        state_low = MCATState(
            theta=theta, covariance=cov_low, standard_error=se_low, n_items=10
        )

        assert rule.should_stop(state_low)

    def test_max_items_mcat_stop(self):
        rule = MaxItemsMCATStop(max_items=10)

        theta = np.array([0.0, 0.0])
        cov = np.eye(2)
        se = np.sqrt(np.diag(cov))

        state_below = MCATState(
            theta=theta, covariance=cov, standard_error=se, n_items=5
        )
        assert not rule.should_stop(state_below)

        state_at = MCATState(theta=theta, covariance=cov, standard_error=se, n_items=10)
        assert rule.should_stop(state_at)

    def test_theta_change_mcat_stop(self):
        rule = ThetaChangeMCATStop(threshold=0.01, n_stable=2)

        theta1 = np.array([0.0, 0.0])
        cov = np.eye(2)
        se = np.sqrt(np.diag(cov))
        state1 = MCATState(theta=theta1, covariance=cov, standard_error=se, n_items=1)
        assert not rule.should_stop(state1)

        theta2 = np.array([0.5, 0.3])
        state2 = MCATState(theta=theta2, covariance=cov, standard_error=se, n_items=2)
        assert not rule.should_stop(state2)

        theta3 = np.array([0.505, 0.305])
        state3 = MCATState(theta=theta3, covariance=cov, standard_error=se, n_items=3)
        assert not rule.should_stop(state3)

        theta4 = np.array([0.508, 0.308])
        state4 = MCATState(theta=theta4, covariance=cov, standard_error=se, n_items=4)
        assert rule.should_stop(state4)

    def test_combined_mcat_stop(self):
        rule = CombinedMCATStop(
            rules=[
                CovarianceTraceStop(threshold=0.3),
                MaxItemsMCATStop(max_items=10),
            ],
            operator="or",
            min_items=3,
        )

        theta = np.array([0.0, 0.0])
        cov = np.eye(2) * 0.1
        se = np.sqrt(np.diag(cov))

        state_early = MCATState(
            theta=theta, covariance=cov, standard_error=se, n_items=2
        )
        assert not rule.should_stop(state_early)

        state_ok = MCATState(theta=theta, covariance=cov, standard_error=se, n_items=5)
        assert rule.should_stop(state_ok)

    def test_create_mcat_stopping_rule(self):
        rule = create_mcat_stopping_rule("trace", threshold=0.4)
        assert isinstance(rule, CovarianceTraceStop)

        rule = create_mcat_stopping_rule("max_items", max_items=15)
        assert isinstance(rule, MaxItemsMCATStop)

        with pytest.raises(ValueError):
            create_mcat_stopping_rule("invalid")


class TestMCATEngine:
    """Tests for MCATEngine."""

    def test_engine_init(self, fitted_mirt_model):
        engine = MCATEngine(
            model=fitted_mirt_model,
            item_selection="D-optimality",
            trace_threshold=0.5,
            max_items=15,
        )

        assert engine.n_factors == 2
        assert engine.model.n_items == 20
        assert not engine._is_complete

    def test_engine_init_unfitted_model(self):
        model = MultidimensionalModel(n_items=10, n_factors=2)

        with pytest.raises(ValueError, match="fitted"):
            MCATEngine(model)

    def test_engine_init_unidimensional_model(self):
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=10)
        model._is_fitted = True

        with pytest.raises(ValueError, match="n_factors >= 2"):
            MCATEngine(model)

    def test_get_current_state(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=10)
        state = engine.get_current_state()

        assert isinstance(state, MCATState)
        assert state.n_factors == 2
        assert state.n_items == 0
        assert not state.is_complete
        assert state.next_item is not None

    def test_select_next_item(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model)
        item = engine.select_next_item()

        assert 0 <= item < fitted_mirt_model.n_items

    def test_administer_item(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=10)

        state1 = engine.get_current_state()
        item1 = state1.next_item

        state2 = engine.administer_item(1)

        assert state2.n_items == 1
        assert item1 in state2.items_administered
        assert 1 in state2.responses

    def test_simulation(self, fitted_mirt_model):
        engine = MCATEngine(
            fitted_mirt_model,
            trace_threshold=0.3,
            max_items=15,
            seed=42,
        )

        true_theta = np.array([0.5, -0.3])
        result = engine.run_simulation(true_theta)

        assert isinstance(result, MCATResult)
        assert result.n_factors == 2
        assert result.n_items_administered > 0
        assert result.n_items_administered <= 15
        assert len(result.theta_history) == result.n_items_administered

    def test_batch_simulation(self, fitted_mirt_model):
        engine = MCATEngine(
            fitted_mirt_model,
            trace_threshold=0.5,
            max_items=10,
            seed=42,
        )

        true_thetas = np.array(
            [
                [0.0, 0.0],
                [1.0, -0.5],
                [-1.0, 0.5],
            ]
        )

        results = engine.run_batch_simulation(true_thetas, n_replications=2)

        assert len(results) == 6
        assert all(isinstance(r, MCATResult) for r in results)

    def test_reset(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=5)

        engine.run_simulation(np.array([0.0, 0.0]))

        assert engine._is_complete

        engine.reset()

        assert not engine._is_complete
        assert len(engine._items_administered) == 0
        assert len(engine._responses) == 0

    def test_custom_initial_theta(self, fitted_mirt_model):
        initial_theta = np.array([1.0, -0.5])
        engine = MCATEngine(
            fitted_mirt_model,
            initial_theta=initial_theta,
            max_items=5,
        )

        state = engine.get_current_state()
        assert_allclose(state.theta, initial_theta)

    def test_custom_initial_covariance(self, fitted_mirt_model):
        initial_cov = np.array([[0.5, 0.1], [0.1, 0.5]])
        engine = MCATEngine(
            fitted_mirt_model,
            initial_covariance=initial_cov,
            max_items=5,
        )

        state = engine.get_current_state()
        assert_allclose(state.covariance, initial_cov)

    def test_3factor_model(self, fitted_3factor_model):
        engine = MCATEngine(
            fitted_3factor_model,
            trace_threshold=0.6,
            max_items=20,
            seed=42,
        )

        true_theta = np.array([0.5, -0.3, 0.2])
        result = engine.run_simulation(true_theta)

        assert result.n_factors == 3
        assert result.n_items_administered > 0

    def test_different_selection_strategies(self, fitted_mirt_model):
        strategies = ["D-optimality", "A-optimality", "KL", "Bayesian", "random"]

        for strategy_name in strategies:
            engine = MCATEngine(
                fitted_mirt_model,
                item_selection=strategy_name,
                max_items=5,
                seed=42,
            )

            result = engine.run_simulation(np.array([0.0, 0.0]))
            assert result.n_items_administered > 0

    def test_stopping_when_pool_exhausted(self, fitted_mirt_model):
        engine = MCATEngine(
            fitted_mirt_model,
            trace_threshold=0.001,
            max_items=None,
            seed=42,
        )

        result = engine.run_simulation(np.array([0.0, 0.0]))

        assert result.n_items_administered == fitted_mirt_model.n_items
        assert "exhausted" in result.stopping_reason.lower()


class TestMCATEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_available_items(self, fitted_mirt_model):
        strategy = DOptimality()

        with pytest.raises(ValueError, match="No available items"):
            strategy.select_item(
                fitted_mirt_model,
                np.array([0.0, 0.0]),
                np.eye(2),
                set(),
            )

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            CovarianceTraceStop(threshold=-0.5)

        with pytest.raises(ValueError):
            MaxItemsMCATStop(max_items=0)

    def test_invalid_initial_theta_shape(self, fitted_mirt_model):
        with pytest.raises(ValueError, match="shape"):
            MCATEngine(
                fitted_mirt_model,
                initial_theta=np.array([0.0, 0.0, 0.0]),
            )

    def test_invalid_simulation_theta_shape(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=5)

        with pytest.raises(ValueError, match="shape"):
            engine.run_simulation(np.array([0.0, 0.0, 0.0]))

    def test_session_already_complete(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=3)
        engine.run_simulation(np.array([0.0, 0.0]))

        with pytest.raises(RuntimeError, match="complete"):
            engine.select_next_item()

        with pytest.raises(RuntimeError, match="complete"):
            engine.administer_item(1)

    def test_result_before_complete(self, fitted_mirt_model):
        engine = MCATEngine(fitted_mirt_model, max_items=10)

        with pytest.raises(RuntimeError, match="not complete"):
            engine.get_result()
