"""Tests for parameter constraints system."""

import numpy as np
import pytest

from mirt import ThreeParameterLogistic, TwoParameterLogistic
from mirt.estimation.constraints import (
    BoundConstraint,
    ConstraintSet,
    CustomConstraint,
    EqualityConstraint,
    FixedConstraint,
    LinearConstraint,
    ParameterConstraint,
    bound_parameter,
    create_1pl_constraints,
    create_rasch_constraints,
    equal_parameters,
    fix_parameter,
    mean_constraint,
)


@pytest.fixture
def fitted_model():
    """Create a fitted 2PL model for constraint testing."""
    model = TwoParameterLogistic(n_items=5)
    model._initialize_parameters()
    model._parameters["discrimination"] = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    model._parameters["difficulty"] = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    model._is_fitted = True
    return model


@pytest.fixture
def fitted_3pl_model():
    """Create a fitted 3PL model for constraint testing."""
    model = ThreeParameterLogistic(n_items=5)
    model._initialize_parameters()
    model._parameters["discrimination"] = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    model._parameters["difficulty"] = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    model._parameters["guessing"] = np.array([0.2, 0.15, 0.25, 0.1, 0.2])
    model._is_fitted = True
    return model


class TestParameterConstraintBase:
    """Tests for base ParameterConstraint class."""

    def test_base_class_not_implemented(self, fitted_model):
        """Test that base class methods raise NotImplementedError."""
        constraint = ParameterConstraint("difficulty")

        with pytest.raises(NotImplementedError):
            constraint.apply(fitted_model)

        with pytest.raises(NotImplementedError):
            constraint.is_satisfied(fitted_model)

        with pytest.raises(NotImplementedError):
            constraint.penalty(fitted_model)


class TestFixedConstraint:
    """Tests for FixedConstraint class."""

    def test_fix_single_item(self, fitted_model):
        """Test fixing a single item's parameter."""
        constraint = FixedConstraint("discrimination", [0], value=1.0)
        original = fitted_model.parameters["discrimination"].copy()

        constraint.apply(fitted_model)

        assert fitted_model.parameters["discrimination"][0] == 1.0
        assert np.allclose(fitted_model.parameters["discrimination"][1:], original[1:])

    def test_fix_multiple_items(self, fitted_model):
        """Test fixing multiple items' parameters."""
        constraint = FixedConstraint("discrimination", [0, 2, 4], value=1.5)

        constraint.apply(fitted_model)

        assert fitted_model.parameters["discrimination"][0] == 1.5
        assert fitted_model.parameters["discrimination"][2] == 1.5
        assert fitted_model.parameters["discrimination"][4] == 1.5

    def test_fix_all_items(self, fitted_model):
        """Test fixing all items' parameters."""
        constraint = FixedConstraint("discrimination", value=1.0)

        constraint.apply(fitted_model)

        assert np.allclose(fitted_model.parameters["discrimination"], 1.0)

    def test_is_satisfied_true(self, fitted_model):
        """Test is_satisfied returns True when constraint is met."""
        constraint = FixedConstraint("discrimination", [0], value=1.0)
        constraint.apply(fitted_model)

        assert constraint.is_satisfied(fitted_model)

    def test_is_satisfied_false(self, fitted_model):
        """Test is_satisfied returns False when constraint is not met."""
        constraint = FixedConstraint("discrimination", [0], value=2.0)

        assert not constraint.is_satisfied(fitted_model)

    def test_is_satisfied_nonexistent_param(self, fitted_model):
        """Test is_satisfied for nonexistent parameter."""
        constraint = FixedConstraint("nonexistent", [0], value=1.0)

        assert constraint.is_satisfied(fitted_model)

    def test_penalty_zero_when_satisfied(self, fitted_model):
        """Test penalty is zero when constraint is satisfied."""
        constraint = FixedConstraint("discrimination", [0], value=1.0)
        constraint.apply(fitted_model)

        assert constraint.penalty(fitted_model) == pytest.approx(0.0)

    def test_penalty_positive_when_violated(self, fitted_model):
        """Test penalty is positive when constraint is violated."""
        constraint = FixedConstraint("discrimination", [0], value=2.0)

        penalty = constraint.penalty(fitted_model)
        expected = (1.0 - 2.0) ** 2
        assert penalty == pytest.approx(expected)

    def test_penalty_nonexistent_param(self, fitted_model):
        """Test penalty for nonexistent parameter."""
        constraint = FixedConstraint("nonexistent", value=1.0)

        assert constraint.penalty(fitted_model) == 0.0

    def test_apply_nonexistent_param(self, fitted_model):
        """Test apply does nothing for nonexistent parameter."""
        original_params = {k: v.copy() for k, v in fitted_model.parameters.items()}
        constraint = FixedConstraint("nonexistent", value=1.0)

        constraint.apply(fitted_model)

        for name, values in fitted_model.parameters.items():
            assert np.allclose(values, original_params[name])


class TestEqualityConstraint:
    """Tests for EqualityConstraint class."""

    def test_equality_subset_items(self, fitted_model):
        """Test equality constraint on subset of items."""
        constraint = EqualityConstraint("discrimination", [0, 1, 2])

        constraint.apply(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert disc[0] == pytest.approx(disc[1])
        assert disc[1] == pytest.approx(disc[2])

    def test_equality_all_items(self, fitted_model):
        """Test equality constraint on all items."""
        constraint = EqualityConstraint("discrimination")

        constraint.apply(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert np.allclose(disc, disc[0])

    def test_equality_preserves_mean(self, fitted_model):
        """Test that equality constraint preserves mean."""
        original_mean = np.mean(fitted_model.parameters["discrimination"][:3])
        constraint = EqualityConstraint("discrimination", [0, 1, 2])

        constraint.apply(fitted_model)

        new_mean = np.mean(fitted_model.parameters["discrimination"][:3])
        assert new_mean == pytest.approx(original_mean)

    def test_is_satisfied_true(self, fitted_model):
        """Test is_satisfied when constraint is met."""
        constraint = EqualityConstraint("discrimination", [0, 1, 2])
        constraint.apply(fitted_model)

        assert constraint.is_satisfied(fitted_model)

    def test_is_satisfied_false(self, fitted_model):
        """Test is_satisfied when constraint is not met."""
        constraint = EqualityConstraint("discrimination", [0, 1, 2])

        assert not constraint.is_satisfied(fitted_model)

    def test_is_satisfied_single_item(self, fitted_model):
        """Test is_satisfied with single item (always true)."""
        constraint = EqualityConstraint("discrimination", [0])

        assert constraint.is_satisfied(fitted_model)

    def test_penalty_zero_when_satisfied(self, fitted_model):
        """Test penalty is zero when constraint is satisfied."""
        constraint = EqualityConstraint("discrimination", [0, 1, 2])
        constraint.apply(fitted_model)

        assert constraint.penalty(fitted_model) == pytest.approx(0.0, abs=1e-10)

    def test_penalty_positive_when_violated(self, fitted_model):
        """Test penalty is positive when constraint is violated."""
        constraint = EqualityConstraint("discrimination", [0, 1, 2])

        penalty = constraint.penalty(fitted_model)
        assert penalty > 0

    def test_penalty_single_item(self, fitted_model):
        """Test penalty with single item."""
        constraint = EqualityConstraint("discrimination", [0])

        assert constraint.penalty(fitted_model) == 0.0


class TestBoundConstraint:
    """Tests for BoundConstraint class."""

    def test_lower_bound(self, fitted_model):
        """Test lower bound constraint."""
        fitted_model.parameters["discrimination"][0] = 0.1
        constraint = BoundConstraint("discrimination", lower=0.5)

        constraint.apply(fitted_model)

        assert np.all(fitted_model.parameters["discrimination"] >= 0.5)

    def test_upper_bound(self, fitted_model):
        """Test upper bound constraint."""
        fitted_model.parameters["discrimination"][0] = 5.0
        constraint = BoundConstraint("discrimination", upper=3.0)

        constraint.apply(fitted_model)

        assert np.all(fitted_model.parameters["discrimination"] <= 3.0)

    def test_both_bounds(self, fitted_model):
        """Test both lower and upper bounds."""
        fitted_model._parameters["discrimination"] = np.array([0.1, 5.0, 1.0, 1.5, 1.1])
        constraint = BoundConstraint("discrimination", lower=0.5, upper=2.0)

        constraint.apply(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert np.all(disc >= 0.5)
        assert np.all(disc <= 2.0)

    def test_bounds_on_subset(self, fitted_model):
        """Test bounds on subset of items."""
        fitted_model._parameters["discrimination"] = np.array([0.1, 0.2, 1.0, 1.5, 1.1])
        constraint = BoundConstraint("discrimination", [0, 1], lower=0.5)

        constraint.apply(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert disc[0] == 0.5
        assert disc[1] == 0.5
        assert disc[2] == 1.0  # Unchanged

    def test_is_satisfied_true(self, fitted_model):
        """Test is_satisfied when within bounds."""
        constraint = BoundConstraint("discrimination", lower=0.5, upper=2.0)
        constraint.apply(fitted_model)

        assert constraint.is_satisfied(fitted_model)

    def test_is_satisfied_false_lower(self, fitted_model):
        """Test is_satisfied when below lower bound."""
        fitted_model._parameters["discrimination"][0] = 0.1
        constraint = BoundConstraint("discrimination", lower=0.5)

        assert not constraint.is_satisfied(fitted_model)

    def test_is_satisfied_false_upper(self, fitted_model):
        """Test is_satisfied when above upper bound."""
        fitted_model._parameters["discrimination"][0] = 5.0
        constraint = BoundConstraint("discrimination", upper=3.0)

        assert not constraint.is_satisfied(fitted_model)

    def test_penalty_zero_when_satisfied(self, fitted_model):
        """Test penalty is zero when within bounds."""
        constraint = BoundConstraint("discrimination", lower=0.5, upper=2.0)
        constraint.apply(fitted_model)

        assert constraint.penalty(fitted_model) == pytest.approx(0.0)

    def test_penalty_positive_lower_violation(self, fitted_model):
        """Test penalty for lower bound violation."""
        fitted_model._parameters["discrimination"][0] = 0.1
        constraint = BoundConstraint("discrimination", lower=0.5)

        penalty = constraint.penalty(fitted_model)
        expected = (0.5 - 0.1) ** 2
        assert penalty == pytest.approx(expected)

    def test_penalty_positive_upper_violation(self, fitted_model):
        """Test penalty for upper bound violation."""
        fitted_model._parameters["discrimination"][0] = 5.0
        constraint = BoundConstraint("discrimination", upper=3.0)

        penalty = constraint.penalty(fitted_model)
        expected = (5.0 - 3.0) ** 2
        assert penalty == pytest.approx(expected)


class TestLinearConstraint:
    """Tests for LinearConstraint class."""

    def test_mean_constraint(self, fitted_model):
        """Test mean constraint."""
        constraint = LinearConstraint("difficulty", target=0.0, constraint_type="mean")

        constraint.apply(fitted_model)

        mean_diff = np.mean(fitted_model.parameters["difficulty"])
        assert mean_diff == pytest.approx(0.0, abs=1e-10)

    def test_sum_constraint(self, fitted_model):
        """Test sum constraint."""
        constraint = LinearConstraint("difficulty", target=0.0, constraint_type="sum")

        constraint.apply(fitted_model)

        sum_diff = np.sum(fitted_model.parameters["difficulty"])
        assert sum_diff == pytest.approx(0.0, abs=1e-10)

    def test_mean_constraint_subset(self, fitted_model):
        """Test mean constraint on subset of items modifies parameters."""
        original_vals = fitted_model.parameters["difficulty"][:3].copy()

        constraint = LinearConstraint(
            "difficulty", [0, 1, 2], target=1.0, constraint_type="mean"
        )
        constraint.apply(fitted_model)

        new_vals = fitted_model.parameters["difficulty"][:3]
        assert not np.allclose(new_vals, original_vals), "Parameters should be modified"

    def test_is_satisfied_true(self, fitted_model):
        """Test is_satisfied when constraint is met."""
        constraint = LinearConstraint("difficulty", target=0.0, constraint_type="mean")
        constraint.apply(fitted_model)

        assert constraint.is_satisfied(fitted_model)

    def test_is_satisfied_false(self, fitted_model):
        """Test is_satisfied when constraint is not met."""
        constraint = LinearConstraint("difficulty", target=5.0, constraint_type="mean")

        assert not constraint.is_satisfied(fitted_model)

    def test_penalty_zero_when_satisfied(self, fitted_model):
        """Test penalty is zero when constraint is satisfied."""
        constraint = LinearConstraint("difficulty", target=0.0, constraint_type="mean")
        constraint.apply(fitted_model)

        assert constraint.penalty(fitted_model) == pytest.approx(0.0, abs=1e-10)

    def test_penalty_positive_when_violated(self, fitted_model):
        """Test penalty is positive when constraint is violated."""
        constraint = LinearConstraint("difficulty", target=5.0, constraint_type="mean")

        penalty = constraint.penalty(fitted_model)
        current_mean = np.mean(fitted_model.parameters["difficulty"])
        expected = (current_mean - 5.0) ** 2
        assert penalty == pytest.approx(expected)


class TestCustomConstraint:
    """Tests for CustomConstraint class."""

    def test_custom_apply(self, fitted_model):
        """Test custom apply function."""

        def set_first_to_one(model):
            model._parameters["discrimination"][0] = 1.0

        constraint = CustomConstraint("custom", apply_func=set_first_to_one)

        constraint.apply(fitted_model)

        assert fitted_model.parameters["discrimination"][0] == 1.0

    def test_custom_check(self, fitted_model):
        """Test custom check function."""

        def is_first_one(model):
            return model.parameters["discrimination"][0] == 1.0

        constraint = CustomConstraint("custom", check_func=is_first_one)

        assert constraint.is_satisfied(fitted_model)  # It is 1.0

        fitted_model._parameters["discrimination"][0] = 2.0
        assert not constraint.is_satisfied(fitted_model)

    def test_custom_penalty(self, fitted_model):
        """Test custom penalty function."""

        def compute_penalty(model):
            return abs(model.parameters["discrimination"][0] - 1.0)

        constraint = CustomConstraint("custom", penalty_func=compute_penalty)

        assert constraint.penalty(fitted_model) == 0.0  # Already 1.0

        fitted_model._parameters["discrimination"][0] = 2.0
        assert constraint.penalty(fitted_model) == 1.0

    def test_none_functions(self, fitted_model):
        """Test that None functions work gracefully."""
        constraint = CustomConstraint("custom")

        constraint.apply(fitted_model)
        assert constraint.is_satisfied(fitted_model) is True
        assert constraint.penalty(fitted_model) == 0.0


class TestConstraintSet:
    """Tests for ConstraintSet class."""

    def test_add_constraints(self, fitted_model):
        """Test adding constraints to set."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=1.0))
        constraint_set.add(EqualityConstraint("difficulty", [0, 1, 2]))

        assert len(constraint_set) == 2

    def test_apply_all(self, fitted_model):
        """Test applying all constraints."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=2.0))
        constraint_set.add(LinearConstraint("difficulty", target=0.0))

        constraint_set.apply_all(fitted_model)

        assert fitted_model.parameters["discrimination"][0] == 2.0
        assert np.mean(fitted_model.parameters["difficulty"]) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_all_satisfied_true(self, fitted_model):
        """Test all_satisfied when all constraints are met."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=1.0))
        constraint_set.add(LinearConstraint("difficulty", target=0.0))
        constraint_set.apply_all(fitted_model)

        assert constraint_set.all_satisfied(fitted_model)

    def test_all_satisfied_false(self, fitted_model):
        """Test all_satisfied when some constraint is not met."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=2.0))

        assert not constraint_set.all_satisfied(fitted_model)

    def test_total_penalty(self, fitted_model):
        """Test total penalty computation."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=2.0))
        constraint_set.add(FixedConstraint("discrimination", [1], value=2.0))

        total = constraint_set.total_penalty(fitted_model)

        expected = (1.0 - 2.0) ** 2 + (1.2 - 2.0) ** 2
        assert total == pytest.approx(expected)

    def test_summary(self, fitted_model):
        """Test constraint summary."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=1.0))
        constraint_set.add(EqualityConstraint("difficulty", [0, 1, 2]))

        summary = constraint_set.summary(fitted_model)

        assert len(summary) == 2
        assert summary[0]["constraint"] == "FixedConstraint"
        assert summary[0]["param"] == "discrimination"
        assert summary[0]["satisfied"] is True
        assert summary[1]["constraint"] == "EqualityConstraint"

    def test_iteration(self, fitted_model):
        """Test iterating over constraint set."""
        constraint_set = ConstraintSet()
        constraint_set.add(FixedConstraint("discrimination", [0], value=1.0))
        constraint_set.add(EqualityConstraint("difficulty"))

        constraints = list(constraint_set)
        assert len(constraints) == 2
        assert isinstance(constraints[0], FixedConstraint)
        assert isinstance(constraints[1], EqualityConstraint)

    def test_chained_add(self, fitted_model):
        """Test chained add calls."""
        constraint_set = (
            ConstraintSet()
            .add(FixedConstraint("discrimination", [0], value=1.0))
            .add(EqualityConstraint("difficulty"))
        )

        assert len(constraint_set) == 2


class TestHelperFunctions:
    """Tests for constraint helper functions."""

    def test_fix_parameter(self, fitted_model):
        """Test fix_parameter helper."""
        constraint = fix_parameter("discrimination", 1.0, items=[0, 1])
        constraint.apply(fitted_model)

        assert fitted_model.parameters["discrimination"][0] == 1.0
        assert fitted_model.parameters["discrimination"][1] == 1.0

    def test_equal_parameters(self, fitted_model):
        """Test equal_parameters helper."""
        constraint = equal_parameters("difficulty", items=[0, 1, 2])
        constraint.apply(fitted_model)

        diff = fitted_model.parameters["difficulty"]
        assert diff[0] == pytest.approx(diff[1])
        assert diff[1] == pytest.approx(diff[2])

    def test_bound_parameter(self, fitted_model):
        """Test bound_parameter helper."""
        fitted_model._parameters["discrimination"][0] = 0.1
        constraint = bound_parameter("discrimination", lower=0.5, upper=3.0)
        constraint.apply(fitted_model)

        assert np.all(fitted_model.parameters["discrimination"] >= 0.5)
        assert np.all(fitted_model.parameters["discrimination"] <= 3.0)

    def test_mean_constraint_helper(self, fitted_model):
        """Test mean_constraint helper."""
        constraint = mean_constraint("difficulty", target=0.0)
        constraint.apply(fitted_model)

        assert np.mean(fitted_model.parameters["difficulty"]) == pytest.approx(
            0.0, abs=1e-10
        )


class TestPresetConstraints:
    """Tests for preset constraint configurations."""

    def test_1pl_constraints(self, fitted_model):
        """Test 1PL constraint preset."""
        constraints = create_1pl_constraints()

        assert len(constraints) == 2
        constraints.apply_all(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert np.allclose(disc, disc[0])
        assert np.mean(fitted_model.parameters["difficulty"]) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_rasch_constraints(self, fitted_model):
        """Test Rasch constraint preset."""
        constraints = create_rasch_constraints()

        assert len(constraints) == 2
        constraints.apply_all(fitted_model)

        assert np.allclose(fitted_model.parameters["discrimination"], 1.0)
        assert np.mean(fitted_model.parameters["difficulty"]) == pytest.approx(
            0.0, abs=1e-10
        )


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_item_indices(self, fitted_model):
        """Test with empty item indices list."""
        constraint = FixedConstraint("discrimination", [], value=1.0)

        constraint.apply(fitted_model)

    def test_constraint_with_3pl_guessing(self, fitted_3pl_model):
        """Test constraints on guessing parameter."""
        constraint = BoundConstraint("guessing", lower=0.0, upper=0.35)
        constraint.apply(fitted_3pl_model)

        guess = fitted_3pl_model.parameters["guessing"]
        assert np.all(guess >= 0.0)
        assert np.all(guess <= 0.35)

    def test_multiple_constraints_same_param(self, fitted_model):
        """Test multiple constraints on same parameter."""
        constraint_set = ConstraintSet()
        constraint_set.add(BoundConstraint("discrimination", lower=0.5))
        constraint_set.add(BoundConstraint("discrimination", upper=2.0))

        fitted_model._parameters["discrimination"] = np.array([0.1, 5.0, 1.0, 1.5, 1.1])
        constraint_set.apply_all(fitted_model)

        disc = fitted_model.parameters["discrimination"]
        assert np.all(disc >= 0.5)
        assert np.all(disc <= 2.0)

    def test_tolerance_in_is_satisfied(self, fitted_model):
        """Test tolerance parameter in is_satisfied."""
        constraint = FixedConstraint("discrimination", [0], value=1.0)
        fitted_model._parameters["discrimination"][0] = 1.1

        assert constraint.is_satisfied(fitted_model, tol=0.2)
        assert not constraint.is_satisfied(fitted_model, tol=0.05)
