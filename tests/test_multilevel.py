"""Tests for multilevel IRT models."""

import numpy as np
import pytest

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multilevel import (
    CrossedRandomEffectsModel,
    MultilevelIRTModel,
    NestedHierarchy,
    RandomEffectSpec,
    ThreeLevelIRTModel,
)


class TestMultilevelIRTModel:
    """Tests for two-level IRT model."""

    def test_init_basic(self):
        """Test basic initialization."""
        base_model = TwoParameterLogistic(n_items=10)
        group_membership = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        assert model.n_persons == 10
        assert model.n_groups == 3
        assert model.n_items == 10
        assert model.group_means.shape == (3,)

    def test_init_with_labels(self):
        """Test initialization with group labels."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1, 1])
        labels = ["School A", "School B"]

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
            group_labels=labels,
        )

        assert model.group_labels == labels

    def test_set_group_means(self):
        """Test setting group means."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        new_means = np.array([0.5, -0.3])
        model.set_group_means(new_means)

        np.testing.assert_array_equal(model.group_means, new_means)

    def test_set_variance_components(self):
        """Test setting variance components."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        model.set_variance_components(between=0.5, within=0.8)

        assert model.between_variance == 0.5
        assert model.within_variance == 0.8

    def test_icc(self):
        """Test ICC computation."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        model.set_variance_components(between=0.25, within=0.75)

        expected_icc = 0.25 / (0.25 + 0.75)
        assert model.icc == pytest.approx(expected_icc)

    def test_person_prior_mean(self):
        """Test person prior mean based on group membership."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        model.set_group_means(np.array([0.5, -0.3]))
        prior_means = model.person_prior_mean()

        expected = np.array([0.5, 0.5, -0.3, -0.3, -0.3])
        np.testing.assert_array_almost_equal(prior_means, expected)

    def test_group_sizes(self):
        """Test group size computation."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 0, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )

        sizes = model.group_sizes()
        expected = np.array([3, 2])
        np.testing.assert_array_equal(sizes, expected)

    def test_copy(self):
        """Test model copying."""
        base_model = TwoParameterLogistic(n_items=5)
        group_membership = np.array([0, 0, 1, 1])

        model = MultilevelIRTModel(
            base_model=base_model,
            group_membership=group_membership,
        )
        model.set_group_means(np.array([0.5, -0.3]))
        model.set_variance_components(between=0.3)

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.group_means, model.group_means)
        assert model_copy.between_variance == model.between_variance


class TestThreeLevelIRTModel:
    """Tests for three-level IRT model."""

    def test_init_basic(self):
        """Test basic initialization."""
        base_model = TwoParameterLogistic(n_items=5)

        level2_membership = np.array([0, 0, 1, 1, 2, 2])
        level3_membership = np.array([0, 0, 1])

        model = ThreeLevelIRTModel(
            base_model=base_model,
            level2_membership=level2_membership,
            level3_membership=level3_membership,
        )

        assert model.n_persons == 6
        assert model.n_level2_units == 3
        assert model.n_level3_units == 2

    def test_variance_components(self):
        """Test variance component retrieval."""
        base_model = TwoParameterLogistic(n_items=5)
        level2_membership = np.array([0, 0, 1, 1])
        level3_membership = np.array([0, 0])

        model = ThreeLevelIRTModel(
            base_model=base_model,
            level2_membership=level2_membership,
            level3_membership=level3_membership,
        )

        vc = model.variance_components
        assert "within" in vc
        assert "level2" in vc
        assert "level3" in vc

    def test_icc_levels(self):
        """Test ICC at different levels."""
        base_model = TwoParameterLogistic(n_items=5)
        level2_membership = np.array([0, 0, 1, 1])
        level3_membership = np.array([0, 0])

        model = ThreeLevelIRTModel(
            base_model=base_model,
            level2_membership=level2_membership,
            level3_membership=level3_membership,
        )

        icc_l2 = model.icc("level2")
        icc_l3 = model.icc("level3")
        icc_total = model.icc("total")

        assert icc_l2 >= 0
        assert icc_l3 >= 0
        assert icc_total == pytest.approx(icc_l2 + icc_l3)


class TestCrossedRandomEffectsModel:
    """Tests for crossed random effects model."""

    def test_init_basic(self):
        """Test basic initialization."""
        base_model = TwoParameterLogistic(n_items=10)

        model = CrossedRandomEffectsModel(
            base_model=base_model,
            n_raters=5,
        )

        assert model.n_raters == 5
        assert model.rater_effects.shape == (5,)

    def test_set_rater_effects(self):
        """Test setting rater effects."""
        base_model = TwoParameterLogistic(n_items=5)
        model = CrossedRandomEffectsModel(base_model=base_model, n_raters=3)

        effects = np.array([0.2, -0.1, -0.1])
        model.set_rater_effects(effects)

        np.testing.assert_array_equal(model.rater_effects, effects)

    def test_set_variance_components(self):
        """Test setting variance components."""
        base_model = TwoParameterLogistic(n_items=5)
        model = CrossedRandomEffectsModel(base_model=base_model, n_raters=3)

        model.set_variance_components(person=1.5, item=0.3, rater=0.2)

        vc = model.variance_components
        assert vc["person"] == 1.5
        assert vc["item"] == 0.3
        assert vc["rater"] == 0.2

    def test_probability_without_rater(self):
        """Test probability without rater effect."""
        base_model = TwoParameterLogistic(n_items=3)
        model = CrossedRandomEffectsModel(base_model=base_model, n_raters=2)

        theta = np.array([[0.0], [1.0]])
        probs = model.probability(theta)

        assert probs.shape == (2, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_with_rater(self):
        """Test probability with rater effect."""
        base_model = TwoParameterLogistic(n_items=3)
        model = CrossedRandomEffectsModel(base_model=base_model, n_raters=2)
        model.set_rater_effects(np.array([0.5, -0.5]))

        theta = np.array([[0.0]])

        prob_lenient = model.probability(theta, item_idx=0, rater_idx=0)
        prob_strict = model.probability(theta, item_idx=0, rater_idx=1)

        assert prob_lenient[0] > prob_strict[0]


class TestRandomEffectSpec:
    """Tests for random effect specification."""

    def test_init(self):
        """Test initialization."""
        spec = RandomEffectSpec(
            name="school",
            type="nested",
            n_levels=20,
        )

        assert spec.name == "school"
        assert spec.type == "nested"
        assert spec.n_levels == 20

    def test_variance_prior(self):
        """Test default variance prior."""
        spec = RandomEffectSpec(name="rater", type="crossed", n_levels=5)

        assert spec.variance_prior == (1.0, 1.0)


class TestNestedHierarchy:
    """Tests for nested hierarchy specification."""

    def test_init(self):
        """Test initialization."""
        levels = ["student", "classroom", "school"]
        memberships = [
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 1]),
        ]

        hierarchy = NestedHierarchy(levels=levels, memberships=memberships)

        assert hierarchy.n_levels == 3

    def test_get_unit_counts(self):
        """Test unit count retrieval."""
        levels = ["student", "classroom", "school"]
        memberships = [
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 1]),
        ]

        hierarchy = NestedHierarchy(levels=levels, memberships=memberships)
        counts = hierarchy.get_unit_counts()

        assert counts["student"] == 6
        assert counts["classroom"] == 3
        assert counts["school"] == 2

    def test_get_full_path(self):
        """Test full path retrieval."""
        levels = ["student", "classroom", "school"]
        memberships = [
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 1]),
        ]

        hierarchy = NestedHierarchy(levels=levels, memberships=memberships)

        path = hierarchy.get_full_path(2, level=0)
        assert path == [2, 1, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
