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
from mirt.models.polytomous import GradedResponseModel


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

    def test_sparse_group_identifiers(self):
        base_model = TwoParameterLogistic(n_items=2)
        model = MultilevelIRTModel(base_model, np.array([7, 2, 7, 2, 7]))
        model.set_group_means(np.array([-0.25, 0.75]))

        np.testing.assert_array_equal(model.group_ids, [2, 7])
        np.testing.assert_array_equal(model.group_sizes(), [2, 3])
        np.testing.assert_allclose(
            model.person_prior_mean(), [0.75, -0.25, 0.75, -0.25, 0.75]
        )
        assert model.group_labels == ["Group_2", "Group_7"]

    @pytest.mark.parametrize(
        "membership",
        [
            np.array([]),
            np.array([[0, 1]]),
            np.array([0.0, 1.0]),
            np.array([False, True]),
            np.array([0, -1]),
        ],
    )
    def test_rejects_invalid_membership(self, membership):
        with pytest.raises(ValueError):
            MultilevelIRTModel(TwoParameterLogistic(n_items=2), membership)

    def test_copies_membership_and_means(self):
        membership = np.array([2, 2, 7])
        means = np.array([0.1, 0.2])
        model = MultilevelIRTModel(TwoParameterLogistic(n_items=2), membership)
        model.set_group_means(means)

        membership[:] = 99
        means[:] = 99
        returned_membership = model.group_membership
        returned_membership[:] = 99

        np.testing.assert_array_equal(model.group_membership, [2, 2, 7])
        np.testing.assert_allclose(model.group_means, [0.1, 0.2])

    def test_rejects_invalid_labels_and_nonfinite_means(self):
        base_model = TwoParameterLogistic(n_items=2)
        with pytest.raises(ValueError, match="labels length"):
            MultilevelIRTModel(base_model, np.array([2, 7]), ["only one"])

        model = MultilevelIRTModel(base_model, np.array([2, 7]))
        with pytest.raises(ValueError, match="finite"):
            model.set_group_means(np.array([0.0, np.nan]))
        np.testing.assert_array_equal(model.group_means, [0.0, 0.0])

    @pytest.mark.parametrize(
        ("between", "within"),
        [(np.nan, 1.0), (np.inf, 1.0), (-0.1, 1.0), (0.1, 0.0)],
    )
    def test_variance_update_is_validated_atomically(self, between, within):
        model = MultilevelIRTModel(TwoParameterLogistic(n_items=2), np.array([0, 1]))
        before = (model.between_variance, model.within_variance)

        with pytest.raises(ValueError):
            model.set_variance_components(between, within)

        assert (model.between_variance, model.within_variance) == before

    def test_log_likelihood_matches_base_model(self):
        base_model = TwoParameterLogistic(n_items=2)
        model = MultilevelIRTModel(base_model, np.array([2, 7]))
        responses = np.array([[0, 1], [1, 1]])
        theta = np.array([[-0.5], [0.5]])

        expected = np.sum(model.base_model.log_likelihood(responses, theta))
        assert model.log_likelihood(responses, theta) == pytest.approx(expected)


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

    def test_sparse_identifiers_effects_and_sizes(self):
        model = ThreeLevelIRTModel(
            TwoParameterLogistic(n_items=2),
            level2_membership=np.array([7, 2, 7, 11, 2]),
            level3_membership=np.array([9, 4, 9]),
        )
        model.set_level_effects(
            level2=np.array([0.1, 0.2, 0.3]), level3=np.array([-1.0, 1.0])
        )

        np.testing.assert_array_equal(model.level2_ids, [2, 7, 11])
        np.testing.assert_array_equal(model.level3_ids, [4, 9])
        np.testing.assert_array_equal(model.level2_sizes(), [2, 2, 1])
        np.testing.assert_array_equal(model.level3_sizes(), [1, 2])
        np.testing.assert_allclose(
            model.person_prior_mean(), [-0.8, 1.1, -0.8, 1.3, 1.1]
        )

    def test_custom_labels_and_membership_properties(self):
        model = ThreeLevelIRTModel(
            TwoParameterLogistic(n_items=2),
            np.array([2, 7]),
            np.array([4, 9]),
            level2_labels=["A", "B"],
            level3_labels=["East", "West"],
        )

        assert model.level2_labels == ["A", "B"]
        assert model.level3_labels == ["East", "West"]
        np.testing.assert_array_equal(model.level2_membership, [2, 7])
        np.testing.assert_array_equal(model.level3_membership, [4, 9])

    @pytest.mark.parametrize(
        ("level2", "level3"),
        [
            (np.array([[0, 1]]), np.array([0, 1])),
            (np.array([0.0, 1.0]), np.array([0, 1])),
            (np.array([0, -1]), np.array([0, 1])),
            (np.array([0, 1]), np.array([0.0, 1.0])),
            (np.array([0, 1]), np.array([0])),
        ],
    )
    def test_rejects_invalid_memberships(self, level2, level3):
        with pytest.raises(ValueError):
            ThreeLevelIRTModel(TwoParameterLogistic(n_items=2), level2, level3)

    def test_effect_update_is_validated_atomically(self):
        model = ThreeLevelIRTModel(
            TwoParameterLogistic(n_items=2), np.array([0, 1]), np.array([0, 1])
        )
        with pytest.raises(ValueError, match="finite"):
            model.set_level_effects(
                level2=np.array([1.0, 2.0]), level3=np.array([0.0, np.nan])
            )

        np.testing.assert_array_equal(model.level2_effects, [0.0, 0.0])
        np.testing.assert_array_equal(model.level3_effects, [0.0, 0.0])

    def test_variance_update_and_icc_validation(self):
        model = ThreeLevelIRTModel(
            TwoParameterLogistic(n_items=2), np.array([0, 1]), np.array([0, 1])
        )
        model.set_variance_components(level2=0.2, level3=0.3, within=0.5)

        assert model.icc("level2") == pytest.approx(0.2)
        assert model.icc("level3") == pytest.approx(0.3)
        with pytest.raises(ValueError, match="level must"):
            model.icc("unknown")

        before = model.variance_components
        with pytest.raises(ValueError):
            model.set_variance_components(level2=0.8, level3=np.nan)
        assert model.variance_components == before

    def test_copy_is_independent(self):
        model = ThreeLevelIRTModel(
            TwoParameterLogistic(n_items=2), np.array([2, 7]), np.array([4, 9])
        ).set_level_effects(np.array([0.1, 0.2]), np.array([0.3, 0.4]))

        copied = model.copy().set_level_effects(level2=np.array([9.0, 9.0]))

        np.testing.assert_allclose(model.level2_effects, [0.1, 0.2])
        np.testing.assert_allclose(copied.level3_effects, [0.3, 0.4])


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

    def test_flat_assignments_are_reshaped_and_copied(self):
        assignments = np.array([0, 1, 1, 0, 0, 0])
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=3), 2, assignments
        )
        assignments[:] = 1

        assert model.n_persons == 2
        np.testing.assert_array_equal(model.rater_assignments, [[0, 1, 1], [0, 0, 0]])
        assert model.get_rater_for_observation(1, 2) == 0

    @pytest.mark.parametrize("n_raters", [0, -1, 1.5, True])
    def test_rejects_invalid_rater_count(self, n_raters):
        with pytest.raises(ValueError):
            CrossedRandomEffectsModel(TwoParameterLogistic(n_items=2), n_raters)

    def test_rejects_polytomous_base_model(self):
        with pytest.raises(ValueError, match="dichotomous"):
            CrossedRandomEffectsModel(GradedResponseModel(n_items=2, n_categories=3), 2)

    @pytest.mark.parametrize(
        "assignments",
        [
            np.array([]),
            np.array([0, 1, 0]),
            np.array([[0], [1]]),
            np.array([[[0, 1]]]),
            np.array([[0.0, 1.0]]),
            np.array([[0, -1]]),
            np.array([[0, 2]]),
        ],
    )
    def test_rejects_invalid_assignments(self, assignments):
        with pytest.raises(ValueError):
            CrossedRandomEffectsModel(TwoParameterLogistic(n_items=2), 2, assignments)

    @pytest.mark.parametrize(
        ("person_idx", "item_idx"), [(-1, 0), (2, 0), (0, -1), (0, 2)]
    )
    def test_observation_indices_are_bounds_checked(self, person_idx, item_idx):
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2, np.array([[0, 1], [1, 0]])
        )
        with pytest.raises(IndexError):
            model.get_rater_for_observation(person_idx, item_idx)

    def test_assigned_probability_matches_scalar_path(self):
        assignments = np.array([[0, 1, 2], [2, 0, 1]])
        theta = np.array([[-0.75], [0.5]])
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=3), 3, assignments
        ).set_rater_effects(np.array([-0.4, 0.0, 0.6]))

        expected = np.empty(assignments.shape)
        for person_idx in range(assignments.shape[0]):
            for item_idx in range(assignments.shape[1]):
                expected[person_idx, item_idx] = model.probability(
                    theta[person_idx : person_idx + 1],
                    item_idx=item_idx,
                    rater_idx=int(assignments[person_idx, item_idx]),
                )[0]

        np.testing.assert_allclose(model.assigned_probability(theta), expected)

    def test_assigned_probability_requires_matching_assignments(self):
        no_assignments = CrossedRandomEffectsModel(TwoParameterLogistic(n_items=2), 2)
        with pytest.raises(ValueError, match="required"):
            no_assignments.assigned_probability(np.array([[0.0]]))

        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2, np.array([[0, 1], [1, 0]])
        )
        with pytest.raises(ValueError, match="probability shape"):
            model.assigned_probability(np.array([[0.0]]))

    def test_rater_adjustment_is_stable_at_extreme_abilities(self):
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2
        ).set_rater_effects(np.array([-1000.0, 1000.0]))
        theta = np.array([[-20.0], [20.0]])

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            low = model.probability(theta, rater_idx=0)
            high = model.probability(theta, rater_idx=1)

        assert np.all(np.isfinite(low))
        assert np.all(np.isfinite(high))
        assert np.all((0.0 <= low) & (low <= 1.0))
        assert np.all((0.0 <= high) & (high <= 1.0))

    def test_disabled_rater_effects_return_base_probabilities(self):
        assignments = np.array([[0, 1], [1, 0]])
        theta = np.array([[-0.5], [0.5]])
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2),
            2,
            assignments,
            include_rater_effects=False,
        ).set_rater_effects(np.array([-10.0, 10.0]))

        expected = model.base_model.probability(theta)
        np.testing.assert_allclose(model.assigned_probability(theta), expected)

    def test_log_likelihood_uses_assignments_and_missing_values(self):
        assignments = np.array([[0, 1], [1, 0]])
        responses = np.array([[1, 0], [-1, 1]])
        theta = np.array([[-0.5], [0.5]])
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2, assignments
        ).set_rater_effects(np.array([-0.4, 0.6]))
        probabilities = model.assigned_probability(theta)
        expected = (
            np.log(probabilities[0, 0])
            + np.log1p(-probabilities[0, 1])
            + np.log(probabilities[1, 1])
        )

        assert model.log_likelihood(responses, theta) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "responses",
        [np.array([0, 1]), np.array([[0, 2]]), np.array([[0.0, np.nan]])],
    )
    def test_log_likelihood_validates_responses(self, responses):
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2, np.array([[0, 1]])
        )
        with pytest.raises(ValueError):
            model.log_likelihood(responses, np.array([[0.0]]))

    def test_variance_partition_respects_enabled_components(self):
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2),
            2,
            include_item_effects=False,
        ).set_variance_components(person=3.0, rater=1.0)

        assert model.variance_partition == {"person": 0.75, "rater": 0.25}

    def test_setters_validate_finite_values_atomically(self):
        model = CrossedRandomEffectsModel(TwoParameterLogistic(n_items=2), 2)
        with pytest.raises(ValueError, match="finite"):
            model.set_rater_effects(np.array([0.0, np.inf]))
        np.testing.assert_array_equal(model.rater_effects, [0.0, 0.0])

        before = model.variance_components
        with pytest.raises(ValueError):
            model.set_variance_components(person=2.0, item=np.nan)
        assert model.variance_components == before

    def test_copy_is_independent(self):
        model = CrossedRandomEffectsModel(
            TwoParameterLogistic(n_items=2), 2, np.array([[0, 1]])
        ).set_rater_effects(np.array([0.1, 0.2]))

        copied = model.copy().set_rater_effects(np.array([9.0, 9.0]))

        np.testing.assert_allclose(model.rater_effects, [0.1, 0.2])
        np.testing.assert_array_equal(copied.rater_assignments, [[0, 1]])


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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "", "type": "nested", "n_levels": 1},
            {"name": "school", "type": "other", "n_levels": 1},
            {"name": "school", "type": "nested", "n_levels": 0},
            {"name": "school", "type": "nested", "n_levels": True},
            {
                "name": "school",
                "type": "nested",
                "n_levels": 1,
                "variance_prior": (1.0, np.nan),
            },
        ],
    )
    def test_rejects_invalid_specification(self, kwargs):
        with pytest.raises(ValueError):
            RandomEffectSpec(**kwargs)


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

    def test_sparse_identifiers_follow_sorted_unit_order(self):
        hierarchy = NestedHierarchy(
            levels=["person", "class", "school"],
            memberships=[np.array([2, 2, 7, 7]), np.array([4, 9])],
        )

        assert hierarchy.get_full_path(0) == [0, 2, 4]
        assert hierarchy.get_full_path(2) == [2, 7, 9]
        assert hierarchy.get_unit_counts() == {"person": 4, "class": 2, "school": 2}

    @pytest.mark.parametrize(
        ("levels", "memberships"),
        [
            (["person"], []),
            (["person", "person"], [np.array([0])]),
            (["person", "class"], []),
            (["person", "class"], [np.array([0.0])]),
            (["person", "class"], [np.array([-1])]),
            (
                ["person", "class", "school"],
                [np.array([2, 7]), np.array([4])],
            ),
        ],
    )
    def test_rejects_invalid_hierarchy(self, levels, memberships):
        with pytest.raises(ValueError):
            NestedHierarchy(levels, memberships)

    @pytest.mark.parametrize(("unit_idx", "level"), [(-1, 0), (4, 0), (0, -1), (0, 3)])
    def test_path_indices_are_bounds_checked(self, unit_idx, level):
        hierarchy = NestedHierarchy(
            ["person", "class", "school"],
            [np.array([2, 2, 7, 7]), np.array([4, 9])],
        )
        with pytest.raises(IndexError):
            hierarchy.get_full_path(unit_idx, level)

    def test_memberships_are_copied(self):
        first = np.array([2, 7])
        second = np.array([4, 9])
        hierarchy = NestedHierarchy(["person", "class", "school"], [first, second])
        first[:] = 99
        second[:] = 99

        assert hierarchy.get_full_path(1) == [1, 7, 9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
