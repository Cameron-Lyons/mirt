"""Tests for Many-Facet Rasch Model."""

import numpy as np
import pytest

from mirt.models.mfrm import (
    Facet,
    ManyFacetRaschModel,
    MFRMResult,
    PolytomousMFRM,
)


class TestFacet:
    """Tests for Facet dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        facet = Facet(name="rater", n_levels=5)

        assert facet.name == "rater"
        assert facet.n_levels == 5
        assert len(facet.labels) == 5
        assert facet.labels[0] == "rater_0"
        assert facet.is_anchored is True
        assert facet.anchor_value == 0.0

    def test_init_with_labels(self):
        """Test initialization with custom labels."""
        labels = ["Rater A", "Rater B", "Rater C"]
        facet = Facet(name="rater", n_levels=3, labels=labels)

        assert facet.labels == labels

    def test_init_label_mismatch(self):
        """Test error on label count mismatch."""
        with pytest.raises(ValueError, match="labels length"):
            Facet(name="rater", n_levels=5, labels=["A", "B"])

    def test_init_not_anchored(self):
        """Test unanchored facet."""
        facet = Facet(name="rater", n_levels=3, is_anchored=False)

        assert facet.is_anchored is False


class TestManyFacetRaschModel:
    """Tests for binary MFRM."""

    def test_init_basic(self):
        """Test basic initialization."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=5, facets=facets)

        assert model.n_items == 5
        assert model.n_facets == 1
        assert model.facet_names == ["rater"]
        assert model.item_difficulty.shape == (5,)
        assert model.facet_parameters["rater"].shape == (3,)

    def test_init_multiple_facets(self):
        """Test initialization with multiple facets."""
        facets = [
            Facet(name="rater", n_levels=3),
            Facet(name="task", n_levels=4),
            Facet(name="criterion", n_levels=2),
        ]
        model = ManyFacetRaschModel(n_items=10, facets=facets)

        assert model.n_facets == 3
        assert set(model.facet_names) == {"rater", "task", "criterion"}
        assert model.facet_parameters["rater"].shape == (3,)
        assert model.facet_parameters["task"].shape == (4,)
        assert model.facet_parameters["criterion"].shape == (2,)

    def test_init_invalid_n_items(self):
        """Test error on invalid n_items."""
        facets = [Facet(name="rater", n_levels=3)]
        with pytest.raises(ValueError, match="n_items must be at least 1"):
            ManyFacetRaschModel(n_items=0, facets=facets)

    def test_init_with_item_names(self):
        """Test initialization with custom item names."""
        facets = [Facet(name="rater", n_levels=2)]
        item_names = ["Q1", "Q2", "Q3"]
        model = ManyFacetRaschModel(n_items=3, facets=facets, item_names=item_names)

        assert model.item_names == item_names

    def test_init_item_name_mismatch(self):
        """Test error on item name count mismatch."""
        facets = [Facet(name="rater", n_levels=2)]
        with pytest.raises(ValueError, match="item_names length"):
            ManyFacetRaschModel(n_items=5, facets=facets, item_names=["A", "B"])

    def test_set_item_difficulty(self):
        """Test setting item difficulty."""
        facets = [Facet(name="rater", n_levels=2)]
        model = ManyFacetRaschModel(n_items=3, facets=facets)

        difficulty = np.array([-0.5, 0.0, 0.5])
        model.set_item_difficulty(difficulty)

        np.testing.assert_array_equal(model.item_difficulty, difficulty)

    def test_set_item_difficulty_wrong_shape(self):
        """Test error on wrong shape item difficulty."""
        facets = [Facet(name="rater", n_levels=2)]
        model = ManyFacetRaschModel(n_items=3, facets=facets)

        with pytest.raises(ValueError, match="difficulty shape"):
            model.set_item_difficulty(np.array([0.0, 1.0]))

    def test_set_facet_parameters(self):
        """Test setting facet parameters."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=2, facets=facets)

        params = np.array([0.3, -0.1, -0.2])
        model.set_facet_parameters("rater", params)

        result = model.facet_parameters["rater"]
        assert np.isclose(result.mean(), 0.0)

    def test_set_facet_parameters_unknown_facet(self):
        """Test error on unknown facet."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=2, facets=facets)

        with pytest.raises(ValueError, match="Unknown facet"):
            model.set_facet_parameters("task", np.array([0.0, 0.0]))

    def test_get_facet(self):
        """Test get_facet method."""
        facets = [
            Facet(name="rater", n_levels=3),
            Facet(name="task", n_levels=4),
        ]
        model = ManyFacetRaschModel(n_items=5, facets=facets)

        rater = model.get_facet("rater")
        assert rater.name == "rater"
        assert rater.n_levels == 3

    def test_get_facet_unknown(self):
        """Test error on unknown facet."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=5, facets=facets)

        with pytest.raises(ValueError, match="Unknown facet"):
            model.get_facet("criterion")

    def test_log_odds(self):
        """Test log-odds computation."""
        facets = [Facet(name="rater", n_levels=2)]
        model = ManyFacetRaschModel(n_items=3, facets=facets)
        model.set_item_difficulty(np.array([0.0, 0.5, -0.5]))
        model.set_facet_parameters("rater", np.array([0.2, -0.2]))

        theta = np.array([0.0, 1.0, -1.0])
        log_odds = model.log_odds(theta, item_idx=0, facet_indices={"rater": 0})

        expected = theta - 0.0 - 0.2
        np.testing.assert_array_almost_equal(log_odds, expected)

    def test_probability(self):
        """Test probability computation."""
        facets = [Facet(name="rater", n_levels=2)]
        model = ManyFacetRaschModel(n_items=3, facets=facets)

        theta = np.array([0.0])
        prob = model.probability(theta, item_idx=0, facet_indices={"rater": 0})

        assert 0 < prob[0] < 1
        assert prob.shape == (1,)

    def test_probability_rater_effect(self):
        """Test probability varies with rater."""
        facets = [Facet(name="rater", n_levels=2, is_anchored=False)]
        model = ManyFacetRaschModel(n_items=2, facets=facets)
        model.set_facet_parameters("rater", np.array([0.5, -0.5]))

        theta = np.array([0.0])
        prob_strict = model.probability(theta, item_idx=0, facet_indices={"rater": 0})
        prob_lenient = model.probability(theta, item_idx=0, facet_indices={"rater": 1})

        assert prob_lenient[0] > prob_strict[0]

    def test_information(self):
        """Test information computation."""
        facets = [Facet(name="rater", n_levels=2)]
        model = ManyFacetRaschModel(n_items=3, facets=facets)

        theta = np.array([0.0])
        info = model.information(theta, item_idx=0, facet_indices={"rater": 0})

        assert 0 < info[0] < 0.26

    def test_copy(self):
        """Test model copying."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=5, facets=facets)
        model.set_item_difficulty(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        model.set_facet_parameters("rater", np.array([0.1, 0.0, -0.1]))

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.item_difficulty, model.item_difficulty)
        np.testing.assert_array_equal(
            model_copy.facet_parameters["rater"], model.facet_parameters["rater"]
        )


class TestPolytomousMFRM:
    """Tests for polytomous MFRM."""

    def test_init_basic(self):
        """Test basic initialization."""
        facets = [Facet(name="rater", n_levels=3)]
        model = PolytomousMFRM(n_items=5, n_categories=4, facets=facets)

        assert model.n_items == 5
        assert model.n_categories == 4
        assert model.category_structure == "rating_scale"
        assert model.thresholds.shape == (3,)

    def test_init_partial_credit(self):
        """Test partial credit structure."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(
            n_items=3,
            n_categories=4,
            facets=facets,
            category_structure="partial_credit",
        )

        assert model.category_structure == "partial_credit"
        assert model.thresholds.shape == (3, 3)

    def test_init_invalid_categories(self):
        """Test error on invalid n_categories."""
        facets = [Facet(name="rater", n_levels=2)]
        with pytest.raises(ValueError, match="n_categories must be at least 2"):
            PolytomousMFRM(n_items=3, n_categories=1, facets=facets)

    def test_set_thresholds_rating_scale(self):
        """Test setting thresholds for rating scale."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        thresholds = np.array([-1.0, 0.0, 1.0])
        model.set_thresholds(thresholds)

        np.testing.assert_array_equal(model.thresholds, thresholds)

    def test_set_thresholds_partial_credit(self):
        """Test setting thresholds for partial credit."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(
            n_items=2,
            n_categories=3,
            facets=facets,
            category_structure="partial_credit",
        )

        thresholds = np.array([[-0.5, 0.5], [-1.0, 1.0]])
        model.set_thresholds(thresholds)

        np.testing.assert_array_equal(model.thresholds, thresholds)

    def test_set_thresholds_wrong_shape(self):
        """Test error on wrong shape thresholds."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        with pytest.raises(ValueError, match="thresholds shape"):
            model.set_thresholds(np.array([0.0, 1.0]))

    def test_category_probability(self):
        """Test category probability computation."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        theta = np.array([0.0, 1.0])
        prob = model.category_probability(
            theta, item_idx=0, category=1, facet_indices={"rater": 0}
        )

        assert prob.shape == (2,)
        assert np.all(prob >= 0)
        assert np.all(prob <= 1)

    def test_probability_all_categories(self):
        """Test all category probabilities."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        theta = np.array([0.0, 1.0, -1.0])
        probs = model.probability(theta, item_idx=0, facet_indices={"rater": 0})

        assert probs.shape == (3, 4)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(3))

    def test_expected_score(self):
        """Test expected score computation."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        theta = np.array([0.0, 1.0, -1.0])
        expected = model.expected_score(theta, item_idx=0, facet_indices={"rater": 0})

        assert expected.shape == (3,)
        assert np.all(expected >= 0)
        assert np.all(expected <= 3)

    def test_expected_score_monotonic(self):
        """Test expected score is monotonic in theta."""
        facets = [Facet(name="rater", n_levels=2)]
        model = PolytomousMFRM(n_items=3, n_categories=4, facets=facets)

        theta = np.linspace(-3, 3, 20)
        expected = model.expected_score(theta, item_idx=0, facet_indices={"rater": 0})

        assert np.all(np.diff(expected) > 0)

    def test_copy(self):
        """Test model copying."""
        facets = [Facet(name="rater", n_levels=3)]
        model = PolytomousMFRM(n_items=5, n_categories=4, facets=facets)
        model.set_thresholds(np.array([-1.5, 0.0, 1.5]))

        model_copy = model.copy()

        assert model_copy is not model
        assert model_copy.n_categories == model.n_categories
        assert model_copy.category_structure == model.category_structure
        np.testing.assert_array_equal(model_copy.thresholds, model.thresholds)


class TestMFRMResult:
    """Tests for MFRM result dataclass."""

    def test_init(self):
        """Test result initialization."""
        facets = [Facet(name="rater", n_levels=3)]
        model = ManyFacetRaschModel(n_items=5, facets=facets)

        result = MFRMResult(
            model=model,
            facet_parameters={"rater": np.array([0.1, 0.0, -0.1])},
            facet_se={"rater": np.array([0.05, 0.04, 0.05])},
            infit={"rater": np.array([1.0, 1.1, 0.9])},
            outfit={"rater": np.array([1.0, 1.2, 0.8])},
            log_likelihood=-500.0,
            n_iterations=25,
            converged=True,
        )

        assert result.model is model
        assert result.log_likelihood == -500.0
        assert result.n_iterations == 25
        assert result.converged is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
