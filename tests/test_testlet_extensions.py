"""Tests for Testlet Model Extensions."""

import numpy as np
import pytest

from mirt.models.testlet import (
    BifactorTestletModel,
    RandomTestletEffectsModel,
    TestletModel,
    compute_testlet_q3,
    create_testlet_structure,
)


class TestCreateTestletStructure:
    """Tests for create_testlet_structure function."""

    def test_basic(self):
        """Test basic testlet structure creation."""
        membership = create_testlet_structure(10, [3, 3, 1, 3])

        expected = np.array([0, 0, 0, 1, 1, 1, -1, 2, 2, 2])
        np.testing.assert_array_equal(membership, expected)

    def test_all_testlets(self):
        """Test structure with no standalone items."""
        membership = create_testlet_structure(6, [3, 3])

        expected = np.array([0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(membership, expected)

    def test_all_standalone(self):
        """Test structure with all standalone items."""
        membership = create_testlet_structure(5, [1, 1, 1, 1, 1])

        expected = np.array([-1, -1, -1, -1, -1])
        np.testing.assert_array_equal(membership, expected)

    def test_mismatch_error(self):
        """Test error when sizes don't sum to n_items."""
        with pytest.raises(ValueError, match="Sum of testlet_sizes"):
            create_testlet_structure(10, [3, 3])


class TestTestletModel:
    """Tests for base TestletModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        membership = np.array([0, 0, 0, 1, 1, -1])
        model = TestletModel(n_items=6, testlet_membership=membership)

        assert model.n_items == 6
        assert model.n_testlets == 2
        np.testing.assert_array_equal(model.testlet_membership, membership)

    def test_init_wrong_length(self):
        """Test error on membership length mismatch."""
        with pytest.raises(ValueError, match="testlet_membership length"):
            TestletModel(n_items=6, testlet_membership=[0, 0, 1])

    def test_get_testlet_items(self):
        """Test getting items in a testlet."""
        membership = np.array([0, 0, 0, 1, 1, -1])
        model = TestletModel(n_items=6, testlet_membership=membership)

        items_0 = model.get_testlet_items(0)
        items_1 = model.get_testlet_items(1)

        assert items_0 == [0, 1, 2]
        assert items_1 == [3, 4]

    def test_probability_general_only(self):
        """Test probability with only general factor."""
        membership = np.array([0, 0, 1, 1])
        model = TestletModel(n_items=4, testlet_membership=membership)

        theta = np.array([0.0, 1.0, -1.0])
        probs = model._marginal_probability(theta)

        assert probs.shape == (3, 4)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_full(self):
        """Test probability with full theta specification."""
        membership = np.array([0, 0, 1, 1])
        model = TestletModel(n_items=4, testlet_membership=membership)

        theta = np.array([[0.0, 0.5, -0.5], [1.0, 0.0, 0.0]])
        probs = model.probability(theta)

        assert probs.shape == (2, 4)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_testlet_reliability(self):
        """Test testlet reliability computation."""
        membership = np.array([0, 0, 0, 1, 1, 1])
        model = TestletModel(n_items=6, testlet_membership=membership)

        reliability = model.testlet_reliability()

        assert 0 in reliability
        assert 1 in reliability
        assert 0 < reliability[0] < 1
        assert 0 < reliability[1] < 1


class TestBifactorTestletModel:
    """Tests for BifactorTestletModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(n_items=4, testlet_membership=membership)

        assert model.n_items == 4
        assert model.n_testlets == 2
        assert model.constrain_testlet_loadings is False

    def test_init_constrained(self):
        """Test initialization with constrained loadings."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(
            n_items=4, testlet_membership=membership, constrain_testlet_loadings=True
        )

        assert model.constrain_testlet_loadings is True

    def test_general_loadings(self):
        """Test getting and setting general loadings."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(n_items=4, testlet_membership=membership)

        loadings = np.array([0.8, 0.9, 1.0, 1.1])
        model.set_general_loadings(loadings)

        np.testing.assert_array_almost_equal(model.general_loadings, loadings)

    def test_set_testlet_loadings(self):
        """Test setting testlet loadings."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(n_items=4, testlet_membership=membership)

        loadings = np.array([0.5, 0.6, 0.4, 0.3])
        model.set_testlet_loadings(loadings)

        np.testing.assert_array_almost_equal(model.testlet_loadings, loadings)

    def test_set_testlet_loadings_constrained(self):
        """Test setting testlet loadings with constraint."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(
            n_items=4, testlet_membership=membership, constrain_testlet_loadings=True
        )

        loadings = np.array([0.4, 0.6, 0.3, 0.5])
        model.set_testlet_loadings(loadings)

        result = model.testlet_loadings
        assert result[0] == result[1]
        assert result[2] == result[3]

    def test_explained_variance(self):
        """Test variance decomposition."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(n_items=4, testlet_membership=membership)

        var_decomp = model.explained_variance()

        assert "general" in var_decomp
        assert "testlet" in var_decomp
        assert "unique" in var_decomp
        assert "total_common" in var_decomp
        assert var_decomp["general"] + var_decomp["testlet"] + var_decomp[
            "unique"
        ] == pytest.approx(1.0)

    def test_omega_hierarchical(self):
        """Test omega hierarchical computation."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(n_items=4, testlet_membership=membership)

        omega = model.omega_hierarchical()

        assert 0 < omega < 1

    def test_copy(self):
        """Test model copying."""
        membership = np.array([0, 0, 1, 1])
        model = BifactorTestletModel(
            n_items=4, testlet_membership=membership, constrain_testlet_loadings=True
        )

        model_copy = model.copy()

        assert model_copy is not model
        assert model_copy.constrain_testlet_loadings == model.constrain_testlet_loadings
        np.testing.assert_array_equal(
            model_copy.testlet_membership, model.testlet_membership
        )


class TestRandomTestletEffectsModel:
    """Tests for RandomTestletEffectsModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        assert model.n_items == 4
        assert model.n_quadpts == 11

    def test_init_custom_quadpts(self):
        """Test initialization with custom quadrature points."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(
            n_items=4, testlet_membership=membership, n_quadpts=21
        )

        assert model.n_quadpts == 21

    def test_testlet_effect_variance(self):
        """Test getting testlet effect variance."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        var = model.testlet_effect_variance
        assert var.shape == (2,)

    def test_set_testlet_variance(self):
        """Test setting specific testlet variance."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        model.set_testlet_variance(0, 0.5)

        assert model.testlet_effect_variance[0] == 0.5

    def test_set_testlet_variance_negative(self):
        """Test error on negative variance."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        with pytest.raises(ValueError, match="non-negative"):
            model.set_testlet_variance(0, -0.5)

    def test_set_all_testlet_variances(self):
        """Test setting all testlet variances."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        model.set_all_testlet_variances(0.25)

        np.testing.assert_array_almost_equal(
            model.testlet_effect_variance, np.array([0.25, 0.25])
        )

    def test_integrate_out_testlet_effects(self):
        """Test integrating out testlet effects."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(n_items=4, testlet_membership=membership)

        responses = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]])
        theta = np.array([0.0, 0.5, 1.0])

        ll = model.integrate_out_testlet_effects(responses, theta)

        assert ll.shape == (3,)
        assert np.all(ll <= 0)

    def test_estimate_testlet_variances(self):
        """Test estimating testlet variances."""
        np.random.seed(42)
        membership = np.array([0, 0, 0, 1, 1, 1])
        model = RandomTestletEffectsModel(n_items=6, testlet_membership=membership)

        n_persons = 100
        responses = np.random.binomial(1, 0.5, (n_persons, 6))
        theta = np.random.normal(0, 1, n_persons)

        variances = model.estimate_testlet_variances(responses, theta)

        assert variances.shape == (2,)
        assert np.all(variances >= 0)

    def test_copy(self):
        """Test model copying."""
        membership = np.array([0, 0, 1, 1])
        model = RandomTestletEffectsModel(
            n_items=4, testlet_membership=membership, n_quadpts=15
        )

        model_copy = model.copy()

        assert model_copy is not model
        assert model_copy.n_quadpts == model.n_quadpts


class TestComputeTestletQ3:
    """Tests for compute_testlet_q3 function."""

    def test_basic(self):
        """Test basic Q3 computation."""
        np.random.seed(42)
        n_persons = 100
        n_items = 6

        responses = np.random.binomial(1, 0.5, (n_persons, n_items))
        theta = np.random.normal(0, 1, n_persons)
        discrimination = np.ones(n_items)
        difficulty = np.zeros(n_items)
        membership = np.array([0, 0, 0, 1, 1, 1])

        result = compute_testlet_q3(
            responses, theta, discrimination, difficulty, membership
        )

        assert "q3_matrix" in result
        assert "within_testlet_mean" in result
        assert "between_testlet_mean" in result
        assert "within_testlet_max" in result
        assert "between_testlet_max" in result

        assert result["q3_matrix"].shape == (n_items, n_items)

    def test_no_testlets(self):
        """Test Q3 with no testlets (all standalone)."""
        np.random.seed(42)
        n_persons = 50
        n_items = 4

        responses = np.random.binomial(1, 0.5, (n_persons, n_items))
        theta = np.random.normal(0, 1, n_persons)
        discrimination = np.ones(n_items)
        difficulty = np.zeros(n_items)
        membership = np.array([-1, -1, -1, -1])

        result = compute_testlet_q3(
            responses, theta, discrimination, difficulty, membership
        )

        assert np.isnan(result["within_testlet_mean"])
        assert not np.isnan(result["between_testlet_mean"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
