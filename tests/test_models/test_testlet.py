"""Tests for Testlet IRT model."""

import numpy as np

from mirt import TestletModel, create_testlet_structure


class TestTestletModelTests:
    """Tests for TestletModel."""

    def test_init(self, testlet_responses):
        """Test TestletModel initialization."""
        membership = testlet_responses["testlet_membership"]
        model = TestletModel(
            n_items=testlet_responses["n_items"],
            testlet_membership=membership,
        )

        assert model.n_items == 12
        assert model.n_testlets == 3
        assert model.model_name == "Testlet"

    def test_testlet_structure(self):
        """Test testlet structure creation helper."""
        structure = create_testlet_structure(12, [4, 4, 4])
        assert len(structure) == 12
        assert structure[0] == 0
        assert structure[4] == 1
        assert structure[8] == 2

    def test_probability(self, testlet_responses):
        """Test probability computation."""
        membership = testlet_responses["testlet_membership"]
        model = TestletModel(
            n_items=testlet_responses["n_items"],
            testlet_membership=membership,
        )
        model._initialize_parameters()

        n_factors = model.n_factors
        theta = np.zeros((3, n_factors))
        theta[:, 0] = [0.0, 1.0, -1.0]

        probs = model.probability(theta)

        assert probs.shape == (3, 12)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_information(self, testlet_responses):
        """Test information computation."""
        membership = testlet_responses["testlet_membership"]
        model = TestletModel(
            n_items=testlet_responses["n_items"],
            testlet_membership=membership,
        )
        model._initialize_parameters()

        n_factors = model.n_factors
        theta = np.zeros((1, n_factors))

        info = model.information(theta)

        assert info.shape == (1, 12)
        assert np.all(info >= 0)

    def test_testlet_loadings(self, testlet_responses):
        """Test that testlet loadings are properly initialized."""
        membership = testlet_responses["testlet_membership"]
        model = TestletModel(
            n_items=testlet_responses["n_items"],
            testlet_membership=membership,
        )
        model._initialize_parameters()

        assert "discrimination" in model._parameters
        disc = model._parameters["discrimination"]
        assert disc.shape[0] == 12

    def test_copy(self, testlet_responses):
        """Test model copying."""
        membership = testlet_responses["testlet_membership"]
        model = TestletModel(
            n_items=testlet_responses["n_items"],
            testlet_membership=membership,
        )
        model._initialize_parameters()

        copied = model.copy()
        assert copied.n_items == model.n_items
        assert copied.n_testlets == model.n_testlets
        np.testing.assert_array_equal(
            copied._parameters["difficulty"],
            model._parameters["difficulty"],
        )
