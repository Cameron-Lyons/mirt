"""Tests for Unfolding/Ideal Point IRT models."""

import numpy as np

from mirt import GeneralizedGradedUnfolding, HyperbolicCosineModel, IdealPointModel


class TestGeneralizedGradedUnfolding:
    """Tests for GGUM model."""

    def test_init(self):
        """Test GGUM initialization."""
        model = GeneralizedGradedUnfolding(n_items=10, n_categories=5)
        assert model.n_items == 10
        assert model.model_name == "GGUM"

    def test_parameters(self):
        """Test parameter initialization."""
        model = GeneralizedGradedUnfolding(n_items=10, n_categories=5)
        model._initialize_parameters()

        assert "discrimination" in model._parameters
        assert "location" in model._parameters
        assert "thresholds" in model._parameters

        thresholds = model._parameters["thresholds"]
        assert thresholds.shape[0] == 10

    def test_probability(self):
        """Test unfolding probability computation."""
        model = GeneralizedGradedUnfolding(n_items=5, n_categories=4)
        model._initialize_parameters()

        theta = np.array([[-2.0], [0.0], [2.0]])
        probs = model.probability(theta)

        assert probs.shape == (3, 5, 4)
        assert np.allclose(probs.sum(axis=2), 1.0)

    def test_single_peaked(self):
        """Test that response function is single-peaked."""
        model = GeneralizedGradedUnfolding(n_items=1, n_categories=5)
        model._initialize_parameters()
        model._parameters["location"] = np.array([0.0])

        theta = np.linspace(-3, 3, 50).reshape(-1, 1)
        probs = model.probability(theta)

        expected_scores = np.sum(probs[:, 0, :] * np.arange(5), axis=1)
        peak_idx = np.argmax(expected_scores)

        assert 15 < peak_idx < 35


class TestIdealPointModel:
    """Tests for Ideal Point model."""

    def test_init(self):
        """Test IdealPointModel initialization."""
        model = IdealPointModel(n_items=10)
        assert model.n_items == 10
        assert model.model_name == "IdealPoint"

    def test_probability_single_peaked(self):
        """Test that probability is single-peaked at item location."""
        model = IdealPointModel(n_items=1)
        model._initialize_parameters()
        model._parameters["location"] = np.array([0.0])
        model._parameters["discrimination"] = np.array([2.0])

        theta = np.linspace(-3, 3, 100).reshape(-1, 1)
        probs = model.probability(theta)

        peak_idx = np.argmax(probs[:, 0])
        assert 45 < peak_idx < 55

    def test_distance_based(self):
        """Test distance-based probability function."""
        model = IdealPointModel(n_items=1)
        model._initialize_parameters()
        model._parameters["location"] = np.array([1.0])

        theta_at = np.array([[1.0]])
        theta_far = np.array([[3.0]])

        prob_at = model.probability(theta_at)[0, 0]
        prob_far = model.probability(theta_far)[0, 0]

        assert prob_at > prob_far


class TestHyperbolicCosineModel:
    """Tests for Hyperbolic Cosine Model."""

    def test_init(self):
        """Test HCM initialization."""
        model = HyperbolicCosineModel(n_items=10)
        assert model.n_items == 10
        assert model.model_name == "HCM"

    def test_probability(self):
        """Test HCM probability computation."""
        model = HyperbolicCosineModel(n_items=5)
        model._initialize_parameters()

        theta = np.array([[0.0], [1.0], [-1.0]])
        probs = model.probability(theta)

        assert probs.shape == (3, 5)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_symmetric(self):
        """Test that HCM is symmetric around item location."""
        model = HyperbolicCosineModel(n_items=1)
        model._initialize_parameters()
        model._parameters["location"] = np.array([0.0])

        theta_pos = np.array([[1.0]])
        theta_neg = np.array([[-1.0]])

        prob_pos = model.probability(theta_pos)[0, 0]
        prob_neg = model.probability(theta_neg)[0, 0]

        np.testing.assert_almost_equal(prob_pos, prob_neg, decimal=5)
