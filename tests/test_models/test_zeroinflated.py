"""Tests for Zero-Inflated IRT models."""

import numpy as np

from mirt import HurdleIRT, ZeroInflated2PL, ZeroInflated3PL


class TestZeroInflated2PL:
    """Tests for ZeroInflated2PL model."""

    def test_init(self):
        """Test ZeroInflated2PL initialization."""
        model = ZeroInflated2PL(n_items=10)
        assert model.n_items == 10
        assert model.model_name == "ZI-2PL"

    def test_parameters(self):
        """Test parameter initialization."""
        model = ZeroInflated2PL(n_items=10)
        model._initialize_parameters()

        assert "discrimination" in model._parameters
        assert "difficulty" in model._parameters
        assert "zero_inflation" in model._parameters

        pi = model._parameters["zero_inflation"]
        assert len(pi) == 10
        assert np.all((pi >= 0) & (pi <= 1))

    def test_probability(self):
        """Test probability computation with zero inflation."""
        model = ZeroInflated2PL(n_items=5)
        model._initialize_parameters()

        # Set high zero inflation
        model._parameters["zero_inflation"] = np.full(5, 0.5)

        theta = np.array([[0.0], [2.0]])
        probs = model.probability(theta)

        assert probs.shape == (2, 5)
        assert np.all((probs >= 0) & (probs <= 1))
        assert np.all(probs < 0.9)

    def test_log_likelihood(self, dichotomous_responses):
        """Test log-likelihood computation."""
        model = ZeroInflated2PL(n_items=10)
        model._initialize_parameters()

        responses = dichotomous_responses["responses"]
        theta = dichotomous_responses["theta"].reshape(-1, 1)

        ll = model.log_likelihood(responses, theta)
        assert ll.shape == (len(responses),)
        assert np.all(ll <= 0)


class TestZeroInflated3PL:
    """Tests for ZeroInflated3PL model."""

    def test_init(self):
        """Test ZeroInflated3PL initialization."""
        model = ZeroInflated3PL(n_items=10)
        assert model.model_name == "ZI-3PL"

    def test_parameters(self):
        """Test that 3PL includes guessing."""
        model = ZeroInflated3PL(n_items=10)
        model._initialize_parameters()

        assert "guessing" in model._parameters
        c = model._parameters["guessing"]
        assert np.all((c >= 0) & (c <= 1))


class TestHurdleIRT:
    """Tests for HurdleIRT model."""

    def test_init(self):
        """Test HurdleIRT initialization."""
        model = HurdleIRT(n_items=10)
        assert model.n_items == 10
        assert "Hurdle" in model.model_name

    def test_probability(self):
        """Test probability computation."""
        model = HurdleIRT(n_items=5)
        model._initialize_parameters()

        theta = np.array([[0.0], [1.0], [-1.0]])

        # Full probability
        probs = model.probability(theta)
        assert probs.shape == (3, 5)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_copy(self):
        """Test model copying."""
        model = HurdleIRT(n_items=5)
        model._initialize_parameters()

        copied = model.copy()
        assert copied.n_items == model.n_items
        np.testing.assert_array_equal(
            copied._parameters["difficulty"],
            model._parameters["difficulty"],
        )
