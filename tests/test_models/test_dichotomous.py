"""Tests for dichotomous IRT models."""

import numpy as np
import pytest

from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


class TestTwoParameterLogistic:
    """Tests for 2PL model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TwoParameterLogistic(n_items=10)

        assert model.n_items == 10
        assert model.n_factors == 1
        assert model.model_name == "2PL"
        assert not model.is_fitted
        assert len(model.item_names) == 10

    def test_probability_shape(self):
        """Test probability output shape."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.linspace(-3, 3, 100)
        probs = model.probability(theta)

        assert probs.shape == (100, 5)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_single_item(self):
        """Test probability for single item."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.array([0.0])
        prob = model.probability(theta, item_idx=0)

        assert prob.shape == (1,)
        assert 0 <= prob[0] <= 1

    def test_icc_at_difficulty(self):
        """Test ICC equals 0.5 at difficulty parameter."""
        model = TwoParameterLogistic(n_items=1)
        model.set_parameters(difficulty=np.array([1.0]))

        theta = np.array([1.0])
        prob = model.icc(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.5, decimal=5)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = TwoParameterLogistic(n_items=3)

        responses = np.array([[1, 0, 1], [0, 1, 0]])
        theta = np.array([[0.0], [0.0]])

        ll = model.log_likelihood(responses, theta)

        assert ll.shape == (2,)
        assert np.all(ll <= 0)

    def test_information(self):
        """Test Fisher information computation."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50, 5)
        assert np.all(info >= 0)

    def test_multidimensional(self):
        """Test multidimensional 2PL."""
        model = TwoParameterLogistic(n_items=10, n_factors=2)

        assert model.n_factors == 2
        assert model.discrimination.shape == (10, 2)

        theta = np.random.randn(20, 2)
        probs = model.probability(theta)

        assert probs.shape == (20, 10)


class TestOneParameterLogistic:
    """Tests for 1PL/Rasch model."""

    def test_discrimination_fixed(self):
        """Test that discrimination is fixed to 1."""
        model = OneParameterLogistic(n_items=5)

        np.testing.assert_array_equal(model.discrimination, np.ones(5))

    def test_cannot_set_discrimination(self):
        """Test that discrimination cannot be modified."""
        model = OneParameterLogistic(n_items=5)

        with pytest.raises(ValueError, match="Cannot set discrimination"):
            model.set_parameters(discrimination=np.array([2.0] * 5))

    def test_no_multidimensional(self):
        """Test that 1PL doesn't support multidimensional."""
        with pytest.raises(ValueError, match="unidimensional"):
            OneParameterLogistic(n_items=5, n_factors=2)


class TestThreeParameterLogistic:
    """Tests for 3PL model."""

    def test_guessing_parameter(self):
        """Test guessing parameter effect."""
        model = ThreeParameterLogistic(n_items=1)
        model.set_parameters(
            guessing=np.array([0.25]),
            difficulty=np.array([10.0]),
        )

        theta = np.array([-10.0])
        prob = model.probability(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.25, decimal=2)

    def test_probability_bounds(self):
        """Test probabilities are bounded by guessing and 1."""
        model = ThreeParameterLogistic(n_items=5)

        theta = np.linspace(-5, 5, 100)
        probs = model.probability(theta)

        guessing = model.guessing

        for i in range(5):
            assert np.all(probs[:, i] >= guessing[i] - 1e-6)


class TestFourParameterLogistic:
    """Tests for 4PL model."""

    def test_upper_asymptote(self):
        """Test upper asymptote effect."""
        model = FourParameterLogistic(n_items=1)
        model.set_parameters(
            upper=np.array([0.9]),
            difficulty=np.array([-10.0]),
        )

        theta = np.array([10.0])
        prob = model.probability(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.9, decimal=2)

    def test_probability_bounds(self):
        """Test probabilities are bounded by guessing and upper."""
        model = FourParameterLogistic(n_items=3)
        model.set_parameters(
            guessing=np.array([0.1, 0.2, 0.15]),
            upper=np.array([0.95, 0.9, 0.85]),
        )

        theta = np.linspace(-5, 5, 100)
        probs = model.probability(theta)

        for i in range(3):
            assert np.all(probs[:, i] >= model.guessing[i] - 1e-6)
            assert np.all(probs[:, i] <= model.upper[i] + 1e-6)
