"""Tests for polytomous IRT models."""

import numpy as np
import pytest

from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


class TestGradedResponseModel:
    """Tests for GRM."""

    def test_initialization(self):
        """Test model initialization."""
        model = GradedResponseModel(n_items=10, n_categories=5)

        assert model.n_items == 10
        assert model.n_factors == 1
        assert model.model_name == "GRM"
        assert model.n_categories == [5] * 10
        assert model.max_categories == 5
        assert not model.is_fitted

    def test_varying_categories(self):
        """Test model with varying categories per item."""
        n_cats = [3, 4, 5, 3, 4]
        model = GradedResponseModel(n_items=5, n_categories=n_cats)

        assert model.n_categories == n_cats
        assert model.max_categories == 5

    def test_probability_shape(self):
        """Test probability output shape."""
        model = GradedResponseModel(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 100)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (100, 4)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = GradedResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_cumulative_probability(self):
        """Test cumulative probability computation."""
        model = GradedResponseModel(n_items=3, n_categories=4)

        theta = np.array([0.0])

        # Cumulative probabilities should be monotonically decreasing
        for item_idx in range(model.n_items):
            cum_probs = [
                model.cumulative_probability(theta, item_idx, k)[0]
                for k in range(model.n_categories[item_idx] - 1)
            ]

            for i in range(len(cum_probs) - 1):
                assert cum_probs[i] >= cum_probs[i + 1]

    def test_information(self):
        """Test Fisher information computation."""
        model = GradedResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)

    def test_expected_score(self):
        """Test expected score computation."""
        model = GradedResponseModel(n_items=3, n_categories=5)

        theta = np.linspace(-3, 3, 20)
        expected = model.expected_score(theta)

        assert expected.shape == (20,)
        # Expected score should be between 0 and n_items * (max_cat - 1)
        assert np.all(expected >= 0)
        assert np.all(expected <= model.n_items * (model.max_categories - 1))

    def test_multidimensional(self):
        """Test multidimensional GRM."""
        model = GradedResponseModel(n_items=6, n_categories=4, n_factors=2)

        assert model.n_factors == 2
        assert model.discrimination.shape == (6, 2)

        theta = np.random.randn(20, 2)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (20, 4)


class TestGeneralizedPartialCredit:
    """Tests for GPCM."""

    def test_initialization(self):
        """Test model initialization."""
        model = GeneralizedPartialCredit(n_items=8, n_categories=5)

        assert model.n_items == 8
        assert model.model_name == "GPCM"
        assert model.max_categories == 5

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_steps_property(self):
        """Test step parameters exist."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=4)

        steps = model.steps
        assert steps.shape == (5, 3)  # n_items x (n_categories - 1)

    def test_information(self):
        """Test Fisher information computation."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)


class TestPartialCreditModel:
    """Tests for PCM (Rasch-like GPCM)."""

    def test_discrimination_fixed(self):
        """Test that discrimination is fixed to 1."""
        model = PartialCreditModel(n_items=5, n_categories=4)

        np.testing.assert_array_equal(model.discrimination, np.ones(5))

    def test_cannot_set_discrimination(self):
        """Test that discrimination cannot be modified."""
        model = PartialCreditModel(n_items=5, n_categories=4)

        with pytest.raises(ValueError, match="Cannot set discrimination"):
            model.set_parameters(discrimination=np.array([2.0] * 5))

    def test_no_multidimensional(self):
        """Test that PCM doesn't support multidimensional."""
        with pytest.raises(ValueError, match="unidimensional"):
            PartialCreditModel(n_items=5, n_categories=4, n_factors=2)


class TestNominalResponseModel:
    """Tests for NRM."""

    def test_initialization(self):
        """Test model initialization."""
        model = NominalResponseModel(n_items=6, n_categories=4)

        assert model.n_items == 6
        assert model.model_name == "NRM"

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = NominalResponseModel(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_slopes_and_intercepts(self):
        """Test slopes and intercepts properties."""
        model = NominalResponseModel(n_items=5, n_categories=4)

        slopes = model.slopes
        intercepts = model.intercepts

        assert slopes.shape[0] == 5
        assert intercepts.shape == (5, 4)

    def test_information(self):
        """Test Fisher information computation."""
        model = NominalResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)

    def test_multidimensional(self):
        """Test multidimensional NRM."""
        model = NominalResponseModel(n_items=6, n_categories=4, n_factors=2)

        assert model.n_factors == 2

        theta = np.random.randn(20, 2)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (20, 4)


class TestPolytomousLogLikelihood:
    """Tests for polytomous log-likelihood computation."""

    def test_grm_log_likelihood(self, polytomous_responses):
        """Test GRM log-likelihood computation."""
        responses = polytomous_responses["responses"]
        theta = polytomous_responses["theta"]
        n_items = polytomous_responses["n_items"]
        n_categories = polytomous_responses["n_categories"]

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)

        theta_2d = theta.reshape(-1, 1)
        ll = model.log_likelihood(responses, theta_2d)

        assert ll.shape == (len(theta),)
        assert np.all(ll <= 0)  # Log-likelihood is always negative

    def test_gpcm_log_likelihood(self, polytomous_responses):
        """Test GPCM log-likelihood computation."""
        responses = polytomous_responses["responses"]
        theta = polytomous_responses["theta"]
        n_items = polytomous_responses["n_items"]
        n_categories = polytomous_responses["n_categories"]

        model = GeneralizedPartialCredit(n_items=n_items, n_categories=n_categories)

        theta_2d = theta.reshape(-1, 1)
        ll = model.log_likelihood(responses, theta_2d)

        assert ll.shape == (len(theta),)
        assert np.all(ll <= 0)


class TestCategoryProbabilityBounds:
    """Test that category probabilities handle edge cases."""

    @pytest.mark.parametrize(
        "ModelClass",
        [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel],
    )
    def test_extreme_theta(self, ModelClass):
        """Test probabilities at extreme theta values."""
        model = ModelClass(n_items=3, n_categories=4)

        theta_extreme = np.array([-10.0, 0.0, 10.0])

        for item_idx in range(model.n_items):
            probs = model.probability(theta_extreme, item_idx=item_idx)

            # Probabilities should be valid
            assert np.all(probs >= 0)
            assert np.all(probs <= 1)

            # Should sum to 1
            np.testing.assert_array_almost_equal(
                probs.sum(axis=1), np.ones(3), decimal=5
            )

    @pytest.mark.parametrize(
        "ModelClass",
        [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel],
    )
    def test_invalid_category(self, ModelClass):
        """Test that invalid category raises error."""
        model = ModelClass(n_items=3, n_categories=4)

        theta = np.array([0.0])

        with pytest.raises(ValueError, match="out of range"):
            model.category_probability(theta, item_idx=0, category=5)

        with pytest.raises(ValueError, match="out of range"):
            model.category_probability(theta, item_idx=0, category=-1)
