"""Tests for explanatory IRT models (LLTM, latent regression)."""

import numpy as np
import pytest

from mirt.models.explanatory import (
    LLTM,
    ExplanatoryIRT,
    LatentRegressionModel,
    RaschLLTM,
)


class TestLLTM:
    """Tests for Linear Logistic Test Model."""

    def test_init_basic(self):
        """Test basic LLTM initialization."""
        n_items = 10
        n_features = 3
        item_features = np.random.randn(n_items, n_features)

        model = LLTM(n_items=n_items, item_features=item_features)

        assert model.n_items == n_items
        assert model.n_features == n_features
        assert model.item_features.shape == (n_items, n_features)
        assert model.feature_weights.shape == (n_features,)
        assert model.discrimination.shape == (n_items,)

    def test_init_with_names(self):
        """Test LLTM initialization with names."""
        n_items = 5
        n_features = 2
        item_features = np.random.randn(n_items, n_features)
        feature_names = ["Operations", "Content"]
        item_names = [f"Item{i}" for i in range(n_items)]

        model = LLTM(
            n_items=n_items,
            item_features=item_features,
            feature_names=feature_names,
            item_names=item_names,
        )

        assert model.feature_names == feature_names
        assert model.item_names == item_names

    def test_difficulty_from_features(self):
        """Test that difficulty is computed from features."""
        n_items = 5
        item_features = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [2, 0],
                [0, 2],
            ],
            dtype=np.float64,
        )
        feature_weights = np.array([0.5, -0.3])

        model = LLTM(n_items=n_items, item_features=item_features)
        model.set_feature_weights(feature_weights)

        expected_difficulty = item_features @ feature_weights
        np.testing.assert_array_almost_equal(model.difficulty, expected_difficulty)

    def test_probability(self):
        """Test probability computation."""
        n_items = 3
        item_features = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        model = LLTM(n_items=n_items, item_features=item_features)
        model.set_feature_weights(np.array([0.5, -0.3]))

        theta = np.array([[-1.0], [0.0], [1.0]])
        probs = model.probability(theta)

        assert probs.shape == (3, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_single_item(self):
        """Test probability for single item."""
        n_items = 3
        item_features = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        model = LLTM(n_items=n_items, item_features=item_features)

        theta = np.array([[0.0], [1.0]])
        prob_item0 = model.probability(theta, item_idx=0)

        assert prob_item0.shape == (2,)

    def test_information(self):
        """Test information computation."""
        n_items = 3
        item_features = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        model = LLTM(n_items=n_items, item_features=item_features)

        theta = np.array([[0.0]])
        info = model.information(theta)

        assert info.shape == (1, 3)
        assert np.all(info >= 0)

    def test_copy(self):
        """Test model copying."""
        n_items = 5
        item_features = np.random.randn(n_items, 2)
        model = LLTM(n_items=n_items, item_features=item_features)
        model.set_feature_weights(np.array([1.0, -1.0]))

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.feature_weights, model.feature_weights)
        np.testing.assert_array_equal(model_copy.item_features, model.item_features)


class TestRaschLLTM:
    """Tests for Rasch-constrained LLTM."""

    def test_discrimination_fixed(self):
        """Test that discrimination is fixed to 1."""
        n_items = 5
        item_features = np.random.randn(n_items, 2)
        model = RaschLLTM(n_items=n_items, item_features=item_features)

        np.testing.assert_array_equal(model.discrimination, np.ones(n_items))

    def test_cannot_set_discrimination(self):
        """Test that setting discrimination raises error."""
        n_items = 5
        item_features = np.random.randn(n_items, 2)
        model = RaschLLTM(n_items=n_items, item_features=item_features)

        with pytest.raises(ValueError, match="Cannot set discrimination"):
            model.set_parameters(discrimination=np.ones(n_items) * 2)


class TestLatentRegressionModel:
    """Tests for latent regression model."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = LatentRegressionModel(n_covariates=3)

        assert model.n_covariates == 3
        assert model.include_intercept
        assert model.regression_weights.shape == (4,)

    def test_init_no_intercept(self):
        """Test initialization without intercept."""
        model = LatentRegressionModel(n_covariates=3, include_intercept=False)

        assert not model.include_intercept
        assert model.regression_weights.shape == (3,)

    def test_predict_mean(self):
        """Test prediction of mean ability."""
        model = LatentRegressionModel(n_covariates=2)
        model.set_regression_weights(np.array([0.5, 1.0, -0.5]))

        covariates = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        predictions = model.predict_mean(covariates)

        expected = np.array([1.5, 0.0, 1.0])
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_set_residual_variance(self):
        """Test setting residual variance."""
        model = LatentRegressionModel(n_covariates=2)
        model.set_residual_variance(0.5)

        assert model.residual_variance == 0.5

    def test_log_prior_density(self):
        """Test log prior density computation."""
        model = LatentRegressionModel(n_covariates=1)
        model.set_regression_weights(np.array([0.0, 1.0]))
        model.set_residual_variance(1.0)

        theta = np.array([1.0])
        covariates = np.array([[1.0]])

        log_p = model.log_prior_density(theta, covariates)

        expected = -0.5 * np.log(2 * np.pi) - 0.0
        np.testing.assert_almost_equal(log_p[0], expected)


class TestExplanatoryIRT:
    """Tests for combined explanatory IRT model."""

    def test_init_basic(self):
        """Test basic initialization."""
        n_items = 10
        item_features = np.random.randn(n_items, 3)

        model = ExplanatoryIRT(
            n_items=n_items,
            item_features=item_features,
            n_person_covariates=2,
        )

        assert model.n_items == n_items
        assert model.n_item_features == 3
        assert model.n_person_covariates == 2

    def test_probability_given_covariates(self):
        """Test probability given covariates."""
        n_items = 5
        item_features = np.zeros((n_items, 2))
        model = ExplanatoryIRT(
            n_items=n_items,
            item_features=item_features,
            n_person_covariates=1,
        )

        model.set_regression_weights(np.array([0.0, 1.0]))

        covariates = np.array([[1.0], [0.0], [-1.0]])
        probs = model.probability_given_covariates(covariates)

        assert probs.shape == (3, n_items)
        assert probs[0, 0] > probs[1, 0] > probs[2, 0]

    def test_copy(self):
        """Test model copying."""
        n_items = 5
        item_features = np.random.randn(n_items, 2)
        model = ExplanatoryIRT(
            n_items=n_items,
            item_features=item_features,
            n_person_covariates=2,
        )
        model.set_regression_weights(np.array([0.5, 1.0, -0.5]))
        model.set_residual_variance(0.8)

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(
            model_copy.regression_weights, model.regression_weights
        )
        assert model_copy.residual_variance == model.residual_variance


class TestLLTMRecovery:
    """Integration tests for LLTM parameter recovery."""

    def test_difficulty_prediction(self):
        """Test that difficulties are correctly predicted from features."""
        rng = np.random.default_rng(42)

        true_weights = np.array([0.8, -0.5, 0.3])
        n_items = 20
        item_features = rng.standard_normal((n_items, 3))
        true_difficulties = item_features @ true_weights

        model = LLTM(n_items=n_items, item_features=item_features)
        model.set_feature_weights(true_weights)

        np.testing.assert_array_almost_equal(model.difficulty, true_difficulties)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
