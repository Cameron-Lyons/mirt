"""Tests for Sparse Bayesian MIRT estimation."""

import numpy as np
import pytest

from mirt.estimation.sparse_bayesian import (
    SparseBayesianEstimator,
    SparseBayesianResult,
    SpikeSlabLassoPrior,
)
from mirt.models.dichotomous import TwoParameterLogistic


class TestSpikeSlabLassoPrior:
    """Tests for spike-slab LASSO prior."""

    def test_initialization(self):
        """Test prior initialization."""
        prior = SpikeSlabLassoPrior(lambda_0=0.04, lambda_1=1.0, theta=0.5)

        assert prior.lambda_0 == 0.04
        assert prior.lambda_1 == 1.0
        assert prior.theta == 0.5
        assert prior.mean == 0.0

    def test_invalid_lambda_order(self):
        """Test that lambda_0 >= lambda_1 raises error."""
        with pytest.raises(ValueError, match="lambda_0 must be smaller"):
            SpikeSlabLassoPrior(lambda_0=1.0, lambda_1=0.5)

    def test_invalid_theta(self):
        """Test that invalid theta raises error."""
        with pytest.raises(ValueError, match="theta must be between"):
            SpikeSlabLassoPrior(theta=1.5)

    def test_log_pdf_shape(self):
        """Test log_pdf returns correct shape."""
        prior = SpikeSlabLassoPrior()
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])

        log_pdf = prior.log_pdf(x)

        assert log_pdf.shape == x.shape
        assert np.all(np.isfinite(log_pdf))

    def test_log_pdf_symmetry(self):
        """Test log_pdf is symmetric around zero."""
        prior = SpikeSlabLassoPrior()
        x = np.array([0.5, 1.0, 2.0])

        log_pdf_pos = prior.log_pdf(x)
        log_pdf_neg = prior.log_pdf(-x)

        np.testing.assert_allclose(log_pdf_pos, log_pdf_neg)

    def test_soft_threshold_zero(self):
        """Test soft-thresholding sets small values to zero."""
        x = np.array([0.1, 0.5, 1.0, -0.1, -0.5])
        penalty = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

        result = SpikeSlabLassoPrior.soft_threshold(x, penalty)

        assert result[0] == 0.0
        assert result[3] == 0.0
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.7)

    def test_soft_threshold_preserves_sign(self):
        """Test soft-thresholding preserves sign."""
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        penalty = np.array([0.5, 0.5, 0.5, 0.5])

        result = SpikeSlabLassoPrior.soft_threshold(x, penalty)

        assert result[0] < 0
        assert result[1] < 0
        assert result[2] > 0
        assert result[3] > 0

    def test_posterior_inclusion_bounds(self):
        """Test posterior inclusion probability is in [0, 1]."""
        prior = SpikeSlabLassoPrior()
        x = np.array([-5.0, -1.0, 0.0, 0.001, 1.0, 5.0])

        gamma = prior.compute_posterior_inclusion(x)

        assert np.all(gamma >= 0)
        assert np.all(gamma <= 1)

    def test_posterior_inclusion_increases_with_magnitude(self):
        """Test that larger magnitudes have higher inclusion probability."""
        prior = SpikeSlabLassoPrior()
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

        gamma = prior.compute_posterior_inclusion(x)

        for i in range(len(gamma) - 1):
            assert gamma[i + 1] >= gamma[i] - 1e-6

    def test_sample_shape(self):
        """Test sample returns correct shape."""
        prior = SpikeSlabLassoPrior()
        rng = np.random.default_rng(42)

        samples_1d = prior.sample(100, rng)
        samples_2d = prior.sample((10, 5), rng)

        assert samples_1d.shape == (100,)
        assert samples_2d.shape == (10, 5)

    def test_adaptive_theta_update(self):
        """Test adaptive theta update."""
        prior = SpikeSlabLassoPrior(theta=0.5, adaptive=True)
        gamma = np.array([[0.3, 0.1], [0.4, 0.2], [0.35, 0.05]])

        prior.update_theta(gamma)

        assert prior.theta < 0.5


class TestSparseBayesianEstimator:
    """Tests for Sparse Bayesian MIRT estimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = SparseBayesianEstimator(k_max=3, lambda_0=0.05, lambda_1=2.0)

        assert estimator.k_max == 3
        assert estimator._ssl_prior.lambda_0 == 0.05
        assert estimator._ssl_prior.lambda_1 == 2.0

    def test_invalid_k_max(self):
        """Test that k_max < 1 raises error."""
        with pytest.raises(ValueError, match="k_max must be at least 1"):
            SparseBayesianEstimator(k_max=0)

    def test_fit_basic(self, rng):
        """Test basic model fitting."""
        n_persons, n_items, k_max = 100, 10, 3

        theta = rng.standard_normal((n_persons, 2))
        a = np.zeros((n_items, 2))
        a[:5, 0] = 1.2
        a[5:, 1] = 1.0
        b = rng.uniform(-1, 1, n_items)

        z = theta @ a.T - np.sum(a, axis=1) * b
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=k_max)
        estimator = SparseBayesianEstimator(k_max=k_max, max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert isinstance(result, SparseBayesianResult)
        assert result.model is model
        assert result.elbo < 0
        assert result.n_iterations > 0
        assert result.loadings.shape == (n_items, k_max)
        assert result.sparse_loadings.shape == (n_items, k_max)

    def test_elbo_increases(self, rng):
        """Test that ELBO generally increases."""
        n_persons, n_items = 80, 8

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.5, n_items)
        b = rng.uniform(-1.5, 1.5, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=2)
        estimator = SparseBayesianEstimator(k_max=2, max_iter=30, tol=1e-6)

        result = estimator.fit(model, responses)

        elbo_history = result.elbo_history
        assert len(elbo_history) > 1

        increases = sum(
            elbo_history[i] >= elbo_history[i - 1] - 0.5
            for i in range(1, len(elbo_history))
        )
        assert increases > len(elbo_history) * 0.7

    def test_sparsity_recovery(self, rng):
        """Test recovery of sparse structure."""
        n_persons, n_items = 300, 12
        k_true = 2
        k_max = 4

        theta = rng.standard_normal((n_persons, k_true))

        a_true = np.zeros((n_items, k_true))
        a_true[:6, 0] = rng.uniform(1.0, 1.5, 6)
        a_true[6:, 1] = rng.uniform(1.0, 1.5, 6)
        b = rng.uniform(-1, 1, n_items)

        z = theta @ a_true.T - np.sum(a_true, axis=1) * b
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=k_max)
        estimator = SparseBayesianEstimator(
            k_max=k_max,
            lambda_0=0.02,
            lambda_1=1.5,
            max_iter=100,
            sparsity_threshold=0.5,
        )

        result = estimator.fit(model, responses)

        assert result.effective_dimensionality <= k_true + 1
        assert result.effective_dimensionality >= 1

        sparsity_ratio = 1 - np.mean(result.sparsity_pattern)
        assert sparsity_ratio > 0.3

    def test_dimensionality_selection(self, rng):
        """Test automatic dimensionality selection."""
        n_persons, n_items = 200, 10
        k_true = 1
        k_max = 5

        theta = rng.standard_normal((n_persons, k_true))
        a = np.ones((n_items, k_true)) * 1.2
        b = np.linspace(-1.5, 1.5, n_items)

        z = theta @ a.T - a.sum(axis=1) * b
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=k_max)
        estimator = SparseBayesianEstimator(k_max=k_max, max_iter=100)

        result = estimator.fit(model, responses)

        assert result.effective_dimensionality <= k_max
        assert result.effective_dimensionality >= 1

    def test_inclusion_probabilities_shape(self, rng):
        """Test inclusion probabilities have correct shape."""
        n_persons, n_items, k_max = 50, 6, 2

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.2, n_items)
        b = rng.uniform(-1, 1, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=k_max)
        estimator = SparseBayesianEstimator(k_max=k_max, max_iter=30)

        result = estimator.fit(model, responses)

        assert result.inclusion_probabilities.shape == (n_items, k_max)
        assert np.all(result.inclusion_probabilities >= 0)
        assert np.all(result.inclusion_probabilities <= 1)

    def test_result_summary(self, rng):
        """Test result summary generation."""
        n_persons, n_items = 50, 6

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.2, n_items)
        b = rng.uniform(-1, 1, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=2)
        estimator = SparseBayesianEstimator(k_max=2, max_iter=20)

        result = estimator.fit(model, responses)
        summary = result.summary()

        assert "Sparse Bayesian MIRT" in summary
        assert "Effective dimensionality" in summary
        assert "Sparsity ratio" in summary

    def test_missing_data(self, rng):
        """Test handling of missing data."""
        n_persons, n_items = 80, 8

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.2, n_items)
        b = rng.uniform(-1, 1, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        missing_mask = rng.random((n_persons, n_items)) < 0.1
        responses[missing_mask] = -1

        model = TwoParameterLogistic(n_items=n_items, n_factors=2)
        estimator = SparseBayesianEstimator(k_max=2, max_iter=30)

        result = estimator.fit(model, responses)

        assert result.n_iterations > 0
        assert np.all(np.isfinite(result.sparse_loadings))

    def test_unsupported_model(self, dichotomous_responses):
        """Test that unsupported models raise error."""
        from mirt.models.dichotomous import ThreeParameterLogistic

        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = ThreeParameterLogistic(n_items=n_items)
        estimator = SparseBayesianEstimator()

        with pytest.raises(ValueError, match="supports 2PL"):
            estimator.fit(model, responses)

    def test_loading_table(self, rng):
        """Test loading table output."""
        n_persons, n_items = 50, 6

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.2, n_items)
        b = rng.uniform(-1, 1, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=2)
        estimator = SparseBayesianEstimator(k_max=2, max_iter=20)

        result = estimator.fit(model, responses)
        table = result.loading_table(threshold=0.1)

        assert table.shape == result.sparse_loadings.shape
        assert np.sum(np.abs(table) < 0.1) >= np.sum(
            np.abs(result.sparse_loadings) < 0.1
        )

    def test_ebic_computed(self, rng):
        """Test that EBIC is computed."""
        n_persons, n_items = 50, 6

        theta = rng.standard_normal(n_persons)
        a = rng.uniform(0.8, 1.2, n_items)
        b = rng.uniform(-1, 1, n_items)
        probs = 1 / (1 + np.exp(-a * (theta[:, None] - b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=2)
        estimator = SparseBayesianEstimator(k_max=2, max_iter=20)

        result = estimator.fit(model, responses)

        assert np.isfinite(result.bic)
        assert np.isfinite(result.ebic)
        assert result.ebic >= result.bic
