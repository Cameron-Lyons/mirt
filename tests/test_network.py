"""Tests for Network Psychometrics models."""

import numpy as np
import pytest

from mirt.models.network import (
    GaussianGraphicalModel,
    IsingModel,
    compare_networks,
    fit_ggm,
    fit_ising,
)


class TestIsingModel:
    """Tests for IsingModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = IsingModel(n_nodes=3)

        assert model.n_nodes == 3
        np.testing.assert_array_equal(model.thresholds, np.zeros(3))
        np.testing.assert_array_equal(model.interactions, np.zeros((3, 3)))

    def test_init_with_node_names(self):
        """Test initialization with node names."""
        model = IsingModel(n_nodes=3, node_names=["A", "B", "C"])
        assert model.node_names == ["A", "B", "C"]

    def test_init_too_few_nodes(self):
        """Test error with too few nodes."""
        with pytest.raises(ValueError, match="at least 2"):
            IsingModel(n_nodes=1)

    def test_set_thresholds(self):
        """Test setting thresholds."""
        model = IsingModel(n_nodes=3)
        thresholds = np.array([0.0, 0.5, -0.5])
        model.set_thresholds(thresholds)

        np.testing.assert_array_equal(model.thresholds, thresholds)

    def test_set_interactions(self):
        """Test setting interactions."""
        model = IsingModel(n_nodes=3)
        interactions = np.array(
            [
                [0.0, 0.3, 0.2],
                [0.3, 0.0, 0.1],
                [0.2, 0.1, 0.0],
            ]
        )
        model.set_interactions(interactions)

        np.testing.assert_array_almost_equal(model.interactions, interactions)

    def test_set_interactions_symmetrizes(self):
        """Test that setting interactions symmetrizes the matrix."""
        model = IsingModel(n_nodes=2)
        interactions = np.array(
            [
                [0.0, 0.4],
                [0.2, 0.0],
            ]
        )
        model.set_interactions(interactions)

        result = model.interactions
        assert result[0, 1] == result[1, 0]

    def test_conditional_probability(self):
        """Test conditional probability computation."""
        model = IsingModel(n_nodes=2)
        model.set_thresholds(np.array([0.0, 0.0]))
        model.set_interactions(np.array([[0.0, 1.0], [1.0, 0.0]]))

        prob_given_0 = model.conditional_probability(0, np.array([[0, 0]]))[0]
        prob_given_1 = model.conditional_probability(0, np.array([[0, 1]]))[0]

        assert prob_given_0 < prob_given_1
        assert 0 < prob_given_0 < 1
        assert 0 < prob_given_1 < 1

    def test_pseudo_likelihood(self):
        """Test pseudo-likelihood computation."""
        model = IsingModel(n_nodes=3)
        model.set_thresholds(np.array([0.0, 0.0, 0.0]))
        model.set_interactions(
            np.array(
                [
                    [0.0, 0.5, 0.3],
                    [0.5, 0.0, 0.2],
                    [0.3, 0.2, 0.0],
                ]
            )
        )

        responses = np.array(
            [
                [1, 1, 1],
                [0, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )

        ll = model.pseudo_likelihood(responses)
        assert ll < 0
        assert np.isfinite(ll)

    def test_sample_shape(self):
        """Test sample generation shape."""
        model = IsingModel(n_nodes=4)

        samples = model.sample(n_samples=100, n_burnin=50, seed=42)

        assert samples.shape == (100, 4)
        assert np.all((samples == 0) | (samples == 1))

    def test_sample_reproducibility(self):
        """Test sample reproducibility with seed."""
        model = IsingModel(n_nodes=2)
        model.set_interactions(np.array([[0.0, 0.5], [0.5, 0.0]]))

        samples1 = model.sample(n_samples=50, seed=42)
        samples2 = model.sample(n_samples=50, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_statistics(self):
        """Test sample statistics are reasonable."""
        model = IsingModel(n_nodes=2)
        model.set_thresholds(np.array([1.0, -1.0]))

        samples = model.sample(n_samples=1000, n_burnin=100, seed=42)

        mean0 = samples[:, 0].mean()
        mean1 = samples[:, 1].mean()

        assert mean0 > 0.5
        assert mean1 < 0.5

    def test_edge_weights(self):
        """Test edge weight computation."""
        model = IsingModel(n_nodes=3)
        model.set_interactions(
            np.array(
                [
                    [0.0, 0.3, 0.2],
                    [0.3, 0.0, 0.1],
                    [0.2, 0.1, 0.0],
                ]
            )
        )

        weights = model.edge_weights()
        assert weights[0, 1] == 0.3
        assert weights[0, 2] == 0.2
        assert weights[1, 0] == 0.0

    def test_degree_centrality(self):
        """Test degree centrality computation."""
        model = IsingModel(n_nodes=3)
        model.set_interactions(
            np.array(
                [
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ]
            )
        )

        centrality = model.degree_centrality()
        assert centrality[0] > centrality[1]

    def test_expected_influence(self):
        """Test expected influence computation."""
        model = IsingModel(n_nodes=3)
        model.set_interactions(
            np.array(
                [
                    [0.0, 0.3, 0.2],
                    [0.3, 0.0, 0.1],
                    [0.2, 0.1, 0.0],
                ]
            )
        )

        influence = model.expected_influence()
        assert influence.shape == (3,)

    def test_copy(self):
        """Test model copying."""
        model = IsingModel(n_nodes=2)
        model.set_thresholds(np.array([0.0, 0.5]))
        model.set_interactions(np.array([[0.0, 0.3], [0.3, 0.0]]))

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.thresholds, model.thresholds)
        np.testing.assert_array_equal(model_copy.interactions, model.interactions)


class TestGaussianGraphicalModel:
    """Tests for GaussianGraphicalModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = GaussianGraphicalModel(n_nodes=3)

        assert model.n_nodes == 3
        np.testing.assert_array_equal(model.means, np.zeros(3))
        np.testing.assert_array_equal(model.precision_matrix, np.eye(3))

    def test_init_with_node_names(self):
        """Test initialization with node names."""
        model = GaussianGraphicalModel(n_nodes=3, node_names=["X", "Y", "Z"])
        assert model.node_names == ["X", "Y", "Z"]

    def test_init_too_few_nodes(self):
        """Test error with too few nodes."""
        with pytest.raises(ValueError, match="at least 2"):
            GaussianGraphicalModel(n_nodes=1)

    def test_set_means(self):
        """Test setting means."""
        model = GaussianGraphicalModel(n_nodes=3)
        means = np.array([1.0, 2.0, 3.0])
        model.set_means(means)

        np.testing.assert_array_equal(model.means, means)

    def test_set_precision_matrix(self):
        """Test setting precision matrix."""
        model = GaussianGraphicalModel(n_nodes=2)
        precision = np.array(
            [
                [2.0, -0.5],
                [-0.5, 2.0],
            ]
        )
        model.set_precision_matrix(precision)

        np.testing.assert_array_almost_equal(model.precision_matrix, precision)

    def test_set_precision_not_positive_definite(self):
        """Test error on non-positive-definite precision."""
        model = GaussianGraphicalModel(n_nodes=2)
        precision = np.array(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
        with pytest.raises(ValueError, match="positive definite"):
            model.set_precision_matrix(precision)

    def test_covariance_matrix(self):
        """Test covariance matrix computation."""
        model = GaussianGraphicalModel(n_nodes=2)
        precision = np.array(
            [
                [2.0, -0.5],
                [-0.5, 2.0],
            ]
        )
        model.set_precision_matrix(precision)

        cov = model.covariance_matrix
        reconstructed = np.linalg.inv(cov)

        np.testing.assert_array_almost_equal(reconstructed, precision)

    def test_partial_correlations(self):
        """Test partial correlation computation."""
        model = GaussianGraphicalModel(n_nodes=3)
        precision = np.array(
            [
                [1.0, 0.3, 0.0],
                [0.3, 1.0, 0.2],
                [0.0, 0.2, 1.0],
            ]
        )
        model.set_precision_matrix(precision)

        partial_corr = model.partial_correlations()

        assert partial_corr.shape == (3, 3)
        np.testing.assert_almost_equal(partial_corr[0, 2], 0.0)

    def test_partial_correlations_range(self):
        """Test partial correlations are in valid range."""
        model = GaussianGraphicalModel(n_nodes=4)
        precision = np.eye(4) * 2.0
        precision[0, 1] = precision[1, 0] = 0.5
        precision[1, 2] = precision[2, 1] = 0.3
        model.set_precision_matrix(precision)

        partial_corr = model.partial_correlations()

        assert np.all(partial_corr >= -1)
        assert np.all(partial_corr <= 1)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = GaussianGraphicalModel(n_nodes=2)

        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [-1.0, -1.0],
            ]
        )

        ll = model.log_likelihood(data)
        assert ll < 0
        assert np.isfinite(ll)

    def test_sample_shape(self):
        """Test sample generation shape."""
        model = GaussianGraphicalModel(n_nodes=3)
        model.set_means(np.array([1.0, 2.0, 3.0]))

        samples = model.sample(n_samples=100, seed=42)

        assert samples.shape == (100, 3)

    def test_sample_statistics(self):
        """Test sample statistics are reasonable."""
        model = GaussianGraphicalModel(n_nodes=2)
        model.set_means(np.array([5.0, -5.0]))

        samples = model.sample(n_samples=1000, seed=42)

        sample_means = samples.mean(axis=0)
        assert sample_means[0] > 3.0
        assert sample_means[1] < -3.0

    def test_edge_weights(self):
        """Test edge weight computation."""
        model = GaussianGraphicalModel(n_nodes=3)
        precision = np.array(
            [
                [1.0, 0.3, 0.0],
                [0.3, 1.0, 0.2],
                [0.0, 0.2, 1.0],
            ]
        )
        model.set_precision_matrix(precision)

        weights = model.edge_weights()
        assert weights.shape == (3, 3)

    def test_degree_centrality(self):
        """Test degree centrality computation."""
        model = GaussianGraphicalModel(n_nodes=3)
        precision = np.array(
            [
                [1.0, 0.3, 0.3],
                [0.3, 1.0, 0.0],
                [0.3, 0.0, 1.0],
            ]
        )
        model.set_precision_matrix(precision)

        centrality = model.degree_centrality()
        assert centrality.shape == (3,)

    def test_copy(self):
        """Test model copying."""
        model = GaussianGraphicalModel(n_nodes=2)
        model.set_means(np.array([0.0, 1.0]))

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.means, model.means)


class TestFitIsing:
    """Tests for fit_ising function."""

    def test_fit_basic(self):
        """Test basic Ising model fitting."""
        np.random.seed(42)
        n_samples = 200
        n_nodes = 4
        responses = np.random.binomial(1, 0.5, (n_samples, n_nodes))

        model, psl = fit_ising(responses)

        assert isinstance(model, IsingModel)
        assert model.n_nodes == n_nodes
        assert psl < 0

    def test_fit_with_regularization(self):
        """Test fitting with regularization."""
        np.random.seed(42)
        responses = np.random.binomial(1, 0.5, (100, 3))

        model, _ = fit_ising(responses, regularization=0.5)

        assert isinstance(model, IsingModel)
        interaction_sum = np.abs(model.interactions).sum()
        assert interaction_sum < 10

    def test_fit_convergence(self):
        """Test that fitting converges."""
        np.random.seed(42)
        responses = np.random.binomial(1, 0.5, (150, 3))

        model, _ = fit_ising(responses, max_iter=100, tol=1e-4)

        assert model is not None


class TestFitGGM:
    """Tests for fit_ggm function."""

    def test_fit_basic(self):
        """Test basic GGM fitting."""
        np.random.seed(42)
        data = np.random.randn(100, 4)

        model, ll = fit_ggm(data)

        assert isinstance(model, GaussianGraphicalModel)
        assert model.n_nodes == 4
        assert ll < 0

    def test_fit_with_regularization(self):
        """Test fitting with regularization."""
        np.random.seed(42)
        data = np.random.randn(100, 3)

        model, _ = fit_ggm(data, regularization=0.5)

        assert isinstance(model, GaussianGraphicalModel)

    def test_fit_sparse(self):
        """Test sparse estimation."""
        np.random.seed(42)
        data = np.random.randn(200, 5)

        model, _ = fit_ggm(data, regularization=0.3)

        partial_corr = model.partial_correlations()
        np.fill_diagonal(partial_corr, 0)
        n_nonzero = np.sum(np.abs(partial_corr) > 0.01)
        assert n_nonzero <= 20


class TestCompareNetworks:
    """Tests for compare_networks function."""

    def test_compare_ising(self):
        """Test comparing Ising models."""
        model1 = IsingModel(n_nodes=2)
        model1.set_thresholds(np.array([0.0, 0.0]))
        model1.set_interactions(np.array([[0.0, 0.5], [0.5, 0.0]]))

        model2 = IsingModel(n_nodes=2)
        model2.set_thresholds(np.array([0.1, 0.1]))
        model2.set_interactions(np.array([[0.0, 0.3], [0.3, 0.0]]))

        result = compare_networks(model1, model2)

        assert "edge_correlation" in result
        assert "degree_correlation" in result
        assert "mean_edge_difference" in result
        assert "max_edge_difference" in result

    def test_compare_ggm(self):
        """Test comparing GGM models."""
        model1 = GaussianGraphicalModel(n_nodes=3)
        precision1 = np.eye(3)
        precision1[0, 1] = precision1[1, 0] = 0.3
        model1.set_precision_matrix(precision1)

        model2 = GaussianGraphicalModel(n_nodes=3)
        precision2 = np.eye(3)
        precision2[1, 2] = precision2[2, 1] = 0.2
        model2.set_precision_matrix(precision2)

        result = compare_networks(model1, model2)

        assert "edge_correlation" in result
        assert "degree_correlation" in result

    def test_compare_different_types_error(self):
        """Test error when comparing different model types."""
        ising = IsingModel(n_nodes=2)
        ggm = GaussianGraphicalModel(n_nodes=2)

        with pytest.raises(ValueError, match="same type"):
            compare_networks(ising, ggm)

    def test_compare_different_sizes_error(self):
        """Test error when comparing different sized models."""
        model1 = IsingModel(n_nodes=2)
        model2 = IsingModel(n_nodes=3)

        with pytest.raises(ValueError, match="same number of nodes"):
            compare_networks(model1, model2)


class TestNetworkIntegration:
    """Integration tests for network models."""

    def test_ising_round_trip(self):
        """Test generating and fitting Ising model."""
        np.random.seed(42)
        true_model = IsingModel(n_nodes=4)
        true_model.set_thresholds(np.array([0.5, -0.3, 0.0, 0.2]))
        true_model.set_interactions(
            np.array(
                [
                    [0.0, 0.4, 0.0, 0.0],
                    [0.4, 0.0, 0.3, 0.0],
                    [0.0, 0.3, 0.0, 0.2],
                    [0.0, 0.0, 0.2, 0.0],
                ]
            )
        )

        samples = true_model.sample(n_samples=500, n_burnin=200, seed=42)

        fitted_model, _ = fit_ising(samples, regularization=0.1)

        threshold_corr = np.corrcoef(true_model.thresholds, fitted_model.thresholds)[
            0, 1
        ]
        assert threshold_corr > 0.5

    def test_ggm_round_trip(self):
        """Test generating and fitting GGM."""
        np.random.seed(42)
        true_model = GaussianGraphicalModel(n_nodes=3)
        true_model.set_means(np.array([0.0, 1.0, 2.0]))
        true_model.set_precision_matrix(
            np.array(
                [
                    [1.5, 0.3, 0.0],
                    [0.3, 2.0, 0.4],
                    [0.0, 0.4, 1.0],
                ]
            )
        )

        samples = true_model.sample(n_samples=500, seed=42)

        fitted_model, _ = fit_ggm(samples, regularization=0.1)

        mean_error = np.abs(fitted_model.means - true_model.means).max()
        assert mean_error < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
