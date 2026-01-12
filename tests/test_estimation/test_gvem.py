"""Tests for GVEM estimation algorithm."""

import numpy as np
import pytest

from mirt.estimation.gvem import GVEMEstimator
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic


class TestGVEMEstimator:
    """Tests for Gaussian Variational EM algorithm."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = GVEMEstimator(max_iter=100, tol=1e-3, n_inner_iter=5)

        assert estimator.max_iter == 100
        assert estimator.tol == 1e-3
        assert estimator.n_inner_iter == 5

    def test_invalid_n_inner_iter(self):
        """Test that invalid n_inner_iter raises error."""
        with pytest.raises(ValueError, match="n_inner_iter must be at least 1"):
            GVEMEstimator(n_inner_iter=0)

    def test_fit_basic(self, dichotomous_responses):
        """Test basic model fitting."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert result.log_likelihood < 0
        assert result.n_iterations > 0
        assert result.aic > 0
        assert result.bic > 0

    def test_elbo_increases(self, dichotomous_responses):
        """Test that ELBO increases monotonically (or stays nearly constant)."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=30, tol=1e-6)

        estimator.fit(model, responses)

        elbo_history = estimator.elbo_history
        assert len(elbo_history) > 1

        for i in range(1, len(elbo_history)):
            assert elbo_history[i] >= elbo_history[i - 1] - 0.1

    def test_convergence_history(self, dichotomous_responses):
        """Test convergence history is tracked."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=20)

        estimator.fit(model, responses)

        history = estimator.convergence_history
        assert len(history) > 0

    def test_parameter_recovery(self, rng):
        """Test recovery of known parameters."""
        n_persons, n_items = 500, 10

        true_a = np.ones(n_items) * 1.5
        true_b = np.linspace(-2, 2, n_items)

        theta = rng.standard_normal(n_persons)
        probs = 1 / (1 + np.exp(-true_a * (theta[:, None] - true_b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=200, tol=1e-4)
        result = estimator.fit(model, responses)

        est_b = result.model.difficulty
        correlation = np.corrcoef(true_b, est_b)[0, 1]
        assert correlation > 0.8

    def test_standard_errors(self, dichotomous_responses):
        """Test standard errors are computed."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=50)

        result = estimator.fit(model, responses)

        assert "discrimination" in result.standard_errors
        assert "difficulty" in result.standard_errors

        se_disc = result.standard_errors["discrimination"]
        se_diff = result.standard_errors["difficulty"]
        assert np.all((se_disc > 0) | np.isnan(se_disc))
        assert np.all((se_diff > 0) | np.isnan(se_diff))

    def test_missing_data(self, dichotomous_responses, rng):
        """Test handling of missing data."""
        responses = dichotomous_responses["responses"].copy()
        n_items = dichotomous_responses["n_items"]

        mask = rng.random(responses.shape) < 0.1
        responses[mask] = -1

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=50)

        result = estimator.fit(model, responses)
        assert result.converged or result.n_iterations == 50

    def test_variational_parameters_available(self, dichotomous_responses):
        """Test that variational parameters are accessible after fitting."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]
        n_persons = dichotomous_responses["n_persons"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=20)

        estimator.fit(model, responses)

        mu = estimator.variational_means
        sigma = estimator.variational_covariances

        assert mu is not None
        assert sigma is not None
        assert mu.shape == (n_persons, 1)
        assert sigma.shape == (n_persons, 1, 1)

    def test_1pl_model(self, dichotomous_responses):
        """Test GVEM works with 1PL model."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = OneParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator(max_iter=50)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert np.allclose(model.discrimination, 1.0)

    def test_unsupported_model(self, dichotomous_responses):
        """Test that unsupported models raise an error."""
        from mirt.models.dichotomous import ThreeParameterLogistic

        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = ThreeParameterLogistic(n_items=n_items)
        estimator = GVEMEstimator()

        with pytest.raises(ValueError, match="currently only supports 2PL"):
            estimator.fit(model, responses)

    def test_multidimensional(self, rng):
        """Test GVEM with multidimensional model."""
        n_persons, n_items, n_factors = 200, 10, 2

        theta = rng.standard_normal((n_persons, n_factors))
        a = rng.uniform(0.5, 1.5, (n_items, n_factors))
        b = rng.uniform(-1.5, 1.5, n_items)

        z = theta @ a.T - np.sum(a, axis=1) * b
        probs = 1 / (1 + np.exp(-z))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items, n_factors=n_factors)
        estimator = GVEMEstimator(max_iter=100, tol=1e-4)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert model.discrimination.shape == (n_items, n_factors)

        mu = estimator.variational_means
        assert mu.shape == (n_persons, n_factors)

    def test_compare_with_em(self, rng):
        """Test that GVEM produces similar estimates to standard EM."""
        from mirt.estimation.em import EMEstimator

        n_persons, n_items = 300, 10
        theta = rng.standard_normal(n_persons)
        true_a = np.ones(n_items) * 1.2
        true_b = np.linspace(-1.5, 1.5, n_items)
        probs = 1 / (1 + np.exp(-true_a * (theta[:, None] - true_b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model_gvem = TwoParameterLogistic(n_items=n_items)
        estimator_gvem = GVEMEstimator(max_iter=200, tol=1e-5)
        result_gvem = estimator_gvem.fit(model_gvem, responses)

        model_em = TwoParameterLogistic(n_items=n_items)
        estimator_em = EMEstimator(n_quadpts=21, max_iter=200, tol=1e-5)
        result_em = estimator_em.fit(model_em, responses)

        b_corr = np.corrcoef(result_gvem.model.difficulty, result_em.model.difficulty)[
            0, 1
        ]
        assert b_corr > 0.95

    def test_lambda_function(self):
        """Test the Jaakkola-Jordan lambda function."""
        estimator = GVEMEstimator()

        xi = np.array([0.0, 0.001, 0.1, 1.0, 10.0])
        lam = estimator._lambda(xi)

        assert lam.shape == xi.shape
        assert np.all(lam > 0)
        assert np.all(lam <= 0.125)

        assert np.isclose(lam[0], 0.125, atol=1e-3)

        assert lam[1] < lam[0]
        assert lam[2] < lam[1]
