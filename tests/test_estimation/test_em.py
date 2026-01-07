"""Tests for EM estimation algorithm."""

import numpy as np

from mirt.estimation.em import EMEstimator
from mirt.models.dichotomous import TwoParameterLogistic


class TestEMEstimator:
    """Tests for EM algorithm."""

    def test_initialization(self):
        """Test estimator initialization."""
        estimator = EMEstimator(n_quadpts=21, max_iter=100, tol=1e-3)

        assert estimator.n_quadpts == 21
        assert estimator.max_iter == 100
        assert estimator.tol == 1e-3

    def test_fit_basic(self, dichotomous_responses):
        """Test basic model fitting."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert result.log_likelihood < 0
        assert result.n_iterations > 0
        assert result.aic > 0
        assert result.bic > 0

    def test_convergence_history(self, dichotomous_responses):
        """Test convergence history is tracked."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=20)

        estimator.fit(model, responses)

        history = estimator.convergence_history
        assert len(history) > 0
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 0.1

    def test_parameter_recovery(self, rng):
        """Test recovery of known parameters."""
        n_persons, n_items = 500, 10

        true_a = np.ones(n_items) * 1.5
        true_b = np.linspace(-2, 2, n_items)

        theta = rng.standard_normal(n_persons)
        probs = 1 / (1 + np.exp(-true_a * (theta[:, None] - true_b)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=21, max_iter=200, tol=1e-4)
        result = estimator.fit(model, responses)

        est_b = result.model.difficulty
        correlation = np.corrcoef(true_b, est_b)[0, 1]
        assert correlation > 0.8

    def test_standard_errors(self, dichotomous_responses):
        """Test standard errors are computed."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

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
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)
        assert result.converged or result.n_iterations == 50
