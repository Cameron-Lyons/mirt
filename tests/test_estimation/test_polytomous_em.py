"""Tests for EM estimation with polytomous models."""

import numpy as np

from mirt.estimation.em import EMEstimator
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
)


class TestPolytomousEM:
    """Tests for EM algorithm with polytomous models."""

    def test_grm_em_basic(self, rng):
        """Test basic GRM fitting with EM."""
        n_persons = 200
        n_items = 8
        n_categories = 5

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert result.log_likelihood < 0
        assert result.n_iterations > 0
        assert result.aic > 0
        assert result.bic > 0

    def test_gpcm_em_basic(self, rng):
        """Test basic GPCM fitting with EM."""
        n_persons = 200
        n_items = 8
        n_categories = 4

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))

        model = GeneralizedPartialCredit(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=50, verbose=False)

        result = estimator.fit(model, responses)

        assert result.model is model
        assert model.is_fitted
        assert result.log_likelihood < 0

    def test_grm_standard_errors(self, rng):
        """Test that standard errors are computed for GRM."""
        n_persons = 200
        n_items = 5
        n_categories = 4

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)

        assert "discrimination" in result.standard_errors
        assert "thresholds" in result.standard_errors

        se_disc = result.standard_errors["discrimination"]
        assert np.all((se_disc > 0) | np.isnan(se_disc))

    def test_grm_missing_data(self, rng):
        """Test GRM with missing data."""
        n_persons = 200
        n_items = 8
        n_categories = 5

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))

        mask = rng.random(responses.shape) < 0.1
        responses[mask] = -1

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)
        assert result.converged or result.n_iterations == 50

    def test_varying_categories(self, rng):
        """Test GRM with varying categories per item."""
        n_persons = 200
        n_categories = [3, 4, 5, 3, 4, 5]
        n_items = len(n_categories)

        responses = np.zeros((n_persons, n_items), dtype=int)
        for i, n_cat in enumerate(n_categories):
            responses[:, i] = rng.integers(0, n_cat, size=n_persons)

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=50)

        result = estimator.fit(model, responses)

        assert model.is_fitted
        assert result.log_likelihood < 0

    def test_convergence_improves(self, rng):
        """Test that log-likelihood improves during fitting."""
        n_persons = 300
        n_items = 6
        n_categories = 4

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)
        estimator = EMEstimator(n_quadpts=15, max_iter=30)

        estimator.fit(model, responses)

        history = estimator.convergence_history
        assert len(history) > 0

        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 0.5
