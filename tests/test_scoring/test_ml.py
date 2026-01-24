"""Tests for ML scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.scoring.ml import MLScorer


class TestMLScorerInitialization:
    """Tests for MLScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = MLScorer()

        assert scorer.theta_bounds == (-6.0, 6.0)
        assert scorer.n_jobs == 1

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        scorer = MLScorer(theta_bounds=(-4.0, 4.0))

        assert scorer.theta_bounds == (-4.0, 4.0)

    def test_parallel_n_jobs(self):
        """Test initialization with parallel processing."""
        scorer = MLScorer(n_jobs=2)

        assert scorer.n_jobs == 2

    def test_repr(self):
        """Test __repr__ method."""
        scorer = MLScorer(theta_bounds=(-5.0, 5.0))
        repr_str = repr(scorer)

        assert "MLScorer" in repr_str
        assert "-5" in repr_str


class TestMLScorerScoring:
    """Tests for MLScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic ML scoring."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "ML"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive_or_inf(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive or infinite."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_within_bounds(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are within bounds."""
        model = fitted_2pl_model.model
        bounds = (-4.0, 4.0)
        scorer = MLScorer(theta_bounds=bounds)

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta >= bounds[0])
        assert np.all(result.theta <= bounds[1])

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = MLScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestMLScorerPerfectZeroScores:
    """Tests for ML with perfect and zero scores."""

    def test_perfect_score_at_upper_bound(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that perfect score returns upper bound."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 1

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == scorer.theta_bounds[1]
        assert np.isinf(result.standard_error[0])

    def test_zero_score_at_lower_bound(self, fitted_2pl_model, dichotomous_responses):
        """Test that zero score returns lower bound."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 0

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == scorer.theta_bounds[0]
        assert np.isinf(result.standard_error[0])


class TestMLScorerMultidimensional:
    """Tests for ML with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional ML scoring."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)
        n_persons = 50
        n_items = 8

        theta = rng.standard_normal((n_persons, 2))
        loading = rng.uniform(0.5, 1.5, (n_items, 2))
        diff = rng.normal(0, 1, n_items)

        logit = theta @ loading.T - diff
        prob = 1 / (1 + np.exp(-logit))
        responses = (rng.random((n_persons, n_items)) < prob).astype(int)

        result = fit_mirt(responses, model="2PL", n_factors=2, max_iter=15, n_quadpts=5)

        scorer = MLScorer()
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)


class TestMLScorerConsistency:
    """Tests for ML scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)

        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])
        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        scorer = MLScorer()
        result = scorer.score(model, responses)

        correlation = np.corrcoef(result.theta[not_extreme], sum_scores[not_extreme])[
            0, 1
        ]

        assert correlation > 0.7


class TestMLScorerPolytomous:
    """Tests for ML with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test ML scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = MLScorer()
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)


class TestMLVsEAP:
    """Tests comparing ML and EAP estimates."""

    def test_ml_eap_correlation(self, fitted_2pl_model, dichotomous_responses):
        """Test that ML and EAP estimates are highly correlated."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])

        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        ml_result = ml_scorer.score(model, responses)
        eap_result = eap_scorer.score(model, responses)

        correlation = np.corrcoef(
            ml_result.theta[not_extreme], eap_result.theta[not_extreme]
        )[0, 1]

        assert correlation > 0.9

    def test_ml_more_extreme_than_eap(self, fitted_2pl_model, dichotomous_responses):
        """Test that ML estimates are more extreme than EAP."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        ml_result = ml_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        ml_var = np.var(ml_result.theta)
        eap_var = np.var(eap_result.theta)

        assert ml_var >= eap_var * 0.9


class TestMLScorerEmptyResponses:
    """Tests for ML with empty/missing responses."""

    def test_all_missing_returns_zero(self, fitted_2pl_model):
        """Test that all missing returns zero theta."""
        model = fitted_2pl_model.model
        responses = np.full((1, model.n_items), -1, dtype=int)

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == 0.0
        assert np.isinf(result.standard_error[0])
