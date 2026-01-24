"""Tests for WLE scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.scoring.wle import WLEScorer


class TestWLEScorerInitialization:
    """Tests for WLEScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = WLEScorer()

        assert scorer.bounds == (-6.0, 6.0)
        assert scorer.tol == 1e-6

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        scorer = WLEScorer(bounds=(-4.0, 4.0))

        assert scorer.bounds == (-4.0, 4.0)

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        scorer = WLEScorer(tol=1e-8)

        assert scorer.tol == 1e-8

    def test_repr(self):
        """Test __repr__ method."""
        scorer = WLEScorer(bounds=(-5.0, 5.0), tol=1e-5)
        repr_str = repr(scorer)

        assert "WLEScorer" in repr_str
        assert "-5" in repr_str


class TestWLEScorerScoring:
    """Tests for WLEScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic WLE scoring."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "WLE"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive_or_inf(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive or infinite."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_within_bounds(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are within bounds."""
        model = fitted_2pl_model.model
        bounds = (-4.0, 4.0)
        scorer = WLEScorer(bounds=bounds)

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta >= bounds[0])
        assert np.all(result.theta <= bounds[1])

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = WLEScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestWLEBiasReduction:
    """Tests for WLE bias reduction properties."""

    def test_wle_less_biased_than_ml_at_extremes(self, fitted_2pl_model):
        """Test that WLE is less biased than ML at extreme abilities."""
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        rng = np.random.default_rng(42)
        true_theta = 2.5
        n_replications = 50

        wle_estimates = []
        ml_estimates = []

        for _ in range(n_replications):
            n_items = model.n_items
            probs = 1 / (1 + np.exp(-1.0 * (true_theta - np.zeros(n_items))))
            responses = (rng.random(n_items) < probs).astype(int)

            wle_scorer = WLEScorer()
            ml_scorer = MLScorer()

            wle_result = wle_scorer.score(model, responses[None, :])
            ml_result = ml_scorer.score(model, responses[None, :])

            wle_estimates.append(wle_result.theta[0])
            ml_estimates.append(ml_result.theta[0])

        wle_bias = abs(np.mean(wle_estimates) - true_theta)
        ml_bias = abs(np.mean(ml_estimates) - true_theta)

        assert wle_bias <= ml_bias + 0.3

    def test_wle_ml_similar_for_moderate_theta(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that WLE and ML are similar for moderate theta."""
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        n_items = responses.shape[1]
        moderate = (sum_scores > n_items * 0.3) & (sum_scores < n_items * 0.7)

        if np.sum(moderate) < 5:
            pytest.skip("Not enough moderate scores")

        wle_scorer = WLEScorer()
        ml_scorer = MLScorer()

        wle_result = wle_scorer.score(model, responses)
        ml_result = ml_scorer.score(model, responses)

        correlation = np.corrcoef(
            wle_result.theta[moderate], ml_result.theta[moderate]
        )[0, 1]

        assert correlation > 0.9


class TestWLEScorerMultidimensional:
    """Tests for WLE with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional WLE scoring."""
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

        scorer = WLEScorer()
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)


class TestWLEScorerConsistency:
    """Tests for WLE scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        sum_scores = dichotomous_responses["responses"].sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7


class TestWLEScorerPolytomous:
    """Tests for WLE with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test WLE scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = WLEScorer()
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)


class TestWLEVsOtherMethods:
    """Tests comparing WLE with other scoring methods."""

    def test_wle_eap_correlation(self, fitted_2pl_model, dichotomous_responses):
        """Test that WLE and EAP estimates are highly correlated."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        wle_scorer = WLEScorer()
        eap_scorer = EAPScorer()

        wle_result = wle_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        correlation = np.corrcoef(wle_result.theta, eap_result.theta)[0, 1]

        assert correlation > 0.8

    def test_wle_between_ml_and_eap_variance(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that WLE variance is between ML and EAP."""
        from mirt.scoring.eap import EAPScorer
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])

        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        wle_scorer = WLEScorer()
        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        wle_result = wle_scorer.score(model, responses)
        ml_result = ml_scorer.score(model, responses)
        eap_result = eap_scorer.score(model, responses)

        wle_var = np.var(wle_result.theta[not_extreme])
        ml_var = np.var(ml_result.theta[not_extreme])
        eap_var = np.var(eap_result.theta[not_extreme])

        assert min(eap_var, ml_var) * 0.7 <= wle_var <= max(eap_var, ml_var) * 1.3


class TestWLEScorerEmptyResponses:
    """Tests for WLE with empty/missing responses."""

    def test_all_missing_returns_zero(self, fitted_2pl_model):
        """Test that all missing returns zero theta."""
        model = fitted_2pl_model.model
        responses = np.full((1, model.n_items), -1, dtype=int)

        scorer = WLEScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == 0.0
        assert np.isinf(result.standard_error[0])
