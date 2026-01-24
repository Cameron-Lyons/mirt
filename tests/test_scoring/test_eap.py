"""Tests for EAP scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.scoring.eap import EAPScorer


class TestEAPScorerInitialization:
    """Tests for EAPScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = EAPScorer()

        assert scorer.n_quadpts == 49
        assert scorer.prior_mean is None
        assert scorer.prior_cov is None

    def test_custom_n_quadpts(self):
        """Test initialization with custom quadrature points."""
        scorer = EAPScorer(n_quadpts=31)

        assert scorer.n_quadpts == 31

    def test_invalid_n_quadpts(self):
        """Test that invalid n_quadpts raises error."""
        with pytest.raises(ValueError, match="at least 5"):
            EAPScorer(n_quadpts=3)

    def test_custom_prior_mean(self):
        """Test initialization with custom prior mean."""
        prior_mean = np.array([0.5])
        scorer = EAPScorer(prior_mean=prior_mean)

        assert_allclose(scorer.prior_mean, prior_mean)

    def test_custom_prior_cov(self):
        """Test initialization with custom prior covariance."""
        prior_cov = np.array([[2.0]])
        scorer = EAPScorer(prior_cov=prior_cov)

        assert_allclose(scorer.prior_cov, prior_cov)

    def test_repr(self):
        """Test __repr__ method."""
        scorer = EAPScorer(n_quadpts=21)
        repr_str = repr(scorer)

        assert "EAPScorer" in repr_str
        assert "21" in repr_str


class TestEAPScorerScoring:
    """Tests for EAPScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic EAP scoring."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "EAP"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_reasonable_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are in reasonable range."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta > -6)
        assert np.all(result.theta < 6)

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = EAPScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestEAPScorerCustomPrior:
    """Tests for EAP with custom prior parameters."""

    def test_shifted_prior_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test that shifted prior mean affects estimates."""
        model = fitted_2pl_model.model
        scorer_default = EAPScorer()
        scorer_shifted = EAPScorer(prior_mean=np.array([1.0]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_shifted = scorer_shifted.score(model, dichotomous_responses["responses"])

        mean_diff = result_shifted.theta.mean() - result_default.theta.mean()
        assert mean_diff > 0

    def test_larger_prior_variance(self, fitted_2pl_model, dichotomous_responses):
        """Test effect of larger prior variance."""
        model = fitted_2pl_model.model
        scorer_default = EAPScorer()
        scorer_large_var = EAPScorer(prior_cov=np.array([[4.0]]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_large_var = scorer_large_var.score(
            model, dichotomous_responses["responses"]
        )

        var_default = np.var(result_default.theta)
        var_large = np.var(result_large_var.theta)
        assert var_large >= var_default * 0.9


class TestEAPScorerMultidimensional:
    """Tests for EAP with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional EAP scoring."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)
        n_persons = 100
        n_items = 12

        theta = rng.standard_normal((n_persons, 2))
        loading = np.zeros((n_items, 2))
        loading[:6, 0] = rng.uniform(0.5, 1.5, 6)
        loading[6:, 1] = rng.uniform(0.5, 1.5, 6)
        diff = rng.normal(0, 1, n_items)

        logit = theta @ loading.T - diff
        prob = 1 / (1 + np.exp(-logit))
        responses = (rng.random((n_persons, n_items)) < prob).astype(int)

        result = fit_mirt(responses, model="2PL", n_factors=2, max_iter=20, n_quadpts=7)

        scorer = EAPScorer(n_quadpts=7)
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)

    def test_multidimensional_custom_prior(self):
        """Test multidimensional EAP with custom prior."""
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

        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.eye(2) * 2.0

        scorer = EAPScorer(n_quadpts=5, prior_mean=prior_mean, prior_cov=prior_cov)
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)


class TestEAPScorerConsistency:
    """Tests for EAP scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)
        assert_allclose(result1.standard_error, result2.standard_error)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        sum_scores = dichotomous_responses["responses"].sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7


class TestEAPScorerPolytomous:
    """Tests for EAP with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test EAP scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = EAPScorer(n_quadpts=15)
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)
        assert np.all(scores.standard_error > 0)
