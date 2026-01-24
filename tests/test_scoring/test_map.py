"""Tests for MAP scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.scoring.map import MAPScorer


class TestMAPScorerInitialization:
    """Tests for MAPScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = MAPScorer()

        assert scorer.prior_mean is None
        assert scorer.prior_cov is None
        assert scorer.theta_bounds == (-6.0, 6.0)
        assert scorer.n_jobs == 1

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        scorer = MAPScorer(theta_bounds=(-4.0, 4.0))

        assert scorer.theta_bounds == (-4.0, 4.0)

    def test_custom_prior_mean(self):
        """Test initialization with custom prior mean."""
        prior_mean = np.array([0.5])
        scorer = MAPScorer(prior_mean=prior_mean)

        assert_allclose(scorer.prior_mean, prior_mean)

    def test_custom_prior_cov(self):
        """Test initialization with custom prior covariance."""
        prior_cov = np.array([[2.0]])
        scorer = MAPScorer(prior_cov=prior_cov)

        assert_allclose(scorer.prior_cov, prior_cov)

    def test_parallel_n_jobs(self):
        """Test initialization with parallel processing."""
        scorer = MAPScorer(n_jobs=2)

        assert scorer.n_jobs == 2

    def test_repr(self):
        """Test __repr__ method."""
        scorer = MAPScorer(theta_bounds=(-5.0, 5.0))
        repr_str = repr(scorer)

        assert "MAPScorer" in repr_str
        assert "-5" in repr_str


class TestMAPScorerScoring:
    """Tests for MAPScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic MAP scoring."""
        model = fitted_2pl_model.model
        scorer = MAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "MAP"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive."""
        model = fitted_2pl_model.model
        scorer = MAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        valid_se = result.standard_error[~np.isnan(result.standard_error)]
        assert np.all(valid_se > 0)

    def test_theta_within_bounds(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are within bounds."""
        model = fitted_2pl_model.model
        bounds = (-4.0, 4.0)
        scorer = MAPScorer(theta_bounds=bounds)

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta >= bounds[0])
        assert np.all(result.theta <= bounds[1])

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = MAPScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestMAPScorerCustomPrior:
    """Tests for MAP with custom prior parameters."""

    def test_shifted_prior_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test that shifted prior mean affects estimates."""
        model = fitted_2pl_model.model
        scorer_default = MAPScorer()
        scorer_shifted = MAPScorer(prior_mean=np.array([1.0]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_shifted = scorer_shifted.score(model, dichotomous_responses["responses"])

        mean_diff = result_shifted.theta.mean() - result_default.theta.mean()
        assert mean_diff > 0

    def test_smaller_prior_variance_shrinkage(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that smaller prior variance causes shrinkage."""
        model = fitted_2pl_model.model
        scorer_default = MAPScorer()
        scorer_small_var = MAPScorer(prior_cov=np.array([[0.25]]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_small_var = scorer_small_var.score(
            model, dichotomous_responses["responses"]
        )

        var_default = np.var(result_default.theta)
        var_small = np.var(result_small_var.theta)
        assert var_small <= var_default


class TestMAPScorerMultidimensional:
    """Tests for MAP with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional MAP scoring."""
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

        scorer = MAPScorer()
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)


class TestMAPScorerConsistency:
    """Tests for MAP scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = MAPScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        scorer = MAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        sum_scores = dichotomous_responses["responses"].sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7


class TestMAPScorerPolytomous:
    """Tests for MAP with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test MAP scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = MAPScorer()
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)


class TestMAPVsEAP:
    """Tests comparing MAP and EAP estimates."""

    def test_map_eap_correlation(self, fitted_2pl_model, dichotomous_responses):
        """Test that MAP and EAP estimates are highly correlated."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        map_scorer = MAPScorer()
        eap_scorer = EAPScorer()

        map_result = map_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        correlation = np.corrcoef(map_result.theta, eap_result.theta)[0, 1]

        assert correlation > 0.95

    def test_map_eap_similar_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test that MAP and EAP have similar means."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        map_scorer = MAPScorer()
        eap_scorer = EAPScorer()

        map_result = map_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        mean_diff = abs(map_result.theta.mean() - eap_result.theta.mean())

        assert mean_diff < 0.5
