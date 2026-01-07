"""Tests for person scoring methods."""

import numpy as np
import pytest

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.scoring import fscores
from mirt.scoring.eap import EAPScorer

try:
    import pandas  # noqa: F401

    HAS_DATAFRAME = True
except ImportError:
    try:
        import polars  # noqa: F401

        HAS_DATAFRAME = True
    except ImportError:
        HAS_DATAFRAME = False


class TestFscores:
    """Tests for fscores function."""

    @pytest.fixture
    def fitted_model(self, dichotomous_responses):
        """Create a fitted model for scoring tests."""
        from mirt.estimation.em import EMEstimator

        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=30)
        estimator.fit(model, responses)

        return model, responses

    def test_eap_scoring(self, fitted_model):
        """Test EAP scoring."""
        model, responses = fitted_model

        result = fscores(model, responses, method="EAP")

        assert result.n_persons == responses.shape[0]
        assert result.method == "EAP"
        assert result.theta.shape == (responses.shape[0],)
        assert result.standard_error.shape == (responses.shape[0],)

        assert np.all(result.standard_error > 0)

    def test_map_scoring(self, fitted_model):
        """Test MAP scoring."""
        model, responses = fitted_model

        result = fscores(model, responses, method="MAP")

        assert result.method == "MAP"
        assert result.theta.shape == (responses.shape[0],)

    def test_ml_scoring(self, fitted_model):
        """Test ML scoring."""
        model, responses = fitted_model

        result = fscores(model, responses, method="ML")

        assert result.method == "ML"
        assert result.theta.shape == (responses.shape[0],)

    def test_theta_correlation_with_sum_score(self, fitted_model):
        """Test that theta correlates with sum score."""
        model, responses = fitted_model

        result = fscores(model, responses, method="EAP")

        sum_scores = responses.sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_to_dataframe(self, fitted_model):
        """Test DataFrame conversion."""
        model, responses = fitted_model

        result = fscores(model, responses, method="EAP")
        df = result.to_dataframe()

        assert "theta" in df.columns
        assert "se" in df.columns
        assert len(df) == responses.shape[0]


class TestEAPScorer:
    """Tests for EAP scorer."""

    def test_initialization(self):
        """Test scorer initialization."""
        scorer = EAPScorer(n_quadpts=49)
        assert scorer.n_quadpts == 49

    def test_custom_prior(self, dichotomous_responses):
        """Test scoring with custom prior."""
        from mirt.estimation.em import EMEstimator

        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]

        model = TwoParameterLogistic(n_items=n_items)
        estimator = EMEstimator(n_quadpts=15, max_iter=30)
        estimator.fit(model, responses)

        scorer_default = EAPScorer()
        scorer_shifted = EAPScorer(prior_mean=np.array([1.0]))

        result_default = scorer_default.score(model, responses)
        result_shifted = scorer_shifted.score(model, responses)

        mean_diff = result_shifted.theta.mean() - result_default.theta.mean()
        assert mean_diff > 0
