"""Tests for model comparison functions."""

from types import SimpleNamespace

import numpy as np
import pytest

from mirt import (
    anova_irt,
    compare_models,
    fit_mirt,
    information_criteria,
    vuong_test,
)
from mirt.diagnostics.comparison import _compute_person_loglik
from mirt.scoring._common import build_quadrature
from mirt.utils.numeric import logsumexp_axis1

try:
    import pandas as pandas

    HAS_DATAFRAME = True
except ImportError:
    try:
        import polars as polars

        HAS_DATAFRAME = True
    except ImportError:
        HAS_DATAFRAME = False


class _BatchLogLikelihoodModel:
    """Minimal model exposing controlled quadrature-grid likelihoods."""

    is_polytomous = False
    n_factors = 1

    def __init__(
        self,
        name: str,
        *,
        zero_loglik: float,
        one_loglik: float,
        node_slope: float = 0.0,
        n_items: int = 1,
    ) -> None:
        self.model_name = name
        self.zero_loglik = zero_loglik
        self.one_loglik = one_loglik
        self.node_slope = node_slope
        self.n_items = n_items

    def log_likelihood_batch(self, responses, theta):
        offsets = np.where(responses[:, 0] == 1, self.one_loglik, self.zero_loglik)
        slopes = np.where(responses[:, 0] == 1, self.node_slope, -self.node_slope)
        return offsets[:, None] + slopes[:, None] * theta[None, :, 0]

    def log_likelihood(self, responses, theta):
        raise AssertionError("the vectorized marginal-likelihood path should be used")


def _result(model):
    return SimpleNamespace(model=model)


class TestAnovaIRT:
    """Tests for likelihood ratio test / anova."""

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_anova_nested_models(self, dichotomous_responses):
        """Test LRT for nested models (1PL vs 2PL)."""
        responses = dichotomous_responses["responses"]

        result_1pl = fit_mirt(responses, model="1PL", max_iter=50)
        result_2pl = fit_mirt(responses, model="2PL", max_iter=50)

        comparison = anova_irt(result_1pl, result_2pl)

        assert hasattr(comparison, "columns")
        cols = list(comparison.columns)
        assert "Model" in cols or "model" in cols
        assert any("LogLik" in c or "log" in c.lower() for c in cols)

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_anova_multiple_models(self, dichotomous_responses):
        """Test comparing multiple models."""
        responses = dichotomous_responses["responses"]

        result_1pl = fit_mirt(responses, model="1PL", max_iter=30)
        result_2pl = fit_mirt(responses, model="2PL", max_iter=30)
        result_3pl = fit_mirt(responses, model="3PL", max_iter=30)

        comparison = anova_irt(result_1pl, result_2pl, result_3pl)

        assert comparison is not None

    def test_rejects_unknown_method(self):
        """Reject comparison methods that are not implemented."""
        with pytest.raises(ValueError, match="method must be 'LRT'"):
            anova_irt(object(), object(), method="AIC")


class TestCompareModels:
    """Tests for non-nested model comparison."""

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_compare_aic_bic(self, dichotomous_responses):
        """Test AIC/BIC comparison."""
        responses = dichotomous_responses["responses"]

        result_1pl = fit_mirt(responses, model="1PL", max_iter=50)
        result_2pl = fit_mirt(responses, model="2PL", max_iter=50)

        comparison = compare_models([result_1pl, result_2pl])

        assert "AIC" in comparison or hasattr(comparison, "columns")
        assert "BIC" in comparison or hasattr(comparison, "columns")

    def test_compare_criteria(self, dichotomous_responses):
        """Test multiple information criteria."""
        responses = dichotomous_responses["responses"]

        result = fit_mirt(responses, model="2PL", max_iter=50)
        criteria = information_criteria(result)

        assert "AIC" in criteria
        assert "BIC" in criteria


class TestVuongTest:
    """Tests for Vuong test for non-nested models."""

    def test_vuong_test(self, dichotomous_responses):
        """Test Vuong test computation."""
        responses = dichotomous_responses["responses"]

        result_2pl = fit_mirt(responses, model="2PL", max_iter=50)
        result_3pl = fit_mirt(responses, model="3PL", max_iter=50)

        vuong_result = vuong_test(result_2pl, result_3pl, responses)

        assert "statistic" in vuong_result or "z" in vuong_result
        assert "p_value" in vuong_result

    def test_vuong_interpretation(self, dichotomous_responses):
        """Test Vuong test interpretation."""
        responses = dichotomous_responses["responses"]

        result_2pl = fit_mirt(responses, model="2PL", max_iter=50)
        result_3pl = fit_mirt(responses, model="3PL", max_iter=50)

        vuong_result = vuong_test(result_2pl, result_3pl, responses)

        assert 0 <= vuong_result["p_value"] <= 1

    def test_person_loglik_is_marginalized_over_quadrature(self):
        """Integrate response-pattern likelihoods instead of plugging in scores."""
        responses = np.array([[0], [1], [1]], dtype=int)
        model = _BatchLogLikelihoodModel(
            "curved",
            zero_loglik=-2.0,
            one_loglik=-1.0,
            node_slope=0.6,
        )

        actual = _compute_person_loglik(model, responses, 9)
        nodes, weights = build_quadrature(
            n_quadpts=9,
            n_factors=1,
            prior_mean=None,
            prior_cov=None,
        )
        conditional = model.log_likelihood_batch(responses, nodes)
        expected = logsumexp_axis1(conditional + np.log(weights)[None, :])

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_stable_tail_probability_and_diagnostics(self):
        """Retain representable two-sided probabilities for large statistics."""
        responses = np.concatenate(
            [np.ones((5_500, 1), dtype=int), np.zeros((4_500, 1), dtype=int)]
        )
        first = _BatchLogLikelihoodModel("first", zero_loglik=-3.0, one_loglik=-1.0)
        second = _BatchLogLikelihoodModel("second", zero_loglik=-2.0, one_loglik=-2.0)

        result = vuong_test(_result(first), _result(second), responses, n_quadpts=5)
        differences = np.where(responses[:, 0] == 1, 1.0, -1.0)
        expected_z = float(
            np.mean(differences)
            / (np.std(differences, ddof=1) / np.sqrt(differences.size))
        )

        assert result["z"] == pytest.approx(expected_z)
        assert 0.0 < result["p_value"] < 1e-20
        assert result["preferred"] == "first"
        assert result["mean_log_likelihood_difference"] == pytest.approx(0.1)
        assert result["standard_error"] == pytest.approx(
            result["mean_log_likelihood_difference"] / result["z"]
        )

    def test_alpha_controls_preference(self):
        """Use the requested significance level when interpreting the result."""
        responses = np.concatenate(
            [np.ones((7, 1), dtype=int), np.zeros((3, 1), dtype=int)]
        )
        first = _BatchLogLikelihoodModel("first", zero_loglik=-3.0, one_loglik=-1.0)
        second = _BatchLogLikelihoodModel("second", zero_loglik=-2.0, one_loglik=-2.0)

        default = vuong_test(_result(first), _result(second), responses, n_quadpts=5)
        relaxed = vuong_test(
            _result(first), _result(second), responses, alpha=0.2, n_quadpts=5
        )

        assert default["preferred"] == "neither"
        assert relaxed["preferred"] == "first"

    @pytest.mark.parametrize("n_quadpts", [True, 4, 5.0])
    def test_rejects_invalid_quadrature_count(self, n_quadpts):
        """Require an integer quadrature resolution with useful accuracy."""
        model = _BatchLogLikelihoodModel("model", zero_loglik=-2.0, one_loglik=-1.0)
        responses = np.array([[0], [1]], dtype=int)

        with pytest.raises(ValueError, match="n_quadpts"):
            vuong_test(
                _result(model),
                _result(model),
                responses,
                n_quadpts=n_quadpts,
            )

    @pytest.mark.parametrize("alpha", [True, 0.0, 1.0, np.nan])
    def test_rejects_invalid_alpha(self, alpha):
        """Require a finite significance level strictly between zero and one."""
        model = _BatchLogLikelihoodModel("model", zero_loglik=-2.0, one_loglik=-1.0)
        responses = np.array([[0], [1]], dtype=int)

        with pytest.raises(ValueError, match="alpha"):
            vuong_test(_result(model), _result(model), responses, alpha=alpha)

    @pytest.mark.parametrize(
        ("responses", "message"),
        [
            (np.array([0, 1]), "responses must be 2D"),
            (np.array([[0]]), "at least two persons"),
            (np.array([[0], [2]]), "dichotomous responses"),
            (np.array([[0.0], [np.nan]]), "finite values"),
        ],
    )
    def test_rejects_invalid_responses(self, responses, message):
        """Validate response dimensions, values, and sample size."""
        model = _BatchLogLikelihoodModel("model", zero_loglik=-2.0, one_loglik=-1.0)

        with pytest.raises(ValueError, match=message):
            vuong_test(_result(model), _result(model), responses)

    def test_rejects_models_with_different_item_counts(self):
        """Only compare models fitted to the same observed variables."""
        first = _BatchLogLikelihoodModel("first", zero_loglik=-2.0, one_loglik=-1.0)
        second = _BatchLogLikelihoodModel(
            "second", zero_loglik=-2.0, one_loglik=-1.0, n_items=2
        )

        with pytest.raises(ValueError, match="same number of items"):
            vuong_test(
                _result(first),
                _result(second),
                np.array([[0], [1]], dtype=int),
            )

    def test_rejects_models_with_different_response_categories(self):
        """Require both models to describe the same response outcomes."""
        first = _BatchLogLikelihoodModel("first", zero_loglik=-2.0, one_loglik=-1.0)
        second = _BatchLogLikelihoodModel("second", zero_loglik=-2.0, one_loglik=-1.0)
        second.is_polytomous = True
        second.n_categories = [3]

        with pytest.raises(ValueError, match="same response categories"):
            vuong_test(
                _result(first),
                _result(second),
                np.array([[0], [1]], dtype=int),
            )

    def test_rejects_a_different_fitting_sample(self):
        """Match the supplied response count to each fitted result."""
        model = _BatchLogLikelihoodModel("model", zero_loglik=-2.0, one_loglik=-1.0)
        first = SimpleNamespace(model=model, n_observations=2)
        second = SimpleNamespace(model=model, n_observations=3)

        with pytest.raises(ValueError, match="observations used to fit"):
            vuong_test(first, second, np.array([[0], [1]], dtype=int))
