"""Tests for model comparison functions."""

import pytest

from mirt import (
    anova_irt,
    compare_models,
    fit_mirt,
    information_criteria,
    vuong_test,
)

try:
    import pandas  # noqa: F401

    HAS_DATAFRAME = True
except ImportError:
    try:
        import polars  # noqa: F401

        HAS_DATAFRAME = True
    except ImportError:
        HAS_DATAFRAME = False


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
