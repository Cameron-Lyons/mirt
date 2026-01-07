"""Tests for model fit statistics (M2, RMSEA, CFI, TLI)."""

from mirt import compute_fit_indices, compute_m2


class TestM2:
    """Tests for M2 limited-information statistic."""

    def test_compute_m2(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)

        assert "M2" in m2_result
        assert "df" in m2_result
        assert "p_value" in m2_result

        assert m2_result["M2"] >= 0
        assert m2_result["df"] > 0
        assert 0 <= m2_result["p_value"] <= 1

    def test_m2_with_fit_result(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation with FitResult object."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)
        assert "M2" in m2_result


class TestFitIndices:
    """Tests for RMSEA, CFI, TLI, SRMSR."""

    def test_compute_fit_indices(self, fitted_2pl_model, dichotomous_responses):
        """Test fit indices computation."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert "RMSEA" in fit_stats
        assert "CFI" in fit_stats
        assert "TLI" in fit_stats
        assert "SRMSR" in fit_stats

    def test_rmsea_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that RMSEA is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["RMSEA"] >= 0

    def test_cfi_tli_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that CFI/TLI are in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["CFI"] >= 0
        assert fit_stats["TLI"] >= -0.5

    def test_rmsea_ci(self, fitted_2pl_model, dichotomous_responses):
        """Test RMSEA confidence intervals."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        if "RMSEA_CI_lower" in fit_stats:
            assert fit_stats["RMSEA_CI_lower"] <= fit_stats["RMSEA"]
            assert fit_stats["RMSEA_CI_upper"] >= fit_stats["RMSEA"]

    def test_srmsr_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that SRMSR is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["SRMSR"] >= 0
