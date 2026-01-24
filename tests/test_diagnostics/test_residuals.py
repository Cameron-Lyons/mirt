"""Tests for residual analysis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.residuals import (
    ResidualAnalysisResult,
    analyze_residuals,
    compute_outfit_infit,
    compute_residuals,
    identify_misfitting_patterns,
)


class TestComputeResiduals:
    """Tests for compute_residuals function."""

    def test_raw_residuals(self, fitted_2pl_model, dichotomous_responses):
        """Test raw residual computation."""
        model = fitted_2pl_model.model
        residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            residual_type="raw",
        )

        assert residuals.shape == dichotomous_responses["responses"].shape
        assert np.all(residuals >= -1)
        assert np.all(residuals <= 1)

    def test_standardized_residuals(self, fitted_2pl_model, dichotomous_responses):
        """Test standardized residual computation."""
        model = fitted_2pl_model.model
        residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            residual_type="standardized",
        )

        assert residuals.shape == dichotomous_responses["responses"].shape
        valid_resid = residuals[~np.isnan(residuals)]
        assert abs(np.mean(valid_resid)) < 0.5

    def test_pearson_residuals(self, fitted_2pl_model, dichotomous_responses):
        """Test Pearson residual computation."""
        model = fitted_2pl_model.model
        residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            residual_type="pearson",
        )

        assert residuals.shape == dichotomous_responses["responses"].shape

    def test_deviance_residuals(self, fitted_2pl_model, dichotomous_responses):
        """Test deviance residual computation."""
        model = fitted_2pl_model.model
        residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            residual_type="deviance",
        )

        assert residuals.shape == dichotomous_responses["responses"].shape
        valid_resid = residuals[~np.isnan(residuals)]
        assert np.all(np.isfinite(valid_resid))

    def test_invalid_residual_type(self, fitted_2pl_model, dichotomous_responses):
        """Test that invalid residual type raises error."""
        model = fitted_2pl_model.model
        with pytest.raises(ValueError, match="Unknown residual type"):
            compute_residuals(
                model,
                dichotomous_responses["responses"],
                residual_type="invalid",
            )

    def test_default_is_standardized(self, fitted_2pl_model, dichotomous_responses):
        """Test that default residual type is standardized."""
        model = fitted_2pl_model.model
        residuals = compute_residuals(model, dichotomous_responses["responses"])

        std_residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            residual_type="standardized",
        )

        assert_allclose(residuals, std_residuals)

    def test_with_provided_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test residuals with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        residuals = compute_residuals(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert residuals.shape == dichotomous_responses["responses"].shape


class TestAnalyzeResiduals:
    """Tests for analyze_residuals function."""

    def test_basic_analysis(self, fitted_2pl_model, dichotomous_responses):
        """Test basic residual analysis."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        assert isinstance(result, ResidualAnalysisResult)
        assert result.raw_residuals.shape == dichotomous_responses["responses"].shape
        assert (
            result.standardized_residuals.shape
            == dichotomous_responses["responses"].shape
        )

    def test_all_residual_types_computed(self, fitted_2pl_model, dichotomous_responses):
        """Test that all residual types are computed."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        assert result.raw_residuals is not None
        assert result.standardized_residuals is not None
        assert result.pearson_residuals is not None
        assert result.deviance_residuals is not None

    def test_expected_values_computed(self, fitted_2pl_model, dichotomous_responses):
        """Test that expected values are computed."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        assert result.expected_values.shape == dichotomous_responses["responses"].shape
        assert np.all(result.expected_values >= 0)
        assert np.all(result.expected_values <= 1)

    def test_item_residuals_computed(self, fitted_2pl_model, dichotomous_responses):
        """Test that item residuals are computed."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        assert len(result.item_residuals) == dichotomous_responses["n_items"]
        for j, stats in result.item_residuals.items():
            assert "mean" in stats
            assert "sd" in stats
            assert "max_abs_z" in stats

    def test_pattern_residuals_computed(self, fitted_2pl_model, dichotomous_responses):
        """Test that pattern residuals are computed."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        assert len(result.pattern_residuals) > 0
        for pattern, stats in result.pattern_residuals.items():
            assert "mean_z" in stats
            assert "n" in stats

    def test_summary_method(self, fitted_2pl_model, dichotomous_responses):
        """Test summary method."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        summary = result.summary()

        assert "Residual Analysis" in summary
        assert "Item" in summary


class TestComputeOutfitInfit:
    """Tests for compute_outfit_infit function."""

    def test_basic_computation(self, fitted_2pl_model, dichotomous_responses):
        """Test basic outfit/infit computation."""
        model = fitted_2pl_model.model
        result = compute_outfit_infit(model, dichotomous_responses["responses"])

        assert "item_outfit" in result
        assert "item_infit" in result
        assert "person_outfit" in result
        assert "person_infit" in result

    def test_item_statistics_length(self, fitted_2pl_model, dichotomous_responses):
        """Test that item statistics have correct length."""
        model = fitted_2pl_model.model
        result = compute_outfit_infit(model, dichotomous_responses["responses"])

        assert len(result["item_outfit"]) == dichotomous_responses["n_items"]
        assert len(result["item_infit"]) == dichotomous_responses["n_items"]

    def test_person_statistics_length(self, fitted_2pl_model, dichotomous_responses):
        """Test that person statistics have correct length."""
        model = fitted_2pl_model.model
        result = compute_outfit_infit(model, dichotomous_responses["responses"])

        assert len(result["person_outfit"]) == dichotomous_responses["n_persons"]
        assert len(result["person_infit"]) == dichotomous_responses["n_persons"]

    def test_statistics_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that fit statistics are positive."""
        model = fitted_2pl_model.model
        result = compute_outfit_infit(model, dichotomous_responses["responses"])

        assert np.all(result["item_outfit"] > 0)
        assert np.all(result["item_infit"] > 0)
        assert np.all(result["person_outfit"] > 0)
        assert np.all(result["person_infit"] > 0)

    def test_fit_statistics_reasonable(self, fitted_2pl_model, dichotomous_responses):
        """Test that fit statistics are in reasonable range."""
        model = fitted_2pl_model.model
        result = compute_outfit_infit(model, dichotomous_responses["responses"])

        assert np.all(result["item_outfit"] < 5.0)
        assert np.all(result["item_infit"] < 5.0)
        assert np.all(result["person_outfit"] < 5.0)
        assert np.all(result["person_infit"] < 5.0)


class TestIdentifyMisfittingPatterns:
    """Tests for identify_misfitting_patterns function."""

    def test_basic_identification(self, fitted_2pl_model, dichotomous_responses):
        """Test basic misfit identification."""
        model = fitted_2pl_model.model
        result = identify_misfitting_patterns(model, dichotomous_responses["responses"])

        assert "misfitting_persons" in result
        assert "misfitting_items" in result
        assert "aberrant_responses" in result

    def test_misfitting_persons_structure(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test structure of misfitting persons results."""
        model = fitted_2pl_model.model
        result = identify_misfitting_patterns(model, dichotomous_responses["responses"])

        for person_info in result["misfitting_persons"]:
            assert "person" in person_info
            assert "outfit" in person_info
            assert "infit" in person_info

    def test_misfitting_items_structure(self, fitted_2pl_model, dichotomous_responses):
        """Test structure of misfitting items results."""
        model = fitted_2pl_model.model
        result = identify_misfitting_patterns(model, dichotomous_responses["responses"])

        for item_info in result["misfitting_items"]:
            assert "item" in item_info
            assert "outfit" in item_info
            assert "infit" in item_info

    def test_aberrant_responses_structure(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test structure of aberrant responses results."""
        model = fitted_2pl_model.model
        result = identify_misfitting_patterns(model, dichotomous_responses["responses"])

        for resp_info in result["aberrant_responses"]:
            assert "person" in resp_info
            assert "item" in resp_info
            assert "response" in resp_info
            assert "expected" in resp_info
            assert "z" in resp_info

    def test_custom_thresholds(self, fitted_2pl_model, dichotomous_responses):
        """Test with custom thresholds."""
        model = fitted_2pl_model.model
        strict = identify_misfitting_patterns(
            model,
            dichotomous_responses["responses"],
            z_threshold=1.5,
            outfit_threshold=1.2,
        )

        lenient = identify_misfitting_patterns(
            model,
            dichotomous_responses["responses"],
            z_threshold=3.0,
            outfit_threshold=2.0,
        )

        assert len(strict["aberrant_responses"]) >= len(lenient["aberrant_responses"])


class TestResidualAnalysisResult:
    """Tests for ResidualAnalysisResult dataclass."""

    def test_result_summary_method(self, fitted_2pl_model, dichotomous_responses):
        """Test summary method."""
        model = fitted_2pl_model.model
        result = analyze_residuals(model, dichotomous_responses["responses"])

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Overall" in summary or "Residual" in summary


class TestResidualsPolytomous:
    """Tests for residuals with polytomous models."""

    def test_residuals_polytomous(self, polytomous_responses):
        """Test residuals with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        residuals = compute_residuals(result.model, polytomous_responses["responses"])

        assert residuals.shape == polytomous_responses["responses"].shape

    def test_outfit_infit_polytomous(self, polytomous_responses):
        """Test outfit/infit with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        fit_stats = compute_outfit_infit(
            result.model, polytomous_responses["responses"]
        )

        assert len(fit_stats["item_outfit"]) == polytomous_responses["n_items"]
