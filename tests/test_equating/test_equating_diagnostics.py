"""Tests for equating diagnostics."""

import numpy as np
import pytest

from mirt.equating.diagnostics import (
    bootstrap_linking_se,
    compare_linking_methods,
    compute_linking_fit,
    delta_method_se,
    linking_summary,
    parameter_recovery_summary,
)
from mirt.equating.linking import LinkingFitStatistics, link


@pytest.fixture
def linked_models_pair():
    """Create a pair of linked models for testing."""
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(42)
    n_items = 10

    disc = np.abs(rng.normal(1.0, 0.3, n_items))
    diff_old = rng.normal(0, 1, n_items)
    diff_new = diff_old * 1.1 + 0.3

    model_old = TwoParameterLogistic(n_items=n_items)
    model_old._parameters = {
        "discrimination": disc.copy(),
        "difficulty": diff_old.copy(),
    }
    model_old._is_fitted = True
    model_old._n_factors = 1

    model_new = TwoParameterLogistic(n_items=n_items)
    model_new._parameters = {
        "discrimination": disc.copy() / 1.1,
        "difficulty": diff_new.copy(),
    }
    model_new._is_fitted = True
    model_new._n_factors = 1

    anchors = list(range(5))

    return model_old, model_new, anchors


class TestBootstrapLinkingSE:
    """Tests for bootstrap_linking_se function."""

    def test_basic_bootstrap(self, linked_models_pair):
        """Test basic bootstrap SE computation."""
        model_old, model_new, anchors = linked_models_pair

        se_a, se_b, a_samples, b_samples = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=50,
            seed=42,
        )

        assert se_a > 0
        assert se_b > 0
        assert len(a_samples) == 50
        assert len(b_samples) == 50

    def test_bootstrap_reproducibility(self, linked_models_pair):
        """Test bootstrap reproducibility with seed."""
        model_old, model_new, anchors = linked_models_pair

        se_a1, se_b1, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=20,
            seed=42,
        )

        se_a2, se_b2, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=20,
            seed=42,
        )

        assert se_a1 == pytest.approx(se_a2)
        assert se_b1 == pytest.approx(se_b2)

    @pytest.mark.parametrize(
        "method",
        ["mean_sigma", "mean_mean", "stocking_lord", "haebara"],
    )
    def test_bootstrap_different_methods(self, linked_models_pair, method):
        """Test bootstrap with different linking methods."""
        model_old, model_new, anchors = linked_models_pair

        se_a, se_b, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            method=method,
            n_bootstrap=20,
            seed=42,
        )

        assert se_a > 0
        assert se_b > 0


class TestDeltaMethodSE:
    """Tests for delta_method_se function."""

    def test_basic_delta_method(self, linked_models_pair):
        """Test basic delta method SE computation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        n_params = 2 * model_old.n_items
        vcov = np.eye(n_params) * 0.01

        se_a, se_b = delta_method_se(
            linking_result,
            vcov,
            vcov,
            anchors,
            anchors,
        )

        assert se_a >= 0
        assert se_b >= 0


class TestComputeLinkingFit:
    """Tests for compute_linking_fit function."""

    def test_basic_fit_computation(self, linked_models_pair):
        """Test basic fit statistic computation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        fit_stats = compute_linking_fit(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert isinstance(fit_stats, LinkingFitStatistics)
        assert fit_stats.rmse_a >= 0
        assert fit_stats.rmse_b >= 0
        assert fit_stats.tcc_rmse >= 0

    def test_fit_statistics_reasonable(self, linked_models_pair):
        """Test that fit statistics are in reasonable range."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        fit_stats = compute_linking_fit(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert fit_stats.rmse_a < 2.0
        assert fit_stats.rmse_b < 2.0
        assert fit_stats.mad_a < 2.0
        assert fit_stats.mad_b < 2.0


class TestLinkingSummary:
    """Tests for linking_summary function."""

    def test_basic_summary(self, linked_models_pair):
        """Test basic summary generation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        summary = linking_summary(linking_result, model_old, model_new)

        assert isinstance(summary, str)
        assert "Linking Summary" in summary
        assert "Transformation" in summary

    def test_summary_contains_constants(self, linked_models_pair):
        """Test that summary contains transformation constants."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = linking_summary(linking_result, model_old, model_new)

        assert "A" in summary
        assert "B" in summary

    def test_summary_contains_fit_statistics(self, linked_models_pair):
        """Test that summary contains fit statistics."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        summary = linking_summary(linking_result, model_old, model_new)

        if linking_result.fit_statistics is not None:
            assert "RMSE" in summary or "Fit" in summary


class TestCompareLinkingMethods:
    """Tests for compare_linking_methods function."""

    def test_basic_comparison(self, linked_models_pair):
        """Test basic method comparison."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=["mean_sigma", "mean_mean"],
        )

        assert "mean_sigma" in results
        assert "mean_mean" in results

    def test_comparison_contains_all_methods(self, linked_models_pair):
        """Test that comparison contains all requested methods."""
        model_old, model_new, anchors = linked_models_pair

        methods = ["mean_sigma", "mean_mean", "stocking_lord"]

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=methods,
        )

        for method in methods:
            assert method in results

    def test_comparison_result_structure(self, linked_models_pair):
        """Test structure of comparison results."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=["mean_sigma"],
        )

        result = results["mean_sigma"]
        assert "A" in result
        assert "B" in result

    def test_default_methods(self, linked_models_pair):
        """Test with default methods."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
        )

        assert len(results) > 2


class TestParameterRecoverySummary:
    """Tests for parameter_recovery_summary function."""

    def test_basic_recovery_summary(self, linked_models_pair):
        """Test basic parameter recovery summary."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert isinstance(summary, str)
        assert "Recovery" in summary or "Parameter" in summary

    def test_recovery_summary_contains_items(self, linked_models_pair):
        """Test that recovery summary contains item information."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert "a_old" in summary or "diff" in summary.lower()

    def test_recovery_summary_contains_rmse(self, linked_models_pair):
        """Test that recovery summary contains RMSE."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert "RMSE" in summary


class TestDiagnosticsWithRealModels:
    """Tests for diagnostics with real fitted models."""

    def test_diagnostics_with_fitted_models(self, dichotomous_responses):
        """Test diagnostics with actual fitted models."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)

        responses = dichotomous_responses["responses"]

        result1 = fit_mirt(responses, model="2PL", max_iter=15, n_quadpts=11)

        responses2 = responses.copy()
        shift = rng.choice([0, 1], size=responses.shape, p=[0.7, 0.3])
        responses2 = np.where(shift, 1 - responses2, responses2)

        result2 = fit_mirt(responses2, model="2PL", max_iter=15, n_quadpts=11)

        anchors = list(range(4))

        linking_result = link(
            result1.model,
            result2.model,
            anchors,
            anchors,
            compute_diagnostics=True,
        )

        summary = linking_summary(linking_result, result1.model, result2.model)

        assert isinstance(summary, str)
        assert "Linking" in summary
