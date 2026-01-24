"""Tests for local dependence statistics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.ld import (
    LDResult,
    compute_ld_chi2,
    compute_ld_statistics,
    compute_q3,
    flag_ld_pairs,
    ld_summary_table,
)


class TestLDResult:
    """Tests for LDResult dataclass."""

    @pytest.fixture
    def sample_ld_result(self):
        """Create sample LDResult for testing."""
        n_items = 5
        q3 = np.random.randn(n_items, n_items) * 0.1
        q3 = (q3 + q3.T) / 2
        np.fill_diagonal(q3, 1.0)

        chi2 = np.abs(np.random.randn(n_items, n_items) * 2)
        chi2 = (chi2 + chi2.T) / 2
        np.fill_diagonal(chi2, np.nan)

        return LDResult(
            q3_matrix=q3,
            ld_chi2_matrix=chi2,
            g2_matrix=chi2.copy(),
            adj_residual_corr=q3 + 1 / (n_items - 1),
            q3_flagged=[(0, 1, 0.25)],
            chi2_flagged=[(0, 2, 5.0, 0.02)],
            item_names=None,
        )

    def test_initialization(self, sample_ld_result):
        """Test LDResult initialization."""
        assert sample_ld_result.q3_matrix.shape == (5, 5)
        assert sample_ld_result.ld_chi2_matrix.shape == (5, 5)
        assert len(sample_ld_result.q3_flagged) == 1
        assert len(sample_ld_result.chi2_flagged) == 1

    def test_summary(self, sample_ld_result):
        """Test summary method."""
        summary = sample_ld_result.summary()

        assert "Local Dependence" in summary
        assert "Q3" in summary


class TestComputeLDStatistics:
    """Tests for compute_ld_statistics function."""

    def test_basic_ld_statistics(self, fitted_2pl_model, dichotomous_responses):
        """Test basic LD statistics computation."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        assert isinstance(result, LDResult)
        n_items = dichotomous_responses["n_items"]
        assert result.q3_matrix.shape == (n_items, n_items)
        assert result.ld_chi2_matrix.shape == (n_items, n_items)

    def test_q3_matrix_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 matrix is symmetric."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        assert_allclose(result.q3_matrix, result.q3_matrix.T)

    def test_chi2_matrix_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that chi-square matrix is symmetric."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(model, dichotomous_responses["responses"])

        valid_mask = ~np.isnan(result.ld_chi2_matrix)
        upper_tri = result.ld_chi2_matrix[valid_mask]
        lower_tri = result.ld_chi2_matrix.T[valid_mask]
        assert_allclose(upper_tri, lower_tri)

    def test_ld_with_provided_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test LD statistics with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_ld_statistics(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert isinstance(result, LDResult)

    def test_ld_flagged_pairs(self, fitted_2pl_model, dichotomous_responses):
        """Test that flagged pairs are returned."""
        model = fitted_2pl_model.model
        result = compute_ld_statistics(
            model,
            dichotomous_responses["responses"],
            q3_threshold=0.05,
        )

        for i, j, q3_val in result.q3_flagged:
            assert abs(q3_val) > 0.05
            assert i < j


class TestComputeQ3:
    """Tests for compute_q3 function."""

    def test_basic_q3(self, fitted_2pl_model, dichotomous_responses):
        """Test basic Q3 computation."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        n_items = dichotomous_responses["n_items"]
        assert q3_matrix.shape == (n_items, n_items)

    def test_q3_diagonal_is_zero(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 diagonal is zero (self-correlation excluded)."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        diagonal = np.diag(q3_matrix)
        assert_allclose(diagonal, np.zeros_like(diagonal))

    def test_q3_symmetric(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 is symmetric."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        assert_allclose(q3_matrix, q3_matrix.T)

    def test_q3_values_bounded(self, fitted_2pl_model, dichotomous_responses):
        """Test that Q3 values are bounded by -1 and 1."""
        model = fitted_2pl_model.model
        q3_matrix = compute_q3(model, dichotomous_responses["responses"])

        non_diagonal = q3_matrix[~np.eye(q3_matrix.shape[0], dtype=bool)]
        assert np.all(non_diagonal >= -1.0)
        assert np.all(non_diagonal <= 1.0)

    def test_q3_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test Q3 with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        q3_matrix = compute_q3(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert q3_matrix.shape[0] == dichotomous_responses["n_items"]


class TestComputeLDChi2:
    """Tests for compute_ld_chi2 function."""

    def test_basic_ld_chi2(self, fitted_2pl_model, dichotomous_responses):
        """Test basic LD chi-square computation."""
        model = fitted_2pl_model.model
        chi2_matrix, p_matrix = compute_ld_chi2(
            model, dichotomous_responses["responses"]
        )

        n_items = dichotomous_responses["n_items"]
        assert chi2_matrix.shape == (n_items, n_items)
        assert p_matrix.shape == (n_items, n_items)

    def test_chi2_non_negative(self, fitted_2pl_model, dichotomous_responses):
        """Test that chi-square values are non-negative."""
        model = fitted_2pl_model.model
        chi2_matrix, _ = compute_ld_chi2(model, dichotomous_responses["responses"])

        valid_chi2 = chi2_matrix[~np.isnan(chi2_matrix)]
        assert np.all(valid_chi2 >= 0)

    def test_p_values_in_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that p-values are in [0, 1]."""
        model = fitted_2pl_model.model
        _, p_matrix = compute_ld_chi2(model, dichotomous_responses["responses"])

        valid_p = p_matrix[~np.isnan(p_matrix)]
        assert np.all(valid_p >= 0)
        assert np.all(valid_p <= 1)


class TestFlagLDPairs:
    """Tests for flag_ld_pairs function."""

    def test_flag_q3(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by Q3."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, q3_threshold=0.05, method="q3")

        assert isinstance(flagged, list)
        for pair in flagged:
            assert len(pair) == 2
            i, j = pair
            assert abs(ld_result.q3_matrix[i, j]) > 0.05

    def test_flag_chi2(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by chi-square."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, chi2_alpha=0.05, method="chi2")

        assert isinstance(flagged, list)

    def test_flag_both(self, fitted_2pl_model, dichotomous_responses):
        """Test flagging by both Q3 and chi-square."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        flagged = flag_ld_pairs(ld_result, method="both")

        assert isinstance(flagged, list)

    def test_stricter_threshold_fewer_flags(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that stricter threshold produces fewer flags."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        lenient = flag_ld_pairs(ld_result, q3_threshold=0.05, method="q3")
        strict = flag_ld_pairs(ld_result, q3_threshold=0.3, method="q3")

        assert len(strict) <= len(lenient)


class TestLDSummaryTable:
    """Tests for ld_summary_table function."""

    def test_summary_table(self, fitted_2pl_model, dichotomous_responses):
        """Test summary table generation."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        table = ld_summary_table(ld_result)

        assert isinstance(table, str)
        assert "Q3" in table
        assert "Item" in table

    def test_summary_table_top_n(self, fitted_2pl_model, dichotomous_responses):
        """Test summary table with top_n parameter."""
        model = fitted_2pl_model.model
        ld_result = compute_ld_statistics(model, dichotomous_responses["responses"])

        table = ld_summary_table(ld_result, top_n=5)

        lines = [
            line
            for line in table.split("\n")
            if line.strip() and not line.startswith("-")
        ]
        assert len(lines) <= 7


class TestLDWithLocallyDependentData:
    """Tests with data having known local dependence."""

    def test_detects_correlated_items(self):
        """Test that LD is detected in correlated items."""
        rng = np.random.default_rng(42)
        n_persons = 200
        n_items = 8

        theta = rng.standard_normal(n_persons)
        diff = rng.normal(0, 1, n_items)

        probs = 1 / (1 + np.exp(-(theta[:, None] - diff)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

        correlation = rng.random((n_persons,)) < 0.7
        responses[correlation, 1] = responses[correlation, 0]

        from mirt import fit_mirt

        result = fit_mirt(responses, model="2PL", max_iter=20, n_quadpts=11)

        ld_result = compute_ld_statistics(result.model, responses)

        q3_01 = abs(ld_result.q3_matrix[0, 1])
        q3_others = np.abs(ld_result.q3_matrix)
        np.fill_diagonal(q3_others, 0)
        q3_others[0, 1] = 0
        q3_others[1, 0] = 0

        assert q3_01 > np.mean(q3_others)
