"""Tests for local dependence statistics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.constants import PROB_EPSILON
from mirt.diagnostics.ld import (
    LDResult,
    _compute_ld_chi2_g2,
    _compute_q3,
    compute_ld_chi2,
    compute_ld_statistics,
    compute_q3,
    flag_ld_pairs,
    ld_summary_table,
)


class _CountingPolytomousModel:
    """Small deterministic model that records probability evaluations."""

    is_polytomous = True

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.item_names = [f"Item_{index}" for index in range(n_items)]
        self.probability_calls = 0

    def positive_probability(self, theta, item_idx):
        theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
        slope = 0.8 + 0.1 * item_idx
        difficulty = -0.6 + 0.25 * item_idx
        logits = slope * (theta_values - difficulty)
        return 1.0 / (1.0 + np.exp(-logits))

    def probability(self, theta, item_idx):
        self.probability_calls += 1
        positive = self.positive_probability(theta, item_idx)
        return np.column_stack((1.0 - positive, positive))


def _reference_ld_chi2_g2(model, responses, theta):
    """Scalar reference implementation for pairwise LD statistics."""
    n_items = responses.shape[1]
    chi2_matrix = np.full((n_items, n_items), np.nan)
    g2_matrix = np.full((n_items, n_items), np.nan)

    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
            if valid.sum() < 10:
                continue

            first_response = responses[valid, first] > 0
            second_response = responses[valid, second] > 0
            first_probability = model.positive_probability(theta[valid], first)
            second_probability = model.positive_probability(theta[valid], second)

            observed = np.array(
                [
                    np.sum(~first_response & ~second_response),
                    np.sum(~first_response & second_response),
                    np.sum(first_response & ~second_response),
                    np.sum(first_response & second_response),
                ]
            )
            expected = np.array(
                [
                    np.sum((1.0 - first_probability) * (1.0 - second_probability)),
                    np.sum((1.0 - first_probability) * second_probability),
                    np.sum(first_probability * (1.0 - second_probability)),
                    np.sum(first_probability * second_probability),
                ]
            )
            expected = np.maximum(expected, 0.5)

            chi2 = np.sum((observed - expected) ** 2 / expected)
            g2 = 2.0 * np.sum(observed * np.log(observed / expected + PROB_EPSILON))
            chi2_matrix[first, second] = chi2_matrix[second, first] = chi2
            g2_matrix[first, second] = g2_matrix[second, first] = g2

    return chi2_matrix, g2_matrix


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


class TestVectorizedLDCalculations:
    """Regression tests for the shared pairwise matrix calculations."""

    def test_q3_matches_pairwise_reference_with_missing_responses(self):
        rng = np.random.default_rng(90210)
        responses = rng.integers(0, 2, size=(40, 7))
        responses[rng.random(responses.shape) < 0.2] = -1
        residuals = rng.normal(size=responses.shape)
        residuals[responses < 0] = np.nan

        expected = np.zeros((responses.shape[1], responses.shape[1]))
        for first in range(responses.shape[1]):
            for second in range(first + 1, responses.shape[1]):
                valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
                valid &= ~np.isnan(residuals[:, first])
                valid &= ~np.isnan(residuals[:, second])
                if valid.sum() > 2:
                    correlation = np.corrcoef(
                        residuals[valid, first], residuals[valid, second]
                    )[0, 1]
                    expected[first, second] = expected[second, first] = correlation

        actual = _compute_q3(residuals, responses)

        assert_allclose(actual, expected, rtol=1e-13, atol=1e-14, equal_nan=True)

    def test_chi2_and_g2_match_pairwise_reference(self):
        rng = np.random.default_rng(314159)
        n_persons, n_items = 50, 6
        theta = rng.normal(size=(n_persons, 1))
        model = _CountingPolytomousModel(n_items)
        probabilities = np.column_stack(
            [model.positive_probability(theta, item) for item in range(n_items)]
        )
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)
        responses[rng.random(responses.shape) < 0.15] = -1

        expected_chi2, expected_g2 = _reference_ld_chi2_g2(model, responses, theta)
        actual_chi2, actual_g2 = _compute_ld_chi2_g2(
            model, responses, theta, n_quadpts=21
        )

        assert model.probability_calls == n_items
        assert_allclose(
            actual_chi2, expected_chi2, rtol=1e-12, atol=1e-12, equal_nan=True
        )
        assert_allclose(actual_g2, expected_g2, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_full_analysis_reuses_residual_correlations(self):
        rng = np.random.default_rng(271828)
        n_persons, n_items = 40, 5
        theta = rng.normal(size=(n_persons, 1))
        model = _CountingPolytomousModel(n_items)
        probabilities = np.column_stack(
            [model.positive_probability(theta, item) for item in range(n_items)]
        )
        responses = (rng.random(probabilities.shape) < probabilities).astype(int)
        responses[rng.random(responses.shape) < 0.1] = -1

        result = compute_ld_statistics(model, responses, theta=theta)

        assert model.probability_calls == 2 * n_items
        assert_allclose(
            result.adj_residual_corr,
            result.q3_matrix + 1.0 / (n_items - 1),
            equal_nan=True,
        )

    def test_single_item_adjustment_remains_finite(self):
        model = _CountingPolytomousModel(1)
        theta = np.linspace(-2.0, 2.0, 12).reshape(-1, 1)
        responses = np.array([[0], [1]] * 6)

        result = compute_ld_statistics(model, responses, theta=theta)

        assert_allclose(result.q3_matrix, np.zeros((1, 1)))
        assert_allclose(result.adj_residual_corr, np.zeros((1, 1)))


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

    def test_q3_threshold_uses_full_matrix(self):
        """A lower threshold can identify pairs absent from cached flags."""
        q3 = np.array(
            [
                [0.0, 0.10, -0.03],
                [0.10, 0.0, -0.08],
                [-0.03, -0.08, 0.0],
            ]
        )
        chi2 = np.full((3, 3), np.nan)
        result = LDResult(q3, chi2, chi2.copy(), q3.copy(), [], [])

        assert flag_ld_pairs(result, q3_threshold=0.05, method="q3") == [
            (0, 1),
            (1, 2),
        ]

    def test_chi2_matches_survival_function_with_nonfinite_values(self):
        """Matrix selection preserves the scalar chi-square definition."""
        q3 = np.zeros((4, 4))
        chi2 = np.array(
            [
                [np.nan, 0.1, 4.0, np.inf],
                [0.1, np.nan, 6.0, 2.0],
                [4.0, 6.0, np.nan, 8.0],
                [np.inf, 2.0, 8.0, np.nan],
            ]
        )
        result = LDResult(q3, chi2, chi2.copy(), q3.copy(), [], [])

        assert flag_ld_pairs(result, method="chi2") == [
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 3),
        ]

    @pytest.mark.parametrize("method", ["", "Q3", "unknown"])
    def test_rejects_unknown_method(self, method):
        """Pair flagging rejects unsupported selection methods."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="method"):
            flag_ld_pairs(result, method=method)

    @pytest.mark.parametrize("method", [None, 1, True])
    def test_rejects_non_string_method(self, method):
        """Pair flagging requires a string selection method."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="method"):
            flag_ld_pairs(result, method=method)

    @pytest.mark.parametrize("threshold", [-0.1, np.inf, np.nan])
    def test_rejects_invalid_q3_threshold(self, threshold):
        """Q3 thresholds must be finite and nonnegative."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="q3_threshold"):
            flag_ld_pairs(result, q3_threshold=threshold, method="q3")

    @pytest.mark.parametrize("threshold", [None, "0.2", True])
    def test_rejects_non_numeric_q3_threshold(self, threshold):
        """Q3 thresholds reject non-numeric and boolean values."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="q3_threshold"):
            flag_ld_pairs(result, q3_threshold=threshold, method="q3")

    @pytest.mark.parametrize("alpha", [0.0, 1.0, np.inf, np.nan])
    def test_rejects_invalid_chi2_alpha(self, alpha):
        """Chi-square levels must be finite probabilities."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="chi2_alpha"):
            flag_ld_pairs(result, chi2_alpha=alpha, method="chi2")


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

    def test_summary_table_stably_orders_ties_and_missing_q3(self):
        """Finite top pairs precede missing values and ties keep matrix order."""
        q3 = np.array(
            [
                [0.0, 0.5, -0.5, np.nan],
                [0.5, 0.0, 0.2, 0.1],
                [-0.5, 0.2, 0.0, 0.3],
                [np.nan, 0.1, 0.3, 0.0],
            ]
        )
        chi2 = np.full((4, 4), np.nan)
        result = LDResult(
            q3,
            chi2,
            chi2.copy(),
            q3.copy(),
            [],
            [],
            ["one", "two", "three", "four"],
        )

        lines = ld_summary_table(result, top_n=3).splitlines()[2:]

        assert [line.split()[:2] for line in lines] == [
            ["one", "two"],
            ["one", "three"],
            ["three", "four"],
        ]

    def test_summary_table_zero_rows(self):
        """A zero limit returns only the table heading."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        assert len(ld_summary_table(result, top_n=0).splitlines()) == 2

    @pytest.mark.parametrize("top_n", [-1, -10])
    def test_summary_table_rejects_negative_limit(self, top_n):
        """Negative display limits are rejected instead of slicing implicitly."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(ValueError, match="top_n"):
            ld_summary_table(result, top_n=top_n)

    @pytest.mark.parametrize("top_n", [None, 1.5, True])
    def test_summary_table_rejects_non_integer_limit(self, top_n):
        """Display limits reject non-integer and boolean values."""
        matrix = np.zeros((2, 2))
        result = LDResult(matrix, matrix, matrix, matrix, [], [])

        with pytest.raises(TypeError, match="top_n"):
            ld_summary_table(result, top_n=top_n)


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
