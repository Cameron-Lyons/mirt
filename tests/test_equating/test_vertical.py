"""Tests for vertical scaling module."""

import numpy as np
import pytest

from mirt.equating.vertical import (
    GradeData,
    VerticalScaleDiagnostics,
    VerticalScaleResult,
    compute_vertical_diagnostics,
    vertical_scale,
    vertical_scale_summary,
)


def generate_grade_responses(
    n_persons: int,
    n_items: int,
    mean_theta: float,
    difficulty_range: tuple[float, float],
    seed: int,
) -> np.ndarray:
    """Generate synthetic IRT responses for a grade level."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean_theta, 1.0, n_persons)
    difficulty = rng.uniform(difficulty_range[0], difficulty_range[1], n_items)
    discrimination = rng.uniform(0.8, 1.5, n_items)

    responses = np.zeros((n_persons, n_items), dtype=np.int_)
    for j in range(n_items):
        prob = 1 / (1 + np.exp(-discrimination[j] * (theta - difficulty[j])))
        responses[:, j] = (rng.random(n_persons) < prob).astype(int)

    return responses


@pytest.fixture
def two_grade_data():
    """Generate data for two adjacent grades."""
    n_persons = 200
    n_items = 15

    responses_g3 = generate_grade_responses(
        n_persons, n_items, mean_theta=0.0, difficulty_range=(-1, 1), seed=42
    )
    responses_g4 = generate_grade_responses(
        n_persons, n_items, mean_theta=0.5, difficulty_range=(-0.5, 1.5), seed=43
    )

    grade_data = [
        GradeData(
            grade_label="Grade 3",
            responses=responses_g3,
            anchor_items_above=[0, 1, 2, 3, 4],
        ),
        GradeData(
            grade_label="Grade 4",
            responses=responses_g4,
            anchor_items_below=[0, 1, 2, 3, 4],
        ),
    ]

    return grade_data


@pytest.fixture
def three_grade_data():
    """Generate data for three adjacent grades."""
    n_persons = 150
    n_items = 12

    responses_g3 = generate_grade_responses(
        n_persons, n_items, mean_theta=-0.3, difficulty_range=(-1.5, 0.5), seed=100
    )
    responses_g4 = generate_grade_responses(
        n_persons, n_items, mean_theta=0.3, difficulty_range=(-1, 1), seed=101
    )
    responses_g5 = generate_grade_responses(
        n_persons, n_items, mean_theta=0.9, difficulty_range=(-0.5, 1.5), seed=102
    )

    grade_data = [
        GradeData(
            grade_label="Grade 3",
            responses=responses_g3,
            anchor_items_above=[0, 1, 2, 3],
        ),
        GradeData(
            grade_label="Grade 4",
            responses=responses_g4,
            anchor_items_below=[0, 1, 2, 3],
            anchor_items_above=[8, 9, 10, 11],
        ),
        GradeData(
            grade_label="Grade 5",
            responses=responses_g5,
            anchor_items_below=[8, 9, 10, 11],
        ),
    ]

    return grade_data


class TestVerticalScale:
    """Tests for vertical_scale function."""

    def test_vertical_scale_returns_result(self, two_grade_data):
        result = vertical_scale(two_grade_data)

        assert isinstance(result, VerticalScaleResult)
        assert len(result.grade_transformations) == 2
        assert len(result.grade_means) == 2
        assert len(result.grade_sds) == 2
        assert len(result.linking_results) == 1
        assert result.method == "chain"

    def test_vertical_scale_methods(self, two_grade_data):
        for method in ["chain", "concurrent", "fixed_anchor", "floating_anchor"]:
            result = vertical_scale(two_grade_data, method=method)

            assert result is not None
            assert result.method == method
            assert len(result.grade_means) == 2

    def test_vertical_scale_reference_grade(self, three_grade_data):
        result_ref0 = vertical_scale(three_grade_data, reference_grade=0)
        result_ref1 = vertical_scale(three_grade_data, reference_grade=1)

        assert result_ref0.grade_transformations["Grade 3"][0] == pytest.approx(1.0)
        assert result_ref0.grade_transformations["Grade 3"][1] == pytest.approx(0.0)

        assert result_ref1.grade_transformations["Grade 4"][0] == pytest.approx(1.0)
        assert result_ref1.grade_transformations["Grade 4"][1] == pytest.approx(0.0)

    def test_monotonicity_check(self, two_grade_data):
        result = vertical_scale(two_grade_data, enforce_monotonicity=True)

        means = list(result.grade_means.values())
        for i in range(len(means) - 1):
            assert means[i + 1] > means[i], "Grade means should be increasing"

    def test_monotonicity_disabled(self, two_grade_data):
        result = vertical_scale(two_grade_data, enforce_monotonicity=False)

        assert result is not None
        assert len(result.grade_means) == 2

    def test_growth_curve_matches_means(self, three_grade_data):
        result = vertical_scale(three_grade_data)

        labels = ["Grade 3", "Grade 4", "Grade 5"]
        expected_curve = [result.grade_means[label] for label in labels]

        np.testing.assert_allclose(result.growth_curve, expected_curve)

    def test_linking_results_populated(self, three_grade_data):
        result = vertical_scale(three_grade_data)

        assert len(result.linking_results) == 2

        for link_result in result.linking_results:
            assert link_result.constants.A != 0
            assert link_result.fit_statistics is not None


class TestVerticalScaleValidation:
    """Tests for validation and error handling."""

    def test_requires_two_grades(self):
        responses = generate_grade_responses(100, 10, 0.0, (-1, 1), 42)
        grade_data = [
            GradeData(grade_label="Grade 3", responses=responses),
        ]

        with pytest.raises(ValueError, match="at least 2 grades"):
            vertical_scale(grade_data)

    def test_anchor_mismatch_raises(self):
        responses_g3 = generate_grade_responses(100, 10, 0.0, (-1, 1), 42)
        responses_g4 = generate_grade_responses(100, 10, 0.5, (-0.5, 1.5), 43)

        grade_data = [
            GradeData(
                grade_label="Grade 3",
                responses=responses_g3,
                anchor_items_above=[0, 1, 2],
            ),
            GradeData(
                grade_label="Grade 4",
                responses=responses_g4,
                anchor_items_below=[0, 1, 2, 3, 4],
            ),
        ]

        with pytest.raises(ValueError, match="Anchor item count mismatch"):
            vertical_scale(grade_data)

    def test_no_anchors_raises(self):
        responses_g3 = generate_grade_responses(100, 10, 0.0, (-1, 1), 42)
        responses_g4 = generate_grade_responses(100, 10, 0.5, (-0.5, 1.5), 43)

        grade_data = [
            GradeData(
                grade_label="Grade 3",
                responses=responses_g3,
            ),
            GradeData(
                grade_label="Grade 4",
                responses=responses_g4,
            ),
        ]

        with pytest.raises(ValueError, match="No anchor items connecting"):
            vertical_scale(grade_data)

    def test_invalid_method_raises(self, two_grade_data):
        with pytest.raises(ValueError, match="Unknown vertical scaling method"):
            vertical_scale(two_grade_data, method="invalid_method")


class TestComputeVerticalDiagnostics:
    """Tests for compute_vertical_diagnostics function."""

    def test_compute_diagnostics(self, three_grade_data):
        result = vertical_scale(three_grade_data)
        diagnostics = compute_vertical_diagnostics(result, three_grade_data)

        assert isinstance(diagnostics, VerticalScaleDiagnostics)
        assert len(diagnostics.grade_separation) == 2
        assert len(diagnostics.growth_per_grade) == 2
        assert len(diagnostics.cumulative_growth) == 3
        assert len(diagnostics.anchor_stability) == 2

    def test_cumulative_growth_starts_at_zero(self, three_grade_data):
        result = vertical_scale(three_grade_data)
        diagnostics = compute_vertical_diagnostics(result, three_grade_data)

        assert diagnostics.cumulative_growth[0] == pytest.approx(0.0)

    def test_grade_separation_positive(self, three_grade_data):
        result = vertical_scale(three_grade_data, enforce_monotonicity=True)
        diagnostics = compute_vertical_diagnostics(result, three_grade_data)

        assert np.all(diagnostics.grade_separation >= 0)


class TestVerticalScaleSummary:
    """Tests for vertical_scale_summary function."""

    def test_summary_returns_string(self, two_grade_data):
        result = vertical_scale(two_grade_data)
        summary = vertical_scale_summary(result)

        assert isinstance(summary, str)
        assert "Vertical Scaling Summary" in summary
        assert "Grade 3" in summary
        assert "Grade 4" in summary
        assert "Mean" in summary

    def test_summary_includes_method(self, two_grade_data):
        result = vertical_scale(two_grade_data, method="chain")
        summary = vertical_scale_summary(result)

        assert "chain" in summary


class TestVerticalScaleEdgeCases:
    """Test edge cases for vertical scaling."""

    def test_two_grade_case(self, two_grade_data):
        result = vertical_scale(two_grade_data)

        assert len(result.grade_means) == 2
        assert len(result.linking_results) == 1

    def test_different_linking_methods(self, two_grade_data):
        for link_method in ["stocking_lord", "mean_sigma", "haebara"]:
            result = vertical_scale(two_grade_data, linking_method=link_method)

            assert result is not None
            assert len(result.grade_means) == 2

    def test_small_sample_size(self):
        n_persons = 50
        n_items = 8

        responses_g3 = generate_grade_responses(
            n_persons, n_items, 0.0, (-1, 1), seed=888
        )
        responses_g4 = generate_grade_responses(
            n_persons, n_items, 0.5, (-0.5, 1.5), seed=889
        )

        grade_data = [
            GradeData(
                grade_label="G3",
                responses=responses_g3,
                anchor_items_above=[0, 1, 2],
            ),
            GradeData(
                grade_label="G4",
                responses=responses_g4,
                anchor_items_below=[0, 1, 2],
            ),
        ]

        result = vertical_scale(grade_data)

        assert result is not None
        assert len(result.grade_means) == 2

    def test_numeric_grade_labels(self):
        n_persons = 100
        n_items = 10

        responses_3 = generate_grade_responses(
            n_persons, n_items, 0.0, (-1, 1), seed=999
        )
        responses_4 = generate_grade_responses(
            n_persons, n_items, 0.5, (-0.5, 1.5), seed=1000
        )

        grade_data = [
            GradeData(
                grade_label=3,
                responses=responses_3,
                anchor_items_above=[0, 1, 2],
            ),
            GradeData(
                grade_label=4,
                responses=responses_4,
                anchor_items_below=[0, 1, 2],
            ),
        ]

        result = vertical_scale(grade_data)

        assert 3 in result.grade_means
        assert 4 in result.grade_means
