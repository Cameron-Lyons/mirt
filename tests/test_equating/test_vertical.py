"""Tests for vertical scaling module."""

import numpy as np
import pytest

import mirt.equating.vertical as vertical_module
from mirt.equating.linking import transform_parameters
from mirt.equating.vertical import (
    GradeData,
    VerticalScaleDiagnostics,
    VerticalScaleResult,
    _enforce_monotonicity,
    _GradeModelInfo,
    compute_vertical_diagnostics,
    vertical_scale,
    vertical_scale_summary,
)
from mirt.models.dichotomous import TwoParameterLogistic


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

    def test_concurrent_respects_linking_method(self, two_grade_data):
        """Concurrent scaling must not silently replace the requested linker."""
        result = vertical_scale(
            two_grade_data,
            method="concurrent",
            linking_method="haebara",
        )

        assert result.linking_results[0].constants.method == "haebara"

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

    def test_chain_maps_upper_grade_to_reference_scale(self, monkeypatch):
        """Pairwise constants must transform upper-grade scores downward."""
        lower_model = TwoParameterLogistic(n_items=6)
        lower_model.set_parameters(
            discrimination=np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.8]),
            difficulty=np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.7]),
        )
        scale, shift = 1.4, 0.6
        upper_model = transform_parameters(lower_model, scale, shift)
        theta_lower = np.array([[0.0], [1.0], [2.0]])
        theta_upper = scale * theta_lower + shift
        grade_data = [
            GradeData(
                "lower",
                np.zeros((3, 6), dtype=int),
                anchor_items_above=[2, 3, 4, 5],
            ),
            GradeData("upper", np.zeros((3, 6), dtype=int)),
        ]
        grade_models = [
            _GradeModelInfo(lower_model, theta_lower, "lower"),
            _GradeModelInfo(upper_model, theta_upper, "upper"),
        ]
        monkeypatch.setattr(
            vertical_module,
            "_fit_grade_models",
            lambda grade_data, models: grade_models,
        )

        result = vertical_scale(
            grade_data,
            models=[lower_model, upper_model],
            enforce_monotonicity=False,
        )

        np.testing.assert_allclose(
            result.grade_transformations["upper"],
            (1.0 / scale, -shift / scale),
            atol=1e-6,
        )
        assert result.grade_means["upper"] == pytest.approx(1.0, abs=1e-6)
        assert result.grade_means["upper"] == pytest.approx(
            result.grade_means["lower"], abs=1e-6
        )
        assert result.linking_results[0].anchor_items == [2, 3, 4, 5]

    def test_chain_composes_around_middle_reference(self, monkeypatch):
        """Multi-grade transforms compose exactly around any reference form."""
        model_0 = TwoParameterLogistic(n_items=5)
        model_0.set_parameters(
            discrimination=np.array([0.7, 0.9, 1.1, 1.4, 1.8]),
            difficulty=np.array([-1.4, -0.6, 0.0, 0.8, 1.5]),
        )
        scale_01, shift_01 = 1.4, 0.6
        scale_12, shift_12 = 0.8, -0.3
        model_1 = transform_parameters(model_0, scale_01, shift_01)
        model_2 = transform_parameters(model_1, scale_12, shift_12)
        theta_0 = np.array([[0.0], [1.0], [2.0]])
        theta_1 = scale_01 * theta_0 + shift_01
        theta_2 = scale_12 * theta_1 + shift_12
        anchors = list(range(5))
        grade_data = [
            GradeData(
                "g0",
                np.zeros((3, 5), dtype=int),
                anchor_items_above=anchors,
            ),
            GradeData(
                "g1",
                np.zeros((3, 5), dtype=int),
                anchor_items_below=anchors,
                anchor_items_above=anchors,
            ),
            GradeData(
                "g2",
                np.zeros((3, 5), dtype=int),
                anchor_items_below=anchors,
            ),
        ]
        grade_models = [
            _GradeModelInfo(model_0, theta_0, "g0"),
            _GradeModelInfo(model_1, theta_1, "g1"),
            _GradeModelInfo(model_2, theta_2, "g2"),
        ]
        monkeypatch.setattr(
            vertical_module,
            "_fit_grade_models",
            lambda grade_data, models: grade_models,
        )

        result = vertical_scale(
            grade_data,
            models=[model_0, model_1, model_2],
            reference_grade=1,
            enforce_monotonicity=False,
        )

        np.testing.assert_allclose(
            result.grade_transformations["g0"],
            (scale_01, shift_01),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.grade_transformations["g1"], (1.0, 0.0), atol=1e-6
        )
        np.testing.assert_allclose(
            result.grade_transformations["g2"],
            (1.0 / scale_12, -shift_12 / scale_12),
            atol=1e-6,
        )
        np.testing.assert_allclose(list(result.grade_means.values()), 2.0, atol=1e-6)

    def test_monotonicity_preserves_reference_and_scale(self):
        """Growth correction shifts locations without rescaling abilities."""
        labels = ["g1", "g2", "g3"]
        grade_data = [GradeData(label, np.zeros((2, 2), dtype=int)) for label in labels]
        result = VerticalScaleResult(
            grade_transformations={
                "g1": (0.8, 0.2),
                "g2": (1.0, 0.0),
                "g3": (1.2, -0.1),
            },
            grade_means={"g1": 3.0, "g2": 2.0, "g3": 1.0},
            grade_sds={"g1": 0.8, "g2": 1.0, "g3": 1.2},
            linking_results=[],
            monotonicity_violations=[],
            growth_curve=np.array([3.0, 2.0, 1.0]),
            method="chain",
            reference_grade=1,
        )

        adjusted = _enforce_monotonicity(result, grade_data, reference_grade=1)

        means = np.array([adjusted.grade_means[label] for label in labels])
        assert np.all(np.diff(means) > 0.0)
        assert adjusted.grade_transformations["g2"] == (1.0, 0.0)
        for label in labels:
            assert adjusted.grade_transformations[label][0] == pytest.approx(
                result.grade_transformations[label][0]
            )
        assert adjusted.grade_sds == result.grade_sds
        assert adjusted.monotonicity_violations == [("g1", "g2"), ("g2", "g3")]


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

    @pytest.mark.parametrize("reference_grade", [-1, 2, 1.5])
    def test_reference_grade_must_be_valid(self, two_grade_data, reference_grade):
        """Reference indices are validated before fitting begins."""
        with pytest.raises(ValueError, match="reference_grade"):
            vertical_scale(two_grade_data, reference_grade=reference_grade)

    def test_grade_labels_must_be_unique(self, two_grade_data):
        """Duplicate labels would silently overwrite result dictionaries."""
        two_grade_data[1].grade_label = two_grade_data[0].grade_label

        with pytest.raises(ValueError, match="labels must be unique"):
            vertical_scale(two_grade_data)

    def test_models_must_match_grades_and_response_width(self, two_grade_data):
        """A partial or dimensionally incompatible model list is rejected."""
        model = TwoParameterLogistic(n_items=15)
        wrong_width = TwoParameterLogistic(n_items=14)

        with pytest.raises(ValueError, match="one model per grade"):
            vertical_scale(two_grade_data, models=[model])
        with pytest.raises(ValueError, match="responses have 15"):
            vertical_scale(two_grade_data, models=[model, wrong_width])

    @pytest.mark.parametrize(
        ("anchors", "message"),
        [([0], "At least 2"), ([0, 0], "unique"), ([-1, 1], "out of range")],
    )
    def test_anchor_indices_are_validated(self, two_grade_data, anchors, message):
        """Invalid anchor mappings fail before model calibration."""
        two_grade_data[0].anchor_items_above = anchors
        two_grade_data[1].anchor_items_below = anchors

        with pytest.raises(ValueError, match=message):
            vertical_scale(two_grade_data)

    def test_invalid_linking_method_fails_before_fitting(self, two_grade_data):
        """Linking configuration is checked at the public boundary."""
        with pytest.raises(ValueError, match="Unknown linking method"):
            vertical_scale(two_grade_data, linking_method="invalid")


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

    def test_cumulative_growth_uses_selected_reference(self):
        """Cumulative growth is centered on the result's reference grade."""
        labels = ["g1", "g2", "g3"]
        grade_data = [GradeData(label, np.zeros((2, 2), dtype=int)) for label in labels]
        result = VerticalScaleResult(
            grade_transformations={label: (1.0, 0.0) for label in labels},
            grade_means={"g1": 1.0, "g2": 3.0, "g3": 5.0},
            grade_sds={label: 1.0 for label in labels},
            linking_results=[],
            monotonicity_violations=[],
            growth_curve=np.array([1.0, 3.0, 5.0]),
            method="chain",
            reference_grade=1,
        )

        diagnostics = compute_vertical_diagnostics(result, grade_data)

        np.testing.assert_allclose(diagnostics.cumulative_growth, [-2.0, 0.0, 2.0])


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
        assert "Reference grade: Grade 3" in summary


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
