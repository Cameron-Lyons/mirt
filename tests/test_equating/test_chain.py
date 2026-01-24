"""Tests for chain linking."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.equating.chain import (
    ChainLinkingResult,
    TimePointModel,
    accumulate_constants,
    chain_link,
    chain_linking_summary,
    concurrent_link,
    detect_longitudinal_drift,
    transform_theta_to_reference,
    transform_to_reference,
)


@pytest.fixture
def linked_models():
    """Create multiple linked models for testing."""
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(42)
    n_items = 10

    disc = np.abs(rng.normal(1.0, 0.3, n_items))
    diff1 = rng.normal(0, 1, n_items)
    diff2 = diff1 + 0.5
    diff3 = diff2 + 0.5

    models = []
    for diff in [diff1, diff2, diff3]:
        model = TwoParameterLogistic(n_items=n_items)
        model._parameters = {
            "discrimination": disc.copy(),
            "difficulty": diff.copy(),
        }
        model._is_fitted = True
        model._n_factors = 1
        models.append(model)

    anchor_pairs = [
        (list(range(5)), list(range(5))),
        (list(range(5)), list(range(5))),
    ]

    return models, anchor_pairs


class TestChainLinkingResult:
    """Tests for ChainLinkingResult dataclass."""

    def test_initialization(self, linked_models):
        """Test ChainLinkingResult initialization."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert isinstance(result, ChainLinkingResult)
        assert len(result.cumulative_A) == 3
        assert len(result.cumulative_B) == 3
        assert len(result.pairwise_results) == 2

    def test_reference_index(self, linked_models):
        """Test that reference index is stored."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, reference_index=1)

        assert result.reference_index == 1


class TestChainLink:
    """Tests for chain_link function."""

    def test_basic_chain_link(self, linked_models):
        """Test basic chain linking."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert len(result.cumulative_A) == len(models)
        assert len(result.cumulative_B) == len(models)

    def test_reference_identity(self, linked_models):
        """Test that reference model has identity transformation."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, reference_index=0)

        assert result.cumulative_A[0] == pytest.approx(1.0)
        assert result.cumulative_B[0] == pytest.approx(0.0)

    def test_different_reference_indices(self, linked_models):
        """Test with different reference indices."""
        models, anchor_pairs = linked_models

        result0 = chain_link(models, anchor_pairs, reference_index=0)
        result1 = chain_link(models, anchor_pairs, reference_index=1)

        assert result0.cumulative_A[0] == pytest.approx(1.0)
        assert result1.cumulative_A[1] == pytest.approx(1.0)

    def test_invalid_anchor_pairs_length(self, linked_models):
        """Test that mismatched anchor pairs raise error."""
        models, _ = linked_models
        wrong_anchors = [(list(range(5)), list(range(5)))]

        with pytest.raises(ValueError, match="Expected .* anchor pairs"):
            chain_link(models, wrong_anchors)

    def test_invalid_reference_index(self, linked_models):
        """Test that invalid reference index raises error."""
        models, anchor_pairs = linked_models

        with pytest.raises(ValueError, match="Invalid reference_index"):
            chain_link(models, anchor_pairs, reference_index=10)

    def test_pairwise_results_computed(self, linked_models):
        """Test that pairwise results are computed."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs)

        assert len(result.pairwise_results) == 2
        for pr in result.pairwise_results:
            assert hasattr(pr, "constants")
            assert pr.constants.A is not None
            assert pr.constants.B is not None

    def test_drift_accumulation_computed(self, linked_models):
        """Test that drift accumulation is computed."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, compute_drift=True)

        assert result.drift_accumulation is not None

    def test_no_drift_accumulation(self, linked_models):
        """Test without drift computation."""
        models, anchor_pairs = linked_models

        result = chain_link(models, anchor_pairs, compute_drift=False)

        assert result.drift_accumulation is None


class TestAccumulateConstants:
    """Tests for accumulate_constants function."""

    def test_identity_at_reference(self):
        """Test that reference has identity transformation."""
        pairwise_A = [1.1, 0.9]
        pairwise_B = [0.2, -0.1]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_B[0] == pytest.approx(0.0)

    def test_accumulation_forward(self):
        """Test forward accumulation."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.0, 0.0]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_A[1] == pytest.approx(1.0)
        assert cum_A[2] == pytest.approx(1.0)

    def test_accumulation_with_shifts(self):
        """Test accumulation with non-identity transformations."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.5, 0.3]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=0)

        assert cum_A[0] == pytest.approx(1.0)
        assert cum_B[0] == pytest.approx(0.0)
        assert cum_B[1] == pytest.approx(-0.5)
        assert cum_B[2] == pytest.approx(-0.8)

    def test_middle_reference(self):
        """Test with middle model as reference."""
        pairwise_A = [1.0, 1.0]
        pairwise_B = [0.5, 0.3]

        cum_A, cum_B = accumulate_constants(pairwise_A, pairwise_B, reference_index=1)

        assert cum_A[1] == pytest.approx(1.0)
        assert cum_B[1] == pytest.approx(0.0)


class TestTransformToReference:
    """Tests for transform_to_reference function."""

    def test_basic_transformation(self, linked_models):
        """Test basic parameter transformation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)

        transformed = transform_to_reference(
            models[2], chain_result, time_index=2, in_place=False
        )

        assert transformed is not models[2]
        assert transformed.is_fitted

    def test_reference_unchanged(self, linked_models):
        """Test that reference model transformation is identity."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)

        original_diff = np.array(models[0].difficulty).copy()

        transformed = transform_to_reference(
            models[0], chain_result, time_index=0, in_place=False
        )

        assert_allclose(transformed.difficulty, original_diff, atol=0.01)


class TestTransformThetaToReference:
    """Tests for transform_theta_to_reference function."""

    def test_basic_theta_transformation(self, linked_models):
        """Test basic theta transformation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)
        theta = np.array([0.0, 1.0, -1.0])

        transformed = transform_theta_to_reference(theta, chain_result, time_index=2)

        assert transformed.shape == theta.shape

    def test_reference_theta_unchanged(self, linked_models):
        """Test that reference theta is unchanged."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, reference_index=0)
        theta = np.array([0.0, 1.0, -1.0])

        transformed = transform_theta_to_reference(theta, chain_result, time_index=0)

        assert_allclose(transformed, theta)


class TestConcurrentLink:
    """Tests for concurrent_link function."""

    def test_basic_concurrent_link(self, linked_models):
        """Test basic concurrent linking."""
        models, _ = linked_models

        anchor_matrices = [
            [[(i, i) for i in range(5)]],
            [[(i, i) for i in range(5)]],
        ]

        result = concurrent_link(models, anchor_matrices)

        assert len(result) == len(models)
        for A, B in result:
            assert isinstance(A, float)
            assert isinstance(B, float)

    def test_reference_identity(self, linked_models):
        """Test that first model has identity transformation."""
        models, _ = linked_models

        anchor_matrices = [
            [[(i, i) for i in range(5)]],
            [[(i, i) for i in range(5)]],
        ]

        result = concurrent_link(models, anchor_matrices)

        assert result[0][0] == pytest.approx(1.0)
        assert result[0][1] == pytest.approx(0.0)


class TestDetectLongitudinalDrift:
    """Tests for detect_longitudinal_drift function."""

    def test_basic_drift_detection(self, linked_models):
        """Test basic drift detection."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, compute_drift=True)

        drift_result = detect_longitudinal_drift(chain_result)

        assert "consistently_flagged" in drift_result
        assert "drift_direction" in drift_result

    def test_drift_detection_no_accumulation(self, linked_models):
        """Test drift detection without drift accumulation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs, compute_drift=False)

        drift_result = detect_longitudinal_drift(chain_result)

        assert drift_result["consistently_flagged"] == []
        assert drift_result["drift_direction"] == []


class TestChainLinkingSummary:
    """Tests for chain_linking_summary function."""

    def test_basic_summary(self, linked_models):
        """Test basic summary generation."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs)

        summary = chain_linking_summary(chain_result)

        assert isinstance(summary, str)
        assert "Chain Linking" in summary
        assert "Cumulative" in summary

    def test_summary_contains_all_time_points(self, linked_models):
        """Test that summary contains all time points."""
        models, anchor_pairs = linked_models

        chain_result = chain_link(models, anchor_pairs)

        summary = chain_linking_summary(chain_result)

        for t in range(len(models)):
            assert str(t) in summary


class TestTimePointModel:
    """Tests for TimePointModel dataclass."""

    def test_initialization(self, linked_models):
        """Test TimePointModel initialization."""
        models, _ = linked_models

        tp = TimePointModel(
            model=models[0],
            anchor_items=[0, 1, 2],
            time_label="T1",
        )

        assert tp.model is models[0]
        assert tp.anchor_items == [0, 1, 2]
        assert tp.time_label == "T1"

    def test_default_time_label(self, linked_models):
        """Test default time label."""
        models, _ = linked_models

        tp = TimePointModel(
            model=models[0],
            anchor_items=[0, 1, 2],
        )

        assert tp.time_label == ""
