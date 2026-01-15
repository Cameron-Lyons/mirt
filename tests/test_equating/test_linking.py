"""Tests for core linking functions."""

import numpy as np
import pytest

from mirt.equating import (
    AnchorDiagnostics,
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
    link,
    transform_parameters,
)
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.fixture
def reference_model():
    """Create a reference 2PL model with known parameters."""
    model = TwoParameterLogistic(n_items=10)
    disc = np.array([1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0, 1.4, 0.7])
    diff = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, 0.8])
    model.set_parameters(discrimination=disc, difficulty=diff)
    model._is_fitted = True
    return model


@pytest.fixture
def scaled_model(reference_model):
    """Create a model with known linear transformation of reference."""
    model = TwoParameterLogistic(n_items=10)
    A_true = 1.2
    B_true = 0.5
    disc = np.asarray(reference_model.discrimination) / A_true
    diff = A_true * np.asarray(reference_model.difficulty) + B_true
    model.set_parameters(discrimination=disc, difficulty=diff)
    model._is_fitted = True
    return model, A_true, B_true


class TestLinkingBasic:
    """Basic linking functionality tests."""

    def test_link_returns_result(self, reference_model, scaled_model):
        """Test that link returns a LinkingResult."""
        new_model, _, _ = scaled_model
        anchors = list(range(5))

        result = link(reference_model, new_model, anchors, anchors)

        assert isinstance(result, LinkingResult)
        assert isinstance(result.constants, LinkingConstants)
        assert result.anchor_items == anchors

    def test_link_recovers_constants(self, reference_model, scaled_model):
        """Test that linking recovers true A and B constants."""
        new_model, A_true, B_true = scaled_model
        anchors = list(range(10))

        result = link(
            new_model, reference_model, anchors, anchors, method="stocking_lord"
        )

        assert abs(result.constants.A - A_true) < 0.1
        assert abs(result.constants.B - B_true) < 0.1

    def test_link_methods(self, reference_model, scaled_model):
        """Test that all linking methods work."""
        new_model, A_true, B_true = scaled_model
        anchors = list(range(10))

        methods = [
            "mean_sigma",
            "mean_mean",
            "stocking_lord",
            "haebara",
            "bisector",
            "orthogonal",
        ]

        for method in methods:
            result = link(reference_model, new_model, anchors, anchors, method=method)

            assert isinstance(result.constants.A, float)
            assert isinstance(result.constants.B, float)
            assert result.constants.method == method

    def test_link_with_diagnostics(self, reference_model, scaled_model):
        """Test that diagnostics are computed when requested."""
        new_model, _, _ = scaled_model
        anchors = list(range(5))

        result = link(
            reference_model, new_model, anchors, anchors, compute_diagnostics=True
        )

        assert result.fit_statistics is not None
        assert isinstance(result.fit_statistics, LinkingFitStatistics)
        assert result.anchor_diagnostics is not None
        assert isinstance(result.anchor_diagnostics, AnchorDiagnostics)

    def test_link_without_diagnostics(self, reference_model, scaled_model):
        """Test that diagnostics can be disabled."""
        new_model, _, _ = scaled_model
        anchors = list(range(5))

        result = link(
            reference_model, new_model, anchors, anchors, compute_diagnostics=False
        )

        assert result.fit_statistics is None
        assert result.anchor_diagnostics is None


class TestLinkingValidation:
    """Validation tests for linking inputs."""

    def test_link_requires_matching_anchors(self, reference_model, scaled_model):
        """Test that anchor lists must have same length."""
        new_model, _, _ = scaled_model

        with pytest.raises(ValueError, match="same length"):
            link(reference_model, new_model, [0, 1, 2], [0, 1])

    def test_link_requires_min_anchors(self, reference_model, scaled_model):
        """Test that at least 2 anchors are required."""
        new_model, _, _ = scaled_model

        with pytest.raises(ValueError, match="At least 2"):
            link(reference_model, new_model, [0], [0])

    def test_link_invalid_method(self, reference_model, scaled_model):
        """Test that invalid method raises error."""
        new_model, _, _ = scaled_model

        with pytest.raises(ValueError, match="Unknown linking method"):
            link(reference_model, new_model, [0, 1], [0, 1], method="invalid")


class TestTransformParameters:
    """Tests for parameter transformation."""

    def test_transform_creates_copy(self, reference_model):
        """Test that transform creates a copy by default."""
        A, B = 1.2, 0.5

        transformed = transform_parameters(reference_model, A, B, in_place=False)

        assert transformed is not reference_model
        assert not np.allclose(
            np.asarray(transformed.discrimination),
            np.asarray(reference_model.discrimination),
        )

    def test_transform_in_place(self, reference_model):
        """Test in-place transformation."""
        A, B = 1.2, 0.5
        original_disc = np.asarray(reference_model.discrimination).copy()

        transformed = transform_parameters(reference_model, A, B, in_place=True)

        assert transformed is reference_model
        assert not np.allclose(np.asarray(transformed.discrimination), original_disc)

    def test_transform_formulas(self, reference_model):
        """Test that transformation formulas are correct."""
        A, B = 1.5, -0.3
        original_disc = np.asarray(reference_model.discrimination).copy()
        original_diff = np.asarray(reference_model.difficulty).copy()

        transformed = transform_parameters(reference_model, A, B, in_place=False)

        expected_disc = original_disc / A
        expected_diff = A * original_diff + B

        np.testing.assert_allclose(
            np.asarray(transformed.discrimination), expected_disc, rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(transformed.difficulty), expected_diff, rtol=1e-10
        )


class TestLinkingRobust:
    """Tests for robust linking options."""

    def test_link_robust_option(self, reference_model, scaled_model):
        """Test that robust linking uses median instead of mean."""
        new_model, _, _ = scaled_model
        anchors = list(range(10))

        result_robust = link(
            reference_model,
            new_model,
            anchors,
            anchors,
            method="mean_sigma",
            robust=True,
        )
        result_normal = link(
            reference_model,
            new_model,
            anchors,
            anchors,
            method="mean_sigma",
            robust=False,
        )

        assert result_robust.constants.A != result_normal.constants.A

    def test_link_with_bootstrap_se(self, reference_model, scaled_model):
        """Test bootstrap standard error computation."""
        new_model, _, _ = scaled_model
        anchors = list(range(10))

        result = link(
            reference_model,
            new_model,
            anchors,
            anchors,
            compute_se=True,
            n_bootstrap=50,
        )

        assert result.constants.A_se is not None
        assert result.constants.B_se is not None
        assert result.constants.A_se > 0
        assert result.constants.B_se > 0


class TestLinkingFitStatistics:
    """Tests for fit statistics computation."""

    def test_fit_statistics_values(self, reference_model, scaled_model):
        """Test that fit statistics have reasonable values."""
        new_model, _, _ = scaled_model
        anchors = list(range(10))

        result = link(
            reference_model, new_model, anchors, anchors, compute_diagnostics=True
        )

        assert result.fit_statistics is not None
        assert result.fit_statistics.rmse_a >= 0
        assert result.fit_statistics.rmse_b >= 0
        assert result.fit_statistics.mad_a >= 0
        assert result.fit_statistics.mad_b >= 0
        assert result.fit_statistics.tcc_rmse >= 0

    def test_perfect_linking_has_zero_rmse(self, reference_model):
        """Test that linking identical models gives near-zero RMSE."""
        anchors = list(range(10))

        result = link(
            reference_model, reference_model, anchors, anchors, compute_diagnostics=True
        )

        assert result.fit_statistics is not None
        assert result.fit_statistics.rmse_a < 0.01
        assert result.fit_statistics.rmse_b < 0.01


class TestAnchorDiagnostics:
    """Tests for anchor item diagnostics."""

    def test_diagnostics_arrays(self, reference_model, scaled_model):
        """Test that diagnostics arrays have correct shapes."""
        new_model, _, _ = scaled_model
        anchors = list(range(5))

        result = link(
            reference_model, new_model, anchors, anchors, compute_diagnostics=True
        )

        assert result.anchor_diagnostics is not None
        assert len(result.anchor_diagnostics.signed_diff_a) == 5
        assert len(result.anchor_diagnostics.signed_diff_b) == 5
        assert len(result.anchor_diagnostics.area_diff) == 5
        assert len(result.anchor_diagnostics.robust_z) == 5
        assert len(result.anchor_diagnostics.flagged) == 5

    def test_no_drift_no_flags(self, reference_model, scaled_model):
        """Test that well-behaved anchors are not flagged."""
        new_model, _, _ = scaled_model
        anchors = list(range(10))

        result = link(
            reference_model, new_model, anchors, anchors, compute_diagnostics=True
        )

        assert result.anchor_diagnostics is not None
        assert np.sum(result.anchor_diagnostics.flagged) == 0
