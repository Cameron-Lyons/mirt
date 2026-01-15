"""Tests for polytomous model linking."""

import numpy as np
import pytest

from mirt.equating import (
    LinkingResult,
    link_gpcm,
    link_grm,
    transform_polytomous_parameters,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel


@pytest.fixture
def grm_model_pair():
    """Create a pair of GRM models for linking."""
    model_old = GradedResponseModel(n_items=5, n_categories=4)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    thresholds_old = np.array(
        [
            [-1.5, -0.5, 0.5],
            [-1.0, 0.0, 1.0],
            [-0.5, 0.5, 1.5],
            [-2.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    model_old.set_parameters(discrimination=disc_old, thresholds=thresholds_old)
    model_old._is_fitted = True

    A_true = 1.2
    B_true = 0.5
    model_new = GradedResponseModel(n_items=5, n_categories=4)
    disc_new = disc_old / A_true
    thresholds_new = A_true * thresholds_old + B_true
    model_new.set_parameters(discrimination=disc_new, thresholds=thresholds_new)
    model_new._is_fitted = True

    return model_old, model_new, A_true, B_true


@pytest.fixture
def gpcm_model_pair():
    """Create a pair of GPCM models for linking."""
    model_old = GeneralizedPartialCredit(n_items=5, n_categories=4)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    steps_old = np.array(
        [
            [-0.5, 0.0, 0.5],
            [-0.3, 0.0, 0.3],
            [-0.4, 0.1, 0.3],
            [-0.6, -0.1, 0.7],
            [-0.2, 0.1, 0.1],
        ]
    )
    model_old.set_parameters(discrimination=disc_old, steps=steps_old)
    model_old._is_fitted = True

    A_true = 1.3
    B_true = -0.2
    model_new = GeneralizedPartialCredit(n_items=5, n_categories=4)
    disc_new = disc_old / A_true
    steps_new = steps_old / A_true
    model_new.set_parameters(discrimination=disc_new, steps=steps_new)
    model_new._is_fitted = True

    return model_old, model_new, A_true, B_true


class TestLinkGRM:
    """Tests for GRM linking."""

    def test_link_grm_returns_result(self, grm_model_pair):
        """Test that link_grm returns LinkingResult."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        result = link_grm(model_old, model_new, anchors, anchors)

        assert isinstance(result, LinkingResult)

    def test_link_grm_recovers_constants(self, grm_model_pair):
        """Test that GRM linking recovers transformation constants."""
        model_old, model_new, A_true, B_true = grm_model_pair
        anchors = list(range(5))

        result = link_grm(
            model_new, model_old, anchors, anchors, method="stocking_lord"
        )

        assert abs(result.constants.A - A_true) < 0.15
        assert abs(result.constants.B - B_true) < 0.15

    def test_link_grm_methods(self, grm_model_pair):
        """Test all GRM linking methods."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        for method in ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]:
            result = link_grm(model_old, model_new, anchors, anchors, method=method)
            assert isinstance(result.constants.A, float)
            assert isinstance(result.constants.B, float)

    def test_link_grm_with_diagnostics(self, grm_model_pair):
        """Test GRM linking with diagnostics."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        result = link_grm(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        assert result.fit_statistics is not None
        assert result.anchor_diagnostics is not None


class TestLinkGPCM:
    """Tests for GPCM linking."""

    def test_link_gpcm_returns_result(self, gpcm_model_pair):
        """Test that link_gpcm returns LinkingResult."""
        model_old, model_new, _, _ = gpcm_model_pair
        anchors = list(range(5))

        result = link_gpcm(model_old, model_new, anchors, anchors)

        assert isinstance(result, LinkingResult)

    def test_link_gpcm_methods(self, gpcm_model_pair):
        """Test GPCM linking methods."""
        model_old, model_new, _, _ = gpcm_model_pair
        anchors = list(range(5))

        for method in ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]:
            result = link_gpcm(model_old, model_new, anchors, anchors, method=method)
            assert isinstance(result.constants.A, float)
            assert isinstance(result.constants.B, float)


class TestTransformPolytomous:
    """Tests for polytomous parameter transformation."""

    def test_transform_grm_creates_copy(self, grm_model_pair):
        """Test that transformation creates copy by default."""
        model_old, _, A_true, B_true = grm_model_pair

        transformed = transform_polytomous_parameters(
            model_old, A_true, B_true, model_type="grm", in_place=False
        )

        assert transformed is not model_old

    def test_transform_grm_in_place(self, grm_model_pair):
        """Test in-place transformation."""
        model_old, _, A_true, B_true = grm_model_pair
        model_copy = model_old.copy()

        transformed = transform_polytomous_parameters(
            model_copy, A_true, B_true, model_type="grm", in_place=True
        )

        assert transformed is model_copy

    def test_transform_grm_formula(self, grm_model_pair):
        """Test GRM transformation formulas."""
        model_old, _, _, _ = grm_model_pair
        A, B = 1.5, 0.3

        original_disc = np.asarray(model_old.discrimination).copy()
        original_thresh = model_old.parameters["thresholds"].copy()

        transformed = transform_polytomous_parameters(
            model_old, A, B, model_type="grm", in_place=False
        )

        expected_disc = original_disc / A
        expected_thresh = A * original_thresh + B

        np.testing.assert_allclose(
            np.asarray(transformed.discrimination), expected_disc, rtol=1e-10
        )
        np.testing.assert_allclose(
            transformed.parameters["thresholds"], expected_thresh, rtol=1e-10
        )


class TestPolytomousValidation:
    """Validation tests for polytomous linking."""

    def test_mismatched_anchors(self, grm_model_pair):
        """Test error on mismatched anchor lengths."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="same length"):
            link_grm(model_old, model_new, [0, 1, 2], [0, 1])

    def test_invalid_method(self, grm_model_pair):
        """Test error on invalid method."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="Unknown method"):
            link_grm(model_old, model_new, [0, 1], [0, 1], method="invalid")
