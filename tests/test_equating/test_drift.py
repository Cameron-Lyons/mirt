"""Tests for drift detection functions."""

import numpy as np
import pytest

from mirt.equating import (
    DriftResult,
    detect_drift,
    purify_anchors,
    signed_area_difference,
)
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.fixture
def stable_models():
    """Create two models with no drift."""
    model_old = TwoParameterLogistic(n_items=10)
    disc = np.array([1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0, 1.4, 0.7])
    diff = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, 0.8])
    model_old.set_parameters(discrimination=disc, difficulty=diff)
    model_old._is_fitted = True

    model_new = TwoParameterLogistic(n_items=10)
    model_new.set_parameters(discrimination=disc.copy(), difficulty=diff.copy())
    model_new._is_fitted = True

    return model_old, model_new


@pytest.fixture
def drifting_models():
    """Create models with intentional drift on some items."""
    model_old = TwoParameterLogistic(n_items=10)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0, 1.4, 0.7])
    diff_old = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.8, 0.3, 0.8])
    model_old.set_parameters(discrimination=disc_old, difficulty=diff_old)
    model_old._is_fitted = True

    model_new = TwoParameterLogistic(n_items=10)
    disc_new = disc_old.copy()
    diff_new = diff_old.copy()
    disc_new[0] = 1.8
    diff_new[0] = -0.5
    disc_new[5] = 1.5
    diff_new[5] = 2.0
    model_new.set_parameters(discrimination=disc_new, difficulty=diff_new)
    model_new._is_fitted = True

    return model_old, model_new


class TestDetectDrift:
    """Tests for detect_drift function."""

    def test_detect_drift_returns_result(self, stable_models):
        """Test that detect_drift returns DriftResult."""
        model_old, model_new = stable_models
        anchors = list(range(10))

        result = detect_drift(model_old, model_new, anchors, anchors)

        assert isinstance(result, DriftResult)

    def test_detect_drift_methods(self, stable_models):
        """Test all drift detection methods."""
        model_old, model_new = stable_models
        anchors = list(range(10))

        for method in ["robust_z", "3sigma", "area"]:
            result = detect_drift(model_old, model_new, anchors, anchors, method=method)

            assert result.method == method
            assert len(result.drift_statistics) == 10

    def test_no_drift_detected_stable(self, stable_models):
        """Test that identical models show no drift."""
        model_old, model_new = stable_models
        anchors = list(range(10))

        result = detect_drift(model_old, model_new, anchors, anchors, method="robust_z")

        assert len(result.flagged_items) == 0

    def test_drift_detected_drifting(self, drifting_models):
        """Test that drifting items are flagged."""
        model_old, model_new = drifting_models
        anchors = list(range(10))

        result = detect_drift(
            model_old, model_new, anchors, anchors, method="area", threshold=0.1
        )

        assert len(result.flagged_items) > 0
        assert 0 in result.flagged_items or 5 in result.flagged_items

    def test_drift_with_transformation(self, stable_models):
        """Test drift detection with explicit A and B."""
        model_old, model_new = stable_models
        anchors = list(range(10))

        result = detect_drift(model_old, model_new, anchors, anchors, A=1.0, B=0.0)

        assert isinstance(result, DriftResult)

    def test_drift_effect_sizes(self, drifting_models):
        """Test that effect sizes are computed."""
        model_old, model_new = drifting_models
        anchors = list(range(10))

        result = detect_drift(model_old, model_new, anchors, anchors)

        assert result.effect_sizes is not None
        assert len(result.effect_sizes) == 10
        assert all(es >= 0 for es in result.effect_sizes)


class TestPurifyAnchors:
    """Tests for anchor purification."""

    def test_purify_returns_tuple(self, drifting_models):
        """Test that purify_anchors returns correct tuple."""
        model_old, model_new = drifting_models
        anchors = list(range(10))

        result = purify_anchors(model_old, model_new, anchors, anchors)

        assert isinstance(result, tuple)
        assert len(result) == 3
        purified_old, purified_new, removed = result
        assert isinstance(purified_old, list)
        assert isinstance(purified_new, list)
        assert isinstance(removed, list)

    def test_purify_removes_drifting(self, drifting_models):
        """Test that drifting items are removed."""
        model_old, model_new = drifting_models
        anchors = list(range(10))

        purified_old, purified_new, removed = purify_anchors(
            model_old, model_new, anchors, anchors, threshold=1.5
        )

        assert len(purified_old) < len(anchors)
        assert len(removed) > 0

    def test_purify_keeps_stable(self, stable_models):
        """Test that purification keeps all items when no drift."""
        model_old, model_new = stable_models
        anchors = list(range(10))

        purified_old, purified_new, removed = purify_anchors(
            model_old, model_new, anchors, anchors
        )

        assert len(purified_old) == len(anchors)
        assert len(removed) == 0

    def test_purify_min_anchors(self, drifting_models):
        """Test minimum anchor constraint."""
        model_old, model_new = drifting_models
        anchors = list(range(10))

        purified_old, _, _ = purify_anchors(
            model_old, model_new, anchors, anchors, min_anchors=5, threshold=0.5
        )

        assert len(purified_old) >= 5


class TestSignedAreaDifference:
    """Tests for signed area computation."""

    def test_signed_area_returns_tuple(self, stable_models):
        """Test that signed_area_difference returns tuple."""
        model_old, model_new = stable_models

        result = signed_area_difference(model_old, model_new, 0, 0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_identical_items_zero_area(self, stable_models):
        """Test that identical items have zero area difference."""
        model_old, model_new = stable_models

        signed, unsigned = signed_area_difference(model_old, model_new, 0, 0)

        assert abs(signed) < 0.01
        assert abs(unsigned) < 0.01

    def test_drifting_items_nonzero_area(self, drifting_models):
        """Test that drifting items have nonzero area."""
        model_old, model_new = drifting_models

        signed, unsigned = signed_area_difference(model_old, model_new, 0, 0)

        assert unsigned > 0.1

    def test_unsigned_always_positive(self, drifting_models):
        """Test that unsigned area is always non-negative."""
        model_old, model_new = drifting_models

        for i in range(10):
            _, unsigned = signed_area_difference(model_old, model_new, i, i)
            assert unsigned >= 0


class TestDriftValidation:
    """Validation tests for drift functions."""

    def test_detect_drift_mismatched_anchors(self, stable_models):
        """Test error on mismatched anchor lengths."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match="same length"):
            detect_drift(model_old, model_new, [0, 1, 2], [0, 1])

    def test_detect_drift_invalid_method(self, stable_models):
        """Test error on invalid method."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match="Unknown"):
            detect_drift(model_old, model_new, [0, 1], [0, 1], method="invalid")

    def test_wald_requires_se(self, stable_models):
        """Test that Wald method requires standard errors."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match="Standard errors required"):
            detect_drift(model_old, model_new, [0, 1], [0, 1], method="wald")
