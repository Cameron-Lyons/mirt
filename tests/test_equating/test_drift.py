"""Tests for drift detection functions."""

import numpy as np
import pytest

from mirt.equating import (
    DriftResult,
    detect_drift,
    purify_anchors,
    signed_area_difference,
)
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


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

    def test_robust_z_flags_outlier_when_mad_is_zero(self):
        """An isolated change must not disappear when most distances are tied."""
        model_old = TwoParameterLogistic(n_items=5)
        model_new = TwoParameterLogistic(n_items=5)
        model_old.set_parameters(discrimination=np.ones(5), difficulty=np.zeros(5))
        model_new.set_parameters(
            discrimination=np.ones(5),
            difficulty=np.array([0.0, 0.0, 0.0, 0.0, 2.0]),
        )

        result = detect_drift(
            model_old,
            model_new,
            list(range(5)),
            list(range(5)),
            method="robust_z",
            A=1.0,
            B=0.0,
        )

        assert result.flagged_items == [4]
        assert np.isinf(result.drift_statistics[4])
        assert result.p_values is not None
        assert result.p_values[4] == 0.0

    def test_area_uses_three_pl_guessing_parameter(self):
        """Area drift must use the actual model curve, including guessing."""
        model_old = ThreeParameterLogistic(n_items=3)
        model_new = ThreeParameterLogistic(n_items=3)
        common = {
            "discrimination": np.ones(3),
            "difficulty": np.zeros(3),
        }
        model_old.set_parameters(**common, guessing=np.full(3, 0.1))
        model_new.set_parameters(**common, guessing=np.array([0.4, 0.1, 0.1]))

        result = detect_drift(
            model_old,
            model_new,
            [0, 1, 2],
            [0, 1, 2],
            method="area",
            threshold=0.01,
            A=1.0,
            B=0.0,
        )

        assert result.flagged_items == [0]
        assert result.drift_statistics[0] > 1.0
        np.testing.assert_allclose(result.drift_statistics[1:], 0.0)

    def test_automatic_linking_preserves_three_pl_curves(self):
        """Estimated constants align a transformed 3PL without losing asymptotes."""
        model_old = ThreeParameterLogistic(n_items=4)
        model_new = ThreeParameterLogistic(n_items=4)
        discrimination = np.array([0.8, 1.0, 1.3, 1.6])
        difficulty = np.array([-1.2, -0.3, 0.5, 1.4])
        guessing = np.array([0.1, 0.15, 0.2, 0.25])
        scale, shift = 1.3, -0.4
        model_old.set_parameters(
            discrimination=discrimination,
            difficulty=difficulty,
            guessing=guessing,
        )
        model_new.set_parameters(
            discrimination=scale * discrimination,
            difficulty=(difficulty - shift) / scale,
            guessing=guessing,
        )

        result = detect_drift(
            model_old,
            model_new,
            list(range(4)),
            list(range(4)),
            method="area",
            threshold=1e-4,
            n_theta=101,
        )

        assert result.flagged_items == []
        np.testing.assert_allclose(result.drift_statistics, 0.0, atol=1e-6)

    def test_area_supports_variable_category_graded_response_models(self):
        """Expected-score areas support polytomous items with unequal categories."""
        model_old = GradedResponseModel(n_items=2, n_categories=[3, 4])
        model_new = GradedResponseModel(n_items=2, n_categories=[3, 4])
        thresholds = np.array([[-1.0, 1.0, 0.0], [-1.5, -0.2, 1.2]])
        model_old.set_parameters(
            discrimination=np.array([1.0, 1.2]), thresholds=thresholds
        )
        changed = thresholds.copy()
        changed[1, :3] += 0.8
        model_new.set_parameters(
            discrimination=np.array([1.0, 1.2]), thresholds=changed
        )

        result = detect_drift(
            model_old,
            model_new,
            [0, 1],
            [0, 1],
            method="area",
            threshold=0.05,
            A=1.0,
            B=0.0,
        )

        assert result.flagged_items == [1]
        np.testing.assert_allclose(result.drift_statistics[0], 0.0, atol=1e-12)
        assert result.drift_statistics[1] > 0.05

    def test_automatic_linking_preserves_graded_response_curves(self):
        """Curve linking also aligns transformed variable-category items."""
        model_old = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model_new = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        discrimination = np.array([0.8, 1.1, 1.5])
        thresholds = np.array([[-1.0, 0.7, 0.0], [-1.5, -0.2, 1.2], [-0.6, 1.4, 0.0]])
        scale, shift = 1.25, -0.35
        model_old.set_parameters(discrimination=discrimination, thresholds=thresholds)
        model_new.set_parameters(
            discrimination=scale * discrimination,
            thresholds=(thresholds - shift) / scale,
        )

        result = detect_drift(
            model_old,
            model_new,
            [0, 1, 2],
            [0, 1, 2],
            method="area",
            threshold=1e-4,
            n_theta=101,
        )

        assert result.flagged_items == []
        np.testing.assert_allclose(result.drift_statistics, 0.0, atol=1e-6)

    def test_area_batches_dense_probability_evaluation(
        self, stable_models, monkeypatch
    ):
        """Dense anchor sets should need one probability call per model."""
        model_old, model_new = stable_models
        calls = {"old": 0, "new": 0}
        probability_old = model_old.probability
        probability_new = model_new.probability

        def counted_old(theta, item_idx=None):
            calls["old"] += 1
            return probability_old(theta, item_idx)

        def counted_new(theta, item_idx=None):
            calls["new"] += 1
            return probability_new(theta, item_idx)

        monkeypatch.setattr(model_old, "probability", counted_old)
        monkeypatch.setattr(model_new, "probability", counted_new)

        detect_drift(
            model_old,
            model_new,
            list(range(10)),
            list(range(10)),
            method="area",
            A=1.0,
            B=0.0,
        )

        assert calls == {"old": 1, "new": 1}

    def test_wald_accepts_full_or_anchor_aligned_standard_errors(self):
        """Wald errors can follow model indexing or the supplied anchor order."""
        model_old = TwoParameterLogistic(n_items=8)
        model_new = TwoParameterLogistic(n_items=8)
        model_old.set_parameters(discrimination=np.ones(8), difficulty=np.zeros(8))
        difficulty_new = np.zeros(8)
        difficulty_new[6] = 0.4
        model_new.set_parameters(discrimination=np.ones(8), difficulty=difficulty_new)
        full_errors = {
            "discrimination": np.full(8, 0.1),
            "difficulty": np.full(8, 0.1),
        }
        anchor_errors = {
            "discrimination": np.full(2, 0.1),
            "difficulty": np.full(2, 0.1),
        }

        full = detect_drift(
            model_old,
            model_new,
            [5, 6],
            [5, 6],
            method="wald",
            A=1.0,
            B=0.0,
            se_old=full_errors,
            se_new=full_errors,
        )
        aligned = detect_drift(
            model_old,
            model_new,
            [5, 6],
            [5, 6],
            method="wald",
            A=1.0,
            B=0.0,
            se_old=anchor_errors,
            se_new=anchor_errors,
        )

        np.testing.assert_allclose(full.drift_statistics, aligned.drift_statistics)
        np.testing.assert_allclose(full.drift_statistics, [0.0, 8.0])
        assert full.flagged_items == [6]


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

    def test_signed_area_uses_three_pl_guessing_parameter(self):
        """Signed area includes lower-asymptote differences."""
        model_old = ThreeParameterLogistic(n_items=1)
        model_new = ThreeParameterLogistic(n_items=1)
        model_old.set_parameters(
            discrimination=np.ones(1),
            difficulty=np.zeros(1),
            guessing=np.array([0.1]),
        )
        model_new.set_parameters(
            discrimination=np.ones(1),
            difficulty=np.zeros(1),
            guessing=np.array([0.4]),
        )

        signed, unsigned = signed_area_difference(model_old, model_new, 0, 0)

        assert signed < -1.0
        assert unsigned == pytest.approx(-signed)

    def test_signed_area_supports_polytomous_expected_scores(self):
        """Polytomous signed area compares expected item-score curves."""
        model_old = GradedResponseModel(n_items=1, n_categories=3)
        model_new = GradedResponseModel(n_items=1, n_categories=3)
        model_old.set_parameters(
            discrimination=np.ones(1), thresholds=np.array([[-1.0, 1.0]])
        )
        model_new.set_parameters(
            discrimination=np.ones(1), thresholds=np.array([[-0.2, 1.8]])
        )

        signed, unsigned = signed_area_difference(model_old, model_new, 0, 0)

        assert signed > 0.5
        assert unsigned == pytest.approx(signed)


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

    @pytest.mark.parametrize("anchors", [[-1, 1], [0, 10], [1, 1]])
    def test_detect_drift_rejects_invalid_anchors(self, stable_models, anchors):
        """Anchor indices must be unique and in bounds."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match="Anchor"):
            detect_drift(model_old, model_new, anchors, [0, 1], A=1.0, B=0.0)

    def test_detect_drift_requires_linking_constants_together(self, stable_models):
        """A partially specified transformation is ambiguous."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match="provided together"):
            detect_drift(model_old, model_new, [0, 1], [0, 1], A=1.0)

    def test_relative_statistics_require_multiple_anchors(self, stable_models):
        """Relative outlier statistics are undefined for a single observation."""
        model_old, model_new = stable_models

        for method in ("robust_z", "3sigma"):
            with pytest.raises(ValueError, match="at least 2"):
                detect_drift(
                    model_old,
                    model_new,
                    [0],
                    [0],
                    method=method,
                    A=1.0,
                    B=0.0,
                )

    @pytest.mark.parametrize(
        ("method", "threshold", "message"),
        [("area", 0.0, "positive"), ("wald", 1.0, "between")],
    )
    def test_detect_drift_validates_threshold(
        self, stable_models, method, threshold, message
    ):
        """Thresholds follow the statistic used by each method."""
        model_old, model_new = stable_models

        with pytest.raises(ValueError, match=message):
            detect_drift(
                model_old,
                model_new,
                [0, 1],
                [0, 1],
                method=method,
                threshold=threshold,
                A=1.0,
                B=0.0,
            )

    def test_wald_rejects_missing_or_nonpositive_standard_errors(self, stable_models):
        """Malformed uncertainty data must fail before computing a statistic."""
        model_old, model_new = stable_models
        missing = {"difficulty": np.full(10, 0.1)}
        invalid = {
            "discrimination": np.full(10, 0.1),
            "difficulty": np.zeros(10),
        }

        with pytest.raises(ValueError, match="Missing discrimination"):
            detect_drift(
                model_old,
                model_new,
                [5, 6],
                [5, 6],
                method="wald",
                A=1.0,
                B=0.0,
                se_old=missing,
                se_new=missing,
            )
        with pytest.raises(ValueError, match="finite and positive"):
            detect_drift(
                model_old,
                model_new,
                [5, 6],
                [5, 6],
                method="wald",
                A=1.0,
                B=0.0,
                se_old=invalid,
                se_new=invalid,
            )
