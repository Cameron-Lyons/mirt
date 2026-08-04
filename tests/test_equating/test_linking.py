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
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


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

    @pytest.mark.parametrize(
        "method",
        [
            "mean_sigma",
            "mean_mean",
            "stocking_lord",
            "haebara",
            "tcc",
            "bisector",
            "orthogonal",
        ],
    )
    def test_all_methods_recover_exact_constants(self, reference_model, method):
        """Every linker must use the same documented transformation direction."""
        scale, shift = 1.7, -0.4
        target_model = transform_parameters(reference_model, scale, shift)
        anchors = list(range(reference_model.n_items))

        result = link(
            target_model,
            reference_model,
            anchors,
            anchors,
            method=method,
            compute_diagnostics=False,
        )

        assert result.constants.A == pytest.approx(scale, abs=1e-6)
        assert result.constants.B == pytest.approx(shift, abs=1e-6)
        assert result.constants.A > 0.0
        if method == "tcc":
            assert result.convergence_info["method"] == "tcc"

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

    @pytest.mark.parametrize(
        ("anchors", "message"),
        [
            ([-1, 1], "out of range"),
            ([0, 10], "out of range"),
            ([1, 1], "unique"),
            ([0, 1.5], "integers"),
        ],
    )
    def test_anchor_indices_are_validated(self, reference_model, anchors, message):
        """Invalid indices cannot silently select or duplicate items."""
        with pytest.raises(ValueError, match=message):
            link(reference_model, reference_model, anchors, [0, 1])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"theta_range": (1.0, -1.0)}, "theta_range"),
            ({"n_theta": 1}, "n_theta"),
            ({"weights": np.ones(3)}, "weights must have shape"),
            ({"weights": np.zeros(61)}, "positive sum"),
            ({"weights": np.r_[np.ones(60), -1.0]}, "non-negative"),
        ],
    )
    def test_curve_grid_is_validated(self, reference_model, kwargs, message):
        """Malformed integration grids fail with clear errors."""
        with pytest.raises(ValueError, match=message):
            link(reference_model, reference_model, [0, 1], [0, 1], **kwargs)

    @pytest.mark.parametrize("n_bootstrap", [0, 1, 1.5])
    def test_bootstrap_count_is_validated(self, reference_model, n_bootstrap):
        """At least two integer replicates are required for a standard error."""
        with pytest.raises(ValueError, match="n_bootstrap"):
            link(
                reference_model,
                reference_model,
                [0, 1],
                [0, 1],
                compute_se=True,
                n_bootstrap=n_bootstrap,
            )

    def test_multidimensional_models_are_rejected(self):
        """Scalar link constants cannot silently discard extra factors."""
        model = TwoParameterLogistic(n_items=3, n_factors=2)

        with pytest.raises(ValueError, match="unidimensional"):
            link(model, model, [0, 1], [0, 1])


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

    @pytest.mark.parametrize(
        ("A", "B", "message"),
        [
            (0.0, 0.0, "A must"),
            (-1.0, 0.0, "A must"),
            (np.inf, 0.0, "A must"),
            (1.0, np.nan, "B must"),
        ],
    )
    def test_transform_rejects_invalid_constants(self, reference_model, A, B, message):
        """Invalid transformations fail before copying or mutating a model."""
        with pytest.raises(ValueError, match=message):
            transform_parameters(reference_model, A, B)


class TestLinkingRobust:
    """Tests for robust linking options."""

    def test_link_robust_option(self, reference_model, scaled_model):
        """Test that robust linking uses median instead of mean."""
        new_model, _, _ = scaled_model
        difficulty = np.asarray(new_model.difficulty).copy()
        difficulty[0] += 8.0
        new_model.set_parameters(difficulty=difficulty)
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

        assert result_robust.constants.A != pytest.approx(result_normal.constants.A)

    def test_link_with_bootstrap_se(self, reference_model, scaled_model):
        """Test bootstrap standard error computation."""
        new_model, _, _ = scaled_model
        difficulty = np.asarray(new_model.difficulty).copy()
        difficulty[0] += 0.5
        new_model.set_parameters(difficulty=difficulty)
        anchors = list(range(10))

        result = link(
            reference_model,
            new_model,
            anchors,
            anchors,
            compute_se=True,
            n_bootstrap=50,
            random_state=42,
        )

        assert result.constants.A_se is not None
        assert result.constants.B_se is not None
        assert result.constants.A_se > 0
        assert result.constants.B_se > 0

    def test_bootstrap_is_reproducible(self, reference_model, scaled_model):
        """Supplying a seed reproduces the same uncertainty estimates."""
        new_model, _, _ = scaled_model
        difficulty = np.asarray(new_model.difficulty).copy()
        difficulty[-1] += 0.4
        new_model.set_parameters(difficulty=difficulty)
        anchors = list(range(10))
        kwargs = {
            "method": "mean_sigma",
            "compute_se": True,
            "n_bootstrap": 100,
            "random_state": 1234,
        }

        first = link(reference_model, new_model, anchors, anchors, **kwargs)
        second = link(reference_model, new_model, anchors, anchors, **kwargs)

        assert first.constants.A_se == second.constants.A_se
        assert first.constants.B_se == second.constants.B_se


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

    def test_three_pl_guessing_drift_affects_curves(self):
        """Lower-asymptote drift must appear in areas and robust flags."""
        model_old = ThreeParameterLogistic(n_items=5)
        model_new = ThreeParameterLogistic(n_items=5)
        common = {
            "discrimination": np.ones(5),
            "difficulty": np.linspace(-1.0, 1.0, 5),
        }
        model_old.set_parameters(**common, guessing=np.full(5, 0.1))
        model_new.set_parameters(**common, guessing=np.array([0.4, 0.1, 0.1, 0.1, 0.1]))

        result = link(model_old, model_new, list(range(5)), list(range(5)))

        diagnostics = result.anchor_diagnostics
        assert diagnostics is not None
        assert diagnostics.area_diff[0] > diagnostics.area_diff[1:].max()
        assert diagnostics.flagged.tolist() == [True, False, False, False, False]

    def test_four_pl_upper_drift_affects_curves(self):
        """Upper-asymptote drift must also contribute to diagnostics."""
        model_old = FourParameterLogistic(n_items=5)
        model_new = FourParameterLogistic(n_items=5)
        common = {
            "discrimination": np.ones(5),
            "difficulty": np.linspace(-1.0, 1.0, 5),
            "guessing": np.full(5, 0.1),
        }
        model_old.set_parameters(**common, upper=np.ones(5))
        model_new.set_parameters(**common, upper=np.array([0.7, 1.0, 1.0, 1.0, 1.0]))

        result = link(model_old, model_new, list(range(5)), list(range(5)))

        diagnostics = result.anchor_diagnostics
        assert diagnostics is not None
        assert diagnostics.area_diff[0] > diagnostics.area_diff[1:].max()
        assert diagnostics.flagged.tolist() == [True, False, False, False, False]
