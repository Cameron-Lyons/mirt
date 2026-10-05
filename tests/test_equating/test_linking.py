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
from mirt.equating.linking import (
    _bisector_link,
    _closed_form_bootstrap_samples,
    _mean_mean_link,
    _mean_sigma_link,
    _orthogonal_link,
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


class TestClosedFormBootstrapBatch:
    """Regression contracts for batched closed-form bootstrap links."""

    @pytest.mark.parametrize(
        ("method", "robust"),
        [
            ("mean_sigma", False),
            ("mean_sigma", True),
            ("mean_mean", False),
            ("mean_mean", True),
            ("bisector", False),
            ("orthogonal", False),
        ],
    )
    def test_batch_matches_scalar_replicates(
        self, reference_model, scaled_model, method, robust
    ):
        """Chunked evaluation preserves every sampled scalar estimate."""
        new_model, _, _ = scaled_model
        disc_old = np.asarray(reference_model.discrimination)
        diff_old = np.asarray(reference_model.difficulty)
        disc_new = np.asarray(new_model.discrimination)
        diff_new = np.asarray(new_model.difficulty).copy()
        diff_new[[0, 3, 7]] += np.array([0.2, -0.1, 0.15])
        n_bootstrap = 37

        expected_a = np.empty(n_bootstrap)
        expected_b = np.empty(n_bootstrap)
        expected_rng = np.random.default_rng(921)
        for replicate in range(n_bootstrap):
            sampled = expected_rng.integers(0, disc_old.size, disc_old.size)
            arrays = (
                disc_old[sampled],
                diff_old[sampled],
                disc_new[sampled],
                diff_new[sampled],
            )
            if method == "mean_sigma":
                A, B, _ = _mean_sigma_link(*arrays, robust=robust)
            elif method == "mean_mean":
                A, B, _ = _mean_mean_link(*arrays, robust=robust)
            elif method == "bisector":
                A, B, _ = _bisector_link(*arrays)
            else:
                A, B, _ = _orthogonal_link(*arrays)
            expected_a[replicate] = A
            expected_b[replicate] = B

        actual_a, actual_b = _closed_form_bootstrap_samples(
            disc_old,
            diff_old,
            disc_new,
            diff_new,
            method,
            n_bootstrap,
            np.random.default_rng(921),
            robust=robust,
            chunk_size=4,
        )

        np.testing.assert_allclose(actual_a, expected_a, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual_b, expected_b, rtol=2e-14, atol=2e-14)

    @pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
    def test_rejects_invalid_chunk_size(
        self, reference_model, scaled_model, chunk_size
    ):
        """The private batching boundary rejects unusable chunk sizes."""
        new_model, _, _ = scaled_model
        with pytest.raises(ValueError, match="chunk_size"):
            _closed_form_bootstrap_samples(
                np.asarray(reference_model.discrimination),
                np.asarray(reference_model.difficulty),
                np.asarray(new_model.discrimination),
                np.asarray(new_model.difficulty),
                "mean_mean",
                10,
                np.random.default_rng(1),
                chunk_size=chunk_size,
            )


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


def _noisy_affine_pair(model_type, n_items=12, seed=7, **shape):
    """Two calibrations of one form related by a noisy affine metric change."""
    rng = np.random.default_rng(seed)
    discrimination = rng.uniform(0.8, 2.0, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    A_true, B_true = 1.3, 0.4
    old = model_type(n_items)
    new = model_type(n_items)
    old.set_parameters(discrimination=discrimination, difficulty=difficulty, **shape)
    new.set_parameters(
        discrimination=discrimination * A_true * np.exp(rng.normal(0, 0.1, n_items)),
        difficulty=(difficulty - B_true) / A_true + rng.normal(0, 0.15, n_items),
        **shape,
    )
    return old, new


def _native_curve_oracle(old, new, method, start):
    """Minimize the curve criterion written directly with model.probability."""
    from scipy import optimize

    theta = np.linspace(-4.0, 4.0, 61)
    weights = np.exp(-0.5 * theta**2)
    weights /= weights.sum()
    curves_old = old.probability(theta[:, None])

    def criterion(parameters):
        scale, shift = np.exp(parameters[0]), parameters[1]
        curves_new = new.probability(((theta - shift) / scale)[:, None])
        if method == "haebara":
            return float(np.sum(weights[:, None] * (curves_old - curves_new) ** 2))
        difference = curves_old.sum(axis=1) - curves_new.sum(axis=1)
        return float(np.sum(weights * difference**2))

    result = optimize.minimize(
        criterion,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-16, "maxiter": 8000},
    )
    assert result.success
    return np.exp(result.x[0]), result.x[1], criterion


class TestModelNativeCurves:
    """Curve methods must use each family's own response function."""

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara", "tcc"])
    @pytest.mark.parametrize("family", ["5PL", "CLL", "NLL", "ULL", "ZI-2PL"])
    def test_curve_links_match_model_probability_oracle(self, family, method):
        from mirt.models.dichotomous import (
            ComplementaryLogLog,
            FiveParameterLogistic,
            NegativeLogLog,
            UnipolarLogLogistic,
        )
        from mirt.models.zeroinflated import ZeroInflated2PL

        if family == "5PL":
            old, new = _noisy_affine_pair(
                FiveParameterLogistic,
                guessing=np.zeros(12),
                asymmetry=np.linspace(0.25, 0.45, 12),
            )
        elif family == "ZI-2PL":
            old, new = _noisy_affine_pair(ZeroInflated2PL)
            old.set_parameters(zero_inflation=np.linspace(0.05, 0.3, 12))
            new.set_parameters(zero_inflation=np.linspace(0.05, 0.3, 12))
        else:
            model_type = {
                "CLL": ComplementaryLogLog,
                "NLL": NegativeLogLog,
                "ULL": UnipolarLogLogistic,
            }[family]
            old, new = _noisy_affine_pair(model_type)
        anchors = list(range(12))

        result = link(old, new, anchors, anchors, method=method)

        A, B, criterion = _native_curve_oracle(
            old, new, method, [np.log(result.constants.A), result.constants.B]
        )
        assert result.constants.A == pytest.approx(A, abs=1e-6)
        assert result.constants.B == pytest.approx(B, abs=1e-6)
        assert result.convergence_info["fun"] == pytest.approx(
            criterion([np.log(A), B]), rel=1e-8, abs=1e-14
        )

    def test_five_parameter_asymmetry_changes_the_estimator(self):
        """Ignoring 5PL asymmetry fits the native curves much worse."""
        from mirt.models.dichotomous import FiveParameterLogistic

        old, new = _noisy_affine_pair(
            FiveParameterLogistic,
            guessing=np.zeros(12),
            asymmetry=np.full(12, 0.3),
        )
        anchors = list(range(12))
        result = link(old, new, anchors, anchors, method="stocking_lord")
        _, _, criterion = _native_curve_oracle(
            old, new, "stocking_lord", [np.log(result.constants.A), result.constants.B]
        )
        symmetric = (TwoParameterLogistic(12), TwoParameterLogistic(12))
        symmetric[0].set_parameters(
            discrimination=old.discrimination, difficulty=old.difficulty
        )
        symmetric[1].set_parameters(
            discrimination=new.discrimination, difficulty=new.difficulty
        )
        ignored = link(*symmetric, anchors, anchors, method="stocking_lord")

        fitted = criterion([np.log(result.constants.A), result.constants.B])
        assert fitted < 0.5 * criterion(
            [np.log(ignored.constants.A), ignored.constants.B]
        )

    def test_native_diagnostics_use_model_curves(self):
        """Anchor areas and TCC fit evaluate the transformed native curves."""
        from mirt.models.dichotomous import ComplementaryLogLog

        old, new = _noisy_affine_pair(ComplementaryLogLog)
        anchors = list(range(12))
        result = link(old, new, anchors, anchors, method="haebara")
        A, B = result.constants.A, result.constants.B
        theta = np.linspace(-4.0, 4.0, 61)
        weights = np.exp(-0.5 * theta**2)
        weights /= weights.sum()
        curves_old = old.probability(theta[:, None])
        curves_new = new.probability(((theta - B) / A)[:, None])

        np.testing.assert_allclose(
            result.anchor_diagnostics.area_diff,
            np.trapezoid(np.abs(curves_old - curves_new), theta, axis=0),
            rtol=1e-12,
            atol=1e-14,
        )
        tcc_difference = curves_old.sum(axis=1) - curves_new.sum(axis=1)
        assert result.fit_statistics.tcc_rmse == pytest.approx(
            np.sqrt(np.sum(weights * tcc_difference**2)), rel=1e-12
        )

    def test_link_matches_diagnostics_estimator(self):
        """link() and the diagnostics estimator share one curve kernel."""
        from mirt.equating.diagnostics import _estimate_constants
        from mirt.equating.linking import _link_form, _validate_curve_grid
        from mirt.models.dichotomous import (
            ComplementaryLogLog,
            FiveParameterLogistic,
        )

        theta, weights = _validate_curve_grid((-4.0, 4.0), 61, None)
        anchors = list(range(12))
        pairs = [
            _noisy_affine_pair(ComplementaryLogLog),
            _noisy_affine_pair(
                FiveParameterLogistic,
                guessing=np.full(12, 0.1),
                asymmetry=np.linspace(0.5, 1.5, 12),
            ),
        ]
        for old, new in pairs:
            for method in ("stocking_lord", "haebara"):
                result = link(old, new, anchors, anchors, method=method)
                expected = _estimate_constants(
                    _link_form(old, anchors, "old"),
                    _link_form(new, anchors, "new"),
                    method,
                    theta,
                    weights,
                )
                assert (result.constants.A, result.constants.B) == expected

    def test_native_bootstrap_matches_least_squares_oracle(self):
        """Native anchor bootstraps resample the same model-native estimator."""
        from scipy.optimize import least_squares

        from mirt.models.dichotomous import ComplementaryLogLog

        old, new = _noisy_affine_pair(ComplementaryLogLog, n_items=6, seed=11)
        anchors = list(range(6))
        seed, n_bootstrap = 5, 8
        fitted = link(
            old,
            new,
            anchors,
            anchors,
            method="stocking_lord",
            compute_se=True,
            n_bootstrap=n_bootstrap,
            random_state=seed,
        )
        theta = np.linspace(-4.0, 4.0, 61)
        weights = np.exp(-0.5 * theta**2)
        weights /= weights.sum()
        curves_old = old.probability(theta[:, None])

        def fit_sample(indices):
            def residual(parameters):
                scale, shift = np.exp(parameters[0]), parameters[1]
                curves_new = new.probability(((theta - shift) / scale)[:, None])
                difference = curves_old[:, indices] - curves_new[:, indices]
                return np.sqrt(weights) * difference.sum(axis=1)

            result = least_squares(
                residual, [0.0, 0.0], xtol=1e-14, ftol=1e-14, gtol=1e-14
            )
            assert result.success
            return np.array([np.exp(result.x[0]), result.x[1]])

        rng = np.random.default_rng(seed)
        samples = np.array(
            [
                fit_sample(rng.choice(6, size=6, replace=True))
                for _ in range(n_bootstrap)
            ]
        )
        np.testing.assert_allclose(
            [fitted.constants.A_se, fitted.constants.B_se],
            samples.std(axis=0, ddof=1),
            rtol=1e-5,
        )


def _nelder_mead_curve_link(old, new, anchors, method):
    """The previous derivative-free optimizer, kept as an equivalence oracle."""
    from scipy import optimize
    from scipy.special import expit

    theta = np.linspace(-4.0, 4.0, 61)
    weights = np.exp(-0.5 * theta**2)
    weights /= weights.sum()
    guessing_old = old.guessing[anchors]
    guessing_new = new.guessing[anchors]
    a_old, b_old = old.discrimination[anchors], old.difficulty[anchors]
    a_new, b_new = new.discrimination[anchors], new.difficulty[anchors]

    def curves(a, b, c):
        return c + (1.0 - c) * expit(a * (theta[:, None] - b))

    curves_old = curves(a_old, b_old, guessing_old)

    def criterion(parameters):
        scale, shift = np.exp(parameters[0]), parameters[1]
        difference = curves_old - curves(
            a_new / scale, scale * b_new + shift, guessing_new
        )
        if method == "haebara":
            return float(np.sum(weights[:, None] * difference**2))
        return float(np.sum(weights * difference.sum(axis=1) ** 2))

    scale = np.std(b_old, ddof=1) / np.std(b_new, ddof=1)
    start = [np.log(scale), np.mean(b_old) - scale * np.mean(b_new)]
    result = optimize.minimize(
        criterion,
        start,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    assert result.success
    return np.exp(result.x[0]), result.x[1]


class TestLeastSquaresCurveLinking:
    """Analytic-Jacobian least squares reproduces the derivative-free optima."""

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara", "tcc"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_nelder_mead_optimum(self, method, seed):
        rng = np.random.default_rng(seed)
        n_items = 15
        discrimination = rng.uniform(0.6, 2.0, n_items)
        difficulty = rng.normal(0.0, 1.0, n_items)
        guessing = rng.uniform(0.05, 0.25, n_items)
        old = ThreeParameterLogistic(n_items)
        new = ThreeParameterLogistic(n_items)
        old.set_parameters(
            discrimination=discrimination, difficulty=difficulty, guessing=guessing
        )
        new.set_parameters(
            discrimination=discrimination * 1.2 * np.exp(rng.normal(0, 0.1, n_items)),
            difficulty=(difficulty - 0.3) / 1.2 + rng.normal(0, 0.1, n_items),
            guessing=guessing,
        )
        anchors = list(range(n_items))

        result = link(old, new, anchors, anchors, method=method)

        A, B = _nelder_mead_curve_link(old, new, anchors, method)
        assert result.convergence_info["optimizer"] == "least_squares"
        assert result.constants.A == pytest.approx(A, abs=1e-6)
        assert result.constants.B == pytest.approx(B, abs=1e-6)

    @pytest.mark.parametrize("failure", ["raises", "status"])
    def test_falls_back_to_nelder_mead(self, reference_model, monkeypatch, failure):
        from types import SimpleNamespace

        import mirt.equating.linking as linking

        old = reference_model
        new = TwoParameterLogistic(10)
        new.set_parameters(
            discrimination=old.discrimination * 1.2 * np.linspace(0.9, 1.1, 10),
            difficulty=(old.difficulty - 0.5) / 1.2 + np.linspace(-0.1, 0.1, 10),
        )
        anchors = list(range(10))
        expected = link(old, new, anchors, anchors, method="haebara")

        def failing_least_squares(*args, **kwargs):
            if failure == "raises":
                raise np.linalg.LinAlgError("singular")
            return SimpleNamespace(x=np.array([np.nan, 0.0]), status=0, njev=1)

        monkeypatch.setattr(linking.optimize, "least_squares", failing_least_squares)
        result = link(old, new, anchors, anchors, method="haebara")

        assert result.convergence_info["optimizer"] == "nelder_mead"
        assert result.constants.A == pytest.approx(expected.constants.A, abs=1e-6)
        assert result.constants.B == pytest.approx(expected.constants.B, abs=1e-6)

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    def test_plateau_stop_falls_back_to_nelder_mead(self, method):
        """LM stalling where the placed curves are flat is not a solution."""
        old = ThreeParameterLogistic(2)
        new = ThreeParameterLogistic(2)
        old.set_parameters(
            discrimination=np.array([2.62, 2.03]),
            difficulty=np.array([1.87, 0.05]),
            guessing=np.array([0.02, 0.22]),
        )
        # Tied new difficulties put the mean/sigma start on near-flat curves,
        # from which one LM step lands where every slope underflows to zero.
        new.set_parameters(
            discrimination=np.array([1.71, 1.71]),
            difficulty=np.array([7.52, 7.44]),
            guessing=np.array([0.26, 0.14]),
        )

        result = link(old, new, [0, 1], [0, 1], method=method)

        A, B = _nelder_mead_curve_link(old, new, [0, 1], method)
        assert result.convergence_info["optimizer"] == "nelder_mead"
        assert result.constants.A == pytest.approx(A, abs=1e-6)
        assert result.constants.B == pytest.approx(B, abs=1e-6)

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    def test_tied_reference_difficulties_start_from_mean_mean(self, method):
        """A form without difficulty spread must not start from A = 0."""
        old = TwoParameterLogistic(4)
        new = TwoParameterLogistic(4)
        old.set_parameters(
            discrimination=np.array([1.0, 1.2, 0.8, 1.5]), difficulty=np.zeros(4)
        )
        new.set_parameters(
            discrimination=np.array([1.1, 1.3, 0.9, 1.4]),
            difficulty=np.array([0.1, -0.1, 0.05, -0.02]),
        )
        anchors = list(range(4))

        result = link(old, new, anchors, anchors, method=method)

        A, B, _ = _native_curve_oracle(old, new, method, [0.0, 0.0])
        assert result.constants.A == pytest.approx(A, abs=1e-6)
        assert result.constants.B == pytest.approx(B, abs=1e-6)

    def test_exact_start_reports_no_optimizer(self, reference_model):
        new = TwoParameterLogistic(10)
        new.set_parameters(
            discrimination=reference_model.discrimination * 1.2,
            difficulty=(reference_model.difficulty - 0.5) / 1.2,
        )
        anchors = list(range(10))

        result = link(reference_model, new, anchors, anchors, method="haebara")

        assert result.convergence_info["optimizer"] == "none"
        assert result.convergence_info["nit"] == 0

    def test_analytic_jacobian_matches_finite_differences(self):
        from mirt.equating.linking import (
            _CurveObjective,
            _link_form,
            _validate_curve_grid,
        )
        from mirt.models.dichotomous import FiveParameterLogistic

        old, new = _noisy_affine_pair(
            FiveParameterLogistic,
            guessing=np.full(12, 0.1),
            upper=np.full(12, 0.9),
            asymmetry=np.linspace(0.4, 1.6, 12),
        )
        anchors = list(range(12))
        theta, weights = _validate_curve_grid((-4.0, 4.0), 61, None)
        parameters = np.array([0.2, -0.3])
        step = 1e-6
        for method in ("stocking_lord", "haebara"):
            objective = _CurveObjective(
                _link_form(old, anchors, "old"),
                _link_form(new, anchors, "new"),
                theta,
                weights,
                method,
            )
            numerical = np.column_stack(
                [
                    (
                        objective.residuals(parameters + step * unit)
                        - objective.residuals(parameters - step * unit)
                    )
                    / (2.0 * step)
                    for unit in np.eye(2)
                ]
            )
            np.testing.assert_allclose(
                objective.jacobian(parameters), numerical, rtol=1e-6, atol=1e-9
            )
