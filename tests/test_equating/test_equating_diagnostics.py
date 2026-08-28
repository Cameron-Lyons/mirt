"""Tests for equating diagnostics."""

import warnings

import numpy as np
import pytest

import mirt.equating.diagnostics as diagnostics
from mirt.equating.diagnostics import (
    bootstrap_linking_se,
    compare_linking_methods,
    compute_linking_fit,
    delta_method_se,
    linking_summary,
    parameter_recovery_summary,
)
from mirt.equating.linking import (
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
    link,
)
from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


@pytest.fixture
def linked_models_pair():
    """Create a pair of linked models for testing."""
    rng = np.random.default_rng(42)
    n_items = 10

    disc = np.abs(rng.normal(1.0, 0.3, n_items))
    diff_old = rng.normal(0, 1, n_items)
    diff_new = diff_old * 1.1 + 0.3

    model_old = TwoParameterLogistic(n_items=n_items)
    model_old._parameters = {
        "discrimination": disc.copy(),
        "difficulty": diff_old.copy(),
    }
    model_old._is_fitted = True
    model_old._n_factors = 1

    model_new = TwoParameterLogistic(n_items=n_items)
    model_new._parameters = {
        "discrimination": disc.copy() / 1.1,
        "difficulty": diff_new.copy(),
    }
    model_new._is_fitted = True
    model_new._n_factors = 1

    anchors = list(range(5))

    return model_old, model_new, anchors


class TestBootstrapLinkingSE:
    """Tests for bootstrap_linking_se function."""

    def test_basic_bootstrap(self, linked_models_pair):
        """Test basic bootstrap SE computation."""
        model_old, model_new, anchors = linked_models_pair

        se_a, se_b, a_samples, b_samples = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=50,
            seed=42,
        )

        assert se_a > 0
        assert se_b > 0
        assert len(a_samples) == 50
        assert len(b_samples) == 50

    def test_bootstrap_reproducibility(self, linked_models_pair):
        """Test bootstrap reproducibility with seed."""
        model_old, model_new, anchors = linked_models_pair

        se_a1, se_b1, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=20,
            seed=42,
        )

        se_a2, se_b2, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            n_bootstrap=20,
            seed=42,
        )

        assert se_a1 == pytest.approx(se_a2)
        assert se_b1 == pytest.approx(se_b2)

    def test_parallel_curve_bootstrap_matches_sequential(self, linked_models_pair):
        """Worker count does not change seeded nonlinear replicates."""
        model_old, model_new, anchors = linked_models_pair
        common = {
            "model_old": model_old,
            "model_new": model_new,
            "responses_old": None,
            "responses_new": None,
            "anchors_old": anchors,
            "anchors_new": anchors,
            "method": "stocking_lord",
            "n_bootstrap": 24,
            "n_theta": 101,
            "seed": 42,
        }

        sequential = bootstrap_linking_se(**common, n_jobs=1)
        parallel = bootstrap_linking_se(**common, n_jobs=3)

        for actual, expected in zip(parallel, sequential, strict=True):
            np.testing.assert_array_equal(actual, expected)

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
    def test_bootstrap_different_methods(self, linked_models_pair, method):
        """Test bootstrap with different linking methods."""
        model_old, model_new, anchors = linked_models_pair

        se_a, se_b, _, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old=None,
            responses_new=None,
            anchors_old=anchors,
            anchors_new=anchors,
            method=method,
            n_bootstrap=20,
            seed=42,
        )

        assert se_a > 0
        assert se_b > 0

    @pytest.mark.parametrize(
        "method", ["mean_sigma", "mean_mean", "bisector", "orthogonal"]
    )
    def test_anchor_bootstrap_batches_closed_form_methods(
        self, linked_models_pair, monkeypatch, method
    ):
        """Closed-form anchor bootstraps bypass scalar replicate dispatch."""
        model_old, model_new, anchors = linked_models_pair

        def unexpected_scalar_dispatch(*args, **kwargs):
            raise AssertionError("scalar replicate dispatch should not run")

        monkeypatch.setattr(
            diagnostics, "_estimate_constants", unexpected_scalar_dispatch
        )
        result = bootstrap_linking_se(
            model_old,
            model_new,
            None,
            None,
            anchors,
            anchors,
            method=method,
            n_bootstrap=25,
            seed=19,
        )

        assert all(np.all(np.isfinite(values)) for values in result)

    def test_mean_sigma_batch_preserves_degenerate_scale_fallback(
        self, linked_models_pair
    ):
        """A zero old-form scale still falls back to discrimination means."""
        model_old, model_new, anchors = linked_models_pair
        model_old.set_parameters(difficulty=np.zeros(model_old.n_items))
        n_bootstrap = 30

        _, _, A_samples, _ = bootstrap_linking_se(
            model_old,
            model_new,
            None,
            None,
            anchors,
            anchors,
            method="mean_sigma",
            n_bootstrap=n_bootstrap,
            seed=73,
        )

        old_disc = np.asarray(model_old.discrimination)[anchors]
        new_disc = np.asarray(model_new.discrimination)[anchors]
        rng = np.random.default_rng(73)
        sampled = rng.integers(0, len(anchors), size=(n_bootstrap, len(anchors)))
        expected = np.mean(new_disc[sampled], axis=1) / np.mean(
            old_disc[sampled], axis=1
        )
        np.testing.assert_allclose(A_samples, expected)

    @pytest.mark.parametrize("n_bootstrap", [0, 1, 1.5, True])
    def test_rejects_invalid_replicate_count(self, linked_models_pair, n_bootstrap):
        """Reject counts that cannot define a sample standard deviation."""
        model_old, model_new, anchors = linked_models_pair

        with pytest.raises(ValueError, match="n_bootstrap"):
            bootstrap_linking_se(
                model_old,
                model_new,
                None,
                None,
                anchors,
                anchors,
                n_bootstrap=n_bootstrap,
            )

    @pytest.mark.parametrize("n_jobs", [0, -1, 1.5, True])
    def test_rejects_invalid_worker_count(self, linked_models_pair, n_jobs):
        """Reject invalid parallel worker counts before sampling."""
        model_old, model_new, anchors = linked_models_pair

        with pytest.raises(ValueError, match="n_jobs"):
            bootstrap_linking_se(
                model_old,
                model_new,
                None,
                None,
                anchors,
                anchors,
                n_jobs=n_jobs,
            )

    def test_rejects_unknown_method(self, linked_models_pair):
        """Do not silently substitute a different estimator."""
        model_old, model_new, anchors = linked_models_pair

        with pytest.raises(ValueError, match="Unknown linking method"):
            bootstrap_linking_se(
                model_old,
                model_new,
                None,
                None,
                anchors,
                anchors,
                method="unknown",
            )

    @pytest.mark.parametrize(
        ("anchors_old", "anchors_new"),
        [([-1, 1], [0, 1]), ([0, 0], [0, 1]), ([0, 1], [0, 1.5])],
    )
    def test_rejects_invalid_anchor_pairs(
        self, linked_models_pair, anchors_old, anchors_new
    ):
        """Validate anchor indices before sampling."""
        model_old, model_new, _ = linked_models_pair

        with pytest.raises(ValueError, match="Anchor"):
            bootstrap_linking_se(
                model_old,
                model_new,
                None,
                None,
                anchors_old,
                anchors_new,
            )

    def test_response_bootstrap_requires_explicit_refit(self, linked_models_pair):
        """Response resampling cannot proceed without recalibration."""
        model_old, model_new, anchors = linked_models_pair
        responses = np.zeros((10, model_old.n_items))

        with pytest.raises(ValueError, match="refit is required"):
            bootstrap_linking_se(
                model_old,
                model_new,
                responses,
                responses,
                anchors,
                anchors,
            )

        with pytest.raises(ValueError, match="supplied together"):
            bootstrap_linking_se(
                model_old,
                model_new,
                responses,
                None,
                anchors,
                anchors,
            )

        with pytest.raises(ValueError, match="refit requires"):
            bootstrap_linking_se(
                model_old,
                model_new,
                None,
                None,
                anchors,
                anchors,
                refit=lambda model, _: model,
            )

    def test_response_bootstrap_resamples_and_refits(self, linked_models_pair):
        """Use the supplied response matrices rather than ignoring them."""
        model_old, model_new, anchors = linked_models_pair
        rng = np.random.default_rng(123)
        probabilities_old = np.linspace(0.2, 0.8, model_old.n_items)
        probabilities_new = np.linspace(0.3, 0.7, model_new.n_items)
        responses_old = rng.binomial(1, probabilities_old, size=(80, model_old.n_items))
        responses_new = rng.binomial(1, probabilities_new, size=(70, model_new.n_items))
        sampled_responses = []

        def refit(model, responses):
            sampled_responses.append(responses.copy())
            fitted = model.copy()
            means = np.clip(np.mean(responses, axis=0), 0.05, 0.95)
            difficulty = np.log((1.0 - means) / means) / fitted.discrimination
            fitted.set_parameters(difficulty=difficulty)
            return fitted

        _, se_b, _, b_samples = bootstrap_linking_se(
            model_old,
            model_new,
            responses_old,
            responses_new,
            anchors,
            anchors,
            method="mean_mean",
            n_bootstrap=16,
            seed=42,
            refit=refit,
        )

        assert len(sampled_responses) == 32
        assert se_b > 0
        assert np.unique(b_samples).size > 1

    def test_parallel_response_refits_match_sequential(self, linked_models_pair):
        """Pre-sampled response replicates stay deterministic across workers."""
        model_old, model_new, anchors = linked_models_pair
        rng = np.random.default_rng(91)
        responses_old = rng.integers(0, 2, size=(40, model_old.n_items))
        responses_new = rng.integers(0, 2, size=(35, model_new.n_items))

        def refit(model, responses):
            means = np.clip(np.mean(responses, axis=0), 0.05, 0.95)
            difficulty = np.log((1.0 - means) / means) / model.discrimination
            return model.set_parameters(difficulty=difficulty)

        common = {
            "model_old": model_old,
            "model_new": model_new,
            "responses_old": responses_old,
            "responses_new": responses_new,
            "anchors_old": anchors,
            "anchors_new": anchors,
            "method": "mean_mean",
            "n_bootstrap": 16,
            "seed": 42,
            "refit": refit,
        }

        sequential = bootstrap_linking_se(**common, n_jobs=1)
        parallel = bootstrap_linking_se(**common, n_jobs=4)

        for actual, expected in zip(parallel, sequential, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_response_bootstrap_preserves_missing_values(self, linked_models_pair):
        """Allow recalibration callbacks to handle missing responses."""
        model_old, model_new, anchors = linked_models_pair
        responses = np.zeros((4, model_old.n_items))
        responses[0, 0] = np.nan
        observed_missing = []

        def refit(model, sampled):
            observed_missing.append(bool(np.isnan(sampled).any()))
            return model

        _, _, a_samples, _ = bootstrap_linking_se(
            model_old,
            model_new,
            responses,
            responses,
            anchors,
            anchors,
            method="mean_mean",
            n_bootstrap=4,
            seed=2,
            refit=refit,
        )

        assert a_samples.size == 4
        assert any(observed_missing)


class TestDeltaMethodSE:
    """Tests for delta_method_se function."""

    def test_basic_delta_method(self, linked_models_pair):
        """Test basic delta method SE computation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        n_params = 2 * model_old.n_items
        vcov = np.eye(n_params) * 0.01

        se_a, se_b = delta_method_se(
            linking_result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        assert se_a >= 0
        assert se_b >= 0

    def test_requires_models(self, linked_models_pair):
        """A result alone cannot define the parameter Jacobian."""
        model_old, model_new, anchors = linked_models_pair
        linking_result = link(model_old, model_new, anchors, anchors)
        vcov = np.eye(2 * model_old.n_items)

        with pytest.raises(ValueError, match="model_old and model_new"):
            delta_method_se(linking_result, vcov, vcov, anchors, anchors)

    def test_matches_mean_mean_analytic_gradient(self, linked_models_pair):
        """Match the closed-form mean/mean delta calculation."""
        model_old, model_new, anchors = linked_models_pair
        old_a = model_old.discrimination[anchors]
        old_b = model_old.difficulty[anchors]
        new_a = model_new.discrimination[anchors]
        new_b = model_new.difficulty[anchors]
        n_anchors = len(anchors)
        A = float(np.mean(new_a) / np.mean(old_a))
        B = float(np.mean(old_b) - A * np.mean(new_b))
        result = LinkingResult(
            constants=LinkingConstants(A=A, B=B, method="mean_mean"),
            anchor_items=anchors,
        )
        variance = 0.01
        vcov = np.eye(2 * model_old.n_items) * variance

        se_a, se_b = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        derivative_a_old = -np.mean(new_a) / (n_anchors * np.mean(old_a) ** 2)
        derivative_a_new = 1.0 / (n_anchors * np.mean(old_a))
        gradient_a = np.concatenate(
            (
                np.full(n_anchors, derivative_a_old),
                np.zeros(n_anchors),
                np.full(n_anchors, derivative_a_new),
                np.zeros(n_anchors),
            )
        )
        gradient_b = np.concatenate(
            (
                np.full(n_anchors, -np.mean(new_b) * derivative_a_old),
                np.full(n_anchors, 1.0 / n_anchors),
                np.full(n_anchors, -np.mean(new_b) * derivative_a_new),
                np.full(n_anchors, -A / n_anchors),
            )
        )
        assert se_a == pytest.approx(np.sqrt(variance * np.sum(gradient_a**2)))
        assert se_b == pytest.approx(np.sqrt(variance * np.sum(gradient_b**2)))

    def test_new_form_covariance_affects_uncertainty(self, linked_models_pair):
        """Propagate uncertainty from both calibrations."""
        model_old, model_new, anchors = linked_models_pair
        result = LinkingResult(
            constants=LinkingConstants(A=1.0, B=0.0, method="mean_mean"),
            anchor_items=anchors,
        )
        size = 2 * model_old.n_items
        zero = np.zeros((size, size))
        small = np.eye(size) * 0.001
        large = np.eye(size)

        small_se = delta_method_se(
            result,
            zero,
            small,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )
        large_se = delta_method_se(
            result,
            zero,
            large,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        assert large_se[0] > small_se[0]
        assert large_se[1] > small_se[1]

    @pytest.mark.parametrize("case", ["nonsquare", "small", "asymmetric", "non_psd"])
    def test_rejects_invalid_covariance(self, linked_models_pair, case):
        """Reject malformed covariance before differentiation."""
        model_old, model_new, anchors = linked_models_pair
        result = LinkingResult(
            constants=LinkingConstants(A=1.0, B=0.0, method="mean_mean"),
            anchor_items=anchors,
        )
        size = 2 * model_old.n_items
        invalid = np.eye(size)
        if case == "nonsquare":
            invalid = np.ones((size, size - 1))
        elif case == "small":
            invalid = np.eye(size - 1)
        elif case == "asymmetric":
            invalid[0, 1] = 0.5
        else:
            invalid[0, 0] = -1.0

        with pytest.raises(ValueError, match="vcov_old"):
            delta_method_se(
                result,
                invalid,
                np.eye(size),
                anchors,
                anchors,
                model_old=model_old,
                model_new=model_new,
            )

    def test_includes_guessing_covariance(self):
        """Propagate 3PL lower-asymptote uncertainty into curve linking."""
        anchors = [0, 1, 2]
        model_old = ThreeParameterLogistic(n_items=3)
        model_new = ThreeParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.1, 1.4]),
            "difficulty": np.array([-1.0, 0.1, 1.2]),
        }
        model_old.set_parameters(**common, guessing=np.array([0.1, 0.15, 0.2]))
        model_new.set_parameters(**common, guessing=np.array([0.2, 0.25, 0.3]))
        result = LinkingResult(
            constants=LinkingConstants(A=1.0, B=0.0, method="stocking_lord"),
            anchor_items=anchors,
        )
        size = model_old.n_parameters
        vcov = np.zeros((size, size))
        vcov[2 * model_old.n_items :, 2 * model_old.n_items :] = np.eye(3) * 0.01

        se_a, se_b = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        assert se_a > 0
        assert se_b > 0

    def test_includes_five_parameter_asymmetry_covariance(self):
        """Propagate 5PL asymmetry uncertainty into curve linking."""
        anchors = [0, 1, 2]
        model_old = FiveParameterLogistic(n_items=3)
        model_new = FiveParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.1, 1.4]),
            "difficulty": np.array([-1.0, 0.1, 1.2]),
            "guessing": np.array([0.1, 0.15, 0.2]),
            "upper": np.array([0.9, 0.92, 0.94]),
        }
        model_old.set_parameters(**common, asymmetry=np.array([0.8, 1.0, 1.2]))
        model_new.set_parameters(**common, asymmetry=np.array([1.1, 1.3, 1.5]))
        result = LinkingResult(
            constants=LinkingConstants(A=1.0, B=0.0, method="stocking_lord"),
            anchor_items=anchors,
        )
        size = model_old.n_parameters
        vcov = np.zeros((size, size))
        offset = 4 * model_old.n_items
        vcov[offset:, offset:] = np.eye(3) * 0.01

        se_a, se_b = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        assert se_a > 0
        assert se_b > 0


class TestComputeLinkingFit:
    """Tests for compute_linking_fit function."""

    def test_basic_fit_computation(self, linked_models_pair):
        """Test basic fit statistic computation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        fit_stats = compute_linking_fit(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert isinstance(fit_stats, LinkingFitStatistics)
        assert fit_stats.rmse_a >= 0
        assert fit_stats.rmse_b >= 0
        assert fit_stats.tcc_rmse >= 0

    def test_fit_statistics_reasonable(self, linked_models_pair):
        """Test that fit statistics are in reasonable range."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        fit_stats = compute_linking_fit(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert fit_stats.rmse_a < 2.0
        assert fit_stats.rmse_b < 2.0
        assert fit_stats.mad_a < 2.0
        assert fit_stats.mad_b < 2.0

    def test_exact_transformation_has_zero_error(self):
        """Recover a known transformation to numerical precision."""
        model_old = TwoParameterLogistic(n_items=4)
        model_new = TwoParameterLogistic(n_items=4)
        disc_old = np.array([0.7, 1.0, 1.3, 1.6])
        diff_old = np.array([-1.5, -0.3, 0.4, 1.2])
        A, B = 1.25, -0.35
        model_old.set_parameters(discrimination=disc_old, difficulty=diff_old)
        model_new.set_parameters(
            discrimination=A * disc_old,
            difficulty=(diff_old - B) / A,
        )

        fit_stats = compute_linking_fit(
            model_old, model_new, [0, 1, 2, 3], [0, 1, 2, 3], A=A, B=B
        )

        assert fit_stats.rmse_a == pytest.approx(0.0, abs=1e-14)
        assert fit_stats.rmse_b == pytest.approx(0.0, abs=1e-14)
        assert fit_stats.tcc_rmse == pytest.approx(0.0, abs=1e-14)

    def test_three_parameter_curves_include_guessing(self):
        """Detect curve misfit that a/b-only diagnostics miss."""
        model_old = ThreeParameterLogistic(n_items=3)
        model_new = ThreeParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.0, 1.2]),
            "difficulty": np.array([-1.0, 0.0, 1.0]),
        }
        model_old.set_parameters(**common, guessing=np.full(3, 0.1))
        model_new.set_parameters(**common, guessing=np.full(3, 0.4))

        fit_stats = compute_linking_fit(
            model_old, model_new, [0, 1, 2], [0, 1, 2], A=1.0, B=0.0
        )

        assert fit_stats.rmse_a == 0.0
        assert fit_stats.rmse_b == 0.0
        assert fit_stats.tcc_rmse > 0.0

    def test_four_parameter_curves_include_upper_asymptote(self):
        """Detect 4PL upper-asymptote differences."""
        model_old = FourParameterLogistic(n_items=3)
        model_new = FourParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.0, 1.2]),
            "difficulty": np.array([-1.0, 0.0, 1.0]),
            "guessing": np.full(3, 0.1),
        }
        model_old.set_parameters(**common, upper=np.full(3, 0.95))
        model_new.set_parameters(**common, upper=np.full(3, 0.75))

        fit_stats = compute_linking_fit(
            model_old, model_new, [0, 1, 2], [0, 1, 2], A=1.0, B=0.0
        )

        assert fit_stats.tcc_rmse > 0.0

    def test_five_parameter_curves_include_asymmetry(self):
        """Use each model's native curve when computing TCC fit."""
        model_old = FiveParameterLogistic(n_items=3)
        model_new = FiveParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.0, 1.2]),
            "difficulty": np.array([-1.0, 0.0, 1.0]),
            "guessing": np.full(3, 0.1),
            "upper": np.full(3, 0.9),
        }
        model_old.set_parameters(**common, asymmetry=np.ones(3))
        model_new.set_parameters(**common, asymmetry=np.full(3, 1.8))

        fit_stats = compute_linking_fit(
            model_old, model_new, [0, 1, 2], [0, 1, 2], A=1.0, B=0.0
        )

        assert fit_stats.tcc_rmse > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"A": 0.0, "B": 0.0}, "A must"),
            ({"A": 1.0, "B": np.inf}, "B must"),
            ({"A": 1.0, "B": 0.0, "theta_range": (1.0, -1.0)}, "theta_range"),
            ({"A": 1.0, "B": 0.0, "n_theta": 1}, "n_theta"),
        ],
    )
    def test_rejects_invalid_fit_configuration(
        self, linked_models_pair, kwargs, message
    ):
        """Reject invalid constants and integration grids."""
        model_old, model_new, anchors = linked_models_pair

        with pytest.raises(ValueError, match=message):
            compute_linking_fit(model_old, model_new, anchors, anchors, **kwargs)

    def test_rejects_invalid_fit_weights(self, linked_models_pair):
        """Require finite, non-negative weights with positive mass."""
        model_old, model_new, anchors = linked_models_pair
        weights = np.ones(61)
        weights[0] = -1.0

        with pytest.raises(ValueError, match="weights"):
            compute_linking_fit(
                model_old,
                model_new,
                anchors,
                anchors,
                A=1.0,
                B=0.0,
                weights=weights,
            )


class TestLinkingSummary:
    """Tests for linking_summary function."""

    def test_basic_summary(self, linked_models_pair):
        """Test basic summary generation."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        summary = linking_summary(linking_result, model_old, model_new)

        assert isinstance(summary, str)
        assert "Linking Summary" in summary
        assert "Transformation" in summary

    def test_summary_contains_constants(self, linked_models_pair):
        """Test that summary contains transformation constants."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = linking_summary(linking_result, model_old, model_new)

        assert "A" in summary
        assert "B" in summary

    def test_summary_contains_fit_statistics(self, linked_models_pair):
        """Test that summary contains fit statistics."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        summary = linking_summary(linking_result, model_old, model_new)

        if linking_result.fit_statistics is not None:
            assert "RMSE" in summary or "Fit" in summary

    def test_summary_states_transformation_direction(self, linked_models_pair):
        """Make score and item transformations explicit and consistent."""
        model_old, model_new, anchors = linked_models_pair
        linking_result = link(model_old, model_new, anchors, anchors)

        summary = linking_summary(linking_result, model_old, model_new)

        assert "Reference model: 2PL (10 items)" in summary
        assert "New model:       2PL (10 items)" in summary
        assert "theta_old =" in summary
        assert "a_new_on_old = a_new /" in summary
        assert "b_new_on_old =" in summary
        assert "theta_new =" not in summary


class TestCompareLinkingMethods:
    """Tests for compare_linking_methods function."""

    def test_basic_comparison(self, linked_models_pair):
        """Test basic method comparison."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=["mean_sigma", "mean_mean"],
        )

        assert "mean_sigma" in results
        assert "mean_mean" in results

    def test_comparison_contains_all_methods(self, linked_models_pair):
        """Test that comparison contains all requested methods."""
        model_old, model_new, anchors = linked_models_pair

        methods = ["mean_sigma", "mean_mean", "stocking_lord"]

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=methods,
        )

        for method in methods:
            assert method in results

    def test_comparison_result_structure(self, linked_models_pair):
        """Test structure of comparison results."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
            methods=["mean_sigma"],
        )

        result = results["mean_sigma"]
        assert "A" in result
        assert "B" in result

    def test_default_methods(self, linked_models_pair):
        """Test with default methods."""
        model_old, model_new, anchors = linked_models_pair

        results = compare_linking_methods(
            model_old,
            model_new,
            anchors,
            anchors,
        )

        assert len(results) > 2

    def test_comparison_uses_full_model_curves(self):
        """Report asymptote misfit in method comparisons."""
        model_old = ThreeParameterLogistic(n_items=3)
        model_new = ThreeParameterLogistic(n_items=3)
        common = {
            "discrimination": np.array([0.8, 1.0, 1.2]),
            "difficulty": np.array([-1.0, 0.0, 1.0]),
        }
        model_old.set_parameters(**common, guessing=np.full(3, 0.1))
        model_new.set_parameters(**common, guessing=np.full(3, 0.4))

        results = compare_linking_methods(
            model_old,
            model_new,
            [0, 1, 2],
            [0, 1, 2],
            methods=["mean_mean"],
        )

        assert results["mean_mean"]["tcc_rmse"] > 0.0


class TestParameterRecoverySummary:
    """Tests for parameter_recovery_summary function."""

    def test_basic_recovery_summary(self, linked_models_pair):
        """Test basic parameter recovery summary."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert isinstance(summary, str)
        assert "Recovery" in summary or "Parameter" in summary

    def test_recovery_summary_contains_items(self, linked_models_pair):
        """Test that recovery summary contains item information."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert "a_old" in summary or "diff" in summary.lower()

    def test_recovery_summary_contains_rmse(self, linked_models_pair):
        """Test that recovery summary contains RMSE."""
        model_old, model_new, anchors = linked_models_pair

        linking_result = link(model_old, model_new, anchors, anchors)

        summary = parameter_recovery_summary(
            model_old,
            model_new,
            anchors,
            anchors,
            A=linking_result.constants.A,
            B=linking_result.constants.B,
        )

        assert "RMSE" in summary

    def test_recovery_rejects_mismatched_anchors(self, linked_models_pair):
        """Report correspondence errors before array operations."""
        model_old, model_new, _ = linked_models_pair

        with pytest.raises(ValueError, match="same length"):
            parameter_recovery_summary(
                model_old, model_new, [0, 1, 2], [0, 1], A=1.0, B=0.0
            )

    def test_recovery_rejects_negative_anchor(self, linked_models_pair):
        """Do not interpret negative indices as valid anchors."""
        model_old, model_new, _ = linked_models_pair

        with pytest.raises(ValueError, match="out of range"):
            parameter_recovery_summary(
                model_old, model_new, [-1, 1], [0, 1], A=1.0, B=0.0
            )

    def test_recovery_handles_constant_parameters_without_warning(
        self, linked_models_pair
    ):
        """Render undefined correlations explicitly."""
        model_old, model_new, anchors = linked_models_pair
        model_old = model_old.copy()
        model_new = model_new.copy()
        model_old.set_parameters(discrimination=np.ones(model_old.n_items))
        model_new.set_parameters(discrimination=np.ones(model_new.n_items))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = parameter_recovery_summary(
                model_old, model_new, anchors, anchors, A=1.0, B=0.0
            )

        assert "Corr(a): n/a" in summary

    def test_recovery_displays_paired_anchor_indices(self, linked_models_pair):
        """Show both form indices for non-identical anchor mappings."""
        model_old, model_new, _ = linked_models_pair

        summary = parameter_recovery_summary(
            model_old, model_new, [0, 1], [2, 3], A=1.0, B=0.0
        )

        assert "     0      2" in summary
        assert "     1      3" in summary


class TestDiagnosticsWithRealModels:
    """Tests for diagnostics with real fitted models."""

    def test_diagnostics_with_fitted_models(self, dichotomous_responses):
        """Test diagnostics with actual fitted models."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)

        responses = dichotomous_responses["responses"]

        result1 = fit_mirt(responses, model="2PL", max_iter=15, n_quadpts=11)

        responses2 = responses.copy()
        shift = rng.choice([0, 1], size=responses.shape, p=[0.7, 0.3])
        responses2 = np.where(shift, 1 - responses2, responses2)

        result2 = fit_mirt(responses2, model="2PL", max_iter=15, n_quadpts=11)

        anchors = list(range(4))

        linking_result = link(
            result1.model,
            result2.model,
            anchors,
            anchors,
            compute_diagnostics=True,
        )

        summary = linking_summary(linking_result, result1.model, result2.model)

        assert isinstance(summary, str)
        assert "Linking" in summary
