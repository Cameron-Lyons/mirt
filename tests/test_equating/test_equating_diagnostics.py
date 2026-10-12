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

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    def test_implicit_curve_derivatives_match_reoptimization(self, method):
        """Implicit-function derivatives equal tightly re-optimized ones."""
        from scipy.optimize import least_squares
        from scipy.special import expit

        rng = np.random.default_rng(4)
        n_items = 5
        old_a = rng.uniform(0.7, 1.8, n_items)
        old_b = np.linspace(-1.5, 1.5, n_items)
        guessing = np.linspace(0.05, 0.25, n_items)
        new_a = old_a * 1.2 * np.exp(rng.normal(0, 0.1, n_items))
        new_b = (old_b - 0.3) / 1.2 + rng.normal(0, 0.1, n_items)
        model_old = ThreeParameterLogistic(n_items=n_items)
        model_new = ThreeParameterLogistic(n_items=n_items)
        model_old.set_parameters(
            discrimination=old_a, difficulty=old_b, guessing=guessing
        )
        model_new.set_parameters(
            discrimination=new_a, difficulty=new_b, guessing=guessing
        )
        anchors = list(range(n_items))
        size = model_old.n_parameters
        root = rng.normal(0.0, 0.05, (size, size))
        vcov = root @ root.T + 1e-3 * np.eye(size)
        result = link(model_old, model_new, anchors, anchors, method=method)

        se = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        theta = np.linspace(-4.0, 4.0, 61)
        root_weights = np.sqrt(np.exp(-0.5 * theta**2) / np.exp(-0.5 * theta**2).sum())

        def fit(parameters):
            a_o, b_o, c_o, a_n, b_n, c_n = np.split(parameters, 6)
            curves_old = c_o + (1 - c_o) * expit(a_o * (theta[:, None] - b_o))

            def residual(x):
                grid = (theta - x[1]) / np.exp(x[0])
                curves_new = c_n + (1 - c_n) * expit(a_n * (grid[:, None] - b_n))
                difference = curves_old - curves_new
                if method == "haebara":
                    return (root_weights[:, None] * difference).ravel()
                return root_weights * difference.sum(axis=1)

            start = [np.log(result.constants.A), result.constants.B]
            solution = least_squares(
                residual, start, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
            )
            return np.array([np.exp(solution.x[0]), solution.x[1]])

        parameters = np.concatenate([old_a, old_b, guessing, new_a, new_b, guessing])
        jacobian = np.empty((2, parameters.size))
        for index in range(parameters.size):
            step = 1e-4 * max(1.0, abs(parameters[index]))
            upper, lower = parameters.copy(), parameters.copy()
            upper[index] += step
            lower[index] -= step
            jacobian[:, index] = (fit(upper) - fit(lower)) / (2.0 * step)
        covariance = np.zeros((2 * size, 2 * size))
        covariance[:size, :size] = vcov
        covariance[size:, size:] = vcov
        expected = np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))

        np.testing.assert_allclose(se, expected, rtol=5e-5)

    def test_native_curve_family_uses_model_curves(self):
        """Non-logistic families propagate through their own response curves."""
        from scipy.optimize import least_squares

        from mirt.models.dichotomous import ComplementaryLogLog

        n_items = 4
        model_old = ComplementaryLogLog(n_items)
        model_new = ComplementaryLogLog(n_items)
        old_a = np.array([0.8, 1.1, 1.4, 1.7])
        old_b = np.array([-1.2, -0.3, 0.4, 1.3])
        model_old.set_parameters(discrimination=old_a, difficulty=old_b)
        model_new.set_parameters(
            discrimination=old_a * np.array([1.25, 1.15, 1.3, 1.2]),
            difficulty=(old_b - 0.3) / 1.2 + np.array([0.1, -0.05, 0.08, -0.1]),
        )
        anchors = list(range(n_items))
        size = model_old.n_parameters
        vcov = np.eye(size) * 0.01
        result = link(model_old, model_new, anchors, anchors)

        se = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )

        theta = np.linspace(-4.0, 4.0, 61)
        root_weights = np.sqrt(np.exp(-0.5 * theta**2) / np.exp(-0.5 * theta**2).sum())

        def fit(parameters):
            old = ComplementaryLogLog(n_items)
            new = ComplementaryLogLog(n_items)
            old.set_parameters(discrimination=parameters[0], difficulty=parameters[1])
            new.set_parameters(discrimination=parameters[2], difficulty=parameters[3])
            curves_old = old.probability(theta[:, None]).sum(axis=1)

            def residual(x):
                grid = (theta - x[1]) / np.exp(x[0])
                return root_weights * (
                    curves_old - new.probability(grid[:, None]).sum(axis=1)
                )

            start = [np.log(result.constants.A), result.constants.B]
            solution = least_squares(
                residual, start, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15
            )
            return np.array([np.exp(solution.x[0]), solution.x[1]])

        parameters = np.array(
            [old_a, old_b, model_new.discrimination, model_new.difficulty]
        )
        jacobian = np.empty((2, parameters.size))
        for index in range(parameters.size):
            row, column = divmod(index, n_items)
            step = 1e-4 * max(1.0, abs(parameters[row, column]))
            upper, lower = parameters.copy(), parameters.copy()
            upper[row, column] += step
            lower[row, column] -= step
            jacobian[:, index] = (fit(upper) - fit(lower)) / (2.0 * step)
        expected = np.sqrt(0.01 * np.sum(jacobian**2, axis=1))

        np.testing.assert_allclose(se, expected, rtol=1e-4)

    def test_native_asymptote_on_its_bound_is_probed_inside_it(self):
        """A zero lower asymptote gets one-sided differences, not a rejection."""
        from mirt.models.zeroinflated import ZeroInflated2PL, ZeroInflated3PL

        n_items = 5
        old_a = np.array([0.8, 1.0, 1.2, 1.4, 1.6])
        old_b = np.array([-1.2, -0.5, 0.0, 0.6, 1.3])
        values = {
            "old": {"discrimination": old_a, "difficulty": old_b},
            "new": {
                "discrimination": old_a * np.array([1.25, 1.15, 1.3, 1.2, 1.1]),
                "difficulty": (old_b - 0.3) / 1.2
                + np.array([0.1, -0.05, 0.08, -0.1, 0.02]),
            },
        }
        zero_inflation = np.array([0.05, 0.1, 0.0, 0.2, 0.1])
        anchors = list(range(n_items))
        rng = np.random.default_rng(9)
        root = rng.normal(0.0, 0.05, (2 * n_items, 2 * n_items))
        location_covariance = root @ root.T

        def standard_errors(model_type, guessing_variance=0.0):
            forms = []
            for label in ("old", "new"):
                model = model_type(n_items)
                model.set_parameters(**values[label], zero_inflation=zero_inflation)
                if model_type is ZeroInflated3PL:
                    # Guessing exactly on its lower bound makes ZI-3PL curves
                    # identical to ZI-2PL curves.
                    model.set_parameters(guessing=np.zeros(n_items))
                forms.append(model)
            vcov = np.zeros((forms[0].n_parameters, forms[0].n_parameters))
            vcov[: 2 * n_items, : 2 * n_items] = location_covariance
            if model_type is ZeroInflated3PL:
                guessing = slice(2 * n_items, 3 * n_items)
                vcov[guessing, guessing] = guessing_variance * np.eye(n_items)
            result = link(*forms, anchors, anchors, method="haebara")
            return np.array(
                delta_method_se(
                    result,
                    vcov,
                    vcov,
                    anchors,
                    anchors,
                    model_old=forms[0],
                    model_new=forms[1],
                )
            )

        without_guessing = standard_errors(ZeroInflated2PL)
        np.testing.assert_allclose(
            standard_errors(ZeroInflated3PL), without_guessing, rtol=1e-8
        )
        with_guessing = standard_errors(ZeroInflated3PL, guessing_variance=1e-4)
        assert np.all(np.isfinite(with_guessing))
        assert np.all(with_guessing > without_guessing)

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    def test_curve_uncertainty_accepts_tied_reference_difficulties(self, method):
        """Tied reference difficulties fall back to a mean/mean start."""
        anchors = [0, 1, 2, 3]
        model_old = TwoParameterLogistic(n_items=4)
        model_new = TwoParameterLogistic(n_items=4)
        model_old.set_parameters(
            discrimination=np.array([1.0, 1.2, 0.8, 1.5]), difficulty=np.zeros(4)
        )
        model_new.set_parameters(
            discrimination=np.array([1.1, 1.3, 0.9, 1.4]),
            difficulty=np.array([0.1, -0.1, 0.05, -0.02]),
        )
        result = link(model_old, model_new, anchors, anchors, method=method)
        vcov = np.eye(model_old.n_parameters) * 0.01

        delta = delta_method_se(
            result,
            vcov,
            vcov,
            anchors,
            anchors,
            model_old=model_old,
            model_new=model_new,
        )
        bootstrap = bootstrap_linking_se(
            model_old,
            model_new,
            None,
            None,
            anchors,
            anchors,
            method=method,
            n_bootstrap=5,
            seed=1,
        )

        assert np.all(np.isfinite(delta)) and np.all(np.asarray(delta) > 0)
        assert np.all(np.isfinite(bootstrap[:2])) and np.all(
            np.asarray(bootstrap[:2]) > 0
        )


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


@pytest.fixture(scope="module")
def fitted_form_pairs():
    """Old and new calibrations whose ``vcov`` uses the free-parameter layout."""
    from mirt import fit_mirt

    rng = np.random.default_rng(0)
    n_items, n_persons = 10, 600
    slopes = rng.uniform(0.8, 1.8, n_items)
    locations = rng.normal(0.0, 1.0, n_items)

    def simulate(mean, guessing=0.0):
        theta = rng.normal(mean, 1.0, n_persons)
        logits = slopes * (theta[:, None] - locations)
        probability = guessing + (1.0 - guessing) / (1.0 + np.exp(-logits))
        return (rng.random((n_persons, n_items)) < probability).astype(int)

    # The first item, which is not an anchor, keeps a fixed slope.
    fixed = {"discrimination": np.arange(n_items) == 0}
    settings = {
        # The unit-slope Rasch model, whose fixed slopes vcov leaves out.
        "1PL": ("1PL", {"fixed": {"discrimination": True}}, 0.0),
        "2PL-fixed": ("2PL", {"fixed": fixed}, 0.0),
        "3PL": ("3PL", {}, 0.15),
    }
    return {
        name: tuple(
            fit_mirt(simulate(mean, guessing), model=model, verbose=False, **options)
            for mean in (0.0, 0.5)
        )
        for name, (model, options, guessing) in settings.items()
    }


def _stored_layout(fit):
    """Scatter a fitted free-parameter covariance into every stored coordinate."""
    masks = fit.model.free_parameter_masks
    free = np.flatnonzero(
        np.concatenate([np.ravel(masks[name]) for name in fit.model.parameters])
    )
    size = sum(values.size for values in fit.model.parameters.values())
    stored = np.zeros((size, size))
    stored[np.ix_(free, free)] = np.nan_to_num(fit.vcov, nan=0.0)
    return stored


class TestDeltaMethodFittedCovariance:
    """``FitResult.vcov`` lists free parameters only, unlike stored arrays."""

    anchors = [1, 2, 3, 4, 5, 6]

    def _delta(self, old, new, vcov_old, vcov_new, **models):
        result = link(old.model, new.model, self.anchors, self.anchors)
        return delta_method_se(
            result, vcov_old, vcov_new, self.anchors, self.anchors, **models
        )

    def test_one_parameter_free_layout_is_mapped(self, fitted_form_pairs):
        """A 1PL covariance omits the fixed slopes instead of failing to index."""
        old, new = fitted_form_pairs["1PL"]
        assert old.vcov.shape[0] == old.model.n_parameters < 2 * old.model.n_items
        models = {"model_old": old.model, "model_new": new.model}

        actual = self._delta(old, new, old.vcov, new.vcov, **models)
        expected = self._delta(
            old, new, _stored_layout(old), _stored_layout(new), **models
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        assert actual[1] > 0.0

    def test_fixed_coordinates_do_not_shift_rows(self, fitted_form_pairs):
        """A fixed non-anchor slope must not misalign the anchors' rows."""
        old, new = fitted_form_pairs["2PL-fixed"]
        assert old.vcov_labels[0] == "discrimination[Item_2]"
        models = {"model_old": old.model, "model_new": new.model}

        actual = self._delta(old, new, old.vcov, new.vcov, **models)
        expected = self._delta(
            old, new, _stored_layout(old), _stored_layout(new), **models
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_rows_at_a_bound_have_zero_variance(self, fitted_form_pairs):
        """NaN rows of asymptotes on their bound are treated as known."""
        old, new = fitted_form_pairs["3PL"]
        assert np.any(np.isnan(np.diag(old.vcov)))
        models = {"model_old": old.model, "model_new": new.model}

        actual = self._delta(old, new, old.vcov, new.vcov, **models)
        expected = self._delta(
            old, new, _stored_layout(old), _stored_layout(new), **models
        )

        assert np.all(np.isfinite(actual)) and np.all(np.asarray(actual) > 0.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.parametrize("name", ["1PL", "2PL-fixed", "3PL"])
    def test_fit_results_supply_covariance_and_models(self, fitted_form_pairs, name):
        old, new = fitted_form_pairs[name]

        from_results = self._delta(old, new, old, new)
        from_arrays = self._delta(
            old,
            new,
            old.vcov,
            new.vcov,
            model_old=old.model,
            model_new=new.model,
        )

        np.testing.assert_allclose(from_results, from_arrays, rtol=1e-12)

    def test_unrecognized_size_names_both_layouts(self, fitted_form_pairs):
        old, new = fitted_form_pairs["1PL"]
        size = old.model.n_parameters + 1

        with pytest.raises(ValueError, match=r"\(10, 10\).*\(20, 20\)"):
            self._delta(
                old,
                new,
                np.eye(size),
                new.vcov,
                model_old=old.model,
                model_new=new.model,
            )

    def test_standard_errors_stand_in_for_a_missing_vcov(self):
        """Results fitted without a covariance use their squared SEs."""
        from mirt import fit_mirt

        rng = np.random.default_rng(0)
        n_items, n_persons = 8, 400
        slopes = rng.uniform(0.8, 1.8, n_items)
        locations = rng.normal(0.0, 1.0, n_items)

        def simulate(mean):
            theta = rng.normal(mean, 1.0, n_persons)
            probability = 1.0 / (1.0 + np.exp(-slopes * (theta[:, None] - locations)))
            return (rng.random((n_persons, n_items)) < probability).astype(int)

        old, new = (
            fit_mirt(
                simulate(mean), model="2PL", verbose=False, se_method="complete_data"
            )
            for mean in (0.0, 0.5)
        )
        assert old.vcov is None and new.vcov is None

        with pytest.warns(UserWarning, match="ignoring covariances"):
            actual = self._delta(old, new, old, new)

        def squared_errors(fit):
            return np.diag(
                np.concatenate(
                    [fit.standard_errors[name] ** 2 for name in fit.model.parameters]
                )
            )

        expected = self._delta(
            old,
            new,
            squared_errors(old),
            squared_errors(new),
            model_old=old.model,
            model_new=new.model,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        assert np.all(np.asarray(actual) > 0.0)

    @pytest.mark.parametrize("layout", ["stored", "free"])
    def test_only_the_anchor_block_is_read(self, layout):
        """Entries of non-anchor coordinates are neither read nor validated."""
        from mirt.models.dichotomous import OneParameterLogistic

        n_items, anchors = 10, [0, 1, 2, 3, 4]
        old, new = OneParameterLogistic(n_items), OneParameterLogistic(n_items)
        locations = np.linspace(-1.5, 1.5, n_items)
        old.set_parameters(difficulty=locations)
        new.set_parameters(difficulty=(locations - 0.3) + np.linspace(-0.1, 0.1, 10))
        result = link(old, new, anchors, anchors)
        # The free layout lists the difficulties only; the stored one adds the
        # fixed slopes in front.
        offset = n_items if layout == "stored" else 0
        clean = np.eye(offset + n_items) * 0.01
        dirty = clean.copy()
        others = offset + np.arange(5, n_items)
        dirty[others[0], others[0]] = -1.0
        dirty[others[1], others[2]] = 0.5
        dirty[others[3], others[4]] = np.nan

        expected = delta_method_se(result, clean, clean, anchors, anchors, old, new)

        assert expected[1] > 0.0
        assert (
            delta_method_se(result, dirty, dirty, anchors, anchors, old, new)
            == expected
        )

    def test_partially_missing_rows_are_rejected(self, linked_models_pair):
        model_old, model_new, anchors = linked_models_pair
        result = link(model_old, model_new, anchors, anchors)
        vcov = np.eye(model_old.n_parameters) * 0.01
        vcov[0, 1] = vcov[1, 0] = np.nan

        with pytest.raises(ValueError, match="finite apart from NaN"):
            delta_method_se(
                result,
                vcov,
                np.eye(model_new.n_parameters) * 0.01,
                anchors,
                anchors,
                model_old=model_old,
                model_new=model_new,
            )


class TestDeltaMethodNativeParameters:
    """Native curve families propagate every per-item curve parameter."""

    def test_zero_inflation_uncertainty_matches_relinking(self):
        from mirt.models.zeroinflated import ZeroInflated2PL

        rng = np.random.default_rng(3)
        n_items = 8
        anchors = list(range(5))
        slopes = rng.uniform(0.8, 2.0, n_items)
        locations = rng.normal(0.0, 1.0, n_items)
        inflation = rng.uniform(0.05, 0.3, n_items)
        old = ZeroInflated2PL(n_items)
        old.set_parameters(
            discrimination=slopes, difficulty=locations, zero_inflation=inflation
        )
        new = ZeroInflated2PL(n_items)
        new.set_parameters(
            discrimination=slopes * 1.2 + rng.normal(0.0, 0.1, n_items),
            difficulty=(locations - 0.3) / 1.2 + rng.normal(0.0, 0.1, n_items),
            zero_inflation=inflation,
        )
        result = link(old, new, anchors, anchors)
        variance = 0.002
        vcov = np.diag(
            np.concatenate([np.zeros(2 * n_items), np.full(n_items, variance)])
        )

        se = delta_method_se(
            result, vcov, vcov, anchors, anchors, model_old=old, model_new=new
        )

        def constants(form, item, step):
            varied = [old.copy(), new.copy()]
            values = varied[form].zero_inflation.copy()
            values[item] += step
            varied[form].set_parameters(zero_inflation=values)
            relinked = link(*varied, anchors, anchors).constants
            return np.array([relinked.A, relinked.B])

        step = 1e-4
        jacobian = np.column_stack(
            [
                (constants(form, item, step) - constants(form, item, -step))
                / (2.0 * step)
                for form in (0, 1)
                for item in anchors
            ]
        )
        expected = np.sqrt(variance * np.sum(jacobian**2, axis=1))

        assert np.all(np.asarray(se) > 0.0)
        np.testing.assert_allclose(se, expected, rtol=1e-3)


def test_free_layout_covariance_of_a_common_slope_1pl_keeps_one_row_per_item():
    """Tied coordinates keep their own vcov rows beyond ``n_parameters``."""
    import mirt
    from mirt.equating.diagnostics import delta_method_se

    def fit(seed: int, shift: float):
        rng = np.random.default_rng(seed)
        responses = mirt.simdata(
            theta=rng.normal(shift, 1.0, 800),
            discrimination=np.full(6, 1.3),
            difficulty=np.linspace(-1.5, 1.5, 6),
            seed=seed,
        )
        return mirt.fit_mirt(responses, model="1PL")

    old, new = fit(1, 0.0), fit(2, 0.4)
    assert old.vcov.shape[0] > old.model.n_parameters
    anchors = list(range(6))
    linking = mirt.link(old.model, new.model, anchors, anchors)

    expected = delta_method_se(
        linking, old, new, anchors, anchors, old.model, new.model
    )
    actual = delta_method_se(
        linking, old.vcov, new.vcov, anchors, anchors, old.model, new.model
    )
    assert actual == pytest.approx(expected)


def _noisy_polytomous_pair(family, seed=11):
    """Two noisy affine copies of one GRM, GPCM, or NRM calibration."""
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
    )

    rng = np.random.default_rng(seed)
    categories = [3, 4, 3, 4, 3]
    n_items = len(categories)
    models = []
    for scale, shift in ((1.0, 0.0), (1.2, 0.3)):
        if family == "NRM":
            model = NominalResponseModel(n_items, categories)
            slopes = model.parameters["slopes"].copy()
            intercepts = model.parameters["intercepts"].copy()
            for item, count in enumerate(categories):
                base = np.linspace(0.0, 1.2 + 0.1 * item, count)
                slopes[item, :count] = base / scale + rng.normal(0.0, 0.03, count)
                intercepts[item, :count] = (
                    np.linspace(0.0, 0.4 * item - 0.5, count)
                    - base * shift / scale
                    + rng.normal(0.0, 0.03, count)
                )
            slopes[:, 0] = intercepts[:, 0] = 0.0
            model.set_parameters(slopes=slopes, intercepts=intercepts)
        else:
            model_class = (
                GradedResponseModel if family == "GRM" else GeneralizedPartialCredit
            )
            model = model_class(n_items, categories)
            name = "thresholds" if family == "GRM" else "steps"
            locations = np.array(model.parameters[name], copy=True)
            for item, count in enumerate(categories):
                base = np.linspace(-1.2, 1.0, count - 1) + 0.2 * item
                locations[item, : count - 1] = (
                    scale * base + shift + rng.normal(0.0, 0.04, count - 1)
                )
            model.set_parameters(
                discrimination=np.linspace(0.8, 1.6, n_items)
                / scale
                * np.exp(rng.normal(0.0, 0.04, n_items)),
                **{name: locations},
            )
        models.append(model)
    return models


def _stored_covariance(model, seed):
    """A covariance over every stored coordinate, zero for fixed ones."""
    rng = np.random.default_rng(seed)
    free = np.concatenate(
        [mask.ravel() for mask in model.free_parameter_masks.values()]
    )
    root = rng.normal(0.0, 0.03, (free.size, free.size))
    covariance = root @ root.T + 1e-4 * np.eye(free.size)
    covariance[~free, :] = covariance[:, ~free] = 0.0
    return covariance


class TestPolytomousLinkingUncertainty:
    """Delta-method and bootstrap standard errors of polytomous links."""

    @pytest.mark.parametrize(
        ("family", "method"),
        [
            ("GRM", "stocking_lord"),
            ("GRM", "haebara"),
            ("GRM", "mean_sigma"),
            ("GPCM", "haebara"),
            ("GPCM", "mean_mean"),
            ("NRM", "stocking_lord"),
            ("NRM", "haebara"),
        ],
    )
    def test_delta_method_matches_relinking(self, family, method):
        """Implicit derivatives equal central differences of relinked constants."""
        from mirt.equating import link_gpcm, link_grm, link_nrm

        linker = {"GRM": link_grm, "GPCM": link_gpcm, "NRM": link_nrm}[family]
        old, new = _noisy_polytomous_pair(family)
        anchors = list(range(old.n_items))
        vcovs = [_stored_covariance(old, 1), _stored_covariance(new, 2)]
        result = linker(old, new, anchors, anchors, method=method)

        se = delta_method_se(
            result, *vcovs, anchors, anchors, model_old=old, model_new=new
        )

        columns = []
        for side, model in enumerate((old, new)):
            names = list(model.parameters)
            stored = np.concatenate([model.parameters[name].ravel() for name in names])
            free = np.concatenate(
                [model.free_parameter_masks[name].ravel() for name in names]
            )
            for coordinate in range(stored.size):
                # Fixed coordinates, such as NRM reference categories and
                # category padding, have zero variance.
                if not free[coordinate]:
                    columns.append(np.zeros(2))
                    continue
                step = 1e-4 * max(1.0, abs(stored[coordinate]))
                constants = []
                for sign in (1.0, -1.0):
                    values = stored.copy()
                    values[coordinate] += sign * step
                    varied = [old.copy(), new.copy()]
                    offset, updates = 0, {}
                    for name in names:
                        size = model.parameters[name].size
                        updates[name] = values[offset : offset + size].reshape(
                            model.parameters[name].shape
                        )
                        offset += size
                    varied[side].set_parameters(**updates)
                    relinked = linker(
                        *varied,
                        anchors,
                        anchors,
                        method=method,
                        compute_diagnostics=False,
                    ).constants
                    constants.append(np.array([relinked.A, relinked.B]))
                columns.append((constants[0] - constants[1]) / (2.0 * step))
        jacobian = np.column_stack(columns)
        sizes = [vcov.shape[0] for vcov in vcovs]
        covariance = np.zeros((sum(sizes), sum(sizes)))
        covariance[: sizes[0], : sizes[0]] = vcovs[0]
        covariance[sizes[0] :, sizes[0] :] = vcovs[1]
        expected = np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))

        assert np.all(np.asarray(se) > 0.0)
        # Relinked curve constants carry Nelder-Mead noise near 1e-8.
        np.testing.assert_allclose(se, expected, rtol=2e-3)

    def test_delta_method_accepts_polytomous_fit_results(self):
        import mirt

        rng = np.random.default_rng(5)
        fits = []
        for shift in (0.0, 0.4):
            theta = rng.normal(shift, 1.0, 600)
            thresholds = (
                np.array([[-1.0, 0.0, 1.0]] * 5) + np.linspace(-0.5, 0.5, 5)[:, None]
            )
            cumulative = 1.0 / (
                1.0 + np.exp(-1.3 * (theta[:, None, None] - thresholds[None]))
            )
            responses = (rng.random((600, 5, 1)) < cumulative).sum(axis=2)
            fits.append(mirt.fit_mirt(responses, model="GRM"))
        anchors = list(range(5))
        result = mirt.equating.link_grm(fits[0].model, fits[1].model, anchors, anchors)

        se = delta_method_se(result, fits[0], fits[1], anchors, anchors)
        from_matrices = delta_method_se(
            result,
            fits[0].vcov,
            fits[1].vcov,
            anchors,
            anchors,
            model_old=fits[0].model,
            model_new=fits[1].model,
        )

        assert np.all(np.isfinite(se)) and np.all(np.asarray(se) > 0.0)
        np.testing.assert_allclose(se, from_matrices, rtol=1e-12)

    def test_polytomous_methods_and_families_are_validated(self):
        from mirt.equating import link_grm
        from mirt.models.polytomous import GeneralizedPartialCredit

        old, new = _noisy_polytomous_pair("GRM")
        anchors = list(range(old.n_items))
        result = link_grm(old, new, anchors, anchors)
        vcov = _stored_covariance(old, 1)
        tcc = LinkingResult(
            constants=LinkingConstants(A=1.0, B=0.0, method="tcc"),
            anchor_items=anchors,
        )

        with pytest.raises(ValueError, match="for GRM forms: tcc"):
            delta_method_se(tcc, vcov, vcov, anchors, anchors, old, new)
        gpcm = GeneralizedPartialCredit(old.n_items, old.n_categories)
        with pytest.raises(ValueError, match="share one response family"):
            delta_method_se(result, vcov, vcov, anchors, anchors, old, gpcm)
        with pytest.raises(ValueError, match="dichotomous or both polytomous"):
            delta_method_se(
                result,
                vcov,
                vcov,
                anchors,
                anchors,
                old,
                TwoParameterLogistic(old.n_items),
            )

    @pytest.mark.parametrize("family", ["GRM", "NRM"])
    def test_anchor_bootstrap_matches_linker_standard_errors(self, family):
        from mirt.equating import link_grm, link_nrm

        linker = link_grm if family == "GRM" else link_nrm
        old, new = _noisy_polytomous_pair(family)
        anchors = list(range(old.n_items))

        se_a, se_b, a_samples, b_samples = bootstrap_linking_se(
            old, new, None, None, anchors, anchors, n_bootstrap=20, seed=3
        )
        linked = linker(
            old, new, anchors, anchors, compute_se=True, n_bootstrap=20, random_state=3
        ).constants

        assert se_a > 0.0 and se_b > 0.0
        assert a_samples.shape == b_samples.shape == (20,)
        assert (linked.A_se, linked.B_se) == pytest.approx((se_a, se_b), rel=1e-12)

    def test_response_bootstrap_refits_polytomous_forms(self):
        old, new = _noisy_polytomous_pair("GPCM")
        anchors = list(range(old.n_items))
        responses = np.random.default_rng(0).integers(0, 3, (30, old.n_items))
        refits = []

        def refit(model, sampled):
            refits.append(sampled.shape)
            return model

        se_a, se_b, a_samples, _ = bootstrap_linking_se(
            old,
            new,
            responses,
            responses,
            anchors,
            anchors,
            method="haebara",
            n_bootstrap=3,
            seed=1,
            refit=refit,
        )

        assert refits == [(30, old.n_items)] * 6
        # Unchanged refits reproduce the point estimate in every replicate.
        assert (se_a, se_b) == pytest.approx((0.0, 0.0), abs=1e-12)
        assert np.ptp(a_samples) == pytest.approx(0.0, abs=1e-12)
