"""Tests for bootstrap standard errors and confidence intervals."""

from types import SimpleNamespace

import numpy as np
import pytest

from mirt import bootstrap_ci, bootstrap_se, parametric_bootstrap
from mirt.backends.rust import _helpers as rust_helpers
from mirt.backends.rust import estimation as rust_estimation
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.dichotomous import FourParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


class TestBootstrapSE:
    """Tests for bootstrap standard errors."""

    def test_bootstrap_se(self, fitted_2pl_model, dichotomous_responses):
        """Test bootstrap SE computation."""
        responses = dichotomous_responses["responses"]

        se = bootstrap_se(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            seed=42,
        )

        assert "discrimination" in se or "discrimination_se" in se.keys()
        assert "difficulty" in se or "difficulty_se" in se.keys()

    def test_bootstrap_se_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that bootstrap SEs are positive."""
        responses = dichotomous_responses["responses"]

        se = bootstrap_se(fitted_2pl_model, responses, n_bootstrap=3, seed=42)

        for key, values in se.items():
            if isinstance(values, np.ndarray):
                assert np.all(values >= 0)

    @pytest.mark.parametrize("n_bootstrap", [0, 1, -1, 1.5, True])
    def test_rejects_invalid_resample_count(
        self, fitted_2pl_model, dichotomous_responses, n_bootstrap
    ):
        responses = dichotomous_responses["responses"]

        with pytest.raises(MirtValidationError, match="n_bootstrap"):
            bootstrap_se(
                fitted_2pl_model,
                responses,
                n_bootstrap=n_bootstrap,
            )

    def test_rejects_unknown_statistic(self, fitted_2pl_model, dichotomous_responses):
        responses = dichotomous_responses["responses"]

        with pytest.raises(MirtValidationError, match="statistic"):
            bootstrap_se(
                fitted_2pl_model,
                responses,
                n_bootstrap=2,
                statistic="unsupported",
            )

    def test_rejects_response_shape_mismatch(self, fitted_2pl_model):
        with pytest.raises(MirtDataError, match="items"):
            bootstrap_se(
                fitted_2pl_model,
                np.ones((5, fitted_2pl_model.model.n_items + 1), dtype=np.int_),
                n_bootstrap=2,
            )

    def test_theta_bootstrap_scores_original_respondents(
        self, fitted_2pl_model, dichotomous_responses, monkeypatch
    ):
        responses = dichotomous_responses["responses"][:8]
        scored_responses = []

        def fake_fit(self, model, boot_responses):
            return SimpleNamespace(model=model)

        def fake_fscores(model, score_responses, method):
            scored_responses.append(score_responses.copy())
            return SimpleNamespace(theta=np.arange(score_responses.shape[0]))

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)
        monkeypatch.setattr("mirt.scoring.fscores", fake_fscores)

        result = bootstrap_se(
            fitted_2pl_model,
            responses,
            n_bootstrap=2,
            statistic="theta",
            seed=42,
        )

        assert result["theta"].shape == (responses.shape[0],)
        assert len(scored_responses) == 2
        assert all(np.array_equal(scored, responses) for scored in scored_responses)

    def test_2pl_parameter_bootstrap_uses_native_parallel_samples(self, monkeypatch):
        """Eligible parameter bootstraps use warm-started native samples."""
        model = TwoParameterLogistic(2)
        model.set_parameters(
            discrimination=np.array([1.25, 0.75]),
            difficulty=np.array([-0.5, 0.25]),
        )
        model._is_fitted = True
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        calls = []

        def fake_bootstrap(responses, **kwargs):
            calls.append((responses.copy(), kwargs))
            n_bootstrap = kwargs["n_bootstrap"]
            values = np.arange(n_bootstrap * 2, dtype=float).reshape(n_bootstrap, 2)
            return values, values + 0.5

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: True)
        monkeypatch.setattr(rust_estimation, "bootstrap_fit_2pl", fake_bootstrap)

        result = bootstrap_se(model, responses, n_bootstrap=10, seed=42)

        values = np.arange(20, dtype=float).reshape(10, 2)
        np.testing.assert_allclose(
            result["discrimination"], np.std(values, axis=0, ddof=1)
        )
        np.testing.assert_allclose(
            result["difficulty"], np.std(values + 0.5, axis=0, ddof=1)
        )
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0][0], responses)
        np.testing.assert_array_equal(
            calls[0][1]["initial_discrimination"], model.discrimination
        )
        np.testing.assert_array_equal(
            calls[0][1]["initial_difficulty"], model.difficulty
        )
        assert calls[0][1]["max_iter"] == 100
        assert calls[0][1]["tol"] == pytest.approx(1e-3)

    def test_native_cold_start_omits_initial_parameters(self, monkeypatch):
        """Cold-start configuration is preserved by the native path."""
        model = TwoParameterLogistic(2)
        model._is_fitted = True
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        calls = []

        def fake_bootstrap(responses, **kwargs):
            calls.append(kwargs)
            return np.ones((2, 2)), np.zeros((2, 2))

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: True)
        monkeypatch.setattr(rust_estimation, "bootstrap_fit_2pl", fake_bootstrap)

        bootstrap_se(
            model,
            responses,
            n_bootstrap=2,
            warm_start=False,
            seed=42,
        )

        assert calls[0]["initial_discrimination"] is None
        assert calls[0]["initial_difficulty"] is None
        assert calls[0]["max_iter"] == 200

    def test_2pl_parameter_bootstrap_falls_back_when_native_is_disabled(
        self, monkeypatch
    ):
        """Global backend selection retains the general implementation."""
        model = TwoParameterLogistic(2)
        model._is_fitted = True
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        fit_calls = 0

        def fake_fit(self, fitted_model, sample):
            nonlocal fit_calls
            fit_calls += 1
            fitted_model._parameters["difficulty"] += 0.1 * fit_calls
            return SimpleNamespace(model=fitted_model)

        def unexpected_native_call(*args, **kwargs):
            raise AssertionError("native bootstrap should not be called")

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: False)
        monkeypatch.setattr(
            rust_estimation, "bootstrap_fit_2pl", unexpected_native_call
        )
        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        result = bootstrap_se(model, responses, n_bootstrap=2, seed=42)

        assert fit_calls == 2
        assert set(result) == {"discrimination", "difficulty"}
        assert np.isfinite(result["difficulty"]).all()


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_percentile(self, fitted_2pl_model, dichotomous_responses):
        """Test percentile bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="percentile",
            alpha=0.05,
            seed=42,
        )

        assert "discrimination" in ci or "difficulty" in ci
        for key, value in ci.items():
            if isinstance(value, tuple):
                assert len(value) == 2

    def test_bootstrap_ci_basic(self, fitted_2pl_model, dichotomous_responses):
        """Test basic bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="basic",
            alpha=0.05,
            seed=42,
        )

        assert ci is not None

    def test_bootstrap_ci_bca(self, fitted_2pl_model, dichotomous_responses):
        """Test BCa bootstrap CI."""
        responses = dichotomous_responses["responses"]

        ci = bootstrap_ci(
            fitted_2pl_model,
            responses,
            n_bootstrap=3,
            method="BCa",
            alpha=0.05,
            seed=42,
        )

        assert ci is not None

    @pytest.mark.parametrize("method", ["bca", "studentized", ""])
    def test_rejects_unknown_method(
        self, fitted_2pl_model, dichotomous_responses, method
    ):
        responses = dichotomous_responses["responses"]

        with pytest.raises(MirtValidationError, match="method"):
            bootstrap_ci(
                fitted_2pl_model,
                responses,
                n_bootstrap=2,
                method=method,
            )

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, True])
    def test_rejects_invalid_alpha(
        self, fitted_2pl_model, dichotomous_responses, alpha
    ):
        responses = dichotomous_responses["responses"]

        with pytest.raises(MirtValidationError, match="alpha"):
            bootstrap_ci(
                fitted_2pl_model,
                responses,
                n_bootstrap=2,
                alpha=alpha,
            )

    def test_rejects_invalid_custom_statistic_result(
        self, fitted_2pl_model, dichotomous_responses
    ):
        responses = dichotomous_responses["responses"]

        with pytest.raises(MirtValidationError, match="non-empty mapping"):
            bootstrap_ci(
                fitted_2pl_model,
                responses,
                n_bootstrap=2,
                statistic=lambda model, data: {},
            )

    def test_bca_supports_matrix_parameters_and_reuses_jackknife(self, monkeypatch):
        model = GradedResponseModel(n_items=2, n_categories=3)
        responses = np.tile(np.array([[0, 1], [1, 2], [2, 0]]), (4, 1))
        fit_calls = 0

        def fake_fit(self, fitted_model, sample):
            nonlocal fit_calls
            fit_calls += 1
            for name, values in fitted_model._parameters.items():
                fitted_model._parameters[name] = values + 0.01 * fit_calls
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        intervals = bootstrap_ci(
            model,
            responses,
            n_bootstrap=10,
            method="BCa",
            seed=42,
        )

        assert fit_calls == 10 + responses.shape[0]
        for name, (lower, upper) in intervals.items():
            assert lower.shape == model.parameters[name].shape
            assert upper.shape == model.parameters[name].shape
            assert np.all(np.isfinite(lower))
            assert np.all(np.isfinite(upper))

    def test_2pl_percentile_ci_uses_native_parallel_samples(self, monkeypatch):
        """Parameter confidence intervals consume native bootstrap draws."""
        model = TwoParameterLogistic(2)
        model._is_fitted = True
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        values = np.arange(20, dtype=float).reshape(10, 2)

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: True)
        monkeypatch.setattr(
            rust_estimation,
            "bootstrap_fit_2pl",
            lambda responses, **kwargs: (values, values + 0.5),
        )

        intervals = bootstrap_ci(
            model,
            responses,
            n_bootstrap=10,
            alpha=0.2,
            method="percentile",
            seed=42,
        )

        np.testing.assert_allclose(
            intervals["discrimination"][0], np.percentile(values, 10, axis=0)
        )
        np.testing.assert_allclose(
            intervals["discrimination"][1], np.percentile(values, 90, axis=0)
        )
        np.testing.assert_allclose(
            intervals["difficulty"][0], np.percentile(values + 0.5, 10, axis=0)
        )
        np.testing.assert_allclose(
            intervals["difficulty"][1], np.percentile(values + 0.5, 90, axis=0)
        )


class TestParametricBootstrap:
    """Tests for parametric bootstrap."""

    def test_parametric_bootstrap(self, fitted_2pl_model):
        """Test parametric bootstrap."""
        bootstrap_results = parametric_bootstrap(
            fitted_2pl_model,
            n_bootstrap=3,
            seed=42,
        )

        assert isinstance(bootstrap_results, dict)
        assert "discrimination" in bootstrap_results
        assert "difficulty" in bootstrap_results

    def test_parametric_bootstrap_variance(self, fitted_2pl_model):
        """Test parametric bootstrap variance estimation."""
        bootstrap_results = parametric_bootstrap(
            fitted_2pl_model,
            n_bootstrap=3,
            seed=42,
        )

        disc_estimates = bootstrap_results["discrimination"]

        variances = np.var(disc_estimates, axis=0)
        assert np.all(variances >= 0)

    @pytest.mark.parametrize("n_persons", [0, -1, 1.5, True])
    def test_rejects_invalid_person_count(self, fitted_2pl_model, n_persons):
        with pytest.raises(MirtValidationError, match="n_persons"):
            parametric_bootstrap(
                fitted_2pl_model,
                n_bootstrap=2,
                n_persons=n_persons,
            )

    def test_supports_multidimensional_polytomous_models(self, monkeypatch):
        model = GradedResponseModel(n_items=2, n_categories=3, n_factors=2)
        simulated = []

        def fake_fit(self, fitted_model, responses):
            simulated.append(responses.copy())
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        result = parametric_bootstrap(
            model,
            n_bootstrap=2,
            n_persons=40,
            seed=42,
        )

        assert set(result) == set(model.parameters)
        assert len(simulated) == 2
        assert all(responses.shape == (40, 2) for responses in simulated)
        assert all(
            np.all((responses >= 0) & (responses <= 2)) for responses in simulated
        )

    def test_four_parameter_simulation_respects_upper_asymptote(self, monkeypatch):
        model = FourParameterLogistic(n_items=3)
        model.set_parameters(
            discrimination=np.ones(3),
            difficulty=np.zeros(3),
            guessing=np.zeros(3),
            upper=np.zeros(3),
        )
        simulated = []

        def fake_fit(self, fitted_model, responses):
            simulated.append(responses.copy())
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        parametric_bootstrap(
            model,
            n_bootstrap=2,
            n_persons=50,
            seed=42,
        )

        assert len(simulated) == 2
        assert all(not np.any(responses) for responses in simulated)

    def test_cold_start_reinitializes_parameters(self, monkeypatch):
        model = FourParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.full(2, 2.0),
            difficulty=np.full(2, 1.0),
            guessing=np.full(2, 0.4),
            upper=np.full(2, 0.8),
        )
        starts = []

        def fake_fit(self, fitted_model, responses):
            starts.append(fitted_model.parameters)
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        parametric_bootstrap(
            model,
            n_bootstrap=2,
            n_persons=10,
            seed=42,
            warm_start=False,
        )

        assert all(
            np.array_equal(start["discrimination"], np.ones(2)) for start in starts
        )
        assert all(np.array_equal(start["difficulty"], np.zeros(2)) for start in starts)
        assert all(
            np.array_equal(start["guessing"], np.full(2, 0.2)) for start in starts
        )
        assert all(np.array_equal(start["upper"], np.ones(2)) for start in starts)
