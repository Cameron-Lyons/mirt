"""Tests for bootstrap standard errors and confidence intervals."""

import pickle
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mirt import bootstrap_ci, bootstrap_se, parametric_bootstrap
from mirt._categorical import draw_item_responses
from mirt.backends.rust import _helpers as rust_helpers
from mirt.backends.rust import estimation as rust_estimation
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.dichotomous import FourParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.bootstrap import (
    _bca_interval,
    _elementwise_percentile,
    _fit_jackknife_task,
    _fit_statistic_task,
    _iter_sample_indices,
    _JackknifeMoments,
    _prepare_bootstrap_model,
    _resample_rng_chunks,
    _run_bootstrap_tasks,
    _StatisticFitTask,
)


def _response_mean_statistic(model, sample):
    """A picklable statistic for real process-worker ordering checks."""
    return {"mean": sample.mean(axis=0)}


def _decimal_jackknife_acceleration(values):
    """Independent central moments with enough precision for extreme offsets."""
    flat = values.reshape(len(values), -1)
    result = []
    with localcontext() as context:
        context.prec = 80
        for column in flat.T:
            decimals = [Decimal(float(value)) for value in column]
            mean = sum(decimals) / len(decimals)
            differences = [mean - value for value in decimals]
            m2 = sum(value**2 for value in differences)
            m3 = sum(value**3 for value in differences)
            result.append(float(m3 / (6 * m2.sqrt() ** 3)) if m2 else 0.0)
    return np.array(result).reshape(values.shape[1:])


@pytest.mark.parametrize("chunk_sizes", [[17], [1] * 17, [2, 5, 4, 6]])
def test_streamed_jackknife_moments_match_decimal_reference(chunk_sizes):
    offsets = np.array([0, 1, 2, 3, 4, 5, 10, 40, 8, 7, 19, 10, -1, 30, 1, 3, 2])
    values = np.column_stack(
        [
            [-1e308] * 8 + [1e308] * 9,
            1e100 + offsets * np.spacing(1e100),
            offsets * -1e-150,
            np.full(17, 42.0),
        ]
    ).reshape(17, 2, 2)
    reference = _decimal_jackknife_acceleration(values)
    merged = None
    start = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for size in chunk_sizes:
            summary = _JackknifeMoments.from_value(values[start])
            for row in values[start + 1 : start + size]:
                summary.add(row)
            if merged is None:
                merged = summary
            else:
                merged.merge(summary)
            start += size
        acceleration = merged.acceleration()
    assert merged.count == len(values)
    np.testing.assert_allclose(acceleration, reference, rtol=2e-13, atol=1e-15)


def test_jackknife_worker_returns_fixed_size_theta_summaries(monkeypatch):
    n_persons = 4096
    responses = np.tile(np.array([[0], [1]]), (n_persons // 2, 1))
    model = TwoParameterLogistic(n_items=1)

    def fake_fit(self, fitted_model, sample):
        fitted_model._parameters["difficulty"][0] = sample.sum()
        return SimpleNamespace(model=fitted_model)

    def fake_scores(fitted, original_responses, method):
        # Replicates are scored through their fit, which carries its population.
        difficulty = fitted.model._parameters["difficulty"][0]
        scores = (np.arange(len(original_responses)) + 1) * difficulty
        return SimpleNamespace(theta=scores.astype(float))

    monkeypatch.setattr(EMEstimator, "fit", fake_fit)
    monkeypatch.setattr("mirt.scoring.fscores", fake_scores)
    result_sizes = []
    for n_omitted in [8, 2048]:
        task = _StatisticFitTask(
            model=model,
            original_params=model.parameters,
            warm_start=True,
            max_iter=1,
            responses=responses,
            statistic="theta",
            omitted_indices=list(range(n_omitted)),
            statistic_shapes={"theta": (n_persons,)},
        )
        summary = _fit_jackknife_task(task)
        assert summary["theta"].count == n_omitted
        np.testing.assert_allclose(summary["theta"].acceleration(), 0.0, atol=1e-14)
        serialized_size = len(pickle.dumps(summary))
        # Five arrays per statistic plus bounded dataclass/pickle metadata;
        # retaining n_omitted theta vectors would exceed this by orders of magnitude.
        assert serialized_size < 5 * n_persons * 8 + 2048
        result_sizes.append(serialized_size)
    assert abs(result_sizes[1] - result_sizes[0]) < 64


def test_jackknife_sample_indices_are_generated_only_when_consumed(monkeypatch):
    responses = np.tile([[0], [1]], (8, 1))
    model = TwoParameterLogistic(n_items=1)
    task = _StatisticFitTask(
        model=model,
        original_params=model.parameters,
        warm_start=True,
        max_iter=1,
        responses=responses,
        statistic="parameters",
        omitted_indices=list(range(len(responses))),
    )
    original_delete = np.delete
    calls = []

    def capture(indices, omitted):
        calls.append(omitted)
        return original_delete(indices, omitted)

    monkeypatch.setattr(np, "delete", capture)
    iterator = _iter_sample_indices(task)
    assert calls == []
    first = next(iterator)
    assert calls == [0]
    np.testing.assert_array_equal(first, np.arange(1, len(responses)))
    next(iterator)
    assert calls == [0, 1]


@pytest.mark.parametrize("alpha", [0.05, 0.0001])
@pytest.mark.parametrize("scale", [1e-150, 1e-9, 1.0, 1e150])
def test_bca_interval_matches_scipy_and_is_independent_of_statistic_units(alpha, scale):
    from scipy.stats import bootstrap

    data = np.array([0.1, 0.2, 0.3, 0.7, 0.9, 1.0, 1.1, 1.4, 2.0, 2.5, 3.0, 5.0, 9.0])
    rng = np.random.default_rng(221)
    samples = np.array(
        [data[rng.integers(0, data.size, size=data.size)].mean() for _ in range(4000)]
    )
    jackknife = [
        np.asarray(np.delete(data, row).mean() * scale) for row in range(data.size)
    ]
    reference = bootstrap(
        (data,),
        np.mean,
        n_resamples=len(samples),
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=np.random.default_rng(221),
    ).confidence_interval

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        lower, upper = _bca_interval(
            samples * scale, np.asarray(data.mean() * scale), jackknife, alpha
        )

    np.testing.assert_allclose(
        np.array([lower, upper]) / scale, [reference.low, reference.high], rtol=1e-12
    )


def test_bca_interval_matches_scipy_for_a_discrete_statistic_with_ties():
    from scipy.stats import bootstrap

    data = np.array([0.0] * 23 + [1.0] * 8)
    rng = np.random.default_rng(221)
    samples = np.array(
        [data[rng.integers(0, data.size, size=data.size)].mean() for _ in range(4000)]
    )
    jackknife = [np.asarray(np.delete(data, row).mean()) for row in range(data.size)]
    reference = bootstrap(
        (data,),
        np.mean,
        n_resamples=len(samples),
        random_state=np.random.default_rng(221),
    ).confidence_interval

    lower, upper = _bca_interval(samples, np.asarray(data.mean()), jackknife, 0.05)

    np.testing.assert_allclose(
        [lower, upper], [reference.low, reference.high], rtol=1e-12
    )


def test_elementwise_percentile_matches_independent_numpy_quantiles():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(31, 4, 5))
    quantiles = rng.uniform(0.01, 0.99, size=(4, 5))

    actual = _elementwise_percentile(samples, quantiles)
    expected = np.empty((4, 5))
    for index in np.ndindex(expected.shape):
        expected[index] = np.percentile(
            samples[(slice(None), *index)], 100 * quantiles[index]
        )

    np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=0.0)


def test_resample_rng_chunks_preserve_the_original_random_stream():
    expected_rng = np.random.default_rng(42)
    expected = [expected_rng.integers(0, 17, size=17) for _ in range(13)]

    rng = np.random.default_rng(42)
    chunks = _resample_rng_chunks(rng, 13, 17, 4)
    actual = []
    for state, chunk_size in chunks:
        chunk_rng = np.random.default_rng()
        chunk_rng.bit_generator.state = state
        actual.extend(chunk_rng.integers(0, 17, size=17) for _ in range(chunk_size))

    assert [chunk_size for _, chunk_size in chunks] == [4, 3, 3, 3]
    for actual_indices, expected_indices in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(
        rng.integers(0, 17, size=17),
        expected_rng.integers(0, 17, size=17),
    )


def test_elementwise_percentile_requires_one_quantile_per_element():
    with pytest.raises(ValueError, match="one value per sample element"):
        _elementwise_percentile(np.ones((10, 2, 3)), np.ones(5))


@pytest.mark.parametrize("quantile", [-0.1, 1.1, np.nan])
def test_elementwise_percentile_rejects_invalid_quantiles(quantile):
    with pytest.raises(ValueError, match="finite values"):
        _elementwise_percentile(np.ones((10, 2)), np.full(2, quantile))


def test_elementwise_percentile_rejects_empty_samples():
    with pytest.raises(ValueError, match="at least one"):
        _elementwise_percentile(np.empty((0, 2)), np.full(2, 0.5))


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

    @pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True])
    def test_rejects_invalid_worker_count(
        self, fitted_2pl_model, dichotomous_responses, n_jobs
    ):
        with pytest.raises(MirtValidationError, match="n_jobs"):
            bootstrap_se(
                fitted_2pl_model,
                dichotomous_responses["responses"],
                n_bootstrap=2,
                n_jobs=n_jobs,
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

    def test_parallel_replicates_match_serial_results(self, monkeypatch):
        model = TwoParameterLogistic(3)
        responses = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: False)

        serial = bootstrap_se(model, responses, n_bootstrap=4, seed=42)
        parallel = bootstrap_se(
            model,
            responses,
            n_bootstrap=4,
            seed=42,
            n_jobs=2,
        )

        assert serial.keys() == parallel.keys()
        for name in serial:
            np.testing.assert_allclose(
                parallel[name], serial[name], rtol=0.0, atol=1e-12
            )

    def test_parallel_replicates_reject_unpicklable_statistics(self, monkeypatch):
        model = TwoParameterLogistic(2)
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: False)

        with pytest.raises(MirtValidationError, match="picklable"):
            bootstrap_se(
                model,
                responses,
                n_bootstrap=2,
                statistic=lambda fitted_model, sample: {"mean": sample.mean()},
                seed=42,
                n_jobs=2,
            )


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

    def test_bca_uses_every_person_in_the_jackknife_and_matches_scipy(
        self, monkeypatch
    ):
        from scipy.stats import bootstrap

        model = TwoParameterLogistic(n_items=1)
        responses = np.array([0] * 23 + [1] * 8).reshape(-1, 1)
        jackknife_sizes = []

        def fake_fit(self, fitted_model, sample):
            if len(sample) < len(responses):
                jackknife_sizes.append(len(sample))
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)
        intervals = bootstrap_ci(
            model,
            responses,
            n_bootstrap=4000,
            statistic=lambda model, data: {"mean": data.mean(axis=0)},
            method="BCa",
            seed=221,
        )
        reference = bootstrap(
            (responses[:, 0],),
            np.mean,
            n_resamples=4000,
            random_state=np.random.default_rng(221),
        ).confidence_interval

        assert jackknife_sizes == [len(responses) - 1] * len(responses)
        np.testing.assert_allclose(
            [intervals["mean"][0][0], intervals["mean"][1][0]],
            [reference.low, reference.high],
            rtol=1e-12,
        )

    def test_bca_requires_all_jackknife_fits_to_succeed(self, monkeypatch):
        model = TwoParameterLogistic(n_items=1)
        responses = np.array([[0], [1], [1], [0]])

        def fake_fit(self, fitted_model, sample):
            if len(sample) < len(responses):
                raise RuntimeError("jackknife failed")
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)
        with pytest.warns(RuntimeWarning, match="full jackknife"):
            intervals = bootstrap_ci(
                model,
                responses,
                n_bootstrap=10,
                statistic=lambda model, data: {"mean": data.mean(axis=0)},
                method="BCa",
                seed=71,
            )

        assert np.isnan(intervals["mean"]).all()

    def test_bca_requires_more_than_one_person(self):
        with pytest.raises(MirtValidationError, match="at least two people"):
            bootstrap_ci(TwoParameterLogistic(1), np.array([[1]]), method="BCa")

    def test_bca_process_workers_match_serial_streaming_acceleration(self):
        model = TwoParameterLogistic(n_items=2)
        responses = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 0], [0, 1]])
        options = {
            "n_bootstrap": 32,
            "statistic": _response_mean_statistic,
            "method": "BCa",
            "seed": 391,
        }
        serial = bootstrap_ci(model, responses, n_jobs=1, **options)
        parallel = bootstrap_ci(model, responses, n_jobs=2, **options)
        for name in serial:
            np.testing.assert_allclose(
                parallel[name], serial[name], rtol=0.0, atol=1e-14
            )

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

    def test_parallel_confidence_intervals_match_serial_results(self, monkeypatch):
        model = TwoParameterLogistic(3)
        responses = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: False)

        serial = bootstrap_ci(
            model,
            responses,
            n_bootstrap=12,
            alpha=0.2,
            method="BCa",
            seed=42,
        )
        parallel = bootstrap_ci(
            model,
            responses,
            n_bootstrap=12,
            alpha=0.2,
            method="BCa",
            seed=42,
            n_jobs=3,
        )

        assert serial.keys() == parallel.keys()
        for name in serial:
            np.testing.assert_allclose(
                parallel[name][0], serial[name][0], rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                parallel[name][1], serial[name][1], rtol=0.0, atol=1e-12
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

    def test_parallel_parametric_bootstrap_matches_serial_results(self):
        model = FourParameterLogistic(n_items=3)

        serial = parametric_bootstrap(
            model,
            n_bootstrap=4,
            n_persons=80,
            seed=42,
        )
        parallel = parametric_bootstrap(
            model,
            n_bootstrap=4,
            n_persons=80,
            seed=42,
            n_jobs=2,
        )

        assert serial.keys() == parallel.keys()
        for name in serial:
            np.testing.assert_allclose(
                parallel[name], serial[name], rtol=0.0, atol=1e-12
            )

    def test_seeded_simulations_preserve_the_serial_random_stream(self, monkeypatch):
        model = GradedResponseModel(n_items=2, n_categories=3, n_factors=2)
        captured = []

        def fake_fit(self, fitted_model, responses):
            captured.append(responses.copy())
            return SimpleNamespace(model=fitted_model)

        monkeypatch.setattr(EMEstimator, "fit", fake_fit)

        expected = []
        rng = np.random.default_rng(42)
        for _ in range(3):
            theta = rng.standard_normal((30, model.n_factors))
            expected.append(draw_item_responses(model, theta, rng))

        parametric_bootstrap(
            model,
            n_bootstrap=3,
            n_persons=30,
            seed=42,
        )

        assert len(captured) == len(expected)
        for actual, reference in zip(captured, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)

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
        from mirt.estimation.base import _initialize_free_parameters

        model = FourParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.full(2, 2.0),
            difficulty=np.full(2, 1.0),
            guessing=np.full(2, 0.4),
            upper=np.full(2, 0.8),
        )
        starts = []

        def fake_fit(self, fitted_model, responses):
            # EM reinitializes the free coordinates of an unfitted model.
            assert not fitted_model.is_fitted
            _initialize_free_parameters(fitted_model)
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


def _worker_backend(_value):
    """Report the backend preference that a process worker resolved."""
    import mirt

    return mirt.get_backend()


@pytest.fixture(scope="module")
def fitted_grm():
    """A converged GRM calibration small enough for repeated refits."""
    from mirt import fit_mirt, simdata

    responses = simdata("GRM", n_persons=150, n_items=5, n_categories=4, seed=11)
    result = fit_mirt(
        responses,
        model="GRM",
        n_categories=4,
        tol=1e-7,
        compute_standard_errors=False,
    )
    return result.model, responses


class TestWarmStart:
    """Replicate fits start from the supplied estimates when requested."""

    def test_warm_start_does_not_reinitialize_estimates(self, fitted_grm, monkeypatch):
        import mirt.estimation.base as base_module

        model, responses = fitted_grm
        calls = []
        original = base_module._initialize_free_parameters

        def record(fitted_model):
            calls.append(fitted_model.n_items)
            original(fitted_model)

        monkeypatch.setattr(base_module, "_initialize_free_parameters", record)

        bootstrap_se(model, responses, n_bootstrap=2, seed=1, warm_start=True)
        assert calls == []

        bootstrap_se(model, responses, n_bootstrap=2, seed=1, warm_start=False)
        assert len(calls) == 2

    def test_first_em_iterate_evaluates_the_original_estimates(self, fitted_grm):
        model, responses = fitted_grm
        sample = responses[np.random.default_rng(3).integers(0, 150, 150)]

        reference_model = model.copy()
        reference = EMEstimator(max_iter=1, tol=1e-3)
        reference.fit(reference_model, sample)

        start = _prepare_bootstrap_model(model, model.parameters, warm_start=True)
        estimator = EMEstimator(max_iter=1, tol=1e-3)
        estimator.fit(start, sample)

        assert estimator._convergence_history[0] == reference._convergence_history[0]

    def test_jackknife_refit_from_converged_grm_converges_quickly(
        self, fitted_grm, monkeypatch
    ):
        model, responses = fitted_grm
        outcomes = []
        original_fit = EMEstimator.fit

        def recording_fit(self, fitted_model, sample, *args, **kwargs):
            result = original_fit(self, fitted_model, sample, *args, **kwargs)
            outcomes.append((result.n_iterations, result.converged))
            return result

        monkeypatch.setattr(EMEstimator, "fit", recording_fit)
        task = _StatisticFitTask(
            model=model,
            original_params=model.parameters,
            warm_start=True,
            max_iter=100,
            responses=responses,
            statistic="parameters",
            omitted_indices=[0, 1, 2],
        )

        results = _fit_statistic_task(task)

        assert all(error is None for _, error in results)
        assert len(outcomes) == 3
        assert all(converged and iterations < 8 for iterations, converged in outcomes)

    @pytest.mark.parametrize("warm_start", [True, False])
    def test_fixed_parameters_keep_their_values_in_replicates(self, warm_start):
        from mirt import simdata

        responses = simdata("2PL", n_persons=120, n_items=5, seed=4)
        model = TwoParameterLogistic(5).set_parameters(discrimination=np.full(5, 1.7))
        masks = model.free_parameter_masks
        masks["discrimination"][:] = False
        model.set_free_parameter_masks(masks)
        task = _StatisticFitTask(
            model=model,
            original_params=model.parameters,
            warm_start=warm_start,
            max_iter=50,
            responses=responses,
            statistic="parameters",
            sample_indices=[np.arange(120), np.arange(60, 120)],
        )

        results = _fit_statistic_task(task)

        assert all(error is None for _, error in results)
        for values, _ in results:
            np.testing.assert_array_equal(values["discrimination"], np.full(5, 1.7))
            assert not np.array_equal(values["difficulty"], np.zeros(5))

    def test_native_2pl_bootstrap_is_skipped_for_restricted_models(self, monkeypatch):
        model = TwoParameterLogistic(2)
        masks = model.free_parameter_masks
        masks["discrimination"][0] = False
        model.set_free_parameter_masks(masks)

        def unexpected_native_call(*args, **kwargs):
            raise AssertionError("restricted models must not use the native path")

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: True)
        monkeypatch.setattr(
            rust_estimation, "bootstrap_fit_2pl", unexpected_native_call
        )
        responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]] * 5)

        result = bootstrap_se(model, responses, n_bootstrap=2, seed=42)

        assert result["discrimination"][0] == 0.0


class TestProcessWorkers:
    """Bootstrap workers share the parent's configuration."""

    @pytest.mark.parametrize("backend", ["auto", "numpy"])
    def test_workers_use_the_parent_backend(self, backend):
        import mirt

        previous = mirt.get_backend()
        mirt.set_backend(backend)
        try:
            reported = _run_bootstrap_tasks(_worker_backend, [0, 1], n_jobs=2)
        finally:
            mirt.set_backend(previous)

        assert reported == [backend, backend]

    def test_seeded_results_match_across_worker_counts_without_native_code(
        self, fitted_grm
    ):
        # The default backend is covered by the parallel tests above; workers
        # that ignored an explicit NumPy request would refit natively. Every
        # bootstrap entry point shares this worker pool.
        import mirt

        model, responses = fitted_grm
        previous = mirt.get_backend()
        mirt.set_backend("numpy")
        try:
            serial = bootstrap_se(model, responses, n_bootstrap=4, seed=5)
            parallel = bootstrap_se(model, responses, n_bootstrap=4, seed=5, n_jobs=2)
        finally:
            mirt.set_backend(previous)

        for name in serial:
            np.testing.assert_array_equal(parallel[name], serial[name])
