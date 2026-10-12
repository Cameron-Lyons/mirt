"""Tests for Bayesian model diagnostics module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.bayesian import (
    PSISResult,
    WAICResult,
    compare_models,
    dic,
    loo_compare,
    psis_loo,
    waic,
)
from mirt.exceptions import MirtValidationError

N_OBS = 10


def _make_psis_result(
    elpd_loo: float = -50.0,
    p_loo: float = 5.0,
    looic: float = 100.0,
    pareto_k_val: float = 0.0,
    n_high_k: int = 0,
    se_elpd: float = 2.0,
) -> PSISResult:
    return PSISResult(
        elpd_loo=elpd_loo,
        p_loo=p_loo,
        looic=looic,
        pointwise=np.zeros(N_OBS),
        pareto_k=np.full(N_OBS, pareto_k_val),
        n_high_k=n_high_k,
        se_elpd=se_elpd,
    )


def _make_waic_result(
    waic_val: float = 100.0,
    elpd_waic: float = -50.0,
    p_waic: float = 5.0,
    se_waic: float = 3.0,
) -> WAICResult:
    return WAICResult(
        waic=waic_val,
        elpd_waic=elpd_waic,
        p_waic=p_waic,
        pointwise=np.zeros(N_OBS),
        se_waic=se_waic,
    )


@pytest.fixture(scope="module")
def log_lik_matrix():
    """Generate a plausible log-likelihood matrix (n_samples, n_obs)."""
    rng = np.random.default_rng(42)
    n_samples, n_obs = 100, 20
    theta = rng.standard_normal((n_samples, 1))
    difficulty = rng.normal(0, 1, (1, n_obs))
    discrimination = np.abs(rng.normal(1, 0.3, (1, n_obs)))
    p = 1 / (1 + np.exp(-discrimination * (theta - difficulty)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    responses = (rng.random((1, n_obs)) < 0.6).astype(float)
    return responses * np.log(p) + (1 - responses) * np.log(1 - p)


class TestPSISResult:
    def test_dataclass_fields(self):
        result = _make_psis_result()
        assert result.elpd_loo == -50.0
        assert result.p_loo == 5.0
        assert result.looic == 100.0
        assert result.n_high_k == 0
        assert result.se_elpd == 2.0

    def test_summary_no_warnings(self):
        result = _make_psis_result(pareto_k_val=0.3)
        summary = result.summary()
        assert "PSIS-LOO" in summary
        assert "elpd_loo" in summary
        assert "Warning" not in summary

    def test_summary_with_warnings(self):
        result = _make_psis_result(pareto_k_val=0.8, n_high_k=N_OBS)
        summary = result.summary()
        assert "Warning" in summary


class TestWAICResult:
    def test_dataclass_fields(self):
        result = _make_waic_result()
        assert result.waic == 100.0
        assert result.elpd_waic == -50.0
        assert result.p_waic == 5.0
        assert result.se_waic == 3.0

    def test_summary(self):
        result = _make_waic_result()
        summary = result.summary()
        assert "WAIC" in summary
        assert "elpd_waic" in summary


class TestPSISLOO:
    def test_basic_output_structure(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        assert isinstance(result, PSISResult)
        assert isinstance(result.elpd_loo, float)
        assert isinstance(result.p_loo, float)
        assert isinstance(result.looic, float)
        assert isinstance(result.se_elpd, float)

    def test_pointwise_shape(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        n_obs = log_lik_matrix.shape[1]
        assert result.pointwise.shape == (n_obs,)
        assert result.pareto_k.shape == (n_obs,)

    def test_looic_equals_neg2_elpd(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        assert_allclose(result.looic, -2 * result.elpd_loo)

    def test_elpd_sum_of_pointwise(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        assert_allclose(result.elpd_loo, np.sum(result.pointwise))

    def test_pareto_k_nonnegative(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        assert np.all(result.pareto_k >= 0)

    def test_n_high_k_count(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix)
        expected = int(np.sum(result.pareto_k > 0.7))
        assert result.n_high_k == expected

    def test_1d_input(self):
        log_lik_1d = np.random.default_rng(42).standard_normal(50)
        result = psis_loo(log_lik_1d)
        assert isinstance(result, PSISResult)

    def test_custom_k_threshold(self, log_lik_matrix):
        result = psis_loo(log_lik_matrix, k_threshold=0.5)
        expected = int(np.sum(result.pareto_k > 0.5))
        assert result.n_high_k == expected


class TestWAIC:
    def test_basic_output_structure(self, log_lik_matrix):
        result = waic(log_lik_matrix)
        assert isinstance(result, WAICResult)
        assert isinstance(result.waic, float)
        assert isinstance(result.elpd_waic, float)
        assert isinstance(result.p_waic, float)
        assert isinstance(result.se_waic, float)

    def test_pointwise_shape(self, log_lik_matrix):
        result = waic(log_lik_matrix)
        n_obs = log_lik_matrix.shape[1]
        assert result.pointwise.shape == (n_obs,)

    def test_waic_equals_neg2_elpd(self, log_lik_matrix):
        result = waic(log_lik_matrix)
        assert_allclose(result.waic, -2 * result.elpd_waic)

    def test_p_waic_nonnegative(self, log_lik_matrix):
        result = waic(log_lik_matrix)
        assert result.p_waic >= 0

    def test_1d_input(self):
        log_lik_1d = np.random.default_rng(42).standard_normal(50)
        result = waic(log_lik_1d)
        assert isinstance(result, WAICResult)

    def test_se_positive(self, log_lik_matrix):
        result = waic(log_lik_matrix)
        assert result.se_waic >= 0


class TestDIC:
    def test_basic_output(self):
        log_lik_at_mean = -100.0
        log_lik_samples = np.random.default_rng(42).normal(-105, 2, 100)
        dic_val, p_dic = dic(log_lik_at_mean, log_lik_samples)
        assert isinstance(dic_val, float)
        assert isinstance(p_dic, float)

    def test_dic_formula(self):
        log_lik_at_mean = -100.0
        log_lik_samples = np.full(50, -110.0)
        dic_val, p_dic = dic(log_lik_at_mean, log_lik_samples)

        deviance_at_mean = -2 * log_lik_at_mean
        mean_deviance = -2 * np.mean(log_lik_samples)
        expected_p = mean_deviance - deviance_at_mean
        expected_dic = deviance_at_mean + 2 * expected_p

        assert_allclose(p_dic, expected_p)
        assert_allclose(dic_val, expected_dic)

    def test_p_dic_positive_when_mean_better(self):
        log_lik_at_mean = -90.0
        log_lik_samples = np.random.default_rng(42).normal(-100, 2, 100)
        _, p_dic = dic(log_lik_at_mean, log_lik_samples)
        assert p_dic > 0


class TestLooCompare:
    @staticmethod
    def _log_likelihoods():
        rng = np.random.default_rng(7)
        # Both models share the observation effects, so their pointwise
        # predictive densities are strongly correlated.
        shared = rng.normal(-1.0, 1.0, (1, 30)) + rng.normal(0.0, 0.3, (400, 30))
        return shared, shared + rng.normal(-0.2, 0.1, (1, 30))

    def test_psis_ranks_by_elpd_with_paired_standard_errors(self):
        better, worse = (psis_loo(values) for values in self._log_likelihoods())
        table = loo_compare(worse, better, names=["Worse", "Better"])

        assert list(table["Model"]) == ["Better", "Worse"]
        assert list(table.columns) == [
            "Model",
            "elpd_diff",
            "se_diff",
            "elpd_loo",
            "se_elpd_loo",
            "p_loo",
            "looic",
            "se_looic",
        ]
        difference = worse.pointwise - better.pointwise
        assert_allclose(table["elpd_diff"], [0.0, worse.elpd_loo - better.elpd_loo])
        assert_allclose(
            table["se_diff"], [0.0, np.sqrt(30 * np.var(difference, ddof=1))]
        )
        # Correlated predictions make the paired error much smaller.
        assert table["se_diff"][1] < 0.5 * worse.se_elpd
        assert_allclose(table["se_elpd_loo"], [better.se_elpd, worse.se_elpd])
        assert_allclose(table["looic"], [better.looic, worse.looic])
        assert_allclose(table["se_looic"], 2 * table["se_elpd_loo"])

    def test_waic_differences_are_on_the_elpd_scale(self):
        better, worse = (waic(values) for values in self._log_likelihoods())
        table = loo_compare(better, worse)

        assert list(table["Model"]) == ["Model 1", "Model 2"]
        # WAIC pointwise values and se_waic are on the deviance scale.
        assert_allclose(table["se_elpd_waic"], [better.se_waic / 2, worse.se_waic / 2])
        assert_allclose(table["se_waic"], [better.se_waic, worse.se_waic])
        difference = (worse.pointwise - better.pointwise) / -2
        assert_allclose(table["se_diff"][1], np.sqrt(30 * np.var(difference, ddof=1)))
        assert_allclose(table["elpd_diff"][1], worse.elpd_waic - better.elpd_waic)

    def test_invalid_inputs_are_rejected(self):
        loo_result = _make_psis_result()
        with pytest.raises(MirtValidationError, match="at least one"):
            loo_compare()
        with pytest.raises(MirtValidationError, match="Expected 1 names"):
            loo_compare(loo_result, names=["A", "B"])
        with pytest.raises(MirtValidationError, match="all be PSISResult"):
            loo_compare(loo_result, _make_waic_result())
        shorter = PSISResult(-50.0, 5.0, 100.0, np.zeros(3), np.zeros(3), 0, 2.0)
        with pytest.raises(MirtValidationError, match="same observations"):
            loo_compare(loo_result, shorter)


class TestDeprecatedCompareModels:
    def test_psis_comparison(self):
        r1 = _make_psis_result(elpd_loo=-50.0, looic=100.0)
        r2 = _make_psis_result(elpd_loo=-60.0, p_loo=8.0, looic=120.0, se_elpd=3.0)
        with pytest.warns(DeprecationWarning, match="loo_compare"):
            output = compare_models(r1, r2, names=["Simple", "Complex"])
        assert "Model Comparison" in output
        assert "Simple" in output
        assert "Complex" in output
        assert "LOOIC" in output

    def test_waic_standard_error_is_on_the_elpd_scale(self):
        r1 = _make_waic_result(waic_val=95.0, elpd_waic=-47.5, p_waic=4.0, se_waic=2.5)
        r2 = _make_waic_result(waic_val=110.0, elpd_waic=-55.0, p_waic=6.0)
        with pytest.warns(DeprecationWarning):
            output = compare_models(r1, r2, names=["A", "B"])
        assert "WAIC" in output
        rows = [
            line.split() for line in output.splitlines() if line[:2] in ("A ", "B ")
        ]
        # Model, WAIC, elpd, SE(elpd), dWAIC, dSE: se_waic 2.5 is 1.25 for elpd.
        assert rows[0][:5] == ["A", "95.00", "-47.50", "1.25", "0.00"]
        assert rows[1][4] == "15.00"

    def test_sorted_by_ic(self):
        r1 = _make_psis_result(elpd_loo=-60.0, p_loo=8.0, looic=120.0, se_elpd=3.0)
        r2 = _make_psis_result(looic=100.0)
        with pytest.warns(DeprecationWarning):
            output = compare_models(r1, r2, names=["Worse", "Better"])
        lines = output.strip().split("\n")
        data_lines = [line for line in lines if "Worse" in line or "Better" in line]
        assert "Better" in data_lines[0]

    def test_mismatched_names_raises(self):
        r1 = _make_psis_result()
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match="Expected 1 names"),
        ):
            compare_models(r1, names=["A", "B"])

    def test_three_models_default_names(self):
        results = [
            _make_psis_result(elpd_loo=-ic / 2, looic=float(ic))
            for ic in [100, 110, 105]
        ]
        with pytest.warns(DeprecationWarning):
            output = compare_models(*results)
        assert "Model 1" in output
        assert "Model 2" in output
        assert "Model 3" in output


class TestCanonicalCompareModels:
    def test_diagnostics_namespace_compares_fits_like_the_top_level(self):
        import mirt
        import mirt.diagnostics

        data = mirt.simdata("2PL", n_persons=300, n_items=6, seed=2)
        fits = [
            mirt.fit_mirt(data, model=name, verbose=False) for name in ("1PL", "2PL")
        ]
        expected = mirt.compare_models(fits, ["AIC"])
        for table in (
            mirt.diagnostics.compare_models(fits, ["AIC"]),
            mirt.diagnostics.compare_models(fits, criteria=["AIC"]),
        ):
            assert list(table.columns) == list(expected.columns)
            assert_allclose(table["AIC"], expected["AIC"])

    def test_diagnostics_namespace_keeps_the_deprecated_loo_table(self):
        import mirt.diagnostics

        r1, r2 = _make_psis_result(), _make_psis_result(elpd_loo=-60.0, looic=120.0)
        with pytest.warns(DeprecationWarning, match="loo_compare") as record:
            output = mirt.diagnostics.compare_models(r1, r2, names=["A", "B"])
        assert record[0].filename == __file__
        assert "LOOIC" in output and "A" in output
        with pytest.raises(TypeError, match="names apply only"):
            mirt.diagnostics.compare_models([], names=["A"])
