"""Tests for the parametric bootstrap likelihood-ratio test."""

import numpy as np
import pytest
from scipy import stats

import mirt.utils.bootstrap as bootstrap_module
from mirt import BootstrapLRResult, bootstrap_lr, fit_mirt, simdata
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic


@pytest.fixture(scope="module")
def nested_fits():
    responses = simdata("2PL", n_persons=200, n_items=5, seed=3)
    responses[np.random.default_rng(4).random(responses.shape) < 0.05] = -1
    reduced = fit_mirt(responses, model="1PL", compute_standard_errors=False)
    full = fit_mirt(responses, model="2PL", compute_standard_errors=False)
    return reduced, full, responses


def test_statistic_comes_from_refits_under_replicate_settings(nested_fits):
    reduced, full, responses = nested_fits

    result = bootstrap_lr(reduced, full, responses, n_bootstrap=3, seed=1)

    assert isinstance(result, BootstrapLRResult)
    assert result.df == full.model.n_parameters - reduced.model.n_parameters == 5
    assert result.statistic == pytest.approx(
        max(0.0, 2 * (result.full_log_likelihood - result.reduced_log_likelihood))
    )
    assert result.reduced_log_likelihood == pytest.approx(
        reduced.log_likelihood, abs=0.05
    )
    assert result.asymptotic_p_value == pytest.approx(
        stats.chi2.sf(result.statistic, result.df)
    )
    assert result.null_statistics.shape == (3,)
    assert result.n_failed == 0
    assert np.all(result.null_statistics >= 0.0)
    exceedances = np.count_nonzero(result.null_statistics >= result.statistic)
    assert result.p_value == (1 + exceedances) / 4


def test_replicates_simulate_the_reduced_model_with_observed_missingness(
    nested_fits, monkeypatch
):
    reduced, full, responses = nested_fits
    fitted = []
    original = bootstrap_module._refit_log_likelihood

    def recording_refit(model, data, warm_start, options):
        fitted.append((type(model), data.copy(), warm_start, dict(options)))
        return original(model, data, warm_start, options)

    monkeypatch.setattr(bootstrap_module, "_refit_log_likelihood", recording_refit)

    bootstrap_lr(
        reduced, full, responses, n_bootstrap=2, seed=8, tol=1e-4, n_quadpts=15
    )

    observed, replicates = fitted[:2], fitted[2:]
    assert [entry[0] for entry in observed] == [
        OneParameterLogistic,
        TwoParameterLogistic,
    ]
    assert all(entry[1] is not None for entry in observed)
    np.testing.assert_array_equal(observed[0][1], responses)
    assert len(replicates) == 4
    for model_type, data, warm_start, options in replicates:
        assert model_type in (OneParameterLogistic, TwoParameterLogistic)
        np.testing.assert_array_equal(data < 0, responses < 0)
        assert warm_start
        assert options["tol"] == 1e-4
        assert options["n_quadpts"] == 15
    # Both models in one replicate are fitted to the same simulated data.
    np.testing.assert_array_equal(replicates[0][1], replicates[1][1])
    assert not np.array_equal(replicates[0][1], replicates[2][1])


def test_failed_replicates_are_counted_and_excluded(nested_fits, monkeypatch):
    reduced, full, responses = nested_fits
    calls = 0
    original = bootstrap_module._refit_log_likelihood

    def flaky_refit(model, data, warm_start, options):
        nonlocal calls
        calls += 1
        if calls in (3, 8):
            raise RuntimeError("replicate failed")
        return original(model, data, warm_start, options)

    monkeypatch.setattr(bootstrap_module, "_refit_log_likelihood", flaky_refit)

    result = bootstrap_lr(reduced, full, responses, n_bootstrap=4, seed=2)

    assert result.n_failed == 2
    assert result.null_statistics.shape == (2,)
    exceedances = np.count_nonzero(result.null_statistics >= result.statistic)
    assert result.p_value == (1 + exceedances) / 3


def test_every_failed_replicate_yields_an_undefined_p_value(nested_fits, monkeypatch):
    reduced, full, responses = nested_fits

    def failing_task(task):
        return [(np.nan, "RuntimeError: failed")] * len(task.seeds)

    monkeypatch.setattr(bootstrap_module, "_fit_lr_task", failing_task)

    with pytest.warns(RuntimeWarning, match="every bootstrap"):
        result = bootstrap_lr(reduced, full, responses, n_bootstrap=2, seed=2)

    assert np.isnan(result.p_value)
    assert result.n_failed == 2
    assert result.null_statistics.size == 0


def test_seeded_results_match_across_worker_counts(nested_fits):
    reduced, full, responses = nested_fits

    serial = bootstrap_lr(reduced, full, responses, n_bootstrap=4, seed=11)
    parallel = bootstrap_lr(reduced, full, responses, n_bootstrap=4, seed=11, n_jobs=2)
    repeated = bootstrap_lr(reduced, full, responses, n_bootstrap=4, seed=11)

    np.testing.assert_array_equal(parallel.null_statistics, serial.null_statistics)
    np.testing.assert_array_equal(repeated.null_statistics, serial.null_statistics)
    assert parallel.p_value == serial.p_value


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bootstrap": 1}, "n_bootstrap"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_quadpts": 2}, "n_quadpts"),
        ({"tol": 0.0}, "tol"),
    ],
)
def test_rejects_invalid_settings(nested_fits, kwargs, message):
    reduced, full, responses = nested_fits
    with pytest.raises(MirtValidationError, match=message):
        bootstrap_lr(reduced, full, responses, **kwargs)


def test_rejects_models_that_are_not_nested_by_parameter_count(nested_fits):
    reduced, full, responses = nested_fits
    with pytest.raises(MirtValidationError, match="more free parameters"):
        bootstrap_lr(full, reduced, responses, n_bootstrap=2)
    with pytest.raises(MirtValidationError, match="same items"):
        bootstrap_lr(reduced, TwoParameterLogistic(6), responses, n_bootstrap=2)
    with pytest.raises(ValueError, match="items"):
        bootstrap_lr(reduced, full, responses[:, :4], n_bootstrap=2)


@pytest.mark.parametrize("argument", ["reduced", "full"])
def test_rejects_objects_that_are_not_item_models(nested_fits, argument):
    reduced, full, responses = nested_fits
    models = {"reduced": reduced, "full": full, argument: "2PL"}

    with pytest.raises(MirtValidationError, match=f"{argument} must be an item model"):
        bootstrap_lr(models["reduced"], models["full"], responses, n_bootstrap=2)


@pytest.mark.slow
def test_bootstrap_p_values_are_uniform_under_the_null():
    p_values = []
    for replication in range(30):
        responses = simdata("1PL", n_persons=150, n_items=4, seed=100 + replication)
        reduced = fit_mirt(responses, model="1PL", compute_standard_errors=False)
        full = fit_mirt(responses, model="2PL", compute_standard_errors=False)
        result = bootstrap_lr(
            reduced, full, responses, n_bootstrap=19, seed=replication, tol=1e-4
        )
        p_values.append(result.p_value)

    assert stats.kstest(p_values, "uniform").pvalue > 0.001
