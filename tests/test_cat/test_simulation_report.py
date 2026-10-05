"""catR-style CAT simulation reports."""

import importlib.util
import json
from itertools import combinations

import numpy as np
import pytest

from mirt.cat import (
    CATEngine,
    CATResult,
    CATSimulationReport,
    MCATResult,
    summarize_cat_simulation,
)
from mirt.cat.exposure import SympsonHetter
from mirt.models import TwoParameterLogistic


def _session(theta, se, items, reason="SE threshold reached"):
    return CATResult(
        theta=theta,
        standard_error=se,
        items_administered=list(items),
        responses=np.ones(len(items), dtype=int),
        n_items_administered=len(items),
        stopping_reason=reason,
    )


@pytest.fixture
def sessions() -> list[CATResult]:
    return [
        _session(0.5, 0.3, [0, 1, 2]),
        _session(-0.2, 0.4, [0, 3], reason="Maximum items reached (2)"),
        _session(1.4, 0.35, [1, 0]),
    ]


def test_hand_computed_statistics(sessions):
    report = summarize_cat_simulation(sessions, [0.0, 0.0, 1.0], n_items=5)

    assert isinstance(report, CATSimulationReport)
    assert report.n_examinees == 3
    assert report.bias == pytest.approx(0.7 / 3)
    assert report.rmse == pytest.approx(np.sqrt(0.15))
    assert report.mae == pytest.approx(1.1 / 3)
    assert report.correlation == pytest.approx(
        np.corrcoef([0.0, 0.0, 1.0], [0.5, -0.2, 1.4])[0, 1]
    )
    assert report.mean_standard_error == pytest.approx(0.35)
    assert report.mean_length == pytest.approx(7 / 3)
    assert report.sd_length == pytest.approx(np.sqrt(1 / 3))
    assert (report.min_length, report.max_length) == (2, 3)
    assert report.stopping_reasons == {
        "SE threshold reached": 2,
        "Maximum items reached (2)": 1,
    }
    np.testing.assert_array_equal(report.selection_counts, [3, 2, 1, 1, 0])
    np.testing.assert_allclose(report.exposure_rates, [1, 2 / 3, 1 / 3, 1 / 3, 0])
    assert report.max_exposure == 1.0
    np.testing.assert_array_equal(report.unused_items, [4])
    # Pairs share 1, 2, and 1 items; the mean test length is 7 / 3.
    assert report.overlap_rate == pytest.approx((4 / 3) / (7 / 3))
    assert report.chi_square == pytest.approx(0.577777777 / (7 / 15))
    assert report.conditional is None


def test_exposure_bounds_match_sympson_hetter_monitoring(sessions):
    report = summarize_cat_simulation(
        sessions, [0.0, 0.0, 1.0], n_items=5, confidence_level=0.9
    )
    monitor = SympsonHetter()
    monitor.record_sessions(np.array([[0, 1, 2], [0, 3, -1], [1, 0, -1]]), n_items=5)
    expected = monitor.exposure_report(n_items=5, confidence_level=0.9)

    np.testing.assert_allclose(report.exposure_lower, expected.confidence_lower)
    np.testing.assert_allclose(report.exposure_upper, expected.confidence_upper)
    np.testing.assert_allclose(report.exposure_rates, expected.exposure_rates)


@pytest.mark.parametrize("seed", range(3))
def test_overlap_rate_matches_pairwise_overlap(seed):
    rng = np.random.default_rng(seed)
    n_items = 30
    sessions = [
        _session(0.0, 0.3, rng.choice(n_items, size=rng.integers(3, 12), replace=False))
        for _ in range(25)
    ]
    lengths = [result.n_items_administered for result in sessions]
    shared = [
        len(set(first.items_administered) & set(second.items_administered))
        for first, second in combinations(sessions, 2)
    ]

    report = summarize_cat_simulation(sessions, np.zeros(25), n_items=n_items)

    assert report.overlap_rate == pytest.approx(np.mean(shared) / np.mean(lengths))


def test_replicated_abilities_follow_batch_order():
    rng = np.random.default_rng(1)
    model = TwoParameterLogistic(n_items=40)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, 40),
        difficulty=rng.normal(0.0, 1.0, 40),
    )
    model._is_fitted = True
    thetas = np.array([-1.0, 0.0, 1.5])
    engine = CATEngine(model, se_threshold=0.4, max_items=10, seed=3)
    results = engine.run_batch_simulation(thetas, n_replications=4)

    compact = summarize_cat_simulation(
        results, thetas, n_items=40, n_replications=4, theta_bins=3
    )
    expanded = summarize_cat_simulation(
        results, np.repeat(thetas, 4), n_items=40, theta_bins=3
    )

    assert json.dumps(compact.to_dict()) == json.dumps(expanded.to_dict())
    np.testing.assert_array_equal(compact.conditional["n"], [4, 4, 4])
    errors = np.array([result.theta for result in results]) - np.repeat(thetas, 4)
    np.testing.assert_allclose(
        compact.conditional["bias"], errors.reshape(3, 4).mean(axis=1)
    )


def test_conditional_bins_use_half_open_intervals_and_closed_last_bin():
    sessions = [_session(theta + 0.1, 0.3, [0]) for theta in (-3.0, -1.0, 0.0, 1.0)]
    true_theta = [-3.0, -1.0, 0.0, 1.0]

    report = summarize_cat_simulation(
        sessions, true_theta, n_items=2, theta_bins=[-1.0, 0.0, 0.5, 1.0]
    )

    table = report.conditional
    np.testing.assert_array_equal(table["lower"], [-1.0, 0.0, 0.5])
    np.testing.assert_array_equal(table["upper"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(table["n"], [1, 1, 1])
    np.testing.assert_allclose(table["bias"], [0.1, 0.1, 0.1])
    np.testing.assert_allclose(table["rmse"], [0.1, 0.1, 0.1])
    assert "Conditional results" in report.summary()


def test_empty_conditional_bins_report_missing_statistics(sessions):
    report = summarize_cat_simulation(
        sessions, [0.0, 0.0, 1.0], n_items=5, theta_bins=[-2.0, -1.0, 2.0]
    )

    np.testing.assert_array_equal(report.conditional["n"], [0, 3])
    assert np.isnan(report.conditional["bias"][0])
    assert np.isnan(report.conditional["mean_length"][0])


def test_constant_abilities_collapse_quantile_bins():
    sessions = [_session(0.1 * k, 0.3, [k]) for k in range(4)]

    report = summarize_cat_simulation(sessions, np.zeros(4), n_items=4, theta_bins=5)

    np.testing.assert_array_equal(report.conditional["n"], [4])
    assert np.isnan(report.correlation)


def test_multidimensional_results_report_each_factor():
    sessions = [
        MCATResult(
            theta=np.array(estimate),
            covariance=np.diag([0.09, 0.16]),
            standard_error=np.array([0.3, 0.4]),
            items_administered=[0, 2],
            responses=np.array([1, 0]),
            n_items_administered=2,
            stopping_reason="Covariance trace threshold reached",
        )
        for estimate in ([0.2, -0.1], [1.1, 0.4], [-0.5, 0.9])
    ]
    true_theta = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])

    report = summarize_cat_simulation(sessions, true_theta, n_items=3)

    errors = np.array([[0.2, -0.1], [0.1, 0.4], [0.5, -0.1]])
    np.testing.assert_allclose(report.bias, errors.mean(axis=0))
    np.testing.assert_allclose(report.rmse, np.sqrt((errors**2).mean(axis=0)))
    np.testing.assert_allclose(report.mean_standard_error, [0.3, 0.4])
    assert report.correlation.shape == (2,)
    assert "[+0.2667, +0.0667]" in report.summary()
    with pytest.raises(ValueError, match="unidimensional"):
        summarize_cat_simulation(sessions, true_theta, n_items=3, theta_bins=2)


def test_single_multidimensional_ability_vector_is_one_examinee():
    session = MCATResult(
        theta=np.array([0.2, 0.3]),
        covariance=np.eye(2),
        standard_error=np.ones(2),
        items_administered=[1],
        responses=np.array([1]),
        n_items_administered=1,
        stopping_reason="done",
    )

    report = summarize_cat_simulation(
        [session, session], [0.0, 0.0], n_items=2, n_replications=2
    )

    np.testing.assert_allclose(report.bias, [0.2, 0.3])


def _mcat_session(theta, items=(0,)) -> MCATResult:
    estimate = np.asarray(theta, dtype=float)
    return MCATResult(
        theta=estimate,
        covariance=np.eye(estimate.size),
        standard_error=np.ones(estimate.size),
        items_administered=list(items),
        responses=np.ones(len(items), dtype=int),
        n_items_administered=len(items),
        stopping_reason="done",
    )


def test_one_factor_mcat_accepts_one_ability_per_examinee():
    sessions = [_mcat_session([0.1]), _mcat_session([0.4], items=(1,))]

    report = summarize_cat_simulation(sessions, [0.0, 0.0], n_items=2)

    assert report.n_examinees == 2
    np.testing.assert_allclose(report.bias, [0.25])


def test_rejects_results_with_different_factor_counts():
    sessions = [_mcat_session([0.1, 0.2]), _mcat_session([0.1, 0.2, 0.3])]

    with pytest.raises(ValueError, match="same number of factors"):
        summarize_cat_simulation(sessions, np.zeros((2, 2)), n_items=2)


def test_dictionary_export_is_json_round_trippable(sessions):
    report = summarize_cat_simulation(
        sessions, [0.0, 0.0, 1.0], n_items=5, theta_bins=2
    )

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["selection_counts"] == [3, 2, 1, 1, 0]
    assert payload["stopping_reasons"]["SE threshold reached"] == 2
    assert set(payload["conditional"]) == {
        "lower",
        "upper",
        "n",
        "bias",
        "rmse",
        "mean_length",
        "mean_standard_error",
    }
    assert payload["bias"] == pytest.approx(report.bias)


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None
    and importlib.util.find_spec("polars") is None,
    reason="no DataFrame backend",
)
def test_dataframe_tables(sessions):
    report = summarize_cat_simulation(
        sessions, [0.0, 0.0, 1.0], n_items=5, theta_bins=2
    )

    items = report.to_dataframe()
    conditional = report.to_dataframe("conditional")

    assert items.shape == (5, 5)
    assert conditional.shape[1] == 7
    with pytest.raises(ValueError, match="table"):
        report.to_dataframe("bogus")
    with pytest.raises(ValueError, match="no conditional"):
        summarize_cat_simulation(sessions, [0, 0, 1], n_items=5).to_dataframe(
            "conditional"
        )


def test_native_batch_results_without_histories():
    rng = np.random.default_rng(4)
    model = TwoParameterLogistic(n_items=30)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, 30),
        difficulty=rng.normal(0.0, 1.0, 30),
    )
    model._is_fitted = True
    thetas = np.linspace(-2.0, 2.0, 5)
    results = CATEngine(model, se_threshold=0.5, seed=1).run_batch_simulation(
        thetas, n_replications=2, use_rust=True
    )

    report = summarize_cat_simulation(results, thetas, n_items=30, n_replications=2)

    assert report.n_examinees == 10
    assert report.selection_counts.sum() == sum(
        result.n_items_administered for result in results
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"results": []}, ValueError, "at least one session"),
        ({"true_theta": [0.0, 1.0]}, ValueError, "describes 2 sessions"),
        ({"n_replications": 2}, ValueError, "describes 6 sessions"),
        ({"n_replications": 0}, ValueError, "n_replications"),
        ({"true_theta": [0.0, np.nan, 1.0]}, ValueError, "finite"),
        ({"true_theta": [[0.0, 1.0]] * 3}, ValueError, "1 value"),
        ({"n_items": 3}, ValueError, "outside n_items"),
        ({"n_items": 0}, ValueError, "n_items"),
        ({"n_items": 5.0}, ValueError, "n_items"),
        ({"confidence_level": 1.0}, ValueError, "confidence_level"),
        ({"confidence_level": True}, ValueError, "confidence_level"),
        ({"theta_bins": 0}, ValueError, "theta_bins"),
        ({"theta_bins": [1.0, 0.0]}, ValueError, "increasing"),
        ({"theta_bins": [[0.0, 1.0]]}, ValueError, "increasing"),
        ({"theta_bins": True}, ValueError, "increasing"),
    ],
)
def test_rejects_invalid_input(sessions, kwargs, error, message):
    arguments = {
        "results": sessions,
        "true_theta": [0.0, 0.0, 1.0],
        "n_items": 5,
    }
    arguments.update(kwargs)

    with pytest.raises(error, match=message):
        summarize_cat_simulation(**arguments)


def test_rejects_mixed_result_types(sessions):
    mcat = MCATResult(
        theta=np.zeros(2),
        covariance=np.eye(2),
        standard_error=np.ones(2),
        items_administered=[0],
        responses=np.array([1]),
        n_items_administered=1,
        stopping_reason="done",
    )

    with pytest.raises(TypeError, match="all be CATResult"):
        summarize_cat_simulation([*sessions, mcat], np.zeros(4), n_items=5)


def test_rejects_repeated_items_within_a_session():
    with pytest.raises(ValueError, match="twice"):
        summarize_cat_simulation([_session(0.0, 0.3, [1, 1])], [0.0], n_items=3)
