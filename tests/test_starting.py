"""Tests for constrained starting values and null-model baselines."""

from __future__ import annotations

import doctest
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import mirt
import mirt.utils as mirt_utils
import mirt.utils.starting as starting
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.models.unfolding import GeneralizedGradedUnfolding, IdealPointModel
from mirt.utils.starting import calc_null, gen_random_pars, multi_start_fit


def test_gen_random_pars_is_reproducible_and_leaves_model_unchanged():
    model = TwoParameterLogistic(n_items=4)
    original = model.parameters

    first = gen_random_pars(model, n_sets=2, seed=18)
    second = gen_random_pars(model, n_sets=2, seed=18)

    for first_set, second_set in zip(first, second, strict=True):
        assert first_set.keys() == second_set.keys()
        for name in first_set:
            np.testing.assert_array_equal(first_set[name], second_set[name])
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)
    assert np.all(first[0]["discrimination"] >= 0.5)
    assert np.all(first[0]["discrimination"] < 2.0)
    assert np.all(first[0]["difficulty"] >= -2.0)
    assert np.all(first[0]["difficulty"] < 2.0)


@pytest.mark.parametrize("n_sets", [0, -1, True, 1.5])
def test_gen_random_pars_rejects_invalid_set_counts(n_sets):
    with pytest.raises(MirtValidationError, match="positive integer"):
        gen_random_pars(TwoParameterLogistic(2), n_sets=n_sets)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_gen_random_pars_rejects_invalid_seeds(seed):
    with pytest.raises(MirtValidationError, match="non-negative integer"):
        gen_random_pars(TwoParameterLogistic(2), seed=seed)


@pytest.mark.parametrize(
    ("keyword", "bounds"),
    [
        ("discrimination_range", (0.0, 1.0)),
        ("difficulty_range", (1.0, -1.0)),
        ("guessing_range", (-0.1, 0.2)),
        ("upper_range", (0.9, np.inf)),
    ],
)
def test_gen_random_pars_rejects_invalid_ranges(keyword, bounds):
    with pytest.raises(MirtValidationError):
        gen_random_pars(TwoParameterLogistic(2), **{keyword: bounds})


@pytest.mark.parametrize(
    "model",
    [OneParameterLogistic(3), PartialCreditModel(3, n_categories=4)],
)
def test_gen_random_pars_omits_fixed_discrimination(model):
    params = gen_random_pars(model, seed=9)[0]

    assert "discrimination" not in params
    model.copy().set_parameters(**params)


def test_gen_random_pars_orders_thresholds_and_retains_padding():
    model = GradedResponseModel(2, n_categories=[3, 5])

    thresholds = gen_random_pars(model, n_sets=25, seed=7)

    for params in thresholds:
        values = params["thresholds"]
        assert np.all(np.diff(values[0, :2]) > 0)
        assert np.all(np.diff(values[1, :4]) > 0)
        np.testing.assert_array_equal(values[0, 2:], [0.0, 0.0])
        model.copy().set_parameters(**params)


@pytest.mark.parametrize("n_categories", [4, [3, 5]])
def test_gen_random_pars_constructs_symmetric_ggum_thresholds(n_categories):
    model = GeneralizedGradedUnfolding(2, n_categories=n_categories)

    starts = gen_random_pars(model, n_sets=25, seed=29)

    for params in starts:
        thresholds = params["thresholds"]
        for item, n_categories in enumerate(model.n_categories):
            n_independent = n_categories - 1
            n_active = 2 * n_independent + 1
            independent = thresholds[item, :n_independent]
            assert np.all(np.diff(independent) > 0.0)
            assert thresholds[item, n_independent] == 0.0
            np.testing.assert_array_equal(
                thresholds[item, n_independent + 1 : n_active],
                -independent[::-1],
            )
            np.testing.assert_array_equal(thresholds[item, n_active:], 0.0)
        model.copy().set_parameters(**params)


def test_gen_random_pars_bounds_ideal_point_peak_heights():
    model = IdealPointModel(20)

    starts = gen_random_pars(
        model,
        n_sets=25,
        seed=31,
        upper_range=(0.0, 0.8),
    )

    for params in starts:
        assert np.all(params["peak_height"] > 0.0)
        assert np.all(params["peak_height"] < 0.8)
        model.copy().set_parameters(**params)


def test_gen_random_pars_preserves_nominal_reference_and_padding():
    model = NominalResponseModel(2, n_categories=[3, 5])

    params = gen_random_pars(model, seed=11)[0]

    for name in ("slopes", "intercepts"):
        np.testing.assert_array_equal(params[name][:, 0], 0.0)
        np.testing.assert_array_equal(params[name][0, 3:], 0.0)
    model.copy().set_parameters(**params)


def test_gen_random_pars_keeps_lower_asymptote_below_upper():
    model = FourParameterLogistic(100)

    params = gen_random_pars(model, n_sets=10, seed=5)

    assert all(np.all(values["guessing"] < values["upper"]) for values in params)
    with pytest.raises(MirtValidationError, match="below upper_range"):
        gen_random_pars(
            model,
            guessing_range=(0.2, 0.7),
            upper_range=(0.6, 0.9),
        )


def test_calc_null_matches_manual_independence_likelihood():
    responses = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])

    result = calc_null(responses)
    expected_ll = 3 * np.log(0.75) + np.log(0.25) + 4 * np.log(0.5)

    assert result["log_likelihood"] == pytest.approx(expected_ll)
    assert result["n_parameters"] == 2
    assert result["aic"] == pytest.approx(-2 * expected_ll + 4)
    assert result["bic"] == pytest.approx(-2 * expected_ll + 2 * np.log(4))


def test_calc_null_intercept_only_uses_one_pooled_probability():
    responses = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])

    independence = calc_null(responses, model_type="independence")
    pooled = calc_null(responses, model_type="intercept_only")
    expected_ll = 5 * np.log(0.625) + 3 * np.log(0.375)

    assert pooled["log_likelihood"] == pytest.approx(expected_ll)
    assert pooled["n_parameters"] == 1
    assert pooled["log_likelihood"] < independence["log_likelihood"]


def test_calc_null_treats_nan_and_negative_values_as_missing():
    with_nan = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, 0.0]])
    with_negative = np.array([[1, -9], [0, 1], [1, 0]])

    assert calc_null(with_nan) == pytest.approx(calc_null(with_negative))


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.array([0, 1]), "2D"),
        (np.array([[0, 2]]), "dichotomous"),
        (np.array([[0, -1], [1, -1]]), "at least one observed"),
    ],
)
def test_calc_null_rejects_invalid_responses(responses, message):
    with pytest.raises(MirtDataError, match=message):
        calc_null(responses)


def test_calc_null_rejects_unknown_model_type():
    with pytest.raises(MirtValidationError, match="model_type"):
        calc_null(np.array([[0, 1]]), model_type="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("n_starts", [0, -1, True, 1.5])
def test_multi_start_fit_rejects_invalid_start_counts(n_starts):
    with pytest.raises(MirtValidationError, match="n_starts"):
        multi_start_fit(
            TwoParameterLogistic(2),
            np.array([[0, 1], [1, 0]]),
            n_starts=n_starts,
        )


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
def test_multi_start_fit_rejects_invalid_worker_counts(n_jobs):
    with pytest.raises(MirtValidationError, match="n_jobs"):
        multi_start_fit(
            TwoParameterLogistic(2),
            np.array([[0, 1], [1, 0]]),
            n_jobs=n_jobs,
        )


def test_multi_start_fit_validates_codes_against_model():
    with pytest.raises(MirtDataError, match="dichotomous"):
        multi_start_fit(TwoParameterLogistic(2), np.array([[0, 2]]))
    with pytest.raises(MirtDataError, match="below n_categories"):
        multi_start_fit(
            GradedResponseModel(2, n_categories=[2, 3]),
            np.array([[0, 3]]),
        )


def test_multi_start_fit_applies_starts_before_fitting(monkeypatch):
    seen_difficulties: list[np.ndarray] = []

    def fake_fit(self, model, responses):
        del self, responses
        assert model.is_fitted
        seen_difficulties.append(model.parameters["difficulty"])
        return SimpleNamespace(log_likelihood=float(len(seen_difficulties)))

    monkeypatch.setattr(EMEstimator, "fit", fake_fit)
    result = multi_start_fit(
        TwoParameterLogistic(3),
        np.array([[0, 1, 0], [1, 0, 1]]),
        n_starts=2,
        seed=23,
    )

    assert result.log_likelihood == 2.0
    assert len(seen_difficulties) == 2
    assert not np.array_equal(seen_difficulties[0], seen_difficulties[1])


def test_multi_start_fit_breaks_likelihood_ties_by_start_order(monkeypatch):
    def fake_start(args):
        index = args[0]
        return index, 5.0, SimpleNamespace(index=index), None

    monkeypatch.setattr(starting, "_fit_single_start", fake_start)
    result = multi_start_fit(
        TwoParameterLogistic(2),
        np.array([[0, 1], [1, 0]]),
        n_starts=3,
        seed=3,
    )

    assert result.index == 0


def test_multi_start_fit_reports_first_setup_failure(monkeypatch):
    def invalid_starts(*args: Any, **kwargs: Any):
        del args, kwargs
        return [{"not_a_parameter": np.zeros(2)}] * 2

    monkeypatch.setattr(starting, "gen_random_pars", invalid_starts)
    with pytest.raises(RuntimeError, match="All 2.*Unknown parameter"):
        multi_start_fit(
            TwoParameterLogistic(2),
            np.array([[0, 1], [1, 0]]),
            n_starts=2,
        )


def test_multi_start_fit_handles_one_parameter_model_without_mutating_input():
    responses = np.tile(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        ),
        (5, 1),
    )
    model = OneParameterLogistic(3)
    original = model.parameters

    result = multi_start_fit(
        model,
        responses,
        n_starts=2,
        seed=17,
        n_quadpts=5,
        max_iter=2,
        use_gpu=False,
    )

    assert np.isfinite(result.log_likelihood)
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_starting_utilities_are_available_from_public_namespaces():
    for name in ("gen_random_pars", "calc_null", "multi_start_fit"):
        assert getattr(mirt_utils, name) is getattr(starting, name)
        assert getattr(mirt, name) is getattr(starting, name)


def test_starting_module_doctests():
    failures, _ = doctest.testmod(starting)
    assert failures == 0
