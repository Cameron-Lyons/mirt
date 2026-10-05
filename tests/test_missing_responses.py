"""NaN and nullable-DataFrame missing responses across entry points.

``fit_mirt`` treats negative codes, ``NaN`` and the nulls of nullable
DataFrame columns as missing. Scoring, posterior, person-fit, item-fit,
residual, model-fit and local-dependence entry points must accept the same
data and give results identical to those for the ``-1``-coded integer matrix.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mirt
from mirt.diagnostics.comparison import vuong_test
from mirt.diagnostics.itemfit import compute_itemfit
from mirt.diagnostics.ld import compute_ld_chi2, compute_ld_statistics, compute_q3
from mirt.diagnostics.personfit import compute_personfit
from mirt.diagnostics.residuals import (
    analyze_residuals,
    compute_outfit_infit,
    compute_residuals,
    identify_misfitting_patterns,
)
from mirt.exceptions import MirtValidationError
from mirt.scoring import EAPSumScorer, ability_posterior, eapsum, eapsum_table
from mirt.utils.data import _missing_coded_responses
from mirt.utils.residuals import LD_X2

_METHODS = ("EAP", "MAP", "ML", "WLE", "EAPsum")
_ITEM_FIT_STATISTICS = [
    "infit",
    "outfit",
    "z_infit",
    "z_outfit",
    "S_X2",
    "X2",
    "G2",
    "PV_Q1",
]


def _missing_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.random.default_rng(11).random(shape) < 0.08
    mask[0] = False
    mask[1, :-1] = True
    return mask


@pytest.fixture(scope="module")
def binary_fit() -> tuple[Any, np.ndarray]:
    data = mirt.simdata(model="2PL", n_persons=240, n_items=6, seed=3)
    coded = np.where(_missing_mask(data.shape), -1, data)
    return mirt.fit_mirt(coded, max_iter=60), coded


@pytest.fixture(scope="module")
def graded_fit() -> tuple[Any, np.ndarray]:
    data = mirt.simdata(model="GRM", n_persons=200, n_items=5, n_categories=4, seed=4)
    coded = np.where(_missing_mask(data.shape), -1, data)
    return mirt.fit_mirt(coded, model="GRM", max_iter=60), coded


def _with_nan(coded: np.ndarray) -> np.ndarray:
    return np.where(coded < 0, np.nan, coded.astype(np.float64))


def _nullable_frame(coded: np.ndarray) -> Any:
    pd = pytest.importorskip("pandas")
    columns = [f"q{index}" for index in range(coded.shape[1])]
    return pd.DataFrame(coded, columns=columns).replace(-1, pd.NA).astype("Int64")


def _polars_frame(coded: np.ndarray) -> Any:
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {
            f"q{index}": [None if value < 0 else int(value) for value in column]
            for index, column in enumerate(coded.T)
        }
    )


_MISSING_INPUTS: dict[str, Callable[[np.ndarray], Any]] = {
    "nan-array": _with_nan,
    "nan-list": lambda coded: _with_nan(coded).tolist(),
    "nullable-pandas": _nullable_frame,
    "polars": _polars_frame,
}


@pytest.fixture(params=sorted(_MISSING_INPUTS))
def to_missing(request: pytest.FixtureRequest) -> Callable[[np.ndarray], Any]:
    return _MISSING_INPUTS[request.param]


def _frame_columns(frame: Any) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(frame[name]) for name in frame.columns}


def _assert_frames_equal(actual: Any, expected: Any) -> None:
    actual_columns = _frame_columns(actual)
    expected_columns = _frame_columns(expected)
    assert actual_columns.keys() == expected_columns.keys()
    for name, values in expected_columns.items():
        assert_array_equal(actual_columns[name], values, err_msg=name)


def _assert_same(actual: Any, expected: Any) -> None:
    """Compare nested results exactly, treating NaN as equal to NaN."""
    if dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            _assert_same(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_same(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_same(actual_value, expected_value)
    else:
        assert_array_equal(actual, expected)


def _assert_same_fit(actual: Any, expected: Any) -> None:
    assert actual.log_likelihood == expected.log_likelihood
    _assert_same(actual.model.parameters, expected.model.parameters)
    _assert_same(actual.standard_errors, expected.standard_errors)


def test_missing_coded_responses_maps_nan_and_nulls_to_missing_code() -> None:
    coded = np.array([[1, -1, 0], [-1, 1, 1]])

    assert_array_equal(_missing_coded_responses(_with_nan(coded)), coded)
    assert_array_equal(_missing_coded_responses(_nullable_frame(coded)), coded)
    assert_array_equal(_missing_coded_responses(_polars_frame(coded)), coded)
    assert_array_equal(
        _missing_coded_responses(_with_nan(coded), missing_code=-9),
        np.where(coded < 0, -9, coded),
    )
    # Integer input is returned as is, and other values are left to callers.
    assert _missing_coded_responses(coded) is coded
    assert np.isinf(_missing_coded_responses([[np.inf, 0.0]])[0, 0])
    assert _missing_coded_responses([["1", "0"]]).dtype.kind == "U"


def test_fit_mirt_treats_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
) -> None:
    expected, coded = binary_fit

    actual = mirt.fit_mirt(to_missing(coded), max_iter=60)

    _assert_same_fit(actual, expected)


@pytest.mark.parametrize("method", _METHODS)
def test_fscores_treats_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
    method: str,
) -> None:
    result, coded = binary_fit
    expected = mirt.fscores(result, coded, method=method)

    actual = mirt.fscores(result, to_missing(coded), method=method)

    assert_array_equal(actual.theta, expected.theta)
    assert_array_equal(actual.standard_error, expected.standard_error)


@pytest.mark.parametrize("method", _METHODS)
def test_polytomous_fscores_treat_nan_as_missing(
    graded_fit: tuple[Any, np.ndarray], method: str
) -> None:
    result, coded = graded_fit
    expected = mirt.fscores(result, coded, method=method)

    actual = mirt.fscores(result, _with_nan(coded), method=method)

    assert_array_equal(actual.theta, expected.theta)
    assert_array_equal(actual.standard_error, expected.standard_error)


def test_eapsum_helpers_treat_nan_as_missing(
    binary_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = binary_fit
    expected = eapsum(result, coded)

    actual = eapsum(result, _nullable_frame(coded))
    scorer_actual = EAPSumScorer().score(result.model, _with_nan(coded))

    assert_array_equal(actual.theta, expected.theta)
    assert_array_equal(actual.standard_error, expected.standard_error)
    assert_array_equal(scorer_actual.theta, expected.theta)
    assert_array_equal(scorer_actual.standard_error, expected.standard_error)


def test_eapsum_table_reports_nan_as_missing_not_invalid(
    binary_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = binary_fit
    complete = coded[~np.any(coded < 0, axis=1)]

    with pytest.raises(MirtValidationError, match="complete responses"):
        eapsum_table(result, _with_nan(coded))

    table = eapsum_table(result, _nullable_frame(complete))
    assert_array_equal(table.observed, eapsum_table(result, complete).observed)


def test_ability_posterior_treats_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
) -> None:
    result, coded = binary_fit
    expected = ability_posterior(result, coded)

    actual = mirt.ability_posterior(result, to_missing(coded))

    assert_array_equal(actual.weights, expected.weights)
    assert_array_equal(actual.log_marginal_likelihood, expected.log_marginal_likelihood)


def test_personfit_treats_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
) -> None:
    result, coded = binary_fit
    statistics = ["infit", "outfit", "z_infit", "z_outfit", "Zh"]
    expected = mirt.personfit(result, coded, statistics=statistics, p_adjust="holm")

    actual = mirt.personfit(
        result, to_missing(coded), statistics=statistics, p_adjust="holm"
    )

    _assert_frames_equal(actual, expected)


def test_polytomous_personfit_treats_nan_as_missing(
    graded_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = graded_fit
    theta = mirt.fscores(result, coded).theta
    expected = compute_personfit(result.model, coded, theta)

    actual = compute_personfit(result.model, _with_nan(coded), theta)
    wrapped = mirt.personfit(result, _with_nan(coded))

    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        assert_array_equal(actual[name], values, err_msg=name)
        assert_array_equal(np.asarray(wrapped[name]), values, err_msg=name)


@pytest.mark.filterwarnings("ignore:z_infit and z_outfit with EAP:UserWarning")
def test_itemfit_treats_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
) -> None:
    result, coded = binary_fit
    options = {"statistics": _ITEM_FIT_STATISTICS, "na_rm": True, "seed": 5}
    expected = mirt.itemfit(result, coded, **options)

    actual = mirt.itemfit(result, to_missing(coded), **options)

    _assert_frames_equal(actual, expected)


@pytest.mark.filterwarnings("ignore:z_infit and z_outfit with EAP:UserWarning")
def test_polytomous_itemfit_treats_nan_as_missing(
    graded_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = graded_fit
    options = {"statistics": _ITEM_FIT_STATISTICS, "na_rm": True, "seed": 5}
    expected = compute_itemfit(result, coded, **options)

    actual = compute_itemfit(result, _nullable_frame(coded), **options)

    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        assert_array_equal(actual[name], values, err_msg=name)


def test_s_x2_still_requires_complete_rows_for_nan(
    binary_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = binary_fit

    with pytest.raises(ValueError, match="complete responses"):
        mirt.itemfit(result, _with_nan(coded), statistics=["S_X2"])


_RESIDUAL_CALLS: dict[str, Callable[[Any, Any], Any]] = {
    "raw": lambda result, data: compute_residuals(result, data, residual_type="raw"),
    "deviance": lambda result, data: compute_residuals(
        result, data, residual_type="deviance"
    ),
    "analyze": analyze_residuals,
    "outfit_infit": lambda result, data: compute_outfit_infit(
        result, data, include_counts=True, include_standardized=True
    ),
    "misfit": identify_misfitting_patterns,
}


@pytest.mark.parametrize("fit", ["binary_fit", "graded_fit"])
@pytest.mark.parametrize("call", sorted(_RESIDUAL_CALLS))
def test_residual_helpers_treat_nan_and_nulls_as_missing(
    request: pytest.FixtureRequest, fit: str, call: str
) -> None:
    result, coded = request.getfixturevalue(fit)
    function = _RESIDUAL_CALLS[call]
    expected = function(result, coded)

    for data in (_with_nan(coded), _nullable_frame(coded)):
        _assert_same(function(result, data), expected)


_MODEL_FIT_AND_LD_CALLS: dict[str, Callable[[Any, Any, np.ndarray], Any]] = {
    "ld_statistics": lambda result, data, _: compute_ld_statistics(result, data),
    "q3": lambda result, data, _: compute_q3(result, data),
    "ld_chi2": lambda result, data, _: compute_ld_chi2(result, data),
    "m2": lambda result, data, _: mirt.compute_m2(result, data),
    "fit_indices": lambda result, data, _: mirt.compute_fit_indices(result, data),
    "utils_residuals": lambda result, data, theta: mirt.residuals(
        result.model, data, theta
    ),
    "utils_q3": lambda result, data, theta: mirt.Q3(result.model, data, theta),
    "utils_ld_x2": lambda result, data, theta: LD_X2(result.model, data, theta),
}


@pytest.mark.parametrize("call", sorted(_MODEL_FIT_AND_LD_CALLS))
def test_model_fit_and_local_dependence_treat_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
    to_missing: Callable[[np.ndarray], Any],
    call: str,
) -> None:
    result, coded = binary_fit
    theta = mirt.fscores(result, coded).theta
    function = _MODEL_FIT_AND_LD_CALLS[call]
    expected = function(result, coded, theta)

    _assert_same(function(result, to_missing(coded), theta), expected)


def test_vuong_test_treats_nan_as_missing(
    binary_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = binary_fit
    rasch = mirt.fit_mirt(coded, model="1PL", max_iter=60)

    expected = vuong_test(rasch, result, coded)
    actual = vuong_test(rasch, result, _with_nan(coded))

    assert actual == expected


def test_plausible_values_and_bootstrap_treat_nan_and_nulls_as_missing(
    binary_fit: tuple[Any, np.ndarray],
) -> None:
    result, coded = binary_fit
    expected_values = mirt.generate_plausible_values(result, coded, seed=2)
    expected_se = mirt.bootstrap_se(result, coded, n_bootstrap=3, seed=2)

    for data in (_with_nan(coded), _nullable_frame(coded)):
        values = mirt.generate_plausible_values(result, data, seed=2)
        standard_errors = mirt.bootstrap_se(result, data, n_bootstrap=3, seed=2)
        assert_array_equal(values, expected_values)
        _assert_same(standard_errors, expected_se)


def test_fit_then_score_a_nullable_frame() -> None:
    """The canonical ``fscores(fit_mirt(df), df)`` workflow with missing data."""
    data = mirt.simdata(model="2PL", n_persons=150, n_items=5, seed=8)
    frame = _nullable_frame(np.where(_missing_mask(data.shape), -1, data))

    result = mirt.fit_mirt(frame, max_iter=40)
    scores = mirt.fscores(result, frame)
    fit = mirt.personfit(result, frame)
    items = mirt.itemfit(result, frame)

    assert result.model.item_names == list(frame.columns)
    assert np.all(np.isfinite(scores.theta))
    assert len(fit) == len(frame)
    assert len(items) == frame.shape[1]


@pytest.mark.parametrize(
    "call",
    [
        lambda result, data: mirt.fscores(result, data),
        lambda result, data: mirt.fscores(result, data, method="EAPsum"),
        lambda result, data: mirt.fscores(result, data, method="WLE"),
        lambda result, data: mirt.ability_posterior(result, data),
        lambda result, data: mirt.personfit(result, data),
        lambda result, data: compute_personfit(
            result.model, data, np.zeros(data.shape[0])
        ),
    ],
    ids=["EAP", "EAPsum", "WLE", "posterior", "personfit", "compute_personfit"],
)
def test_infinite_responses_are_still_rejected(
    binary_fit: tuple[Any, np.ndarray], call: Callable[[Any, np.ndarray], Any]
) -> None:
    result, coded = binary_fit
    data = coded[:5].astype(np.float64)
    data[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        call(result, data)
