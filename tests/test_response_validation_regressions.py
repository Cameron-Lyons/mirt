"""Regression coverage for response validation at public entry points."""

import numpy as np
import pytest

from mirt import fit_mirt, validate_responses
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.mark.parametrize("value", [0.5, np.inf, -np.inf, 1 + 0j, "1"])
def test_validate_responses_rejects_non_integer_numeric_codes(value):
    with pytest.raises(MirtDataError):
        validate_responses(np.array([[0, value]]))


def test_validate_responses_treats_nan_as_missing():
    # NaN (R's NA) used to be rejected as a non-finite code; it now means missing.
    responses = validate_responses(np.array([[0.0, np.nan], [np.nan, 1.0]]))

    np.testing.assert_array_equal(responses, [[0, -1], [-1, 1]])
    assert responses.dtype == np.int_


def test_validate_responses_nan_uses_requested_missing_code():
    responses = validate_responses(np.array([[np.nan, 1.0]]), missing_code=-9)

    np.testing.assert_array_equal(responses, [[-9, 1]])


def test_validate_responses_can_reject_nan_explicitly():
    with pytest.raises(MirtDataError, match="finite response codes"):
        validate_responses(np.array([[0.0, np.nan]]), nan_as_missing=False)
    with pytest.raises(MirtDataError, match="missing data not allowed"):
        validate_responses(np.array([[0.0, np.nan]]), allow_missing=False)


def test_validate_responses_still_rejects_infinite_codes_with_missing_values():
    with pytest.raises(MirtDataError, match="finite response codes"):
        validate_responses(np.array([[np.nan, np.inf]]))


@pytest.mark.parametrize(
    "responses",
    [np.empty((0, 2), dtype=int), np.empty((2, 0), dtype=int)],
)
def test_validate_responses_rejects_empty_dimensions(responses):
    with pytest.raises(MirtDataError, match="at least one"):
        validate_responses(responses)


def test_validate_responses_normalizes_negative_missing_codes():
    responses = validate_responses([[0.0, -9.0], [1.0, 0.0]])

    np.testing.assert_array_equal(responses, [[0, -1], [1, 0]])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.array([[0, 2]]), "dichotomous responses"),
        (np.empty((0, 2), dtype=int), "at least one person"),
        (np.empty((2, 0), dtype=int), "at least one item"),
        (np.array([[0.0, 0.5]]), "integer response codes"),
    ],
)
def test_fit_mirt_rejects_invalid_response_matrices(data, message):
    with pytest.raises(MirtDataError, match=message):
        fit_mirt(data, model="2PL", use_rust=False)


def test_fit_mirt_rejects_polytomous_code_outside_declared_categories():
    with pytest.raises(MirtDataError, match="below n_categories"):
        fit_mirt(
            np.array([[0, 3], [1, 2]]),
            model="GRM",
            n_categories=3,
            use_rust=False,
        )


def test_fit_mirt_requires_categories_for_all_missing_polytomous_data():
    with pytest.raises(MirtValidationError, match="n_categories is required"):
        fit_mirt(np.full((3, 2), -1), model="GPCM", use_rust=False)


def test_fit_mirt_accepts_integral_floats_and_negative_missing_codes():
    result = fit_mirt(
        np.array([[0.0, -9.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        model="2PL",
        n_quadpts=5,
        max_iter=1,
        use_rust=False,
    )

    assert np.isfinite(result.log_likelihood)


def test_direct_estimators_share_response_validation():
    estimator = EMEstimator(n_quadpts=5, max_iter=1)
    model = TwoParameterLogistic(n_items=2)

    with pytest.raises(MirtDataError, match="integer response codes"):
        estimator.fit(model, np.array([[0.0, 0.5]]))
