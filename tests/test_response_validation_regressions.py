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


def _recoded_binary_data() -> np.ndarray:
    from mirt import simdata

    data = np.asarray(simdata(n_persons=80, n_items=4, seed=3))
    data[np.flatnonzero(data[:, 0] == 1)[:5], 0] = 2
    return data


def _two_pl_fit(data: np.ndarray):
    clean = np.where(data > 1, 1, data)
    return fit_mirt(clean, model="2PL", max_iter=20, compute_standard_errors=False)


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(
            lambda data: EMEstimator(n_quadpts=11, max_iter=2).fit(
                TwoParameterLogistic(4), data
            ),
            id="EMEstimator",
        ),
        pytest.param(
            lambda data: _gibbs(use_rust=True).fit(TwoParameterLogistic(4), data),
            id="GibbsSampler-native",
        ),
        pytest.param(
            lambda data: _gibbs(use_rust=False).fit(TwoParameterLogistic(4), data),
            id="GibbsSampler-numpy",
        ),
        pytest.param(
            lambda data: _mhrm(use_rust=True).fit(TwoParameterLogistic(4), data),
            id="MHRMEstimator-native",
        ),
        pytest.param(
            lambda data: _mhrm(use_rust=False).fit(TwoParameterLogistic(4), data),
            id="MHRMEstimator-numpy",
        ),
        pytest.param(
            lambda data: _gvem().fit(TwoParameterLogistic(4), data),
            id="GVEMEstimator",
        ),
        pytest.param(
            lambda data: _bootstrap_se()(_two_pl_fit(data), data, n_bootstrap=2),
            id="bootstrap_se",
        ),
        pytest.param(
            lambda data: _bootstrap_ci()(_two_pl_fit(data), data, n_bootstrap=2),
            id="bootstrap_ci",
        ),
    ],
)
def test_estimators_reject_non_binary_codes_for_dichotomous_models(fit):
    # Regression: only fit_mirt rejected a code 2, which the estimators read
    # as a count (slopes ran to the bound) or, natively, as incorrect.
    with pytest.raises(MirtDataError, match="dichotomous responses must be coded"):
        fit(_recoded_binary_data())


def test_estimators_reject_codes_outside_polytomous_categories():
    from mirt.models.polytomous import GradedResponseModel

    data = np.array([[0, 1], [1, 2], [2, 0], [1, 1]])
    with pytest.raises(MirtDataError, match="below n_categories"):
        EMEstimator(n_quadpts=5, max_iter=1).fit(
            GradedResponseModel(2, n_categories=[3, 2]), data
        )


def _gibbs(*, use_rust: bool):
    from mirt.estimation.mcmc import GibbsSampler

    return GibbsSampler(n_iter=20, burnin=5, seed=1, use_rust=use_rust)


def _mhrm(*, use_rust: bool):
    from mirt.estimation.mcmc import MHRMEstimator

    return MHRMEstimator(
        n_cycles=10,
        burnin=2,
        seed=1,
        use_rust=use_rust,
        compute_standard_errors=False,
    )


def _gvem():
    from mirt.estimation.gvem import GVEMEstimator

    return GVEMEstimator(max_iter=2)


def _bootstrap_se():
    from mirt.utils.bootstrap import bootstrap_se

    return bootstrap_se


def _bootstrap_ci():
    from mirt.utils.bootstrap import bootstrap_ci

    return bootstrap_ci
