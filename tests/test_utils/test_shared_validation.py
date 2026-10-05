"""Shared argument validators and Rubin pooling used across utilities."""

import numpy as np
import pytest

from mirt import averageMI, combine_plausible_values
from mirt.exceptions import MirtValidationError
from mirt.utils import clinical, confidence
from mirt.utils._validation import (
    as_finite_vector,
    validate_alpha,
    validate_finite_scalar,
    validate_positive_scalar,
)
from mirt.utils.imputation import LARGE_DF


def test_clinical_and_confidence_share_one_set_of_validators():
    for module in (clinical, confidence):
        assert module.validate_alpha is validate_alpha
        assert module.validate_finite_scalar is validate_finite_scalar
        assert module.validate_positive_scalar is validate_positive_scalar
        assert module.as_finite_vector is as_finite_vector


@pytest.mark.parametrize("value", [1, 2.5, np.float32(-3.0), np.int64(4)])
def test_finite_scalars_are_returned_as_floats(value):
    result = validate_finite_scalar(value, "value")
    assert result == float(value)
    assert type(result) is float


@pytest.mark.parametrize("value", [True, np.bool_(False), [1.0], "one", np.nan, np.inf])
def test_invalid_scalars_are_rejected(value):
    with pytest.raises(MirtValidationError, match="value must be a finite number"):
        validate_finite_scalar(value, "value")


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_positive_scalars_exclude_zero_and_negatives(value):
    with pytest.raises(MirtValidationError, match="scale must be positive"):
        validate_positive_scalar(value, "scale")
    assert validate_positive_scalar(0.5, "scale") == 0.5


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_lie_strictly_between_zero_and_one(alpha):
    with pytest.raises(MirtValidationError, match="alpha must be between 0 and 1"):
        validate_alpha(alpha)
    assert validate_alpha(0.05) == 0.05


def test_finite_vectors_are_flattened_and_validated():
    np.testing.assert_array_equal(as_finite_vector([[1, 2], [3, 4]], "x"), [1, 2, 3, 4])
    with pytest.raises(MirtValidationError, match="x must contain numeric values"):
        as_finite_vector(["a"], "x")
    with pytest.raises(MirtValidationError, match="x must contain at least one value"):
        as_finite_vector([], "x")
    with pytest.raises(MirtValidationError, match="x must contain only finite values"):
        as_finite_vector([1.0, np.nan], "x")


@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_plausible_value_and_imputation_pooling_agree(shape):
    rng = np.random.default_rng(31)
    estimates = [rng.normal(size=shape) for _ in range(6)]
    variances = [rng.uniform(0.01, 0.2, size=shape) for _ in range(6)]

    pooled_pvs = combine_plausible_values(estimates, variances)
    pooled_mi = averageMI(estimates, variances=variances)

    stacked = np.stack(estimates)
    within = np.mean(variances, axis=0)
    between = np.var(stacked, axis=0, ddof=1)
    total = within + (1 + 1 / 6) * between
    df = 5 * (1 + within / ((1 + 1 / 6) * between)) ** 2
    for pooled_estimate in (pooled_pvs["estimate"], pooled_mi.estimate):
        np.testing.assert_allclose(pooled_estimate, stacked.mean(axis=0), rtol=1e-15)
    np.testing.assert_array_equal(pooled_pvs["estimate"], pooled_mi.estimate)
    np.testing.assert_array_equal(pooled_pvs["between_var"], pooled_mi.between_variance)
    np.testing.assert_array_equal(pooled_pvs["within_var"], pooled_mi.within_variance)
    np.testing.assert_array_equal(pooled_pvs["variance"], pooled_mi.total_variance)
    np.testing.assert_array_equal(pooled_pvs["se"], pooled_mi.standard_error)
    np.testing.assert_array_equal(pooled_pvs["df"], pooled_mi.df)
    np.testing.assert_allclose(pooled_mi.total_variance, total, rtol=1e-13)
    np.testing.assert_allclose(pooled_mi.df, df, rtol=1e-12)


def test_pooling_keeps_each_degrees_of_freedom_convention_without_between_variance():
    estimates = [1.0, 1.0, 1.0]
    variances = [0.04, 0.04, 0.04]

    pooled_pvs = combine_plausible_values(estimates, variances)
    pooled_mi = averageMI(estimates, variances=variances)

    assert pooled_pvs["df"] == np.inf
    assert pooled_mi.df == LARGE_DF
    assert pooled_pvs["se"] == pooled_mi.standard_error == pytest.approx(0.2)
    assert pooled_mi.lambda_hat == 0.0
