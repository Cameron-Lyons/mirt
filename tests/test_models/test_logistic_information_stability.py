"""Independent accuracy and storage contracts for logistic Fisher information."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt.models import dichotomous
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


def _decimal_information(model, theta, item):
    with localcontext() as context:
        context.prec = 100
        parameters = model.parameters
        slopes = np.atleast_1d(parameters["discrimination"][item])
        a = [Decimal.from_float(float(value)) for value in slopes]
        b = Decimal.from_float(float(parameters["difficulty"][item]))
        points = [Decimal.from_float(float(value)) for value in np.atleast_1d(theta)]
        z = sum(slope * (point - b) for slope, point in zip(a, points, strict=True))
        norm = sum(slope * slope for slope in a)
        c = Decimal.from_float(
            float(parameters.get("guessing", np.zeros(model.n_items))[item])
        )
        d = Decimal.from_float(
            float(parameters.get("upper", np.ones(model.n_items))[item])
        )
        if abs(z) > 10_000 or norm == 0 or c == d:
            return 0.0
        tail = (-abs(z)).exp()
        success, failure = 1 / (1 + tail), tail / (1 + tail)
        if z < 0:
            success, failure = failure, success
        probability = c + (d - c) * success
        complement = 1 - d + (d - c) * failure
        numerator = norm * ((d - c) * success * failure) ** 2
        return float(numerator / (probability * complement))


@pytest.mark.parametrize(
    "model_class",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
    ],
)
def test_logistic_information_preserves_both_tails(model_class):
    model = model_class(2)
    if model_class is FourParameterLogistic:
        model.set_parameters(upper=np.array([1.0, 0.9]))
    theta = np.array(
        [-1000.0, -745.0, -710.0, -400.0, -40.0, 0.0, 40.0, 400.0, 710.0, 745.0, 1000.0]
    )
    expected = np.array(
        [
            [_decimal_information(model, point, item) for item in range(2)]
            for point in theta
        ]
    )
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information(theta)
        selected = model.information(theta, 1)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(selected, expected[:, 1], rtol=3e-12, atol=0.0)


@pytest.mark.parametrize(
    "model_class", [TwoParameterLogistic, ThreeParameterLogistic, FourParameterLogistic]
)
@pytest.mark.parametrize(
    "slope,logit",
    [
        (1e308, 1000.0),
        (-1e308, -1000.0),
        (1e200, -400.0),
        (1e200, 400.0),
        (1e-200, 0.0),
        (1e308, 0.0),
        (0.0, 0.0),
    ],
)
def test_slope_rescaling_recovers_information(model_class, slope, logit):
    model = model_class(1).set_parameters(discrimination=np.array([slope]))
    theta = 0.0 if slope == 0.0 else logit / slope
    expected = _decimal_information(model, theta, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information([theta], 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)


@pytest.mark.parametrize(
    "guessing,upper",
    [
        (0.0, 1.0),
        (0.0, 0.9),
        (0.2, 1.0),
        (0.2, 0.9),
        (0.0, np.nextafter(0.0, 1.0)),
        (0.3, 0.3),
        (0.0, 0.0),
    ],
)
def test_asymptote_limits_retain_information(guessing, upper):
    model = FourParameterLogistic(1).set_parameters(
        discrimination=np.array([1e200]),
        guessing=np.array([guessing]),
        upper=np.array([upper]),
    )
    theta = np.array([-400.0, 0.0, 400.0]) / 1e200
    expected = [_decimal_information(model, point, 0) for point in theta]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information(theta, 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)


@pytest.mark.parametrize("scale", [0.0, 1.0, 1e200, 1.7e308])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_multidimensional_information_scales_norm_without_squaring_overflow(
    scale, sign
):
    model = TwoParameterLogistic(1, n_factors=3).set_parameters(
        discrimination=np.array([[scale, sign * scale, scale / 2.0]])
    )
    theta = np.zeros((5, 3))
    theta[:, 0] = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0]) / (scale or 1.0)
    expected = np.array([_decimal_information(model, point, 0) for point in theta])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        all_items = model.information(theta)
        single = model.information(theta, 0)
    np.testing.assert_allclose(all_items[:, 0], expected, rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(single, expected, rtol=3e-12, atol=0.0)


@pytest.mark.parametrize(
    "model_class", [TwoParameterLogistic, ThreeParameterLogistic, FourParameterLogistic]
)
def test_finite_offset_overflow_is_recovered_before_information(model_class):
    largest = np.finfo(float).max
    model = model_class(1).set_parameters(
        discrimination=np.array([0.0]), difficulty=np.array([-largest])
    )
    # A zero slope at an overflowing offset must still describe a flat curve.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        np.testing.assert_array_equal(model.information([largest]), [[0.0]])
    model.set_parameters(discrimination=np.array([1e-308]))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information([largest])
    np.testing.assert_array_equal(actual, [[_decimal_information(model, largest, 0)]])


@pytest.mark.parametrize(
    "model_class,n_factors",
    [
        (OneParameterLogistic, 1),
        (TwoParameterLogistic, 1),
        (TwoParameterLogistic, 3),
        (ThreeParameterLogistic, 1),
        (FourParameterLogistic, 1),
    ],
)
@pytest.mark.parametrize("item_idx", [None, 2])
def test_information_batches_preserve_strided_inputs(
    model_class, n_factors, item_idx, monkeypatch
):
    model = model_class(5, n_factors=n_factors).set_parameters(
        difficulty=np.linspace(-1.0, 1.0, 5)
    )
    theta = np.linspace(-4.0, 4.0, 74 * n_factors).reshape(74, n_factors)[::2]
    original_theta = theta.copy()
    theta.flags.writeable = False
    items = range(5) if item_idx is None else [item_idx]
    expected = np.array(
        [
            [_decimal_information(model, point, item) for item in items]
            for point in theta
        ]
    )
    if item_idx is not None:
        expected = expected[:, 0]
    original = dichotomous._logistic_information
    sizes = []

    def tracked(logits, *args, **kwargs):
        assert logits.size <= 17
        sizes.append(logits.size)
        return original(logits, *args, **kwargs)

    monkeypatch.setattr(dichotomous, "_LOGISTIC_CURVE_CHUNK_ELEMENTS", 17)
    monkeypatch.setattr(dichotomous, "_logistic_information", tracked)
    actual = model.information(theta, item_idx)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
    np.testing.assert_array_equal(theta, original_theta)
    assert sum(sizes) == expected.size and len(sizes) > 1


@pytest.mark.parametrize(
    "model_class",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
    ],
)
def test_information_retains_infinite_limits_and_nan(model_class):
    model = model_class(1)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.information([-np.inf, np.inf, np.nan], 0)
    np.testing.assert_array_equal(actual[:2], [0.0, 0.0])
    assert np.isnan(actual[-1])
