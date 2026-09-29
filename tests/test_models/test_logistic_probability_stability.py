"""High-precision probability and centering contracts for logistic models."""

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

_MAX_FLOAT = np.finfo(float).max


def test_four_parameter_probabilities_preserve_exact_asymptotes():
    rng = np.random.default_rng(79)
    upper = rng.random(1000)
    guessing = upper * rng.random(1000)
    model = FourParameterLogistic(upper.size).set_parameters(
        guessing=guessing, upper=upper
    )
    probability = model.probability([-np.inf, -40.0, 0.0, 40.0, np.inf])
    np.testing.assert_array_equal(probability[0], guessing)
    np.testing.assert_array_equal(probability[-1], upper)
    assert np.all((probability >= guessing) & (probability <= upper))


def test_nonfinite_multidimensional_abilities_do_not_enter_exact_recovery():
    model = TwoParameterLogistic(1, n_factors=3)
    theta = np.array([[np.inf, 0.0, 0.0], [-np.inf, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability(theta, 0)
        information = model.information(theta, 0)
        pairs = model.probability_pairs(theta, [0, 0, 0])
    np.testing.assert_array_equal(probability[:2], [1.0, 0.0])
    np.testing.assert_array_equal(pairs, probability)
    np.testing.assert_array_equal(information[:2], [0.0, 0.0])
    assert np.isnan(probability[-1]) and np.isnan(information[-1])


def _decimal_curve(model, theta, item):
    with localcontext() as context:
        # Exact float products can span more than 2,000 decimal places before
        # cancellation. Reduce precision only after the centered dot product.
        context.prec = 3000
        parameters = model.parameters
        slopes = [
            Decimal.from_float(float(v))
            for v in np.atleast_1d(parameters["discrimination"][item])
        ]
        points = [Decimal.from_float(float(v)) for v in np.atleast_1d(theta)]
        b = Decimal.from_float(float(parameters["difficulty"][item]))
        z = sum(a * (t - b) for a, t in zip(slopes, points, strict=True))
        norm = sum(a * a for a in slopes)
        c = Decimal.from_float(
            float(parameters.get("guessing", np.zeros(model.n_items))[item])
        )
        d = Decimal.from_float(
            float(parameters.get("upper", np.ones(model.n_items))[item])
        )
        context.prec = 100
        if abs(z) > 10_000:
            return float(d if z > 0 else c), 0.0
        tail = (-abs(z)).exp()
        success, failure = 1 / (1 + tail), tail / (1 + tail)
        if z < 0:
            success, failure = failure, success
        probability = c + (d - c) * success
        complement = 1 - d + (d - c) * failure
        if norm == 0 or c == d:
            information = Decimal(0)
        else:
            information = (
                norm * ((d - c) * success * failure) ** 2 / (probability * complement)
            )
        return float(probability), float(information)


@pytest.mark.parametrize(
    "model_class", [TwoParameterLogistic, ThreeParameterLogistic, FourParameterLogistic]
)
@pytest.mark.parametrize(
    "slope,location,theta",
    [
        (1e-308, -_MAX_FLOAT, _MAX_FLOAT),
        (1e-308, _MAX_FLOAT, -_MAX_FLOAT),
        (0.0, _MAX_FLOAT, -_MAX_FLOAT),
        (1e308, 0.0, 2.0),
        (-1e308, 0.0, 2.0),
    ],
)
def test_finite_unidimensional_inputs_preserve_representable_probabilities(
    model_class, slope, location, theta
):
    model = model_class(1).set_parameters(
        discrimination=np.array([slope]), difficulty=np.array([location])
    )
    expected, _ = _decimal_curve(model, theta, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        np.testing.assert_allclose(
            model.probability([theta]), expected, rtol=3e-13, atol=0.0
        )
        np.testing.assert_allclose(
            model.probability([theta], 0), expected, rtol=3e-13, atol=0.0
        )
        np.testing.assert_allclose(
            model.probability_pairs([[theta]], [0]), expected, rtol=3e-13, atol=0.0
        )


@pytest.mark.parametrize(
    "model_class",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
    ],
)
def test_probability_keeps_subnormal_tails_and_nonfinite_limits(model_class):
    model = model_class(1)
    theta = np.array([-1000.0, -745.0, -710.0, -40.0, 0.0, 40.0, 710.0, 1000.0])
    expected = [_decimal_curve(model, point, 0)[0] for point in theta]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.probability(theta, 0)
        limits = model.probability([-np.inf, np.inf, np.nan], 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0.0)
    np.testing.assert_array_equal(
        limits[:2], [model.parameters.get("guessing", [0.0])[0], 1.0]
    )
    assert np.isnan(limits[-1])


@pytest.mark.parametrize(
    "slopes,location,theta",
    [
        ([0.7, 1.1, 1.3], 1e16, [1e16 + 2.0, 1e16, 1e16]),
        ([1.0, 1.0, 1.0], 1e308, [1e308, 1e308, 1e308]),
        ([1e308, 1e308, 1e308], 0.0, [0.0, 0.0, 0.0]),
        ([1e308, 1.0, -1e308], 0.0, [1.0, 1.0, 1.0]),
        ([1e150, -1e150, 1.0], 0.0, [1e200, 1e200, 1000.0]),
        ([1e150, -1e150, 1.0], 0.0, [1e200, 1e200, -1000.0]),
        ([1e150, 1.0, -1e150], 0.0, [1e200, 2.0, 1e200]),
        ([1e-308, 1e-308, 1e-308], -_MAX_FLOAT, [_MAX_FLOAT] * 3),
        ([0.0, 0.0, 0.0], _MAX_FLOAT, [-_MAX_FLOAT] * 3),
        ([-1.0, 2.0, -3.0], 1e308, [1e308] * 3),
    ],
)
def test_multidimensional_centering_and_cancellation_match_decimal(
    slopes, location, theta
):
    model = TwoParameterLogistic(1, n_factors=3).set_parameters(
        discrimination=np.array([slopes]),
        difficulty=np.array([location]),
    )
    expected_p, expected_i = _decimal_curve(model, theta, 0)
    theta = np.array([theta])
    original = theta.copy()
    theta.flags.writeable = False
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for item_idx in (None, 0):
            np.testing.assert_allclose(
                model.probability(theta, item_idx), expected_p, rtol=3e-12, atol=0.0
            )
            np.testing.assert_allclose(
                model.information(theta, item_idx), expected_i, rtol=3e-12, atol=0.0
            )
        np.testing.assert_allclose(
            model.probability_pairs(theta, [0]), expected_p, rtol=3e-12, atol=0.0
        )
    np.testing.assert_array_equal(theta, original)


@pytest.mark.parametrize(
    "method,item_idx",
    [
        ("probability", None),
        ("probability", 1),
        ("information", None),
        ("information", 1),
        ("probability_pairs", None),
    ],
)
def test_multidimensional_recovery_buffers_and_pairs_are_bounded(
    method, item_idx, monkeypatch
):
    model = TwoParameterLogistic(3, n_factors=3).set_parameters(
        discrimination=np.array([[0.7, 1.1, 1.3], [1e150, 1.0, -1e150], [1e308] * 3]),
        difficulty=np.array([1e16, 0.0, 0.0]),
    )
    storage = np.full((78, 3), np.nan)
    storage[::2] = np.resize(
        np.array([[1e16 + 2.0, 1e16, 1e16], [1e200, 2.0, 1e200], [0.0] * 3]), (39, 3)
    )
    theta = storage[::2]
    indices = np.tile([2, 0, 1, 0, 0, 0], 13)[::2]
    theta.flags.writeable = False
    indices.flags.writeable = False
    expected = np.array(
        [[_decimal_curve(model, point, item) for item in range(3)] for point in theta]
    )
    if method == "probability_pairs":
        expected = expected[np.arange(theta.shape[0]), indices, 0]
    else:
        expected = expected[:, :, int(method == "information")]
        if item_idx is not None:
            expected = expected[:, item_idx]
    original = dichotomous._centered_logit_pairs
    sizes = []

    def tracked(points, *parameters):
        assert points.size <= 7
        sizes.append(points.size)
        return original(points, *parameters)

    monkeypatch.setattr(dichotomous, "_LOGISTIC_CURVE_CHUNK_ELEMENTS", 7)
    monkeypatch.setattr(dichotomous, "_centered_logit_pairs", tracked)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = getattr(model, method)(
            theta, indices if method == "probability_pairs" else item_idx
        )
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
    assert len(sizes) > 1
