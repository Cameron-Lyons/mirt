"""Regression coverage for overflow before 5PL shape rescaling."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt.models import dichotomous
from mirt.models.dichotomous import FiveParameterLogistic
from mirt.utils.information import gen_difficulty


def _decimal_curve(theta, slope, location, guessing, upper, shape):
    """Compute an independent high-precision reference, including both tails."""
    with localcontext() as context:
        context.prec = 80
        t, a, b, c, d, e = (
            Decimal.from_float(float(value))
            for value in (theta, slope, location, guessing, upper, shape)
        )
        z = a * (t - b)
        if z > 10_000:
            return float(d), 0.0
        # These approximations have relative errors below exp(-100), far
        # smaller than the precision of the float results being checked.
        if z < -100:
            log_sigmoid, complement = z, Decimal(1)
        elif z > 100:
            complement = (-z).exp()
            log_sigmoid = -complement
        else:
            complement = 1 / (1 + z.exp())
            log_sigmoid = -(1 + (-z).exp()).ln()
        exponent = e * log_sigmoid
        if exponent < -10_000:
            return float(c), 0.0
        power = exponent.exp()
        failure_power = -exponent if abs(exponent) < Decimal("1e-60") else 1 - power
        probability = c + (d - c) * power
        failure = 1 - d + (d - c) * failure_power
        derivative = a * e * (d - c) * power * complement
        information = derivative**2 / (probability * failure)
        return float(probability), float(information)


_MAX_FLOAT = np.finfo(float).max
_MIN_FLOAT = np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("guessing,upper", [(0.0, 1.0), (0.2, 0.95)])
@pytest.mark.parametrize(
    "slope,shape,location,theta",
    [
        (1e308, 1e-308, 0.0, -2.0),
        (-1e308, 1e-308, 0.0, 2.0),
        (1e308, _MIN_FLOAT, 0.0, -2e15),
        (1e-308, 1.0, -_MAX_FLOAT, _MAX_FLOAT),
        (1e-308, 1.0, _MAX_FLOAT, -_MAX_FLOAT),
        (0.0, 2.0, -_MAX_FLOAT, _MAX_FLOAT),
        (1.0, 1e-308, _MAX_FLOAT, -_MAX_FLOAT),
        (1.0, 1.0, _MAX_FLOAT, -_MAX_FLOAT),
        (1.0, _MIN_FLOAT, -_MAX_FLOAT, _MAX_FLOAT),
    ],
)
def test_extreme_finite_inputs_match_decimal_reference(
    slope, shape, location, theta, guessing, upper
):
    model = FiveParameterLogistic(1).set_parameters(
        discrimination=np.array([slope]),
        difficulty=np.array([location]),
        guessing=np.array([guessing]),
        upper=np.array([upper]),
        asymmetry=np.array([shape]),
    )
    points = np.array([[theta]])
    points.flags.writeable = False
    expected_p, expected_i = _decimal_curve(
        theta, slope, location, guessing, upper, shape
    )

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for item_idx in (None, 0):
            np.testing.assert_allclose(
                model.probability(points, item_idx), expected_p, rtol=3e-12, atol=0.0
            )
            np.testing.assert_allclose(
                model.information(points, item_idx), expected_i, rtol=3e-12, atol=0.0
            )
        np.testing.assert_allclose(
            model.probability_pairs(points, np.array([0])),
            expected_p,
            rtol=3e-12,
            atol=0.0,
        )
    np.testing.assert_array_equal(points, [[theta]])


def test_extreme_asymmetry_inverse_round_trip_survives_overflowed_logit():
    model = FiveParameterLogistic(1).set_parameters(
        discrimination=np.array([1e308]),
        guessing=np.array([0.0]),
        asymmetry=np.array([_MIN_FLOAT]),
    )
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        theta = gen_difficulty(model, 0, 0.5, theta_range=(-1e16, 1e16))
        probability = model.probability([theta], 0)[0]
        information = model.information([theta], 0)[0]
    assert probability == pytest.approx(0.5, abs=1e-13)
    assert information == pytest.approx((1e308 * _MIN_FLOAT) ** 2, rel=3e-12, abs=0.0)


def test_recovery_preserves_unaffected_cells_and_item_alignment():
    slopes = np.array([1e308, 1e-308, 0.0, -1e308, 1.0])
    locations = np.array([0.0, -_MAX_FLOAT, _MAX_FLOAT, 0.0, 0.0])
    shapes = np.array([1e-308, 1.0, 2.0, 1e-308, 1.0])
    theta = np.array([-_MAX_FLOAT, -3.0, -1.0, 0.0, 1.0, 3.0, _MAX_FLOAT])
    model = FiveParameterLogistic(5).set_parameters(
        discrimination=slopes,
        difficulty=locations,
        guessing=np.full(5, 0.2),
        upper=np.full(5, 0.95),
        asymmetry=shapes,
    )
    references = np.array(
        [
            [
                _decimal_curve(point, a, b, 0.2, 0.95, e)
                for a, b, e in zip(slopes, locations, shapes, strict=True)
            ]
            for point in theta
        ]
    )
    indices = np.array([3, 0, 2, 4, 1, 3, 0])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probabilities = model.probability(theta)
        information = model.information(theta)
        pairs = model.probability_pairs(theta[:, None], indices)
    np.testing.assert_allclose(probabilities, references[:, :, 0], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(information, references[:, :, 1], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(
        pairs, references[np.arange(theta.size), indices, 0], rtol=3e-12, atol=0.0
    )


@pytest.mark.parametrize("method", ["probability", "information"])
@pytest.mark.parametrize("item_idx", [None, 2])
def test_probability_and_information_use_shared_bounded_evaluation(
    method, item_idx, monkeypatch
):
    model = FiveParameterLogistic(5).set_parameters(
        discrimination=np.linspace(0.6, 2.0, 5),
        difficulty=np.linspace(-1.0, 1.0, 5),
        asymmetry=np.linspace(0.5, 1.5, 5),
    )
    theta = np.linspace(-2.0, 2.0, 74)[::2, None]
    theta.flags.writeable = False
    expected = getattr(model, method)(theta, item_idx)
    original = dichotomous._five_pl_curve
    calls = []

    def tracked(points, slope, *parameters, **kwargs):
        assert len(points) * np.size(slope) <= 17
        calls.append(points.copy())
        return original(points, slope, *parameters, **kwargs)

    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 17)
    monkeypatch.setattr(dichotomous, "_five_pl_curve", tracked)
    actual = getattr(model, method)(theta, item_idx)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(np.concatenate(calls).ravel(), theta.ravel())
    assert len(calls) > 1


def test_nonfinite_input_abilities_are_not_replaced_by_finite_recovery():
    model = FiveParameterLogistic(1)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability([np.inf, -np.inf, np.nan], 0)
        information = model.information([np.inf, -np.inf, np.nan], 0)
    np.testing.assert_array_equal(probability[:2], [1.0, 0.2])
    np.testing.assert_array_equal(information[:2], [0.0, 0.0])
    assert np.isnan(probability[-1]) and np.isnan(information[-1])


def test_paired_batches_preserve_reordered_items_and_extreme_inputs(monkeypatch):
    model = FiveParameterLogistic(3).set_parameters(
        discrimination=np.array([1e308, 1e-308, 1.0]),
        difficulty=np.array([0.0, -_MAX_FLOAT, 0.0]),
        asymmetry=np.array([1e-308, 1.0, 0.5]),
    )
    theta = np.tile([-3.0, 8.0, _MAX_FLOAT, 9.0, 1.0, 10.0], 13)[::2, None]
    indices = np.tile([0, 2, 1, 2, 2, 0], 13)[::2]
    theta.flags.writeable = False
    indices.flags.writeable = False
    expected = np.array(
        [
            _decimal_curve(
                point,
                model.discrimination[index],
                model.difficulty[index],
                model.guessing[index],
                model.upper[index],
                model.asymmetry[index],
            )[0]
            for point, index in zip(theta[:, 0], indices, strict=True)
        ]
    )
    original = dichotomous._five_pl_curve
    calls = []

    def tracked(points, *parameters, item_indices, **kwargs):
        assert len(points) <= 7
        calls.append((points.copy(), item_indices.copy()))
        return original(points, *parameters, item_indices=item_indices, **kwargs)

    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 7)
    monkeypatch.setattr(dichotomous, "_five_pl_curve", tracked)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = model.probability_pairs(theta, indices)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
    np.testing.assert_array_equal(np.concatenate([c[0] for c in calls]), theta[:, 0])
    np.testing.assert_array_equal(np.concatenate([c[1] for c in calls]), indices)
    assert len(calls) > 1
