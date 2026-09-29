"""Tail accuracy and bounded evaluation for the mirrored log-log links."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt.models import dichotomous
from mirt.models.dichotomous import ComplementaryLogLog, NegativeLogLog

_MODELS = (ComplementaryLogLog, NegativeLogLog)
_MAX_FLOAT = np.finfo(float).max


def _decimal_curve(theta, slope, location, negative):
    with localcontext() as context:
        context.prec = 100
        t, a, b = (Decimal.from_float(float(v)) for v in (theta, slope, location))
        z = a * (t - b)
        if negative:
            z = -z
        if z > 10:
            return (0.0 if negative else 1.0), 0.0
        if z < -10_000:
            return (1.0 if negative else 0.0), 0.0
        power = z.exp()
        if z < -100:
            # Relative corrections are smaller than exp(-100), even when a
            # large discrimination rescales an otherwise unrepresentable tail.
            cll_probability = power
            information = a * a * power
            probability = 1 - power if negative else power
        else:
            failure = (-power).exp()
            cll_probability = 1 - failure
            information = a * a * power * power * failure / cll_probability
            probability = failure if negative else cll_probability
        return float(probability), float(information)


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize(
    "slope,logit",
    [
        (1.0, -1000.0),
        (1.0, -745.0),
        (1.0, -710.0),
        (1.0, -400.0),
        (1.0, -40.0),
        (1.0, 0.0),
        (1.0, 4.0),
        (1.0, 6.6),
        (1.0, 7.0),
        (1.0, 1000.0),
        (-2.0, -40.0),
        (1e308, -1000.0),
        (1e308, np.log(1000.0)),
        (1e170, 6.7),
        (1e-150, -40.0),
        (1e308, 0.0),
    ],
)
def test_information_preserves_representable_tails(model_class, slope, logit):
    negative = model_class is NegativeLogLog
    theta = (-logit if negative else logit) / slope
    model = model_class(1).set_parameters(discrimination=np.array([slope]))
    expected_p, expected_i = _decimal_curve(theta, slope, 0.0, negative)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for item_idx in (None, 0):
            np.testing.assert_allclose(
                model.probability([theta], item_idx), expected_p, rtol=3e-12, atol=0.0
            )
            np.testing.assert_allclose(
                model.information([theta], item_idx), expected_i, rtol=3e-12, atol=0.0
            )
        np.testing.assert_allclose(
            model.probability_pairs([[theta]], [0]), expected_p, rtol=3e-12, atol=0.0
        )


@pytest.mark.parametrize("model_class", _MODELS)
def test_overflowed_differences_and_products_keep_their_limits(model_class):
    slopes = np.array([1e-308, 0.0, -1e-308, 1e308])
    locations = np.array([-_MAX_FLOAT, -_MAX_FLOAT, _MAX_FLOAT, 0.0])
    theta = np.array([_MAX_FLOAT, -_MAX_FLOAT, -2.0, 0.0, 2.0])
    model = model_class(4).set_parameters(discrimination=slopes, difficulty=locations)
    theta.flags.writeable = False
    expected = np.array(
        [
            [
                _decimal_curve(t, a, b, model_class is NegativeLogLog)
                for a, b in zip(slopes, locations, strict=True)
            ]
            for t in theta
        ]
    )
    indices = np.array([0, 2, 3, 1, 0])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability(theta)
        information = model.information(theta)
        pairs = model.probability_pairs(theta[:, None], indices)
    np.testing.assert_allclose(probability, expected[:, :, 0], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(information, expected[:, :, 1], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(
        pairs, expected[np.arange(theta.size), indices, 0], rtol=3e-12, atol=0.0
    )
    np.testing.assert_array_equal(theta, [_MAX_FLOAT, -_MAX_FLOAT, -2.0, 0.0, 2.0])


def test_mirrored_information_and_probability_complements():
    parameters = {
        "discrimination": np.array([0.7, -2.0, 0.0]),
        "difficulty": np.array([1.0, -0.5, 2.0]),
    }
    cll = ComplementaryLogLog(3).set_parameters(**parameters)
    nll = NegativeLogLog(3).set_parameters(
        **{**parameters, "difficulty": -parameters["difficulty"]}
    )
    theta = np.array([-1000.0, -40.0, -4.0, 0.0, 4.0, 40.0, 1000.0])
    np.testing.assert_allclose(cll.probability(theta) + nll.probability(-theta), 1.0)
    np.testing.assert_allclose(
        cll.information(theta), nll.information(-theta), atol=0.0
    )


@pytest.mark.parametrize("model_class", _MODELS)
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
def test_bounded_batches_keep_readonly_strided_inputs_aligned(
    model_class, method, item_idx, monkeypatch
):
    model = model_class(3).set_parameters(
        discrimination=np.array([0.7, -2.0, 1.3]),
        difficulty=np.array([1.0, -0.5, 2.0]),
    )
    theta = np.linspace(-40.0, 40.0, 78)[::2, None]
    indices = np.tile([2, 0, 2, 1, 0, 1], 13)[::2]
    theta.flags.writeable = False
    indices.flags.writeable = False
    expected = getattr(model, method)(
        theta, indices if method == "probability_pairs" else item_idx
    )
    original = dichotomous._double_exponential_curve
    calls = []

    def tracked(points, slope, *args, item_indices=None, **kwargs):
        size = points.size if item_indices is not None else points.size * np.size(slope)
        assert size <= 7
        calls.append(
            (points.copy(), None if item_indices is None else item_indices.copy())
        )
        return original(points, slope, *args, item_indices=item_indices, **kwargs)

    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 7)
    monkeypatch.setattr(dichotomous, "_double_exponential_curve", tracked)
    actual = getattr(model, method)(
        theta, indices if method == "probability_pairs" else item_idx
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(
        np.concatenate([c[0] for c in calls]).ravel(), theta.ravel()
    )
    if method == "probability_pairs":
        np.testing.assert_array_equal(np.concatenate([c[1] for c in calls]), indices)
    assert len(calls) > 1


@pytest.mark.parametrize("model_class", _MODELS)
def test_nonfinite_abilities_preserve_limits_and_propagate_nan(model_class):
    model = model_class(1)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probabilities = model.probability([-np.inf, np.inf, np.nan], 0)
        information = model.information([-np.inf, np.inf, np.nan], 0)
    np.testing.assert_array_equal(probabilities[:2], [0.0, 1.0])
    np.testing.assert_array_equal(information[:2], [0.0, 0.0])
    assert np.isnan(probabilities[-1]) and np.isnan(information[-1])
