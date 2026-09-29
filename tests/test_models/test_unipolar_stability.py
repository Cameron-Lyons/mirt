"""Independent tail, peak, and batching contracts for the unipolar curve."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt.models import dichotomous
from mirt.models.dichotomous import UnipolarLogLogistic

_MIN_FLOAT = np.nextafter(0.0, 1.0)
_MAX_FLOAT = np.finfo(float).max


def _decimal_curve(theta, slope, location=0.0):
    with localcontext() as context:
        context.prec = 100
        t, a, b = (Decimal.from_float(float(v)) for v in (theta, slope, location))
        z = abs(a * (t - b))
        if z > 10_000:
            return 0.0, 0.0
        if z < Decimal("1e-30"):
            # The relative corrections are O(z²), far below float precision.
            return 0.25, float(a * a * z * z / 12)
        tail = (-z).exp()
        if z > 100:
            return float(tail), float(a * a * tail)
        probability = tail / (1 + tail) ** 2
        attenuation = (1 - tail) / (1 + tail)
        information = a * a * probability * attenuation**2 / (1 - probability)
        return float(probability), float(information)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize(
    "slope,theta",
    [
        (1.0, 0.0),
        (1.0, 1e-8),
        (1.0, 1e-16),
        (1.0, 1e-160),
        (1.0, 40.0),
        (1.0, 400.0),
        (1.0, 710.0),
        (1.0, 745.0),
        (1.0, 1000.0),
        (1e100, 1e-300),
        (1e160, _MIN_FLOAT),
        (1e308, 1000.0 / 1e308),
        (1e200, 400.0 / 1e200),
        (1e308, 1e-308),
        (_MIN_FLOAT, _MIN_FLOAT),
    ],
)
def test_tails_and_peak_preserve_representable_values(slope, theta, direction):
    theta *= direction
    model = UnipolarLogLogistic(1).set_parameters(discrimination=np.array([slope]))
    probability, information = _decimal_curve(theta, slope)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for item_idx in (None, 0):
            np.testing.assert_allclose(
                model.probability([theta], item_idx), probability, rtol=3e-12, atol=0.0
            )
            np.testing.assert_allclose(
                model.information([theta], item_idx), information, rtol=3e-12, atol=0.0
            )
        np.testing.assert_allclose(
            model.probability_pairs([[theta]], [0]), probability, rtol=3e-12, atol=0.0
        )


def test_overflow_recovery_preserves_mixed_items_and_pair_alignment():
    slopes = np.array([1e-308, 1e308, 1.0])
    locations = np.array([-_MAX_FLOAT, 0.0, _MAX_FLOAT])
    theta = np.array([-_MAX_FLOAT, -2.0, 0.0, 2.0, _MAX_FLOAT])
    model = UnipolarLogLogistic(3).set_parameters(
        discrimination=slopes, difficulty=locations
    )
    expected = np.array(
        [
            [_decimal_curve(t, a, b) for a, b in zip(slopes, locations, strict=True)]
            for t in theta
        ]
    )
    indices = np.array([2, 0, 1, 2, 0])
    theta.flags.writeable = False
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability(theta)
        information = model.information(theta)
        pairs = model.probability_pairs(theta[:, None], indices)
    np.testing.assert_allclose(probability, expected[:, :, 0], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(information, expected[:, :, 1], rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(
        pairs, expected[np.arange(theta.size), indices, 0], rtol=3e-12, atol=0.0
    )


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
def test_batches_preserve_readonly_strided_abilities_and_item_order(
    method, item_idx, monkeypatch
):
    slopes = np.array([0.7, 1.3, 1e160])
    locations = np.array([-0.5, 1.0, 0.0])
    model = UnipolarLogLogistic(3).set_parameters(
        discrimination=slopes, difficulty=locations
    )
    storage = np.empty(78)
    storage[::2] = np.resize([-40.0, -2.0, 0.0, _MIN_FLOAT, 2.0, 40.0, _MAX_FLOAT], 39)
    storage[1::2] = np.nan
    theta = storage[::2, None]
    indices = np.tile([2, 0, 2, 1, 0, 1], 13)[::2]
    original_theta = theta.copy()
    original_indices = indices.copy()
    theta.flags.writeable = False
    indices.flags.writeable = False
    expected = np.array(
        [
            [_decimal_curve(t, a, b) for a, b in zip(slopes, locations, strict=True)]
            for t in theta[:, 0]
        ]
    )
    if method == "probability_pairs":
        expected = expected[np.arange(theta.shape[0]), indices, 0]
    else:
        expected = expected[:, :, int(method == "information")]
        if item_idx is not None:
            expected = expected[:, item_idx]
    original = dichotomous._unipolar_curve
    sizes = []

    def tracked(points, slope, *args, item_indices=None, **kwargs):
        size = points.size if item_indices is not None else points.size * np.size(slope)
        assert size <= 7
        sizes.append(size)
        return original(points, slope, *args, item_indices=item_indices, **kwargs)

    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 7)
    monkeypatch.setattr(dichotomous, "_unipolar_curve", tracked)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = getattr(model, method)(
            theta, indices if method == "probability_pairs" else item_idx
        )
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(indices, original_indices)
    assert sum(sizes) == expected.size and len(sizes) > 1


def test_nonfinite_abilities_retain_tail_limits_and_nan():
    model = UnipolarLogLogistic(1)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        probability = model.probability([-np.inf, np.inf, np.nan], 0)
        information = model.information([-np.inf, np.inf, np.nan], 0)
    np.testing.assert_array_equal(probability[:2], [0.0, 0.0])
    np.testing.assert_array_equal(information[:2], [0.0, 0.0])
    assert np.isnan(probability[-1]) and np.isnan(information[-1])


def test_probability_preserves_symmetry_and_the_exact_peak_bound():
    theta = np.concatenate(([0.0], np.geomspace(1e-20, 1e-2, 1024), [40.0, 710.0]))
    model = UnipolarLogLogistic(1)
    positive = model.probability(theta, 0)
    negative = model.probability(-theta, 0)
    np.testing.assert_array_equal(positive, negative)
    assert np.all((positive >= 0.0) & (positive <= 0.25))
    assert positive[0] == 0.25
