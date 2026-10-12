"""Lucke's unipolar log-logistic curve: R parity, tails, and batching contracts."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from mirt.estimation.em import EMEstimator
from mirt.models import dichotomous
from mirt.models.dichotomous import TwoParameterLogistic, UnipolarLogLogistic

_MIN_FLOAT = np.nextafter(0.0, 1.0)
_MAX_FLOAT = np.finfo(float).max

# R mirt 1.48: an itemtype = "ULL" item with eta1 = 1.39342 and
# log_lambda1 = -0.3722916, evaluated with probtrace() and iteminfo() at the
# unipolar trait values below. iteminfo() is information about tau itself.
_R_ETA = 1.39342
_R_LOG_LAMBDA = -0.3722916
_R_TAU = np.array([0.01, 0.5, 1.0, 2.0, 5.0, 20.0])
_R_PROBABILITY = np.array(
    [
        1.12457167962e-03,
        2.07816637264e-01,
        4.07987406489e-01,
        6.44180077746e-01,
        8.66496615768e-01,
        9.78161955905e-01,
    ]
)
_R_TRAIT_INFORMATION = np.array(
    [
        1.27858646033e00,
        4.68966458907e-01,
        1.11260661604e-01,
        8.98427872003e-03,
        1.03688023201e-04,
    ]
)


def _decimal_curve(theta, slope, location=0.0):
    """Return the logistic probability and Fisher information to 100 digits."""
    with localcontext() as context:
        context.prec = 100
        t, a, b = (Decimal.from_float(float(v)) for v in (theta, slope, location))
        z = a * (t - b)
        if abs(z) > 10_000:
            return (1.0 if z > 0 else 0.0), 0.0
        # Both values use the far tail, so that neither cancels to zero.
        tail = (-abs(z)).exp()
        probability = 1 / (1 + tail) if z >= 0 else tail / (1 + tail)
        information = a * a * tail / (1 + tail) ** 2
        return float(probability), float(information)


def _r_item() -> UnipolarLogLogistic:
    return UnipolarLogLogistic(1).set_parameters(
        discrimination=[_R_ETA], difficulty=[-_R_LOG_LAMBDA / _R_ETA]
    )


def test_probability_matches_r_mirt_on_the_unipolar_trait():
    model = _r_item()

    probability = model.probability(np.log(_R_TAU), 0)

    np.testing.assert_allclose(probability, _R_PROBABILITY, rtol=1e-10, atol=0.0)
    lam = np.exp(_R_LOG_LAMBDA)
    lucke = lam * _R_TAU**_R_ETA / (1.0 + lam * _R_TAU**_R_ETA)
    np.testing.assert_allclose(probability, lucke, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(model.eta, [_R_ETA], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model.log_lambda, [_R_LOG_LAMBDA], rtol=1e-15)


def test_information_converts_to_r_mirt_trait_information():
    model = _r_item()
    tau = _R_TAU[1:]

    information = model.information(np.log(tau), 0)

    probability = model.probability(np.log(tau), 0)
    np.testing.assert_allclose(
        information, _R_ETA**2 * probability * (1.0 - probability), rtol=1e-14
    )
    np.testing.assert_allclose(
        information / tau**2, _R_TRAIT_INFORMATION, rtol=1e-8, atol=0.0
    )


def test_curve_is_monotone_and_not_capped_at_a_quarter():
    """The former logistic-density curve never exceeded 0.25."""
    model = UnipolarLogLogistic(2).set_parameters(
        discrimination=np.array([0.8, 1.5]), difficulty=np.array([-0.5, 1.0])
    )
    theta = np.linspace(-8.0, 8.0, 801)

    probability = model.probability(theta)

    assert np.all(np.diff(probability, axis=0) > 0.0)
    assert probability.max() > 0.99
    np.testing.assert_allclose(
        np.diag(model.probability(model.difficulty)), [0.5, 0.5], rtol=1e-15
    )


def test_marginal_fit_on_the_log_trait_matches_the_logistic_likelihood():
    """A standard normal log trait is R mirt's log-normal ULL density."""
    rng = np.random.default_rng(3)
    eta = np.array([0.8, 1.2, 1.6, 2.0, 1.1])
    log_lambda = np.array([-1.0, -0.3, 0.2, 0.6, 1.4])
    generator = UnipolarLogLogistic(5).set_parameters(
        discrimination=eta, difficulty=-log_lambda / eta
    )
    responses = generator.simulate(rng.normal(size=600), seed=4)
    # Items endorsed by more than a quarter of respondents are now fittable.
    assert responses.mean(axis=0).max() > 0.6

    def fit(model):
        estimator = EMEstimator(tol=1e-8, max_iter=2000, compute_standard_errors=False)
        return estimator.fit(model, responses)

    ull, logistic = fit(UnipolarLogLogistic(5)), fit(TwoParameterLogistic(5))

    assert ull.log_likelihood == pytest.approx(logistic.log_likelihood, abs=1e-5)
    np.testing.assert_allclose(
        ull.model.eta, logistic.model.discrimination, rtol=0.0, atol=1e-3
    )
    assert ull.n_parameters == 10


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize(
    "slope,theta",
    [
        (1.0, 0.0),
        (1.0, 1e-8),
        (1.0, 1e-160),
        (1.0, 40.0),
        (1.0, 400.0),
        (1.0, 710.0),
        (1.0, 745.0),
        (1e100, 1e-300),
        (1e160, _MIN_FLOAT),
        (1e308, 1000.0 / 1e308),
        (1e200, 400.0 / 1e200),
        (1e308, 1e-308),
        (_MIN_FLOAT, _MIN_FLOAT),
    ],
)
def test_tails_preserve_representable_values(slope, theta, direction):
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
    np.testing.assert_array_equal(probability[:2], [0.0, 1.0])
    np.testing.assert_array_equal(information[:2], [0.0, 0.0])
    assert np.isnan(probability[-1]) and np.isnan(information[-1])
