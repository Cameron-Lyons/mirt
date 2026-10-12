"""The shared item-parameter boxes used by EM item optimizers."""

from types import SimpleNamespace

import numpy as np
import pytest

import mirt
from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step
from mirt.estimation.base import _parameter_bounds
from mirt.estimation.em import EMEstimator
from mirt.models.dichotomous import FourParameterLogistic
from mirt.models.polytomous import GradedResponseModel, NominalResponseModel


@pytest.mark.parametrize(
    ("model_name", "name", "expected"),
    [
        ("4PL", "discrimination", (0.1, 5.0)),
        ("4PL", "difficulty", (-6.0, 6.0)),
        ("4PL", "guessing", (0.0, 0.5)),
        ("4PL", "upper", (0.5, 1.0)),
        ("5PL", "asymmetry", (0.1, 5.0)),
        ("GRM", "thresholds", (-6.0, 6.0)),
        ("GPCM", "steps", (-6.0, 6.0)),
        ("NRM", "slopes", (-5.0, 5.0)),
        ("NRM", "intercepts", (-6.0, 6.0)),
        ("Bifactor", "general_slopes", (0.1, 5.0)),
        ("MIRT", "specific_discrimination", (0.1, 5.0)),
        ("Custom", "location", (-6.0, 6.0)),
    ],
)
def test_parameter_bounds_by_name(model_name, name, expected):
    assert _parameter_bounds(SimpleNamespace(model_name=model_name), name) == expected


@pytest.mark.parametrize(
    "model",
    [FourParameterLogistic(2), NominalResponseModel(2, [3, 4], n_factors=2)],
)
def test_item_optimizer_boxes_follow_parameter_bounds(model):
    _, bounds = EMEstimator()._get_item_params_and_bounds(model, 1)
    masks = model.free_parameter_masks
    expected = []
    for name in model.parameters:
        count = int(np.count_nonzero(masks[name][1]))
        expected += [_parameter_bounds(model, name)] * count
    assert bounds == expected


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
@pytest.mark.parametrize(
    ("name", "value", "accepted"),
    [
        ("discrimination", 5.0, True),
        ("discrimination", 5.5, False),
        ("thresholds", -6.5, False),
    ],
)
def test_native_polytomous_m_step_requires_starts_inside_the_boxes(
    name, value, accepted
):
    previous = mirt.get_backend()
    mirt.set_backend("rust")
    try:
        model = GradedResponseModel(2, n_categories=3)
        values = model.parameters[name]
        values.flat[0] = value
        model.set_parameters(**{name: values})
        responses = np.array([[0, 2], [1, 1], [2, 0], [1, 2]])
        points = np.linspace(-2.0, 2.0, 5)[:, None]
        posterior = np.full((4, 5), 0.2)
        assert (
            try_polytomous_m_step(
                model,
                responses,
                posterior,
                points,
                max_iter=20,
                ftol=1e-10,
                epsilon=1e-10,
                n_jobs=1,
            )
            is accepted
        )
    finally:
        mirt.set_backend(previous)


@pytest.mark.parametrize(
    ("model", "name", "expected"),
    [
        (mirt.TwoParameterLogistic(3), "discrimination", (0.1, 20.0)),
        (mirt.TwoParameterLogistic(3), "difficulty", (-30.0, 30.0)),
        # Unrotated exploratory loadings may be negative.
        (mirt.TwoParameterLogistic(3, n_factors=2), "discrimination", (-20.0, 20.0)),
        (mirt.OneParameterLogistic(3), "discrimination", (0.1, 20.0)),
        (mirt.ThreeParameterLogistic(3), "discrimination", (0.1, 5.0)),
        (mirt.ThreeParameterLogistic(3), "difficulty", (-30.0, 30.0)),
        (mirt.MultidimensionalModel(3, 2), "slopes", (-20.0, 20.0)),
        (mirt.MultidimensionalModel(3, 2), "intercepts", (-30.0, 30.0)),
        (GradedResponseModel(3, n_categories=3), "thresholds", (-6.0, 6.0)),
    ],
)
def test_logistic_boxes_are_wide(model, name, expected):
    assert _parameter_bounds(model, name) == expected


def _extreme_item_data():
    rng = np.random.default_rng(4)
    model = mirt.TwoParameterLogistic(10).set_parameters(
        discrimination=np.array([0.5, 0.6, 7.0, 1.0, 1.2, 1.4, 1.1, 0.9, 1.3, 1.0]),
        difficulty=np.array([-7.0, 7.5, 0.3, -1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.2]),
    )
    theta = rng.standard_normal((5000, 1))
    return (rng.random((5000, 10)) < model.probability(theta)).astype(int)


@pytest.mark.parametrize("quadrature", ["auto", "gauss-hermite"])
def test_easy_hard_and_steep_items_are_not_pinned_to_the_old_box(quadrature):
    # R mirt 1.48 (61 rectangular points) estimates b = -7.13 and 7.38 for
    # the low-slope easy and hard items and a = 6.55 for the steep one; the
    # former boxes pinned them at b = -6, 6 and a = 5. A Gauss-Hermite fit
    # leaves the native 2PL kernel, whose box is that old one.
    data = _extreme_item_data()
    result = mirt.fit_mirt(data, quadrature=quadrature, compute_standard_errors=False)
    discrimination = result.model.parameters["discrimination"]
    difficulty = result.model.parameters["difficulty"]
    tolerance = 0.05 if quadrature == "auto" else 0.3
    assert difficulty[0] == pytest.approx(-7.13, abs=tolerance)
    assert difficulty[1] == pytest.approx(7.38, abs=tolerance)
    assert discrimination[2] == pytest.approx(6.55, abs=2 * tolerance)


def test_reverse_keyed_items_warn_on_the_positive_slope_bound():
    rng = np.random.default_rng(3)
    model = mirt.TwoParameterLogistic(6).set_parameters(
        discrimination=np.r_[-1.2, np.full(5, 1.3)],
        difficulty=np.linspace(-1.0, 1.0, 6),
    )
    data = rng.random((1500, 6)) < model.probability(rng.standard_normal((1500, 1)))
    with pytest.warns(UserWarning, match="Item_1 ended on the lower bound 0.1"):
        mirt.fit_mirt(data.astype(int), compute_standard_errors=False)


def test_custom_items_use_their_declared_parameter_bounds():
    from mirt.models.custom import STANDARD_2PL, CustomItemModel

    rng = np.random.default_rng(3)
    true = mirt.TwoParameterLogistic(8).set_parameters(
        discrimination=np.linspace(0.7, 2.0, 8), difficulty=np.linspace(-1.5, 1.5, 8)
    )
    data = true.simulate(rng.normal(size=1500), seed=1)
    custom = CustomItemModel(8, STANDARD_2PL)
    assert _parameter_bounds(custom, "a") == (0.01, 5.0)

    builtin = EMEstimator(compute_standard_errors=False).fit(
        mirt.TwoParameterLogistic(8), data
    )
    fitted = EMEstimator(compute_standard_errors=False).fit(custom, data)
    assert fitted.log_likelihood == pytest.approx(builtin.log_likelihood, abs=1e-2)
    np.testing.assert_allclose(
        fitted.model.parameters["a"], builtin.model.discrimination, atol=5e-3
    )
    np.testing.assert_allclose(
        fitted.model.parameters["b"], builtin.model.difficulty, atol=5e-3
    )


def _family_boxes():
    from mirt.models.nonparametric import (
        MonotonicPolynomialModel,
        MonotonicSplineModel,
    )
    from mirt.models.testlet import TestletModel
    from mirt.models.unfolding import IdealPointModel

    return [
        (IdealPointModel(2), "peak_height", (1e-3, 1.0)),
        (MonotonicSplineModel(2), "lower", (0.0, 0.45)),
        (MonotonicSplineModel(2), "upper", (0.55, 1.0)),
        (MonotonicSplineModel(2), "log_weights", (-10.0, 10.0)),
        (MonotonicPolynomialModel(2), "scale", (0.05, 10.0)),
        (MonotonicPolynomialModel(2), "upper", (0.55, 1.0)),
        (MonotonicPolynomialModel(2), "log_coefficients", (-10.0, 10.0)),
        (TestletModel(2, [0, 0]), "testlet_variances", (0.0, 25.0)),
    ]


@pytest.mark.parametrize(("model", "name", "expected"), _family_boxes())
def test_families_declare_boxes_their_setters_accept(model, name, expected):
    assert _parameter_bounds(model, name) == expected
    # Every corner of the box is a valid value for every item.
    for value in expected:
        updated = model.parameters[name]
        updated[...] = value
        if name == "lower":
            model.set_parameters(lower=updated, upper=np.ones_like(updated))
        else:
            model.set_parameters(**{name: updated})


def test_em_recovers_ideal_point_items():
    from mirt.models.unfolding import IdealPointModel

    rng = np.random.default_rng(21)
    truth = IdealPointModel(8)
    location = np.linspace(-1.5, 1.5, 8)
    truth.set_parameters(
        discrimination=np.full(8, 0.6),
        location=location,
        peak_height=np.full(8, 0.85),
    )
    probability = truth.probability(rng.standard_normal(2000))
    responses = (rng.random(probability.shape) < probability).astype(int)

    result = EMEstimator(max_iter=300).fit(IdealPointModel(8), responses)

    fitted = result.model
    # The likelihood is unchanged by reversing the trait and every location.
    orientation = np.sign(np.corrcoef(fitted.location, location)[0, 1])
    assert result.converged
    np.testing.assert_allclose(orientation * fitted.location, location, atol=0.35)
    np.testing.assert_allclose(fitted.peak_height, 0.85, atol=0.1)
    assert np.all((fitted.peak_height > 0.0) & (fitted.peak_height <= 1.0))


@pytest.mark.parametrize("family", ["spline", "polynomial"])
def test_em_fits_monotone_curve_items_with_free_asymptotes(family):
    from mirt.models.nonparametric import (
        MonotonicPolynomialModel,
        MonotonicSplineModel,
    )

    rng = np.random.default_rng(11)
    slopes, locations = np.array([0.9, 1.4, 1.8, 1.1]), np.array([-1, 0, 0.5, 1])
    theta = rng.standard_normal(1000)
    logits = slopes * (theta[:, None] - locations)
    responses = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    model = (MonotonicSplineModel if family == "spline" else MonotonicPolynomialModel)(
        4
    )
    estimator = EMEstimator(max_iter=25)

    result = estimator.fit(model, responses)

    history = np.asarray(estimator.convergence_history)
    assert np.all(np.diff(history) > -1e-6)
    grid = np.linspace(-2.5, 2.5, 21)
    truth = 1 / (1 + np.exp(-slopes * (grid[:, None] - locations)))
    assert np.max(np.abs(result.model.probability(grid) - truth)) < 0.15
