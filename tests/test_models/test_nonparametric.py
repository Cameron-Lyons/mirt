"""Smoke tests for nonparametric IRT models."""

import numpy as np

from mirt.models.nonparametric import (
    KernelSmoothingModel,
    MonotonicPolynomialModel,
    MonotonicSplineModel,
)


def test_monotonic_spline_probability():
    model = MonotonicSplineModel(n_items=2, n_knots=2, degree=2)
    theta = np.linspace(-2, 2, 5)
    probs = model.probability(theta)

    assert probs.shape == (5, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.all((probs > 0) & (probs < 1))


def test_monotonic_polynomial_probability():
    model = MonotonicPolynomialModel(n_items=3, degree=3)
    theta = np.linspace(-2, 2, 11)
    probs = model.probability(theta)

    assert probs.shape == (11, 3)
    assert np.all((probs > 0) & (probs < 1))


def test_kernel_smoothing_probability():
    model = KernelSmoothingModel(n_items=3, bandwidth=0.5)
    rng = np.random.default_rng(0)
    theta_train = rng.standard_normal(40)
    responses = (rng.random((40, 3)) < 0.5).astype(int)
    model.calibrate(responses, theta_train)

    theta = np.linspace(-2, 2, 11)
    probs = model.probability(theta)

    assert probs.shape == (11, 3)
    assert np.all((probs > 0) & (probs < 1))
