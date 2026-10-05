"""Staggered default slopes of exploratory multidimensional models."""

import numpy as np
import pytest

import mirt
from mirt.estimation import base as estimation_base
from mirt.estimation.base import _initialize_free_parameters
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import NominalResponseModel


def _two_factor_responses(seed, n_persons=500, n_items=10):
    """Simple-structure two-factor 2PL responses with correlated factors."""
    rng = np.random.default_rng(seed)
    half = n_items // 2
    slopes = np.zeros((n_items, 2))
    slopes[:half, 0] = rng.uniform(0.9, 1.7, half)
    slopes[half:, 1] = rng.uniform(0.9, 1.7, n_items - half)
    intercepts = rng.normal(0.0, 1.0, n_items)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], n_persons)
    probability = 1.0 / (1.0 + np.exp(-(theta @ slopes.T + intercepts)))
    return (rng.random(probability.shape) < probability).astype(np.int_)


def _equal_slope_starts(monkeypatch):
    """Restore the former exchangeable starts, which stall at the saddle."""
    monkeypatch.setattr(
        estimation_base,
        "_stagger_factor_slopes",
        lambda model, name, values, free: values,
    )


def test_default_starts_stagger_exploratory_slopes():
    model = TwoParameterLogistic(n_items=5, n_factors=2)
    _initialize_free_parameters(model)
    expected = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 1.0], [1.0, 0.5]])
    np.testing.assert_array_equal(model.parameters["discrimination"], expected)

    nominal = NominalResponseModel(n_items=3, n_categories=3, n_factors=2)
    _initialize_free_parameters(nominal)
    slopes = nominal.parameters["slopes"]
    free = nominal.free_parameter_masks["slopes"]
    # Fixed reference-category slopes keep their values.
    np.testing.assert_array_equal(slopes[~free], 0.0)
    assert not np.allclose(slopes[..., 0], slopes[..., 1])


def test_confirmatory_and_unidimensional_starts_are_unchanged():
    model = TwoParameterLogistic(n_items=4, n_factors=2)
    masks = model.free_parameter_masks
    masks["discrimination"][:2, 1] = False
    masks["discrimination"][2:, 0] = False
    model.set_free_parameter_masks(masks)
    _initialize_free_parameters(model)
    np.testing.assert_array_equal(model.parameters["discrimination"], 1.0)

    unidimensional = TwoParameterLogistic(n_items=4)
    _initialize_free_parameters(unidimensional)
    np.testing.assert_array_equal(unidimensional.parameters["discrimination"], 1.0)


def test_exploratory_fit_leaves_the_equal_slope_saddle(monkeypatch):
    responses = _two_factor_responses(3)
    options = dict(model="2PL", n_factors=2, n_quadpts=11, verbose=False)
    fits = [
        mirt.fit_mirt(responses, accelerate=accelerate, **options)
        for accelerate in ("none", "squarem")
    ]
    _equal_slope_starts(monkeypatch)
    # Precise SQUAREM M-steps keep equal starting slopes equal.
    saddle = mirt.fit_mirt(responses, accelerate="squarem", **options)
    saddle_slopes = saddle.model.parameters["discrimination"]
    np.testing.assert_allclose(saddle_slopes[:, 0], saddle_slopes[:, 1], atol=1e-4)

    for fit in fits:
        slopes = fit.model.parameters["discrimination"]
        assert fit.converged
        assert np.max(np.abs(slopes[:, 0] - slopes[:, 1])) > 0.5
        assert fit.log_likelihood > saddle.log_likelihood + 10.0
    assert fits[1].log_likelihood == pytest.approx(fits[0].log_likelihood, abs=1e-2)
