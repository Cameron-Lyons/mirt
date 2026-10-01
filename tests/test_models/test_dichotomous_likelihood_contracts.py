"""Binary likelihoods preserve public curves, numeric responses, and shapes."""

import tracemalloc

import numpy as np
import pytest

from mirt import models
from mirt.constants import PROB_EPSILON
from mirt.estimation._mc_likelihood import sampled_log_likelihoods
from mirt.estimation.mcem import MCEMEstimator
from mirt.exceptions import MirtDataError
from mirt.models import base as base_module
from mirt.models.dichotomous import (
    ComplementaryLogLog,
    NegativeLogLog,
    UnipolarLogLogistic,
)


def _model(kind):
    if kind == "mirt":
        return models.MultidimensionalModel(3, n_factors=2)
    if kind == "bifactor":
        return models.BifactorModel(3, [0, 1, 0])
    return {
        "1pl": models.OneParameterLogistic,
        "2pl": models.TwoParameterLogistic,
        "3pl": models.ThreeParameterLogistic,
        "4pl": models.FourParameterLogistic,
        "5pl": models.FiveParameterLogistic,
        "ull": UnipolarLogLogistic,
        "cll": ComplementaryLogLog,
        "nll": NegativeLogLog,
    }[kind](3)


def _curve_likelihood(responses, probability):
    p = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)
    values = responses.astype(np.result_type(responses.dtype, p.dtype))
    with np.errstate(invalid="ignore"):
        terms = values * np.log(p) + (1.0 - values) * np.log(1.0 - p)
    return np.where(responses >= 0, terms, 0.0).sum(axis=-1)


@pytest.mark.parametrize(
    "kind", ["1pl", "2pl", "3pl", "4pl", "5pl", "mirt", "bifactor", "ull", "cll", "nll"]
)
@pytest.mark.parametrize("binding", ["class", "instance", "subclass"])
def test_likelihoods_use_original_points_for_public_theta_transforms(
    kind, binding, monkeypatch
):
    model = _model(kind)
    original = type(model)._ensure_theta_2d

    def transform(self, theta):
        return 1.2 * original(self, theta) + 0.4

    if binding == "class":
        monkeypatch.setattr(type(model), "_ensure_theta_2d", transform)
    elif binding == "instance":
        monkeypatch.setattr(model, "_ensure_theta_2d", transform.__get__(model))
    else:
        model.__class__ = type(
            "TransformedModel", (type(model),), {"_ensure_theta_2d": transform}
        )
    rng = np.random.default_rng(982)
    responses = rng.integers(-1, 2, (7, model.n_items))
    theta = rng.normal(size=(7, model.n_factors))
    expected = _curve_likelihood(responses, model.probability(theta))
    np.testing.assert_array_equal(model.log_likelihood(responses, theta), expected)
    expected_batch = _curve_likelihood(
        responses[:, None, :], model.probability(theta)[None, :, :]
    )
    np.testing.assert_allclose(
        model.log_likelihood_batch(responses, theta), expected_batch, atol=1e-13
    )
    samples = rng.normal(size=(7, 50, model.n_factors))
    expected_samples = _curve_likelihood(
        responses[:, None, :],
        model.probability(samples.reshape(-1, model.n_factors)).reshape(
            7, 50, model.n_items
        ),
    )
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_array_equal(actual, expected_samples)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
@pytest.mark.parametrize("layout", ["readonly", "reversed", "strided"])
def test_unsigned_general_responses_match_floating_values(dtype, layout):
    model = models.TwoParameterLogistic(3)
    responses = np.array([[2, 0, 255], [1, 2, 0], [3, 255, 1]], dtype=dtype)
    if layout == "reversed":
        responses = responses[::-1, ::-1]
    elif layout == "strided":
        storage = np.zeros((6, 6), dtype=dtype)
        storage[::2, ::2] = responses
        responses = storage[::2, ::2]
    saved = responses.copy()
    responses.setflags(write=False)
    theta = np.array([[-0.7], [0.2], [1.3]])
    floating = responses.astype(np.float64)
    np.testing.assert_array_equal(
        model.log_likelihood(responses, theta), model.log_likelihood(floating, theta)
    )
    np.testing.assert_array_equal(
        model.log_likelihood_batch(responses, theta),
        model.log_likelihood_batch(floating, theta),
    )
    samples = np.broadcast_to(theta[None, :1, :], (3, 50, 1))
    estimator = MCEMEstimator(n_samples=50)
    np.testing.assert_array_equal(
        estimator._sample_log_likelihoods(model, responses, samples),
        estimator._sample_log_likelihoods(model, floating, samples),
    )
    np.testing.assert_array_equal(responses, saved)


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.array(1), "must be 2D"),
        (np.ones(3), "must be 2D"),
        (np.ones((2, 1, 3)), "must be 2D"),
        (np.ones((2, 2)), "has 2 items, expected 3"),
        (np.full((2, 3), "1"), "numeric values"),
        (np.full((2, 3), 1, dtype=object), "numeric values"),
        (np.ones((2, 3), dtype=complex), "numeric values"),
    ],
)
@pytest.mark.parametrize("method", ["single", "batch", "sampled"])
def test_invalid_response_matrices_have_consistent_data_errors(
    responses, message, method
):
    model = models.TwoParameterLogistic(3)
    with pytest.raises(MirtDataError, match=message):
        if method == "sampled":
            sampled_log_likelihoods(model, responses, np.zeros((2, 5, 1)))
        else:
            likelihood = (
                model.log_likelihood
                if method == "single"
                else model.log_likelihood_batch
            )
            likelihood(responses, np.zeros((2, 1)))


@pytest.mark.parametrize(
    ("n_persons", "n_points"), [(1, 4), (4, 1), (0, 1), (1, 0), (0, 0)]
)
def test_single_likelihood_broadcast_and_empty_row_contract(n_persons, n_points):
    model = models.TwoParameterLogistic(3)
    responses = np.ones((n_persons, 3), dtype=int)
    theta = np.zeros((n_points, 1))
    expected = _curve_likelihood(responses, model.probability(theta))
    np.testing.assert_array_equal(model.log_likelihood(responses, theta), expected)
    assert model.log_likelihood_batch(responses, theta).shape == (n_persons, n_points)


def test_single_likelihood_rejects_incompatible_rows():
    with pytest.raises(MirtDataError, match="matching row counts or a single row"):
        models.TwoParameterLogistic(3).log_likelihood(np.ones((2, 3)), np.zeros((4, 1)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_borrowed_curve_broadcasts_without_modification(dtype, monkeypatch):
    model = models.TwoParameterLogistic(3)
    probability = np.array([[0.2, 0.8, 0.4]], dtype=dtype)
    original = probability.copy()
    probability.setflags(write=False)
    monkeypatch.setattr(model, "probability", lambda theta: probability)
    responses = np.array([[0.25, 1.5, -1], [1, np.nan, 0.75]], dtype=dtype)
    theta = np.zeros((2, 1))
    expected = _curve_likelihood(responses, probability)
    np.testing.assert_array_equal(model.log_likelihood(responses, theta), expected)
    actual = model.log_likelihood_batch(responses, np.zeros((5, 1)))
    assert actual.shape == (2, 5)
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected[:, None], (2, 5)), rtol=1e-6
    )
    np.testing.assert_array_equal(probability, original)


def test_missing_items_do_not_propagate_undefined_curves_to_other_rows(monkeypatch):
    model = models.TwoParameterLogistic(3)
    curves = np.array([[np.nan, 0.2, 0.7], [0.4, np.nan, 0.8], [0.5, 0.6, 0.3]])
    monkeypatch.setattr(
        model, "probability", lambda theta: curves[np.asarray(theta[:, 0], dtype=int)]
    )
    responses = np.array([[-1, 1, 0], [1, -1, 0], [-1, -1, -1], [np.nan, 0, -np.inf]])
    theta = np.arange(3)[:, None]
    with np.errstate(invalid="ignore"):
        actual = model.log_likelihood_batch(responses, theta)
        expected = np.column_stack(
            [model.log_likelihood(responses, point[None, :]) for point in theta]
        )
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual[0, 0]) and np.isfinite(actual[1, 1])
    np.testing.assert_array_equal(actual[2], np.zeros(3))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_boolean_responses_preserve_public_curve_arithmetic(dtype, monkeypatch):
    model = models.TwoParameterLogistic(3)
    probabilities = np.array([[0.2, 0.8, 0.4]], dtype=dtype)
    monkeypatch.setattr(model, "probability", lambda theta: probabilities)
    responses = np.array([[True, False, True], [False, True, False]])
    expected = (
        responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities)
    ).sum(axis=1)
    actual = model.log_likelihood(responses, np.zeros((2, 1)))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    samples = np.zeros((2, 50, 1))
    sampled = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_array_equal(
        sampled, np.broadcast_to(expected[:, None], sampled.shape)
    )


def test_large_responses_preserve_finite_per_item_cancellation(monkeypatch):
    model = models.TwoParameterLogistic(3)
    monkeypatch.setattr(
        model, "probability", lambda theta: np.full((len(theta), 3), 0.5)
    )
    responses = np.array([[1e308, 1e308, -1], [1e308, 0, 1], [np.nan, -np.inf, -1]])
    theta = np.zeros((4, 1))
    with np.errstate(over="raise", invalid="raise"):
        actual = model.log_likelihood_batch(responses, theta)
    with np.errstate(over="raise", invalid="ignore"):
        expected = model.log_likelihood(responses, theta[:1])
    np.testing.assert_array_equal(actual, np.broadcast_to(expected[:, None], (3, 4)))
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("exceptional", [False, True])
def test_batch_scratch_stays_bounded(exceptional, monkeypatch):
    model = models.TwoParameterLogistic(2)
    curves = np.column_stack(
        [np.full(301, np.nan if exceptional else 0.5), np.linspace(0.1, 0.9, 301)]
    )
    responses = np.column_stack([np.full(200, -1), np.arange(200) % 2])
    monkeypatch.setattr(model, "probability", lambda theta: curves)
    monkeypatch.setattr(base_module, "_DICHOTOMOUS_MAX_LIKELIHOOD_VALUES", 256)
    expected = np.log(
        np.where(
            responses[:, 1, None] == 1, curves[None, :, 1], 1.0 - curves[None, :, 1]
        )
    )
    tracemalloc.start()
    try:
        actual = model.log_likelihood_batch(responses, np.zeros((301, 1)))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert peak < actual.nbytes + 150_000
