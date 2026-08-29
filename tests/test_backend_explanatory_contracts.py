"""Contracts for explanatory marginal-likelihood dispatch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.backends.rust.explanatory as explanatory_backend
from mirt._core import sigmoid
from mirt.backends.rust._helpers import RUST_AVAILABLE, mirt_rs
from mirt.constants import PROB_EPSILON
from mirt.models.explanatory import ExplanatoryIRT
from mirt.utils.numeric import standard_normal_quadrature


def _reference_marginal_log_likelihood(
    responses: np.ndarray,
    means: np.ndarray,
    residual_std: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    """Small scalar reference matching the public clipped-probability contract."""
    result = np.empty(responses.shape[0], dtype=np.float64)
    for person in range(responses.shape[0]):
        terms = np.empty(nodes.size, dtype=np.float64)
        for point, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
            theta = means[person] + residual_std * node
            conditional = 0.0
            for item, response in enumerate(responses[person]):
                if response < 0:
                    continue
                probability = float(
                    np.clip(
                        sigmoid(discrimination[item] * (theta - difficulty[item])),
                        PROB_EPSILON,
                        1.0 - PROB_EPSILON,
                    )
                )
                conditional += (
                    np.log(probability) if response == 1 else np.log1p(-probability)
                )
            terms[point] = np.log(weight) + conditional
        maximum = np.max(terms)
        result[person] = maximum + np.log(np.sum(np.exp(terms - maximum)))
    return result


@pytest.fixture
def likelihood_batch() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(521)
    n_persons, n_items = 47, 13
    responses = rng.integers(0, 2, size=(n_persons, n_items), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.17] = -7
    means = rng.normal(size=n_persons)
    nodes, weights = standard_normal_quadrature(17)
    discrimination = rng.uniform(0.35, 2.4, size=n_items)
    difficulty = rng.normal(size=n_items)
    return (
        responses,
        means,
        np.array(0.8),
        nodes,
        weights,
        discrimination,
        difficulty,
    )


def _call_batch(batch: tuple[np.ndarray, ...]) -> np.ndarray:
    responses, means, residual_std, nodes, weights, discrimination, difficulty = batch
    return explanatory_backend.compute_explanatory_marginal_log_likelihood(
        responses,
        means,
        float(residual_std),
        nodes,
        weights,
        discrimination,
        difficulty,
    )


def test_numpy_fallback_matches_scalar_reference(
    likelihood_batch: tuple[np.ndarray, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: False)
    responses, means, residual_std, nodes, weights, discrimination, difficulty = (
        likelihood_batch
    )

    actual = _call_batch(likelihood_batch)
    expected = _reference_marginal_log_likelihood(
        responses,
        means,
        float(residual_std),
        nodes,
        weights,
        discrimination,
        difficulty,
    )

    assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="native extension unavailable")
def test_native_matches_numpy_fallback(
    likelihood_batch: tuple[np.ndarray, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: False)
    expected = _call_batch(likelihood_batch)
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: True)

    actual = _call_batch(likelihood_batch)

    assert_allclose(actual, expected, rtol=2e-14, atol=3e-14)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="native extension unavailable")
def test_extreme_probabilities_preserve_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = (
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.array([1.0e6, -1.0e6]),
        np.array(1.0),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0e6, 1.0e6]),
        np.array([0.0, 0.0]),
    )
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: False)
    expected = _call_batch(batch)
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: True)

    actual = _call_batch(batch)

    assert np.all(np.isfinite(actual))
    assert_allclose(actual, expected, rtol=0.0, atol=2e-7)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="native extension unavailable")
def test_native_binding_rejects_invalid_shapes_and_responses() -> None:
    arguments = (
        np.zeros((2, 3), dtype=np.int32),
        np.zeros(2),
        1.0,
        np.array([0.0]),
        np.array([1.0]),
        np.ones(3),
        np.zeros(3),
    )
    with pytest.raises(ValueError, match="prior_means"):
        mirt_rs.explanatory_marginal_log_likelihood(
            arguments[0],
            np.zeros(1),
            *arguments[2:],
        )

    invalid_responses = arguments[0].copy()
    invalid_responses[0, 0] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        mirt_rs.explanatory_marginal_log_likelihood(
            invalid_responses,
            *arguments[1:],
        )


def test_all_missing_patterns_integrate_to_weight_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: False)
    nodes = np.array([-1.0, 0.0, 1.0])
    weights = np.array([0.2, 0.5, 0.3])

    actual = explanatory_backend.compute_explanatory_marginal_log_likelihood(
        np.full((4, 3), -9),
        np.linspace(-1.0, 1.0, 4),
        0.7,
        nodes,
        weights,
        np.ones(3),
        np.zeros(3),
    )

    assert_allclose(actual, np.log(np.sum(weights)), atol=2e-16)


def test_numpy_fallback_streams_quadrature_reduction(
    likelihood_batch: tuple[np.ndarray, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(explanatory_backend, "rust_enabled", lambda: False)
    calls: list[tuple[int, ...]] = []
    original = explanatory_backend.np.logaddexp

    def recording_logaddexp(
        left: np.ndarray,
        right: np.ndarray,
        *,
        out: np.ndarray,
    ) -> np.ndarray:
        calls.append(np.shape(right))
        return original(left, right, out=out)

    monkeypatch.setattr(explanatory_backend.np, "logaddexp", recording_logaddexp)

    result = _call_batch(likelihood_batch)

    assert result.shape == (likelihood_batch[0].shape[0],)
    assert calls == [(likelihood_batch[0].shape[0],)] * likelihood_batch[3].size


@pytest.mark.parametrize(
    ("replace", "message"),
    [
        (lambda values: values.__setitem__(0, np.zeros(3)), "responses"),
        (lambda values: values.__setitem__(0, np.full((2, 3), np.nan)), "finite"),
        (lambda values: values.__setitem__(0, np.full((2, 3), 2)), "only 0 and 1"),
        (lambda values: values.__setitem__(1, np.zeros(3)), "prior_means"),
        (lambda values: values.__setitem__(2, np.array(0.0)), "residual_std"),
        (lambda values: values.__setitem__(3, np.array([])), "quad_nodes"),
        (lambda values: values.__setitem__(4, np.ones(2)), "quad_weights"),
        (lambda values: values.__setitem__(4, np.array([0.0])), "positive"),
        (lambda values: values.__setitem__(5, np.ones(2)), "discrimination"),
        (lambda values: values.__setitem__(6, np.ones(2)), "difficulty"),
    ],
)
def test_input_validation(
    replace: Callable[[list[np.ndarray]], None],
    message: str,
) -> None:
    values = [
        np.zeros((2, 3), dtype=np.int32),
        np.zeros(2),
        np.array(1.0),
        np.array([0.0]),
        np.array([1.0]),
        np.ones(3),
        np.zeros(3),
    ]
    replace(values)

    with pytest.raises(ValueError, match=message):
        _call_batch(tuple(values))


def test_public_method_passes_latent_distribution_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    model = ExplanatoryIRT(3, item_features, 1)
    model.set_feature_weights(np.array([0.3, -0.2]))
    model.set_regression_weights(np.array([0.4, 0.7]))
    model.set_residual_variance(0.64)
    covariates = np.array([[-1.0], [1.5]])
    responses = np.array([[1, 0, -1], [0, 1, 1]])
    captured: dict[str, object] = {}

    def record(*arguments: object) -> np.ndarray:
        captured["arguments"] = arguments
        return np.array([-2.0, -3.0])

    monkeypatch.setattr(
        explanatory_backend,
        "compute_explanatory_marginal_log_likelihood",
        record,
    )

    result = model.marginal_log_likelihood_given_covariates(
        responses,
        covariates,
        n_quadpts=9,
    )

    assert_allclose(result, [-2.0, -3.0])
    arguments = captured["arguments"]
    assert isinstance(arguments, tuple)
    assert_allclose(arguments[0], responses)
    assert_allclose(arguments[1], model.latent_regression.predict_mean(covariates))
    assert arguments[2] == pytest.approx(0.8)
    assert_allclose(arguments[5], model.discrimination)
    assert_allclose(arguments[6], model.difficulty)
