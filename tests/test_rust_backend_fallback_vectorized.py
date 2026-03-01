"""Tests for vectorized Python fallback paths in ``mirt._rust_backend``."""

from __future__ import annotations

import numpy as np

import mirt._rust_backend as rb
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON


def _slow_log_likelihoods_2pl(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        z = discrimination * (theta - difficulty)
        probs = sigmoid(z)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                if responses[i, j] >= 0:
                    if responses[i, j] == 1:
                        ll += np.log(probs[j])
                    else:
                        ll += np.log(1 - probs[j])
            log_likes[i, q] = ll

    return log_likes


def _slow_log_likelihoods_3pl(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    guessing: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        z = discrimination * (theta - difficulty)
        p_star = sigmoid(z)
        probs = guessing + (1 - guessing) * p_star
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                if responses[i, j] >= 0:
                    if responses[i, j] == 1:
                        ll += np.log(probs[j])
                    else:
                        ll += np.log(1 - probs[j])
            log_likes[i, q] = ll

    return log_likes


def _slow_log_likelihoods_mirt(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> np.ndarray:
    n_persons = responses.shape[0]
    n_quad = quad_points.shape[0]
    n_items = responses.shape[1]

    disc_sums = discrimination.sum(axis=1)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta_q = quad_points[q]
        z = np.dot(discrimination, theta_q) - disc_sums * difficulty

        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                if responses[i, j] >= 0:
                    p = sigmoid(z[j])
                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
                    if responses[i, j] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
            log_likes[i, q] = ll

    return log_likes


def test_2pl_vectorized_fallback_matches_reference(monkeypatch) -> None:
    monkeypatch.setattr(rb, "RUST_AVAILABLE", False)
    monkeypatch.setattr(rb, "_MAX_VECTOR_CHUNK_ENTRIES", 20)

    rng = np.random.default_rng(123)
    responses = rng.integers(0, 3, size=(25, 7), dtype=np.int32)
    responses[rng.random(size=responses.shape) < 0.15] = -1
    quad_points = np.linspace(-3.5, 3.5, 17)
    discrimination = rng.uniform(0.4, 2.2, size=7)
    difficulty = rng.normal(0, 1, size=7)

    expected = _slow_log_likelihoods_2pl(
        responses, quad_points, discrimination, difficulty
    )
    actual = rb.compute_log_likelihoods_2pl(
        responses, quad_points, discrimination, difficulty
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_3pl_vectorized_fallback_matches_reference(monkeypatch) -> None:
    monkeypatch.setattr(rb, "RUST_AVAILABLE", False)
    monkeypatch.setattr(rb, "_MAX_VECTOR_CHUNK_ENTRIES", 18)

    rng = np.random.default_rng(321)
    responses = rng.integers(0, 2, size=(20, 6), dtype=np.int32)
    responses[rng.random(size=responses.shape) < 0.2] = -1
    quad_points = np.linspace(-4.0, 4.0, 19)
    discrimination = rng.uniform(0.5, 2.0, size=6)
    difficulty = rng.normal(0, 1, size=6)
    guessing = rng.uniform(0.05, 0.35, size=6)

    expected = _slow_log_likelihoods_3pl(
        responses, quad_points, discrimination, difficulty, guessing
    )
    actual = rb.compute_log_likelihoods_3pl(
        responses, quad_points, discrimination, difficulty, guessing
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_mirt_vectorized_fallback_matches_reference(monkeypatch) -> None:
    monkeypatch.setattr(rb, "RUST_AVAILABLE", False)
    monkeypatch.setattr(rb, "_MAX_VECTOR_CHUNK_ENTRIES", 24)

    rng = np.random.default_rng(456)
    responses = rng.integers(0, 2, size=(18, 5), dtype=np.int32)
    responses[rng.random(size=responses.shape) < 0.1] = -1
    quad_points = rng.normal(size=(23, 2))
    discrimination = rng.uniform(0.2, 1.6, size=(5, 2))
    difficulty = rng.normal(0, 1, size=5)

    expected = _slow_log_likelihoods_mirt(
        responses, quad_points, discrimination, difficulty
    )
    actual = rb.compute_log_likelihoods_mirt(
        responses, quad_points, discrimination, difficulty
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
