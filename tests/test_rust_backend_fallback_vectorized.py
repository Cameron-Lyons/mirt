"""Tests for vectorized Python fallback paths in ``mirt._rust_backend``."""

from __future__ import annotations

import numpy as np

import mirt
import mirt._rust_backend as rb
import mirt.backends.rust._helpers as helpers
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


def _slow_log_likelihoods_grm(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    thresholds: np.ndarray,
    n_categories: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    log_likes = np.zeros((n_persons, len(quad_points)))

    for q, theta in enumerate(quad_points):
        for i in range(n_persons):
            for j in range(n_items):
                response = responses[i, j]
                if response < 0:
                    continue
                n_cat = n_categories[j]
                if response == 0:
                    probability = 1.0 - sigmoid(
                        discrimination[j] * (theta - thresholds[j, 0])
                    )
                elif response == n_cat - 1:
                    probability = sigmoid(
                        discrimination[j] * (theta - thresholds[j, response - 1])
                    )
                else:
                    probability = sigmoid(
                        discrimination[j] * (theta - thresholds[j, response - 1])
                    ) - sigmoid(discrimination[j] * (theta - thresholds[j, response]))
                log_likes[i, q] += np.log(max(probability, PROB_EPSILON))

    return log_likes


def _slow_log_likelihoods_gpcm(
    responses: np.ndarray,
    quad_points: np.ndarray,
    discrimination: np.ndarray,
    steps: np.ndarray,
    n_categories: np.ndarray,
) -> np.ndarray:
    n_persons, n_items = responses.shape
    log_likes = np.zeros((n_persons, len(quad_points)))

    for q, theta in enumerate(quad_points):
        for i in range(n_persons):
            for j in range(n_items):
                response = responses[i, j]
                if response < 0:
                    continue
                n_cat = n_categories[j]
                numerators = np.zeros(n_cat)
                for category in range(1, n_cat):
                    numerators[category] = numerators[category - 1] + (
                        discrimination[j] * (theta - steps[j, category])
                    )
                max_numerator = np.max(numerators)
                log_denominator = max_numerator + np.log(
                    np.sum(np.exp(numerators - max_numerator))
                )
                probability = np.exp(numerators[response] - log_denominator)
                log_likes[i, q] += np.log(max(probability, PROB_EPSILON))

    return log_likes


def _with_numpy_backend_and_chunk(chunk: int):
    previous = mirt.get_backend()
    old_chunk = helpers._MAX_VECTOR_CHUNK_ENTRIES
    mirt.set_backend("numpy")
    helpers._MAX_VECTOR_CHUNK_ENTRIES = chunk
    return previous, old_chunk


def _restore(previous: str, old_chunk: int) -> None:
    helpers._MAX_VECTOR_CHUNK_ENTRIES = old_chunk
    mirt.set_backend(previous)


def test_2pl_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(20)
    try:
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
    finally:
        _restore(previous, old_chunk)


def test_3pl_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(18)
    try:
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
    finally:
        _restore(previous, old_chunk)


def test_mirt_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(24)
    try:
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
    finally:
        _restore(previous, old_chunk)


def test_eap_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(45)
    try:
        rng = np.random.default_rng(616)
        responses = rng.integers(0, 2, size=(24, 7), dtype=np.int32)
        responses[rng.random(size=responses.shape) < 0.2] = -1
        responses[0] = -1
        quad_points = np.linspace(-3.5, 3.5, 17)
        quad_weights = np.exp(-0.5 * quad_points**2)
        quad_weights[[0, -1]] = 0.0
        quad_weights /= quad_weights.sum()
        discrimination = rng.uniform(0.4, 2.2, size=7)
        difficulty = rng.normal(0, 1, size=7)

        log_likes = _slow_log_likelihoods_2pl(
            responses,
            quad_points,
            discrimination,
            difficulty,
        )
        log_posterior = log_likes + np.log(quad_weights + 1e-300)
        log_posterior -= np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        expected_theta = posterior @ quad_points
        expected_se = np.sqrt(
            np.sum(
                posterior * (quad_points[None, :] - expected_theta[:, None]) ** 2,
                axis=1,
            )
        )

        actual_theta, actual_se = rb.compute_eap_scores(
            responses,
            quad_points,
            quad_weights,
            discrimination,
            difficulty,
        )

        np.testing.assert_allclose(actual_theta, expected_theta, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_se, expected_se, rtol=1e-12, atol=1e-12)
    finally:
        _restore(previous, old_chunk)


def test_grm_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(30)
    try:
        rng = np.random.default_rng(717)
        n_categories = np.array([2, 3, 5, 4, 3, 5], dtype=np.int32)
        responses = np.column_stack(
            [rng.integers(0, n_cat, size=22, dtype=np.int32) for n_cat in n_categories]
        )
        responses[rng.random(size=responses.shape) < 0.18] = -1
        responses[:, 2] = -1
        quad_points = np.linspace(-4.0, 4.0, 19)
        discrimination = rng.uniform(0.5, 2.0, size=len(n_categories))
        thresholds = np.sort(
            rng.normal(size=(len(n_categories), max(n_categories) - 1)),
            axis=1,
        )

        expected = _slow_log_likelihoods_grm(
            responses,
            quad_points,
            discrimination,
            thresholds,
            n_categories,
        )
        actual = rb.compute_log_likelihoods_grm(
            responses,
            quad_points,
            discrimination,
            thresholds,
            n_categories,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        _restore(previous, old_chunk)


def test_gpcm_vectorized_fallback_matches_reference() -> None:
    previous, old_chunk = _with_numpy_backend_and_chunk(30)
    try:
        rng = np.random.default_rng(818)
        n_categories = np.array([2, 3, 5, 4, 3, 5], dtype=np.int32)
        responses = np.column_stack(
            [rng.integers(0, n_cat, size=22, dtype=np.int32) for n_cat in n_categories]
        )
        responses[rng.random(size=responses.shape) < 0.18] = -1
        responses[:, 4] = -1
        quad_points = np.linspace(-4.0, 4.0, 19)
        discrimination = rng.uniform(0.5, 2.0, size=len(n_categories))
        steps = rng.normal(size=(len(n_categories), max(n_categories)))

        expected = _slow_log_likelihoods_gpcm(
            responses,
            quad_points,
            discrimination,
            steps,
            n_categories,
        )
        actual = rb.compute_log_likelihoods_gpcm(
            responses,
            quad_points,
            discrimination,
            steps,
            n_categories,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        _restore(previous, old_chunk)
