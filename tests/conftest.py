"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def dichotomous_responses(rng):
    """Sample dichotomous response data."""
    n_persons, n_items = 200, 10

    # Generate true theta
    theta = rng.standard_normal(n_persons)

    # Generate true parameters
    discrimination = rng.lognormal(0, 0.3, n_items)
    difficulty = rng.normal(0, 1, n_items)

    # Generate responses
    probs = 1 / (1 + np.exp(-discrimination * (theta[:, None] - difficulty)))
    responses = (rng.random((n_persons, n_items)) < probs).astype(int)

    return {
        "responses": responses,
        "theta": theta,
        "discrimination": discrimination,
        "difficulty": difficulty,
        "n_persons": n_persons,
        "n_items": n_items,
    }


@pytest.fixture
def polytomous_responses(rng):
    """Sample polytomous (5-category) response data."""
    n_persons, n_items, n_categories = 200, 8, 5

    # Generate true theta
    theta = rng.standard_normal(n_persons)

    # Generate responses (simplified simulation)
    responses = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        difficulty = rng.normal(0, 1)
        # Map theta to category probabilities
        for p in range(n_persons):
            expected = (theta[p] - difficulty + 2) / 4 * (n_categories - 1)
            expected = np.clip(expected, 0, n_categories - 1)
            responses[p, i] = int(np.round(expected + rng.normal(0, 0.5)))
            responses[p, i] = np.clip(responses[p, i], 0, n_categories - 1)

    return {
        "responses": responses,
        "theta": theta,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
    }
