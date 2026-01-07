"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def q_matrix():
    """Sample Q-matrix for CDM tests (6 items, 3 attributes)."""
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )


@pytest.fixture
def cdm_responses(rng, q_matrix):
    """Sample CDM response data."""
    n_persons = 200
    n_items, n_attrs = q_matrix.shape

    alphas = rng.integers(0, 2, (n_persons, n_attrs))

    slip = np.full(n_items, 0.1)
    guess = np.full(n_items, 0.2)

    responses = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_persons):
        for j in range(n_items):
            eta = np.prod(alphas[i] ** q_matrix[j])
            p = (1 - slip[j]) ** eta * guess[j] ** (1 - eta)
            responses[i, j] = int(rng.random() < p)

    return {
        "responses": responses,
        "alphas": alphas,
        "q_matrix": q_matrix,
        "slip": slip,
        "guess": guess,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_attrs": n_attrs,
    }


@pytest.fixture
def two_group_responses(rng):
    """Two-group response data for DIF/DTF testing."""
    n_per_group = 150
    n_items = 10

    theta_ref = rng.standard_normal(n_per_group)
    disc_ref = np.ones(n_items)
    diff_ref = np.linspace(-2, 2, n_items)

    theta_foc = rng.standard_normal(n_per_group)
    diff_foc = diff_ref.copy()
    diff_foc[2] += 0.8
    diff_foc[3] += 1.0

    probs_ref = 1 / (1 + np.exp(-disc_ref * (theta_ref[:, None] - diff_ref)))
    probs_foc = 1 / (1 + np.exp(-disc_ref * (theta_foc[:, None] - diff_foc)))

    responses_ref = (rng.random((n_per_group, n_items)) < probs_ref).astype(int)
    responses_foc = (rng.random((n_per_group, n_items)) < probs_foc).astype(int)

    responses = np.vstack([responses_ref, responses_foc])
    groups = np.array([0] * n_per_group + [1] * n_per_group)

    return {
        "responses": responses,
        "groups": groups,
        "dif_items": [2, 3],
        "n_persons": 2 * n_per_group,
        "n_items": n_items,
    }


@pytest.fixture
def testlet_responses(rng):
    """Response data with testlet structure."""
    n_persons = 200
    n_testlets = 3
    items_per_testlet = 4
    n_items = n_testlets * items_per_testlet

    theta = rng.standard_normal(n_persons)
    testlet_effects = rng.normal(0, 0.5, (n_persons, n_testlets))

    difficulty = rng.normal(0, 1, n_items)
    responses = np.zeros((n_persons, n_items), dtype=int)

    for t in range(n_testlets):
        start = t * items_per_testlet
        end = start + items_per_testlet
        for j in range(start, end):
            eff_theta = theta + testlet_effects[:, t]
            prob = 1 / (1 + np.exp(-(eff_theta - difficulty[j])))
            responses[:, j] = (rng.random(n_persons) < prob).astype(int)

    testlet_membership = np.repeat(np.arange(n_testlets), items_per_testlet)

    return {
        "responses": responses,
        "testlet_membership": testlet_membership,
        "theta": theta,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_testlets": n_testlets,
    }


@pytest.fixture
def responses_with_missing(dichotomous_responses, rng):
    """Dichotomous responses with missing data."""
    responses = dichotomous_responses["responses"].copy().astype(float)
    n_persons, n_items = responses.shape

    missing_mask = rng.random((n_persons, n_items)) < 0.10
    responses[missing_mask] = -1

    return {
        **dichotomous_responses,
        "responses": responses.astype(int),
        "missing_mask": missing_mask,
    }


@pytest.fixture
def dichotomous_responses(rng):
    """Sample dichotomous response data."""
    n_persons, n_items = 200, 10

    theta = rng.standard_normal(n_persons)

    discrimination = rng.lognormal(0, 0.3, n_items)
    difficulty = rng.normal(0, 1, n_items)

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

    theta = rng.standard_normal(n_persons)

    responses = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        difficulty = rng.normal(0, 1)
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
