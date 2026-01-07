"""Pytest configuration and shared fixtures.

Uses Rust backend for fast parallel data generation where available.
"""

import numpy as np
import pytest

from mirt._rust_backend import RUST_AVAILABLE

if RUST_AVAILABLE:
    from mirt._rust_backend import simulate_dichotomous


@pytest.fixture(scope="module")
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def cdm_responses(q_matrix):
    """Sample CDM response data - vectorized generation."""
    rng = np.random.default_rng(42)
    n_persons = 50
    n_items, n_attrs = q_matrix.shape

    alphas = rng.integers(0, 2, (n_persons, n_attrs))

    slip = np.full(n_items, 0.1)
    guess = np.full(n_items, 0.2)

    eta = np.prod(alphas[:, None, :] ** q_matrix[None, :, :], axis=2)
    p = (1 - slip[None, :]) ** eta * guess[None, :] ** (1 - eta)
    responses = (rng.random((n_persons, n_items)) < p).astype(int)

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


@pytest.fixture(scope="module")
def two_group_responses():
    """Two-group response data for DIF/DTF testing."""
    rng = np.random.default_rng(43)
    n_per_group = 30
    n_items = 8

    theta_ref = rng.standard_normal(n_per_group)
    disc_ref = np.ones(n_items)
    diff_ref = np.linspace(-2, 2, n_items)

    theta_foc = rng.standard_normal(n_per_group)
    diff_foc = diff_ref.copy()
    diff_foc[2] += 0.8
    diff_foc[3] += 1.0

    if RUST_AVAILABLE:
        responses_ref = simulate_dichotomous(theta_ref, disc_ref, diff_ref, seed=100)
        responses_foc = simulate_dichotomous(theta_foc, disc_ref, diff_foc, seed=101)
    else:
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


@pytest.fixture(scope="module")
def testlet_responses():
    """Response data with testlet structure."""
    rng = np.random.default_rng(44)
    n_persons = 50
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
        eff_theta = theta[:, None] + testlet_effects[:, t : t + 1]
        diff_slice = difficulty[start:end]
        prob = 1 / (1 + np.exp(-(eff_theta - diff_slice)))
        responses[:, start:end] = (
            rng.random((n_persons, items_per_testlet)) < prob
        ).astype(int)

    testlet_membership = np.repeat(np.arange(n_testlets), items_per_testlet)

    return {
        "responses": responses,
        "testlet_membership": testlet_membership,
        "theta": theta,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_testlets": n_testlets,
    }


@pytest.fixture(scope="module")
def dichotomous_responses():
    """Sample dichotomous response data - uses Rust for speed."""
    rng = np.random.default_rng(45)
    n_persons, n_items = 50, 8

    theta = rng.standard_normal(n_persons)
    discrimination = rng.lognormal(0, 0.3, n_items)
    difficulty = rng.normal(0, 1, n_items)

    if RUST_AVAILABLE:
        responses = simulate_dichotomous(theta, discrimination, difficulty, seed=200)
    else:
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


@pytest.fixture(scope="module")
def responses_with_missing(dichotomous_responses):
    """Dichotomous responses with missing data."""
    rng = np.random.default_rng(46)
    responses = dichotomous_responses["responses"].copy().astype(float)
    n_persons, n_items = responses.shape

    missing_mask = rng.random((n_persons, n_items)) < 0.10
    responses[missing_mask] = -1

    return {
        **dichotomous_responses,
        "responses": responses.astype(int),
        "missing_mask": missing_mask,
    }


@pytest.fixture(scope="module")
def polytomous_responses():
    """Sample polytomous (4-category) response data - vectorized."""
    rng = np.random.default_rng(47)
    n_persons, n_items, n_categories = 50, 6, 4

    theta = rng.standard_normal(n_persons)
    difficulty = rng.normal(0, 1, n_items)

    expected = (theta[:, None] - difficulty[None, :] + 2) / 3 * (n_categories - 1)
    expected = np.clip(expected, 0, n_categories - 1)
    noise = rng.normal(0, 0.5, (n_persons, n_items))
    responses = np.round(expected + noise).astype(int)
    responses = np.clip(responses, 0, n_categories - 1)

    return {
        "responses": responses,
        "theta": theta,
        "n_persons": n_persons,
        "n_items": n_items,
        "n_categories": n_categories,
    }


@pytest.fixture(scope="module")
def fitted_2pl_model(dichotomous_responses):
    """Pre-fitted 2PL model for tests that need a fitted model."""
    from mirt import fit_mirt

    result = fit_mirt(
        dichotomous_responses["responses"],
        model="2PL",
        max_iter=15,
        n_quadpts=11,
    )
    return result


@pytest.fixture(scope="module")
def fitted_2pl_model_small():
    """Pre-fitted 2PL model on small data for quick tests."""
    from mirt import fit_mirt

    rng = np.random.default_rng(99)
    n_persons, n_items = 30, 6

    theta = rng.standard_normal(n_persons)
    discrimination = rng.lognormal(0, 0.3, n_items)
    difficulty = rng.normal(0, 1, n_items)

    if RUST_AVAILABLE:
        responses = simulate_dichotomous(theta, discrimination, difficulty, seed=300)
    else:
        probs = 1 / (1 + np.exp(-discrimination * (theta[:, None] - difficulty)))
        responses = (rng.random((n_persons, n_items)) < probs).astype(int)

    result = fit_mirt(responses, model="2PL", max_iter=10, n_quadpts=9)
    return {"result": result, "responses": responses, "theta": theta}
