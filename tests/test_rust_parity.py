"""Rust ↔ NumPy numerical parity tests for hot-path backend wrappers."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.backends import rust as rb

pytestmark = pytest.mark.skipif(
    not mirt.is_rust_available(),
    reason="Rust extension required for parity tests",
)


@pytest.fixture
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


def _sample_2pl(seed: int = 7):
    rng = np.random.default_rng(seed)
    responses = rng.integers(0, 2, size=(40, 8), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.1] = -1
    quad = np.linspace(-3.0, 3.0, 21)
    disc = rng.uniform(0.5, 2.0, size=8)
    diff = rng.normal(0.0, 1.0, size=8)
    return responses, quad, disc, diff


def _sample_polytomous(seed: int = 19):
    rng = np.random.default_rng(seed)
    n_categories = np.array([2, 3, 5, 4, 3, 5], dtype=np.int32)
    responses = np.column_stack(
        [rng.integers(0, n_cat, size=32, dtype=np.int32) for n_cat in n_categories]
    )
    responses[rng.random(responses.shape) < 0.15] = -1
    quad = np.linspace(-3.5, 3.5, 19)
    discrimination = rng.uniform(0.5, 2.0, size=len(n_categories))
    thresholds = np.sort(
        rng.normal(size=(len(n_categories), max(n_categories) - 1)),
        axis=1,
    )
    steps = rng.normal(size=(len(n_categories), max(n_categories)))
    return responses, quad, discrimination, thresholds, steps, n_categories


def test_likelihood_2pl_rust_matches_numpy(restore_backend) -> None:
    responses, quad, disc, diff = _sample_2pl()

    mirt.set_backend("auto")
    rust_ll = rb.compute_log_likelihoods_2pl(responses, quad, disc, diff)

    mirt.set_backend("numpy")
    numpy_ll = rb.compute_log_likelihoods_2pl(responses, quad, disc, diff)

    np.testing.assert_allclose(rust_ll, numpy_ll, rtol=1e-10, atol=1e-10)


def test_likelihood_3pl_rust_matches_numpy(restore_backend) -> None:
    rng = np.random.default_rng(11)
    responses = rng.integers(0, 2, size=(30, 6), dtype=np.int32)
    quad = np.linspace(-3.5, 3.5, 19)
    disc = rng.uniform(0.5, 2.0, size=6)
    diff = rng.normal(0.0, 1.0, size=6)
    guessing = rng.uniform(0.05, 0.3, size=6)

    mirt.set_backend("auto")
    rust_ll = rb.compute_log_likelihoods_3pl(responses, quad, disc, diff, guessing)
    mirt.set_backend("numpy")
    numpy_ll = rb.compute_log_likelihoods_3pl(responses, quad, disc, diff, guessing)

    np.testing.assert_allclose(rust_ll, numpy_ll, rtol=1e-10, atol=1e-10)


def test_polytomous_likelihoods_rust_match_numpy(restore_backend) -> None:
    responses, quad, discrimination, thresholds, steps, n_categories = (
        _sample_polytomous()
    )

    mirt.set_backend("auto")
    rust_grm = rb.compute_log_likelihoods_grm(
        responses,
        quad,
        discrimination,
        thresholds,
        n_categories,
    )
    rust_gpcm = rb.compute_log_likelihoods_gpcm(
        responses,
        quad,
        discrimination,
        steps,
        n_categories,
    )

    mirt.set_backend("numpy")
    numpy_grm = rb.compute_log_likelihoods_grm(
        responses,
        quad,
        discrimination,
        thresholds,
        n_categories,
    )
    numpy_gpcm = rb.compute_log_likelihoods_gpcm(
        responses,
        quad,
        discrimination,
        steps,
        n_categories,
    )

    np.testing.assert_allclose(rust_grm, numpy_grm, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rust_gpcm, numpy_gpcm, rtol=1e-10, atol=1e-10)


def test_estep_complete_rust_matches_numpy(restore_backend) -> None:
    responses, quad, disc, diff = _sample_2pl(21)
    weights = np.exp(-0.5 * quad**2)
    weights /= weights.sum()

    mirt.set_backend("auto")
    rust_post, rust_marg = rb.e_step_complete(responses, quad, weights, disc, diff)
    mirt.set_backend("numpy")
    numpy_post, numpy_marg = rb.e_step_complete(responses, quad, weights, disc, diff)

    np.testing.assert_allclose(rust_post, numpy_post, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rust_marg, numpy_marg, rtol=1e-9, atol=1e-9)


def test_expected_counts_rust_matches_numpy(restore_backend) -> None:
    rng = np.random.default_rng(5)
    responses = rng.integers(0, 2, size=40, dtype=np.int32)
    responses[rng.random(40) < 0.1] = -1
    post = rng.random((40, 15))
    post /= post.sum(axis=1, keepdims=True)

    mirt.set_backend("auto")
    rust_r, rust_n = rb.compute_expected_counts(responses, post)
    mirt.set_backend("numpy")
    numpy_r, numpy_n = rb.compute_expected_counts(responses, post)

    np.testing.assert_allclose(rust_r, numpy_r, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rust_n, numpy_n, rtol=1e-10, atol=1e-10)


def test_eap_scores_rust_matches_numpy(restore_backend) -> None:
    responses, quad, disc, diff = _sample_2pl(33)
    weights = np.exp(-0.5 * quad**2)
    weights /= weights.sum()

    mirt.set_backend("auto")
    rust_theta, rust_se = rb.compute_eap_scores(responses, quad, weights, disc, diff)
    mirt.set_backend("numpy")
    numpy_theta, numpy_se = rb.compute_eap_scores(responses, quad, weights, disc, diff)

    np.testing.assert_allclose(rust_theta, numpy_theta, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rust_se, numpy_se, rtol=1e-9, atol=1e-9)


def test_wle_scores_rust_matches_numpy_with_missing_data(restore_backend) -> None:
    responses, _, disc, diff = _sample_2pl(44)
    responses[0] = -1

    mirt.set_backend("auto")
    rust_theta, rust_se = rb.compute_wle_scores(
        responses,
        disc,
        diff,
        theta_min=-5.0,
        theta_max=4.5,
        tol=1e-8,
        n_jobs=3,
    )
    mirt.set_backend("numpy")
    numpy_theta, numpy_se = rb.compute_wle_scores(
        responses,
        disc,
        diff,
        theta_min=-5.0,
        theta_max=4.5,
        tol=1e-8,
        n_jobs=3,
    )

    np.testing.assert_allclose(rust_theta, numpy_theta, rtol=1e-8, atol=5e-8)
    np.testing.assert_allclose(rust_se, numpy_se, rtol=1e-8, atol=5e-8)


def test_q3_and_residuals_rust_match_numpy(restore_backend) -> None:
    rng = np.random.default_rng(99)
    responses = rng.integers(0, 2, size=(50, 10), dtype=np.int32)
    theta = rng.normal(size=50)
    disc = rng.uniform(0.6, 1.8, size=10)
    diff = rng.normal(0.0, 1.0, size=10)

    mirt.set_backend("auto")
    rust_q3 = rb.compute_q3_matrix(responses, theta, disc, diff)
    rust_res = rb.compute_standardized_residuals(responses, theta, disc, diff)

    mirt.set_backend("numpy")
    numpy_q3 = rb.compute_q3_matrix(responses, theta, disc, diff)
    numpy_res = rb.compute_standardized_residuals(responses, theta, disc, diff)

    np.testing.assert_allclose(rust_q3, numpy_q3, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rust_res, numpy_res, rtol=1e-9, atol=1e-9)


def test_set_backend_numpy_disables_rust_dispatch(restore_backend) -> None:
    mirt.set_backend("auto")
    assert mirt.should_use_rust(True) is True
    mirt.set_backend("numpy")
    assert mirt.should_use_rust(True) is False
    assert rb.rust_enabled() is False
