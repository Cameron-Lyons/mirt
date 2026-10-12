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


def test_set_backend_numpy_disables_rust_dispatch(restore_backend) -> None:
    mirt.set_backend("auto")
    assert mirt.should_use_rust(True) is True
    mirt.set_backend("numpy")
    assert mirt.should_use_rust(True) is False
    assert rb.rust_enabled() is False
