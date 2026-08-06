"""Regression tests for stable, vectorized polytomous simulation."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.backends.rust.simulation import simulate_gpcm, simulate_grm


@pytest.fixture
def numpy_backend():
    """Force NumPy wrappers and restore the previous backend afterward."""
    previous = mirt.get_backend()
    mirt.set_backend("numpy")
    try:
        yield
    finally:
        mirt.set_backend(previous)


def test_grm_simdata_returns_valid_reproducible_categories() -> None:
    first = mirt.simdata(
        model="GRM",
        n_persons=500,
        n_items=6,
        n_categories=5,
        seed=42,
    )
    second = mirt.simdata(
        model="GRM",
        n_persons=500,
        n_items=6,
        n_categories=5,
        seed=42,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (500, 6)
    assert np.all((first >= 0) & (first < 5))


def test_grm_simdata_orders_scores_by_ability() -> None:
    theta = np.repeat([-3.0, 3.0], 1_000)
    responses = mirt.simdata(
        model="GRM",
        theta=theta,
        n_items=8,
        n_categories=5,
        discrimination=np.ones(8),
        difficulty=np.zeros(8),
        seed=7,
    )

    assert responses[1_000:].mean() > responses[:1_000].mean() + 2.0


def test_gpcm_simdata_handles_extreme_abilities() -> None:
    responses = mirt.simdata(
        model="GPCM",
        theta=np.array([-1_000.0, 1_000.0]),
        n_items=2,
        n_categories=5,
        discrimination=np.array([5.0, 5.0]),
        thresholds=np.array([[-2.0, -1.0, 1.0, 2.0]] * 2),
        seed=3,
    )

    np.testing.assert_array_equal(responses[0], 0)
    np.testing.assert_array_equal(responses[1], 4)


def test_multidimensional_gpcm_orders_scores_by_ability() -> None:
    theta = np.vstack(
        [
            np.full((1_000, 2), -2.0),
            np.full((1_000, 2), 2.0),
        ]
    )
    responses = mirt.simdata(
        model="GPCM",
        theta=theta,
        n_items=6,
        n_categories=4,
        n_factors=2,
        discrimination=np.ones((6, 2)),
        thresholds=np.array([[-1.0, 0.0, 1.0]] * 6),
        seed=9,
    )

    assert responses[1_000:].mean() > responses[:1_000].mean() + 2.0


def test_numpy_grm_wrapper_returns_valid_categories(numpy_backend) -> None:
    responses = simulate_grm(
        theta=np.linspace(-3.0, 3.0, 200),
        discrimination=np.array([0.8, 1.2]),
        thresholds=np.array([[-1.0, 0.0, 1.0], [-1.5, 0.0, 1.5]]),
        seed=11,
    )

    assert responses.shape == (200, 2)
    assert np.all((responses >= 0) & (responses < 4))


def test_numpy_gpcm_wrapper_handles_extreme_abilities(numpy_backend) -> None:
    responses = simulate_gpcm(
        theta=np.array([-1_000.0, 1_000.0]),
        discrimination=np.array([5.0]),
        thresholds=np.array([[-2.0, -1.0, 1.0, 2.0]]),
        seed=13,
    )

    np.testing.assert_array_equal(responses[:, 0], [0, 4])
