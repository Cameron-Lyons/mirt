"""Regression coverage for the validated 2PL M-step kernel."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
from mirt.backends.rust.mstep import m_step_dichotomous_parallel


@pytest.fixture
def restore_backend() -> Iterator[None]:
    previous = mirt.get_backend()
    try:
        yield
    finally:
        mirt.set_backend(previous)


def _one_item_inputs() -> tuple[np.ndarray, ...]:
    responses = np.array([[0], [1], [-1], [1]], dtype=np.int64)
    posterior = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
            [0.4, 0.4, 0.2],
        ]
    )
    points = np.array([-1.5, 0.0, 1.5])
    return responses, posterior, points, np.array([0.8]), np.array([-0.2])


def _expected_counts(
    responses: np.ndarray,
    posterior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (responses == 1).T @ posterior, (responses >= 0).T @ posterior


def _item_log_likelihoods(
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    correct: np.ndarray,
    total: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    logits = discrimination[:, None] * (points[None, :] - difficulty[:, None])
    return np.sum(
        correct * -np.logaddexp(0.0, -logits)
        + (total - correct) * -np.logaddexp(0.0, logits),
        axis=1,
    )


def test_one_iteration_matches_independent_canonical_newton_step(
    restore_backend: None,
) -> None:
    responses, posterior, points, discrimination, difficulty = _one_item_inputs()
    regularization = 0.01
    damping = 0.5
    correct, total = _expected_counts(responses, posterior)
    intercept = -discrimination[0] * difficulty[0]
    probability = 1.0 / (1.0 + np.exp(-(discrimination[0] * points + intercept)))
    residual = correct[0] - total[0] * probability
    information = total[0] * probability * (1.0 - probability)
    gradient = np.array([residual @ points, np.sum(residual)])
    hessian = np.array(
        [
            [
                -(information @ (points**2)) - regularization,
                -(information @ points),
            ],
            [
                -(information @ points),
                -np.sum(information) - regularization,
            ],
        ]
    )
    delta = np.linalg.solve(hessian, gradient)
    expected_discrimination = discrimination[0] - damping * delta[0]
    expected_intercept = intercept - damping * delta[1]
    expected_difficulty = -expected_intercept / expected_discrimination

    mirt.set_backend("numpy")
    actual_discrimination, actual_difficulty = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        discrimination,
        difficulty,
        max_iter=1,
        tol=1e-12,
        damping=damping,
        regularization=regularization,
    )

    assert_allclose(actual_discrimination, [expected_discrimination], atol=1e-14)
    assert_allclose(actual_difficulty, [expected_difficulty], atol=1e-14)


def test_every_accepted_update_is_likelihood_monotone(
    restore_backend: None,
) -> None:
    rng = np.random.default_rng(918)
    responses = rng.integers(0, 2, size=(120, 48), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.17] = -1
    posterior = rng.random((120, 21))
    posterior /= posterior.sum(axis=1, keepdims=True)
    points = np.linspace(-4.0, 4.0, 21)
    discrimination = rng.uniform(0.2, 4.8, size=48)
    difficulty = rng.uniform(-5.5, 5.5, size=48)
    correct, total = _expected_counts(responses, posterior)
    before = _item_log_likelihoods(discrimination, difficulty, correct, total, points)

    mirt.set_backend("numpy")
    updated_discrimination, updated_difficulty = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        discrimination,
        difficulty,
        max_iter=20,
        tol=1e-10,
        damping=1.0,
    )
    after = _item_log_likelihoods(
        updated_discrimination,
        updated_difficulty,
        correct,
        total,
        points,
    )

    assert np.all(after >= before - 1e-10)
    assert np.all((0.1 <= updated_discrimination) & (updated_discrimination <= 5.0))
    assert np.all((-6.0 <= updated_difficulty) & (updated_difficulty <= 6.0))


def test_optimizer_recovers_interior_expected_counts(
    restore_backend: None,
) -> None:
    points = np.linspace(-3.0, 3.0, 17)
    true_discrimination = 1.45
    true_difficulty = -0.65
    probability = 1.0 / (
        1.0 + np.exp(-true_discrimination * (points - true_difficulty))
    )
    responses = np.tile(np.array([[1], [0]], dtype=np.int32), (points.size, 1))
    posterior = np.zeros((2 * points.size, points.size))
    indices = np.arange(points.size)
    posterior[2 * indices, indices] = 100.0 * probability
    posterior[2 * indices + 1, indices] = 100.0 * (1.0 - probability)

    mirt.set_backend("numpy")
    discrimination, difficulty = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        np.array([0.6]),
        np.array([0.8]),
        max_iter=100,
        tol=1e-12,
        damping=1.0,
        regularization=0.0,
    )

    assert_allclose(discrimination, [true_discrimination], atol=1e-11)
    assert_allclose(difficulty, [true_difficulty], atol=1e-11)


def test_missing_only_and_singular_items_keep_initial_values(
    restore_backend: None,
) -> None:
    responses = np.array([[1, -1], [0, -3], [1, -8]], dtype=np.int32)
    posterior = np.ones((3, 1))
    points = np.array([0.0])
    discrimination = np.array([1.2, 2.3])
    difficulty = np.array([-0.4, 0.7])

    mirt.set_backend("numpy")
    actual = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        discrimination,
        difficulty,
        max_iter=5,
        regularization=0.0,
    )

    assert_array_equal(actual[0], discrimination)
    assert_array_equal(actual[1], difficulty)


def test_finite_negative_missing_codes_are_equivalent(
    restore_backend: None,
) -> None:
    responses, posterior, points, discrimination, difficulty = _one_item_inputs()
    alternate = responses.astype(np.float64)
    alternate[alternate < 0] = -0.25

    mirt.set_backend("numpy")
    expected = m_step_dichotomous_parallel(
        responses, posterior, points, discrimination, difficulty
    )
    actual = m_step_dichotomous_parallel(
        alternate, posterior, points, discrimination, difficulty
    )

    assert_allclose(actual[0], expected[0], atol=0.0)
    assert_allclose(actual[1], expected[1], atol=0.0)


def test_inputs_are_not_mutated_and_outputs_are_normalized(
    restore_backend: None,
) -> None:
    inputs = _one_item_inputs()
    snapshots = tuple(value.copy() for value in inputs)

    mirt.set_backend("numpy")
    outputs = m_step_dichotomous_parallel(*inputs)

    for value, snapshot in zip(inputs, snapshots, strict=True):
        assert_array_equal(value, snapshot)
    for output in outputs:
        assert output.dtype == np.float64
        assert output.flags.c_contiguous


@pytest.mark.skipif(
    not mirt.is_rust_available(),
    reason="native extension required for parity coverage",
)
def test_native_and_numpy_paths_match_randomized_items(
    restore_backend: None,
) -> None:
    rng = np.random.default_rng(1236)
    responses = rng.integers(0, 2, size=(250, 80), dtype=np.int32)
    responses[rng.random(responses.shape) < 0.13] = -7
    posterior = rng.random((250, 31))
    posterior /= posterior.sum(axis=1, keepdims=True)
    points = np.linspace(-4.5, 4.5, 31)
    discrimination = rng.uniform(0.15, 4.9, size=80)
    difficulty = rng.uniform(-5.8, 5.8, size=80)
    options = dict(max_iter=15, tol=1e-9, damping=0.8, regularization=0.02)

    mirt.set_backend("numpy")
    fallback = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        discrimination,
        difficulty,
        **options,
    )
    mirt.set_backend("rust")
    native = m_step_dichotomous_parallel(
        responses,
        posterior,
        points,
        discrimination,
        difficulty,
        **options,
    )

    assert_allclose(native[0], fallback[0], rtol=1e-11, atol=1e-11)
    assert_allclose(native[1], fallback[1], rtol=1e-11, atol=1e-11)


def _valid_arguments() -> dict[str, object]:
    responses, posterior, points, discrimination, difficulty = _one_item_inputs()
    return {
        "responses": responses,
        "posterior_weights": posterior,
        "quad_points": points,
        "discrimination": discrimination,
        "difficulty": difficulty,
    }


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("responses", [0, 1], "two-dimensional"),
        ("responses", np.empty((0, 1)), "non-empty"),
        ("responses", [[0], [1], [np.nan], [1]], "finite"),
        ("responses", [[0], [1], [2], [1]], "0 or 1"),
        ("posterior_weights", np.ones((3, 3)), "one row"),
        ("posterior_weights", np.empty((4, 0)), "quadrature"),
        (
            "posterior_weights",
            [[0.7, 0.2, 0.1], [0.2, -0.5, 0.3], [0.1, 0.3, 0.6], [0.4, 0.4, 0.2]],
            "non-negative",
        ),
        (
            "posterior_weights",
            [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.0, 0.0, 0.0], [0.4, 0.4, 0.2]],
            "positive sum",
        ),
        ("quad_points", [[-1.5, 0.0, 1.5]], "one-dimensional"),
        ("quad_points", [-1.5, 0.0], "posterior column"),
        ("quad_points", [-1.5, np.inf, 1.5], "finite"),
        ("discrimination", [0.8, 1.0], "shape"),
        ("discrimination", [np.nan], "finite"),
        ("discrimination", [0.05], "disc_bounds"),
        ("difficulty", [-6.5], "diff_bounds"),
        ("max_iter", 0, "max_iter"),
        ("max_iter", True, "max_iter"),
        ("tol", 0.0, "tol"),
        ("tol", np.nan, "tol"),
        ("disc_bounds", (0.0, 5.0), "positive lower"),
        ("disc_bounds", (2.0, 1.0), "lower < upper"),
        ("diff_bounds", (-np.inf, 6.0), "finite"),
        ("diff_bounds", (-6.0,), "two finite"),
        ("damping", 0.0, "damping"),
        ("damping", 1.1, "no greater"),
        ("damping", True, "damping"),
        ("regularization", -0.1, "regularization"),
        ("regularization", np.inf, "regularization"),
    ],
)
def test_validation_precedes_dispatch(
    restore_backend: None,
    name: str,
    value: object,
    match: str,
) -> None:
    arguments = _valid_arguments()
    arguments[name] = value

    with pytest.raises(ValueError, match=match):
        m_step_dichotomous_parallel(**arguments)
