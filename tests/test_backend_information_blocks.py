"""Regression tests for covariance-aware 2PL item information blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

import mirt
from mirt.backends import rust as rust_backend


@pytest.fixture(autouse=True)
def restore_backend() -> Iterator[None]:
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


def _sample_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    responses = rng.integers(0, 2, size=(80, 2), dtype=np.int32)
    posterior = rng.random((80, 11))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return {
        "responses": responses,
        "posterior_weights": posterior,
        "quad_points": np.linspace(-3.0, 3.0, 11),
        "discrimination": np.array([0.8, 1.6]),
        "difficulty": np.array([-0.7, 0.9]),
        "h": 1e-5,
    }


def _item_expected_log_likelihood(
    responses: np.ndarray,
    posterior: np.ndarray,
    quad_points: np.ndarray,
    discrimination: float,
    difficulty: float,
    item_idx: int,
) -> float:
    item_responses = responses[:, item_idx]
    valid = item_responses >= 0
    expected_total = posterior[valid].sum(axis=0)
    expected_correct = (item_responses[valid, None] * posterior[valid]).sum(axis=0)
    logits = discrimination * (quad_points - difficulty)
    return float(
        np.sum(
            -expected_correct * np.logaddexp(0.0, -logits)
            - (expected_total - expected_correct) * np.logaddexp(0.0, logits)
        )
    )


def _finite_difference_block(
    payload: dict[str, Any],
    item_idx: int,
    step: float = 1e-4,
) -> np.ndarray:
    responses = payload["responses"]
    posterior = payload["posterior_weights"]
    quad_points = payload["quad_points"]
    a = payload["discrimination"][item_idx]
    b = payload["difficulty"][item_idx]

    def objective(discrimination: float, difficulty: float) -> float:
        return _item_expected_log_likelihood(
            responses,
            posterior,
            quad_points,
            discrimination,
            difficulty,
            item_idx,
        )

    center = objective(a, b)
    hessian_aa = (
        objective(a + step, b) - 2.0 * center + objective(a - step, b)
    ) / step**2
    hessian_bb = (
        objective(a, b + step) - 2.0 * center + objective(a, b - step)
    ) / step**2
    hessian_ab = (
        objective(a + step, b + step)
        - objective(a + step, b - step)
        - objective(a - step, b + step)
        + objective(a - step, b - step)
    ) / (4.0 * step**2)
    return np.array(
        [[hessian_aa, hessian_ab], [hessian_ab, hessian_bb]],
        dtype=np.float64,
    )


def _full_inverse_standard_errors(hessian: np.ndarray) -> np.ndarray:
    errors = []
    for item_idx in range(hessian.shape[0] // 2):
        start = item_idx * 2
        information = -hessian[start : start + 2, start : start + 2]
        if np.linalg.det(information) <= 1e-10:
            errors.append([np.nan, np.nan])
        else:
            errors.append(np.sqrt(np.diag(np.linalg.inv(information))))
    return np.asarray(errors)


def test_numpy_hessian_contains_exact_covariance_blocks() -> None:
    payload = _sample_inputs()
    mirt.set_backend("numpy")

    hessian = rust_backend.compute_hessian_block_diagonal(**payload)

    expected_blocks = [
        _finite_difference_block(payload, item_idx) for item_idx in range(2)
    ]
    np.testing.assert_allclose(hessian[:2, :2], expected_blocks[0], atol=2e-5)
    np.testing.assert_allclose(hessian[2:, 2:], expected_blocks[1], atol=2e-5)
    np.testing.assert_array_equal(hessian[:2, 2:], 0.0)
    np.testing.assert_array_equal(hessian[2:, :2], 0.0)
    assert abs(hessian[0, 1]) > 1.0
    assert abs(hessian[2, 3]) > 1.0


def test_standard_errors_invert_full_blocks_instead_of_only_diagonals() -> None:
    payload = _sample_inputs()
    mirt.set_backend("numpy")
    hessian = rust_backend.compute_hessian_block_diagonal(**payload)

    se_discrimination, se_difficulty = rust_backend.compute_item_se_parallel(**payload)

    actual = np.column_stack([se_discrimination, se_difficulty])
    expected = _full_inverse_standard_errors(hessian)
    diagonal_only = np.sqrt(-1.0 / np.diag(hessian)).reshape(-1, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(actual / diagonal_only - 1.0)) > 0.1


def test_analytic_results_do_not_depend_on_compatibility_step_size() -> None:
    payload = _sample_inputs()
    mirt.set_backend("numpy")

    small_step = {
        function.__name__: function(**{**payload, "h": 1e-8})
        for function in (
            rust_backend.compute_hessian_block_diagonal,
            rust_backend.compute_item_se_parallel,
        )
    }
    large_step = {
        function.__name__: function(**{**payload, "h": 1e-2})
        for function in (
            rust_backend.compute_hessian_block_diagonal,
            rust_backend.compute_item_se_parallel,
        )
    }

    for name in small_step:
        small_values = small_step[name]
        large_values = large_step[name]
        if isinstance(small_values, tuple):
            for small, large in zip(small_values, large_values, strict=True):
                np.testing.assert_array_equal(small, large)
        else:
            np.testing.assert_array_equal(small_values, large_values)


@pytest.mark.skipif(
    not mirt.is_rust_available(),
    reason="native extension required for parity coverage",
)
def test_native_and_numpy_information_blocks_match() -> None:
    payload = _sample_inputs()
    mirt.set_backend("numpy")
    numpy_hessian = rust_backend.compute_hessian_block_diagonal(**payload)
    numpy_errors = rust_backend.compute_item_se_parallel(**payload)

    mirt.set_backend("auto")
    native_hessian = rust_backend.compute_hessian_block_diagonal(**payload)
    native_errors = rust_backend.compute_item_se_parallel(**payload)

    np.testing.assert_allclose(native_hessian, numpy_hessian, rtol=1e-12, atol=1e-12)
    for native, fallback in zip(native_errors, numpy_errors, strict=True):
        np.testing.assert_allclose(native, fallback, rtol=1e-12, atol=1e-12)


def test_missing_only_items_return_singular_blocks_and_unknown_errors() -> None:
    payload = _sample_inputs()
    payload["responses"][:, 0] = -1
    mirt.set_backend("numpy")

    hessian = rust_backend.compute_hessian_block_diagonal(**payload)
    se_discrimination, se_difficulty = rust_backend.compute_item_se_parallel(**payload)

    np.testing.assert_array_equal(hessian[:2, :2], 0.0)
    assert np.isnan(se_discrimination[0])
    assert np.isnan(se_difficulty[0])


def test_nan_responses_use_the_missing_response_contract() -> None:
    payload = _sample_inputs()
    with_nan = {**payload, "responses": payload["responses"].astype(np.float64)}
    with_nan["responses"][0, 0] = np.nan
    with_code = {**with_nan, "responses": with_nan["responses"].copy()}
    with_code["responses"][0, 0] = -1
    mirt.set_backend("numpy")

    actual = rust_backend.compute_hessian_block_diagonal(**with_nan)
    expected = rust_backend.compute_hessian_block_diagonal(**with_code)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("responses", np.array([0, 1]), "non-empty 2D"),
        ("responses", np.empty((0, 2)), "non-empty 2D"),
        (
            "responses",
            np.tile(np.array([[0, 2], [1, 0]]), (40, 1)),
            "only 0 or 1",
        ),
        (
            "responses",
            np.tile(np.array([[0, np.inf], [1, 0]]), (40, 1)),
            "finite values",
        ),
        ("posterior_weights", np.ones((79, 11)), "one row per respondent"),
        ("posterior_weights", np.empty((80, 0)), "one row per respondent"),
        ("posterior_weights", np.full((80, 11), np.nan), "finite non-negative"),
        ("posterior_weights", np.full((80, 11), -0.1), "finite non-negative"),
        ("quad_points", np.ones((11, 1)), "one-dimensional"),
        ("quad_points", np.ones(10), "one value per posterior column"),
        ("quad_points", np.full(11, np.nan), "only finite"),
        ("discrimination", np.ones(3), "shape \\(2,\\)"),
        ("discrimination", np.array([1.0, np.nan]), "only finite"),
        ("difficulty", np.ones(3), "shape \\(2,\\)"),
        ("difficulty", np.array([0.0, np.inf]), "only finite"),
        ("h", 0.0, "finite positive"),
        ("h", np.nan, "finite positive"),
        ("h", True, "finite positive"),
        ("h", "small", "finite positive"),
    ],
)
def test_information_block_functions_validate_inputs(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _sample_inputs()
    payload[field] = value
    mirt.set_backend("numpy")
    functions: tuple[Callable[..., object], ...] = (
        rust_backend.compute_hessian_block_diagonal,
        rust_backend.compute_item_se_parallel,
    )

    for function in functions:
        with pytest.raises(ValueError, match=match):
            function(**payload)
