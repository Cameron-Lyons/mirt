"""Contracts and numerical-stability tests for simulation backend wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mirt.backends.rust import simulation


@pytest.fixture
def numpy_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the local NumPy paths without changing global backend state."""
    monkeypatch.setattr(simulation, "rust_enabled", lambda: False)


def test_dichotomous_column_theta_preserves_person_item_shape(
    numpy_simulation: None,
) -> None:
    responses = simulation.simulate_dichotomous(
        theta=np.array([[-1.0], [0.0], [1.0]]),
        discrimination=np.array([0.8, 1.2]),
        difficulty=np.array([-0.5, 0.5]),
        seed=7,
    )

    assert responses.shape == (3, 2)
    assert np.all((responses == 0) | (responses == 1))


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            simulation.simulate_grm,
            {
                "theta": ["invalid"],
                "discrimination": np.ones(1),
                "thresholds": np.zeros((1, 2)),
            },
            "numeric",
        ),
        (
            simulation.simulate_grm,
            {
                "theta": np.array([]),
                "discrimination": np.ones(1),
                "thresholds": np.zeros((1, 2)),
            },
            "at least one person",
        ),
        (
            simulation.simulate_grm,
            {
                "theta": np.zeros(2),
                "discrimination": np.array([]),
                "thresholds": np.zeros((0, 2)),
            },
            "non-empty",
        ),
        (
            simulation.simulate_grm,
            {
                "theta": np.zeros((2, 2)),
                "discrimination": np.ones(1),
                "thresholds": np.zeros((1, 2)),
            },
            "theta",
        ),
        (
            simulation.simulate_grm,
            {
                "theta": np.zeros(2),
                "discrimination": np.ones(2),
                "thresholds": np.zeros((1, 2)),
            },
            "thresholds",
        ),
        (
            simulation.simulate_gpcm,
            {
                "theta": np.zeros(2),
                "discrimination": np.ones(1),
                "thresholds": np.empty((1, 0)),
            },
            "thresholds",
        ),
        (
            simulation.simulate_dichotomous,
            {
                "theta": np.zeros(2),
                "discrimination": np.ones(2),
                "difficulty": np.zeros(1),
            },
            "difficulty",
        ),
        (
            simulation.simulate_dichotomous,
            {
                "theta": np.zeros(2),
                "discrimination": np.ones(2),
                "difficulty": np.zeros(2),
                "guessing": np.array([0.2]),
            },
            "guessing",
        ),
        (
            simulation.simulate_dichotomous,
            {
                "theta": np.array([0.0, np.nan]),
                "discrimination": np.ones(1),
                "difficulty": np.zeros(1),
            },
            "finite",
        ),
    ],
)
def test_invalid_array_contracts_fail_before_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    function: Any,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setattr(simulation, "rust_enabled", lambda: True)
    monkeypatch.setattr(simulation, "mirt_rs", None)

    with pytest.raises(ValueError, match=message):
        function(**kwargs, seed=1)


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**63])
def test_invalid_seeds_fail_before_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    seed: object,
) -> None:
    monkeypatch.setattr(simulation, "rust_enabled", lambda: True)
    monkeypatch.setattr(simulation, "mirt_rs", None)

    with pytest.raises(ValueError, match="seed"):
        simulation.simulate_dichotomous(
            np.zeros(2),
            np.ones(1),
            np.zeros(1),
            seed=seed,  # type: ignore[arg-type]
        )


def test_guessing_must_be_a_probability(numpy_simulation: None) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulation.simulate_dichotomous(
            np.zeros(2),
            np.ones(1),
            np.zeros(1),
            guessing=np.array([1.1]),
            seed=1,
        )


def test_none_seed_generates_a_valid_seed(numpy_simulation: None) -> None:
    responses = simulation.simulate_dichotomous(
        np.zeros(2),
        np.ones(1),
        np.zeros(1),
    )

    assert responses.shape == (2, 1)


def test_extreme_gpcm_inputs_are_stable_without_runtime_warnings(
    numpy_simulation: None,
) -> None:
    with np.errstate(over="raise", invalid="raise"):
        responses = simulation.simulate_gpcm(
            theta=np.array([-1e308, 1e308]),
            discrimination=np.array([1e308]),
            thresholds=np.array([[-1.0, 1.0]]),
            seed=13,
        )

    np.testing.assert_array_equal(responses[:, 0], [0, 2])


def test_extreme_gpcm_inputs_bypass_unsafe_native_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_native_call(*args: object) -> np.ndarray:
        raise AssertionError("unsafe inputs reached the native function")

    monkeypatch.setattr(simulation, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        simulation,
        "mirt_rs",
        SimpleNamespace(simulate_gpcm=unexpected_native_call),
    )

    responses = simulation.simulate_gpcm(
        theta=np.array([-1e308, 1e308]),
        discrimination=np.array([1e308]),
        thresholds=np.array([[-1.0, 1.0]]),
        seed=13,
    )

    np.testing.assert_array_equal(responses[:, 0], [0, 2])


def test_valid_gpcm_inputs_still_dispatch_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []

    def fake_native(
        theta: np.ndarray,
        discrimination: np.ndarray,
        thresholds: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        calls.append((theta.shape, discrimination.shape, thresholds.shape, seed))
        return np.ones((len(theta), len(discrimination)), dtype=np.int32)

    monkeypatch.setattr(simulation, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        simulation,
        "mirt_rs",
        SimpleNamespace(simulate_gpcm=fake_native),
    )

    responses = simulation.simulate_gpcm(
        theta=np.array([-1.0, 1.0]),
        discrimination=np.array([1.2]),
        thresholds=np.array([[-0.5, 0.5]]),
        seed=17,
    )

    np.testing.assert_array_equal(responses, 1)
    assert calls == [((2, 1), (1,), (1, 2), 17)]


def test_valid_guessing_values_are_forwarded_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_guessing: list[np.ndarray] = []

    def fake_native(
        theta: np.ndarray,
        discrimination: np.ndarray,
        difficulty: np.ndarray,
        guessing: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        received_guessing.append(guessing)
        return np.zeros((len(theta), len(discrimination)), dtype=np.int32)

    monkeypatch.setattr(simulation, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        simulation,
        "mirt_rs",
        SimpleNamespace(simulate_dichotomous=fake_native),
    )

    simulation.simulate_dichotomous(
        theta=np.array([-1.0, 1.0]),
        discrimination=np.array([1.2]),
        difficulty=np.array([0.0]),
        guessing=np.array([0.2]),
        seed=19,
    )

    np.testing.assert_array_equal(received_guessing, [[0.2]])


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            simulation.simulate_grm,
            {
                "theta": np.array([-1.0, 0.0, 1.0]),
                "discrimination": np.array([0.8, 1.2]),
                "thresholds": np.array([[-1.0, 0.5], [-0.5, 1.0]]),
            },
        ),
        (
            simulation.simulate_gpcm,
            {
                "theta": np.array([-1.0, 0.0, 1.0]),
                "discrimination": np.array([0.8, 1.2]),
                "thresholds": np.array([[-1.0, 0.5], [-0.5, 1.0]]),
            },
        ),
        (
            simulation.simulate_dichotomous,
            {
                "theta": np.array([-1.0, 0.0, 1.0]),
                "discrimination": np.array([0.8, 1.2]),
                "difficulty": np.array([-0.5, 0.5]),
            },
        ),
    ],
)
def test_numpy_simulation_remains_seeded(
    numpy_simulation: None,
    function: Any,
    kwargs: dict[str, Any],
) -> None:
    first = function(**kwargs, seed=23)
    second = function(**kwargs, seed=23)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (
            simulation.simulate_grm,
            [
                [3, 0, 0, 0],
                [0, 0, 3, 1],
                [3, 0, 2, 2],
                [2, 3, 1, 1],
                [3, 3, 2, 3],
            ],
        ),
        (
            simulation.simulate_gpcm,
            [
                [2, 0, 0, 0],
                [0, 0, 2, 1],
                [2, 1, 2, 2],
                [2, 3, 1, 1],
                [3, 3, 2, 3],
            ],
        ),
    ],
)
def test_polytomous_simulation_preserves_seed_order_across_chunks(
    numpy_simulation: None,
    monkeypatch: pytest.MonkeyPatch,
    function: Any,
    expected: list[list[int]],
) -> None:
    theta = np.array([-2.0, -0.5, 0.0, 0.75, 2.0])
    discrimination = np.array([0.6, 1.0, 1.4, 2.0])
    thresholds = np.array(
        [
            [-1.5, -0.2, 1.1],
            [-1.0, 0.0, 1.0],
            [-0.7, 0.4, 1.8],
            [-2.0, -0.5, 0.5],
        ]
    )

    unchunked = function(theta, discrimination, thresholds, seed=31)
    monkeypatch.setattr(simulation, "_entry_chunk_size", lambda *_: 1)
    single_item_chunks = function(theta, discrimination, thresholds, seed=31)

    np.testing.assert_array_equal(unchunked, expected)
    np.testing.assert_array_equal(single_item_chunks, expected)
