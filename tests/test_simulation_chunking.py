"""Regression coverage for bounded public polytomous simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import mirt.utils.simulation as simulation

_THETA = np.array(
    [
        [-1.5, -0.5],
        [-0.5, 0.75],
        [0.0, 0.0],
        [0.8, -1.0],
        [1.5, 0.5],
    ]
)
_DISCRIMINATION = np.array(
    [
        [0.7, 0.4],
        [1.0, 0.8],
        [1.3, 0.5],
        [0.6, 1.5],
    ]
)
_THRESHOLDS = np.array(
    [
        [-1.2, 0.0, 1.1],
        [-0.8, 0.2, 1.4],
        [-1.5, -0.1, 0.9],
        [-1.0, 0.5, 1.5],
    ]
)
_NRM_SLOPES = np.stack(
    [
        np.array(
            [
                [0.0, 0.0],
                [0.8, -0.3],
                [-0.4, 0.9],
                [0.6, 0.7],
            ]
        )
        + item * 0.05
        for item in range(4)
    ]
)
_NRM_INTERCEPTS = np.array(
    [
        [0.0, -0.2, 0.3, -0.4],
        [0.0, 0.1, -0.3, 0.2],
        [0.0, -0.4, 0.2, 0.1],
        [0.0, 0.3, -0.2, -0.1],
    ]
)


@pytest.mark.parametrize(
    ("model", "kwargs", "expected"),
    [
        (
            "GRM",
            {
                "theta": _THETA,
                "n_factors": 2,
                "discrimination": _DISCRIMINATION,
                "difficulty": np.zeros(4),
                "thresholds": _THRESHOLDS,
            },
            [
                [2, 0, 1, 0],
                [0, 0, 3, 1],
                [2, 0, 2, 1],
                [2, 3, 1, 0],
                [3, 2, 3, 3],
            ],
        ),
        (
            "GPCM",
            {
                "theta": _THETA,
                "n_factors": 2,
                "discrimination": _DISCRIMINATION,
                "steps": _THRESHOLDS,
            },
            [
                [2, 0, 1, 0],
                [0, 0, 3, 1],
                [2, 1, 2, 1],
                [2, 3, 1, 0],
                [3, 2, 3, 2],
            ],
        ),
        (
            "PCM",
            {
                "theta": _THETA[:, 0],
                "steps": _THRESHOLDS,
            },
            [
                [1, 0, 1, 0],
                [0, 0, 3, 0],
                [2, 0, 2, 1],
                [2, 3, 2, 0],
                [3, 2, 3, 3],
            ],
        ),
        (
            "NRM",
            {
                "theta": _THETA,
                "n_factors": 2,
                "slopes": _NRM_SLOPES,
                "intercepts": _NRM_INTERCEPTS,
            },
            [
                [2, 0, 2, 0],
                [0, 0, 3, 1],
                [2, 0, 2, 1],
                [1, 3, 0, 0],
                [3, 1, 2, 3],
            ],
        ),
    ],
)
def test_polytomous_seed_order_is_independent_of_item_chunks(
    monkeypatch: pytest.MonkeyPatch,
    model: simulation.SimulationModel,
    kwargs: dict[str, Any],
    expected: list[list[int]],
) -> None:
    monkeypatch.setattr(simulation, "_should_use_rust", lambda: False)
    unchunked = simulation.simdata(
        model=model,
        n_items=4,
        n_categories=4,
        seed=31,
        **kwargs,
    )

    monkeypatch.setattr(simulation, "_MAX_POLYTOMOUS_CHUNK_ENTRIES", 1)
    single_item_chunks = simulation.simdata(
        model=model,
        n_items=4,
        n_categories=4,
        seed=31,
        **kwargs,
    )

    np.testing.assert_array_equal(unchunked, expected)
    np.testing.assert_array_equal(single_item_chunks, expected)


@pytest.mark.parametrize("model", ["GRM", "GPCM", "NRM"])
def test_generated_unidimensional_parameters_are_chunk_invariant(
    monkeypatch: pytest.MonkeyPatch,
    model: simulation.SimulationModel,
) -> None:
    monkeypatch.setattr(simulation, "_should_use_rust", lambda: False)
    unchunked = simulation.simdata(
        model=model,
        n_persons=17,
        n_items=7,
        n_categories=6,
        seed=90210,
    )

    monkeypatch.setattr(simulation, "_MAX_POLYTOMOUS_CHUNK_ENTRIES", 1)
    single_item_chunks = simulation.simdata(
        model=model,
        n_persons=17,
        n_items=7,
        n_categories=6,
        seed=90210,
    )

    np.testing.assert_array_equal(single_item_chunks, unchunked)
