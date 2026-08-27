"""Validation contracts for accelerated estimation entry points."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mirt.backends.rust.estimation import bootstrap_fit_2pl, em_fit_2pl


@pytest.mark.parametrize("native_fit", [em_fit_2pl, bootstrap_fit_2pl])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_quadpts": 0}, "n_quadpts must be positive"),
        ({"n_quadpts": True}, "n_quadpts must be positive"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"max_iter": 1.5}, "max_iter must be at least 1"),
        ({"tol": 0.0}, "tol must be positive"),
        ({"tol": np.nan}, "tol must be positive"),
    ],
)
def test_native_fit_rejects_invalid_em_controls(
    native_fit: Callable[..., Any],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    responses = np.array([[0], [1]], dtype=np.int32)

    with pytest.raises(ValueError, match=message):
        native_fit(responses, **kwargs)


@pytest.mark.parametrize("n_bootstrap", [0, False, 1.5])
def test_native_bootstrap_requires_positive_integer_replicates(
    n_bootstrap: Any,
) -> None:
    responses = np.array([[0], [1]], dtype=np.int32)

    with pytest.raises(ValueError, match="n_bootstrap must be at least 1"):
        bootstrap_fit_2pl(responses, n_bootstrap=n_bootstrap)


def test_native_bootstrap_validates_warm_start_parameters() -> None:
    responses = np.array([[0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="provided together"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.ones(2),
        )
    with pytest.raises(ValueError, match="one value per item"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.ones(1),
            initial_difficulty=np.zeros(1),
        )
    with pytest.raises(ValueError, match="finite"):
        bootstrap_fit_2pl(
            responses,
            n_bootstrap=2,
            initial_discrimination=np.array([1.0, np.nan]),
            initial_difficulty=np.zeros(2),
        )


def test_native_bootstrap_accepts_warm_start_parameters() -> None:
    responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int32)
    kwargs = {
        "n_bootstrap": 2,
        "max_iter": 2,
        "seed": 42,
        "initial_discrimination": np.array([1.2, 0.8]),
        "initial_difficulty": np.array([-0.25, 0.25]),
    }

    discrimination, difficulty = bootstrap_fit_2pl(responses, **kwargs)
    repeated_discrimination, repeated_difficulty = bootstrap_fit_2pl(
        responses, **kwargs
    )

    assert discrimination.shape == (2, 2)
    assert difficulty.shape == (2, 2)
    assert np.isfinite(discrimination).all()
    assert np.isfinite(difficulty).all()
    np.testing.assert_array_equal(discrimination, repeated_discrimination)
    np.testing.assert_array_equal(difficulty, repeated_difficulty)
