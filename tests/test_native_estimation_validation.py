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
