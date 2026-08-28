"""Rust backend helpers for score equating.

Fallback mode: optional. Returns ``None`` when Rust is unavailable or disabled;
the public equating API retains its NumPy implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import _ensure_f64, mirt_rs, rust_enabled

FALLBACK_MODE = "optional"
_PROBABILITY_TOLERANCE = 1e-10


def observed_score_distribution_2pl(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Compute conditional sum-score probabilities for a 1PL/2PL form.

    Returns an array with shape ``(n_theta, n_items + 1)`` or ``None`` when
    compiled dispatch is unavailable.
    """
    theta = np.asarray(theta, dtype=np.float64)
    discrimination = np.asarray(discrimination, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array")
    if discrimination.ndim != 1 or discrimination.size == 0:
        raise ValueError("discrimination must be a non-empty one-dimensional array")
    if difficulty.ndim != 1 or difficulty.shape != discrimination.shape:
        raise ValueError("difficulty must match the one-dimensional discrimination")
    if not (
        np.all(np.isfinite(theta))
        and np.all(np.isfinite(discrimination))
        and np.all(np.isfinite(difficulty))
    ):
        raise ValueError("equating inputs must contain only finite values")

    if not rust_enabled():
        return None

    conditional = np.asarray(
        mirt_rs.lord_wingersky_recursion(
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
            _ensure_f64(theta),
        ),
        dtype=np.float64,
    )
    expected_shape = (theta.size, discrimination.size + 1)
    if conditional.shape != expected_shape:
        raise RuntimeError(
            f"native score distribution has shape {conditional.shape}, "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(conditional)) or np.any(
        (conditional < -_PROBABILITY_TOLERANCE)
        | (conditional > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise RuntimeError("native score distribution contains invalid probabilities")
    row_mass = np.sum(conditional, axis=1)
    if not np.allclose(row_mass, 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise RuntimeError("native score distribution rows must sum to one")
    return np.clip(conditional, 0.0, 1.0)
