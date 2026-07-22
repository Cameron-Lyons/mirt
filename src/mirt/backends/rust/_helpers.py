"""Shared helpers and Rust availability for the mirt Rust backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from mirt import mirt_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    mirt_rs = None


def is_rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE


def _ensure_f64(arr: NDArray[np.floating] | None) -> NDArray[np.float64] | None:
    """Convert array to float64 using no-copy if already correct dtype."""
    if arr is None:
        return None
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float64)


def _ensure_i32(arr: NDArray[np.integer] | None) -> NDArray[np.int32] | None:
    """Convert array to int32 using no-copy if already correct dtype."""
    if arr is None:
        return None
    if arr.dtype == np.int32:
        return arr
    return arr.astype(np.int32)


def _ensure_i64(arr: NDArray[np.integer] | None) -> NDArray[np.int64] | None:
    """Convert array to int64 using no-copy if already correct dtype."""
    if arr is None:
        return None
    if arr.dtype == np.int64:
        return arr
    return arr.astype(np.int64)


_MAX_VECTOR_CHUNK_ENTRIES = 1_000_000


def _prepare_binary_response_components(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare binary response matrices used by vectorized log-likelihood code."""
    response_matrix = np.asarray(responses)
    valid = (response_matrix >= 0).astype(np.float64)
    correct = (response_matrix == 1).astype(np.float64)
    return correct, valid


def _quad_chunk_size(n_quad: int, n_items: int) -> int:
    """Select a quadrature chunk size with bounded temporary memory."""
    return max(1, min(n_quad, _MAX_VECTOR_CHUNK_ENTRIES // max(1, n_items)))


class PreparedArrays:
    """Pre-converted arrays for EM estimation to avoid repeated conversions."""

    __slots__ = (
        "responses_i32",
        "quad_points_f64",
        "quad_weights_f64",
        "discrimination_f64",
        "difficulty_f64",
        "guessing_f64",
        "thresholds_f64",
        "n_categories_i32",
    )

    def __init__(
        self,
        responses: NDArray[np.integer] | None = None,
        quad_points: NDArray[np.floating] | None = None,
        quad_weights: NDArray[np.floating] | None = None,
        discrimination: NDArray[np.floating] | None = None,
        difficulty: NDArray[np.floating] | None = None,
        guessing: NDArray[np.floating] | None = None,
        thresholds: NDArray[np.floating] | None = None,
        n_categories: NDArray[np.integer] | None = None,
    ):
        """Pre-convert arrays once for use in EM iteration loops.

        Use this to avoid repeated dtype conversions when calling Rust functions
        multiple times with the same arrays (e.g., in EM iterations where only
        parameters change).
        """
        self.responses_i32 = _ensure_i32(responses)
        self.quad_points_f64 = _ensure_f64(quad_points)
        self.quad_weights_f64 = _ensure_f64(quad_weights)
        self.discrimination_f64 = _ensure_f64(discrimination)
        self.difficulty_f64 = _ensure_f64(difficulty)
        self.guessing_f64 = _ensure_f64(guessing)
        self.thresholds_f64 = _ensure_f64(thresholds)
        self.n_categories_i32 = _ensure_i32(n_categories)

    def update_params(
        self,
        discrimination: NDArray[np.floating] | None = None,
        difficulty: NDArray[np.floating] | None = None,
        guessing: NDArray[np.floating] | None = None,
        thresholds: NDArray[np.floating] | None = None,
    ) -> None:
        """Update parameter arrays (used during M-step)."""
        if discrimination is not None:
            self.discrimination_f64 = _ensure_f64(discrimination)
        if difficulty is not None:
            self.difficulty_f64 = _ensure_f64(difficulty)
        if guessing is not None:
            self.guessing_f64 = _ensure_f64(guessing)
        if thresholds is not None:
            self.thresholds_f64 = _ensure_f64(thresholds)
