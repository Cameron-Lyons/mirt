"""Internal categorical sampling helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_categorical_rows(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Draw one category from every row of a probability matrix."""
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(probabilities.shape[0])
    return np.sum(uniforms[:, None] >= cumulative, axis=1).astype(np.int_)
