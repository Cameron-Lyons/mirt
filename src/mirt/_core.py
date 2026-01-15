"""Core utility functions with no internal dependencies.

This module provides fundamental utility functions that are used throughout
the codebase but have no dependencies on other mirt modules, avoiding
circular import issues.
"""

import numpy as np
from numpy.typing import NDArray


def sigmoid(x: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
    """Compute sigmoid function with numerical stability.

    Uses the identity sigmoid(-x) = 1 - sigmoid(x) to avoid overflow
    for large negative values.

    Parameters
    ----------
    x : array_like or float
        Input values.

    Returns
    -------
    array_like or float
        Sigmoid of input, same shape as input.
    """
    x = np.asarray(x)
    result = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return float(result) if result.ndim == 0 else result
