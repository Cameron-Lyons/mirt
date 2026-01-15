"""Constants for numerical stability and default bounds.

These are true constants that should not be user-configurable.
For configurable values, use function arguments with defaults.
"""

PROB_EPSILON: float = 1e-10
"""Small value to prevent log(0) and division by zero in probability calculations."""

PROB_CLIP_MIN: float = 0.01
"""Minimum probability for clipping to avoid extreme values."""

PROB_CLIP_MAX: float = 0.99
"""Maximum probability for clipping to avoid extreme values."""

REGULARIZATION_EPSILON: float = 1e-6
"""Small value for matrix regularization and conditioning."""
