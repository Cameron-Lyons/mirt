"""Validation and composition shared by CAT and MCAT stopping rules.

The helpers are dimension-agnostic: statistics may be scalars (CAT) or
vectors (MCAT), and combinations accept either rule family.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _Rule(Protocol):
    def should_stop(self, state: Any) -> bool: ...


RuleT = TypeVar("RuleT", bound=_Rule)


def finite_real(value: Any, name: str) -> float:
    """Validate and normalize a finite real-valued rule parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_real(value: Any, name: str) -> float:
    """Validate and normalize a finite positive rule parameter."""
    result = finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def integer(value: Any, name: str, *, minimum: int) -> int:
    """Validate and normalize an integer rule parameter."""
    requirement = "positive" if minimum == 1 else "non-negative"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a {requirement} integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be {requirement}")
    return result


def stable_count(value: Any) -> int:
    """Validate the number of consecutive stable updates a rule requires."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("n_stable must be an integer of at least 1")
    if value < 1:
        raise ValueError("n_stable must be an integer of at least 1")
    return int(value)


def combination_operator(operator: Any) -> Literal["and", "or"]:
    """Validate the logical operator of a combined rule."""
    if operator == "and":
        return "and"
    if operator == "or":
        return "or"
    raise ValueError("operator must be 'and' or 'or'")


def triggered_rule(
    rules: Sequence[RuleT],
    operator: Literal["and", "or"],
    state: Any,
) -> RuleT | None:
    """Return the rule that stops a combination, or ``None``.

    ``"or"`` returns the first rule that stops and does not evaluate later
    rules. ``"and"`` evaluates every rule, so stateful rules observe each
    state, and reports the first rule when all of them stop.
    """
    if operator == "or":
        return next((rule for rule in rules if rule.should_stop(state)), None)
    results = [rule.should_stop(state) for rule in rules]
    return rules[0] if all(results) else None


class ChangeTracker:
    """Count consecutive updates whose largest absolute change is small.

    Scalars and arrays are compared elementwise. A non-finite value on either
    side of a change never counts as stable.
    """

    def __init__(self) -> None:
        self.previous: NDArray[np.float64] | None = None
        self.stable_count = 0

    def update(self, value: ArrayLike, threshold: float) -> int:
        """Record ``value`` and return the current run of stable changes."""
        current = np.array(value, dtype=np.float64)
        previous, self.previous = self.previous, current
        if previous is None:
            return self.stable_count
        stable = (
            bool(np.all(np.isfinite(current)) and np.all(np.isfinite(previous)))
            and float(np.max(np.abs(current - previous))) <= threshold
        )
        self.stable_count = self.stable_count + 1 if stable else 0
        return self.stable_count

    def reset(self) -> None:
        """Forget the previous value and the current stable run."""
        self.previous = None
        self.stable_count = 0
