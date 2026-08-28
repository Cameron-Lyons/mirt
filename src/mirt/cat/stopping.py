"""Stopping rules for computerized adaptive testing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

from scipy.special import ndtri

if TYPE_CHECKING:
    from mirt.cat.results import CATState


def _finite_real(value: Real, name: str) -> float:
    """Validate and normalize a finite real-valued rule parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value: Real, name: str) -> float:
    """Validate and normalize a finite positive rule parameter."""
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Integral, name: str, *, minimum: int) -> int:
    """Validate and normalize an integer rule parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer")
    result = int(value)
    if result < minimum:
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {requirement}")
    return result


class StoppingRule(ABC):
    """Abstract base class for CAT stopping rules.

    Stopping rules determine when a CAT session should terminate
    based on precision achieved, test length, or other criteria.
    """

    @abstractmethod
    def should_stop(self, state: CATState) -> bool:
        """Check if the CAT session should stop.

        Parameters
        ----------
        state : CATState
            Current state of the CAT session.

        Returns
        -------
        bool
            True if the test should stop, False otherwise.
        """
        pass

    @abstractmethod
    def get_reason(self) -> str:
        """Get the reason for stopping.

        Returns
        -------
        str
            Description of why the test stopped.
        """
        pass

    def reset(self) -> None:
        """Reset state before the rule is reused for another session."""


class StandardErrorStop(StoppingRule):
    """Stop when standard error falls below a threshold.

    This is the most common stopping rule in CAT, ensuring
    that ability estimates meet a specified precision criterion.

    Parameters
    ----------
    threshold : float, optional
        Maximum acceptable standard error. Default is 0.3.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = _positive_real(threshold, "SE threshold")
        self._triggered = False

    def should_stop(self, state: CATState) -> bool:
        if state.standard_error <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"SE threshold reached (SE <= {self.threshold})"

    def reset(self) -> None:
        """Clear the prior session's trigger state."""
        self._triggered = False


class MaxItemsStop(StoppingRule):
    """Stop after a maximum number of items.

    Ensures the test does not exceed a specified length,
    which is important for test security and examinee fatigue.

    Parameters
    ----------
    max_items : int
        Maximum number of items to administer.
    """

    def __init__(self, max_items: int):
        self.max_items = _integer(max_items, "max_items", minimum=1)
        self._triggered = False

    def should_stop(self, state: CATState) -> bool:
        if state.n_items >= self.max_items:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Maximum items reached ({self.max_items})"

    def reset(self) -> None:
        """Clear the prior session's trigger state."""
        self._triggered = False


class MinItemsStop(StoppingRule):
    """Require a minimum number of items before other rules can stop.

    This rule by itself never triggers a stop; it is used in
    combination with other rules via CombinedStop to ensure
    a minimum test length.

    Parameters
    ----------
    min_items : int
        Minimum number of items required before stopping.
    """

    def __init__(self, min_items: int):
        self.min_items = _integer(min_items, "min_items", minimum=0)

    def should_stop(self, state: CATState) -> bool:
        return False

    def is_satisfied(self, state: CATState) -> bool:
        """Check if minimum items requirement is met.

        Parameters
        ----------
        state : CATState
            Current CAT state.

        Returns
        -------
        bool
            True if minimum items have been administered.
        """
        return state.n_items >= self.min_items

    def get_reason(self) -> str:
        return f"Minimum items requirement ({self.min_items})"


class ThetaChangeStop(StoppingRule):
    """Stop when theta estimate stabilizes.

    Stops when the change in ability estimate between consecutive
    items falls below a threshold, indicating convergence.

    Parameters
    ----------
    threshold : float, optional
        Maximum change in theta to trigger stop. Default is 0.01.
    n_stable : int, optional
        Number of consecutive stable estimates required. Default is 3.
    """

    def __init__(self, threshold: float = 0.01, n_stable: int = 3):
        self.threshold: float = _positive_real(threshold, "threshold")
        if (
            isinstance(n_stable, bool)
            or not isinstance(n_stable, Integral)
            or n_stable < 1
        ):
            raise ValueError("n_stable must be an integer of at least 1")
        self.n_stable: int = int(n_stable)
        self._stable_count: int = 0
        self._last_theta: float | None = None
        self._triggered: bool = False

    def should_stop(self, state: CATState) -> bool:
        if self._last_theta is None:
            self._last_theta = state.theta
            return False

        change = abs(state.theta - self._last_theta)
        self._last_theta = state.theta

        if change <= self.threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self.n_stable:
            self._triggered = True
            return True
        return False

    def reset(self) -> None:
        """Reset the rule for a new examinee."""
        self._stable_count = 0
        self._last_theta = None
        self._triggered = False

    def get_reason(self) -> str:
        return (
            f"Theta stabilized (change <= {self.threshold} for {self.n_stable} items)"
        )


class ClassificationStop(StoppingRule):
    """Stop when classification decision is confident.

    Used for mastery testing where the goal is to classify
    examinees above or below a cut score with sufficient confidence.

    Parameters
    ----------
    cut_score : float
        The ability cut score for classification.
    confidence : float, optional
        Required confidence level (0-1). Default is 0.95.
    """

    def __init__(self, cut_score: float, confidence: float = 0.95):
        cut_score_value = _finite_real(cut_score, "cut_score")
        confidence_value = _finite_real(confidence, "confidence")
        if not 0.0 < confidence_value < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.cut_score = cut_score_value
        self.confidence = confidence_value
        self._critical_z = float(ndtri(confidence_value))
        self._triggered = False
        self._classification: str | None = None

    def should_stop(self, state: CATState) -> bool:
        theta = float(state.theta)
        standard_error = float(state.standard_error)
        if not math.isfinite(theta):
            raise ValueError("state.theta must be finite")
        if math.isnan(standard_error) or standard_error < 0.0:
            raise ValueError("state.standard_error must be non-negative")

        distance = abs(theta - self.cut_score)
        if standard_error == 0.0 or math.isinf(standard_error):
            confident = distance > 0.0 if standard_error == 0.0 else False
            if self._critical_z <= 0.0:
                confident = True
        else:
            confident = distance >= self._critical_z * standard_error

        if confident:
            self._triggered = True
            self._classification = "above" if theta > self.cut_score else "below"
            return True
        return False

    def get_reason(self) -> str:
        direction = self._classification or "undetermined"
        return (
            f"Classification confidence reached ({self.confidence:.0%} "
            f"confident, {direction} cut score {self.cut_score})"
        )

    def reset(self) -> None:
        """Clear classification details from the prior session."""
        self._triggered = False
        self._classification = None


class CombinedStop(StoppingRule):
    """Combine multiple stopping rules with logical operators.

    Parameters
    ----------
    rules : list[StoppingRule]
        List of stopping rules to combine.
    operator : {"and", "or"}, optional
        Logical operator for combining rules. Default is "or".
        - "or": Stop when ANY rule is satisfied
        - "and": Stop when ALL rules are satisfied
    min_items : int, optional
        Minimum items before stopping rules are evaluated. Default is 0.
    """

    def __init__(
        self,
        rules: list[StoppingRule],
        operator: Literal["and", "or"] = "or",
        min_items: int = 0,
    ):
        if not rules:
            raise ValueError("At least one rule is required")
        if operator not in ("and", "or"):
            raise ValueError("operator must be 'and' or 'or'")

        self.rules = list(rules)
        self.operator = operator
        self.min_items = _integer(min_items, "min_items", minimum=0)
        self._triggered_rule: StoppingRule | None = None

    def should_stop(self, state: CATState) -> bool:
        self._triggered_rule = None
        if state.n_items < self.min_items:
            return False

        if self.operator == "or":
            for rule in self.rules:
                if rule.should_stop(state):
                    self._triggered_rule = rule
                    return True
            return False

        results = [rule.should_stop(state) for rule in self.rules]
        if all(results):
            self._triggered_rule = self.rules[0]
            return True
        return False

    def get_reason(self) -> str:
        if self._triggered_rule is not None:
            return self._triggered_rule.get_reason()
        return f"Combined rule ({self.operator})"

    def reset(self) -> None:
        """Reset every nested rule before a new adaptive session."""
        self._triggered_rule = None
        for rule in self.rules:
            rule.reset()


def create_stopping_rule(
    method: str,
    **kwargs: Any,
) -> StoppingRule:
    """Factory function to create stopping rules.

    Parameters
    ----------
    method : str
        Stopping rule name. One of: "SE", "max_items", "min_items",
        "theta_change", "classification", "combined".
    **kwargs
        Additional keyword arguments passed to the rule constructor.

    Returns
    -------
    StoppingRule
        The requested stopping rule.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    rules = {
        "SE": StandardErrorStop,
        "max_items": MaxItemsStop,
        "min_items": MinItemsStop,
        "theta_change": ThetaChangeStop,
        "classification": ClassificationStop,
        "combined": CombinedStop,
    }

    if method not in rules:
        valid = ", ".join(rules.keys())
        raise ValueError(f"Unknown stopping rule '{method}'. Valid options: {valid}")

    return rules[method](**kwargs)
