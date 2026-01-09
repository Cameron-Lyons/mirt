"""Stopping rules for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mirt.cat.results import CATState


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
        if threshold <= 0:
            raise ValueError("SE threshold must be positive")
        self.threshold = threshold
        self._triggered = False

    def should_stop(self, state: CATState) -> bool:
        if state.standard_error <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"SE threshold reached (SE <= {self.threshold})"


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
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._triggered = False

    def should_stop(self, state: CATState) -> bool:
        if state.n_items >= self.max_items:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Maximum items reached ({self.max_items})"


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
        if min_items < 0:
            raise ValueError("min_items must be non-negative")
        self.min_items = min_items

    def should_stop(self, state: CATState) -> bool:
        # This rule alone never stops the test
        # It's meant to be combined with other rules
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
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if n_stable < 1:
            raise ValueError("n_stable must be at least 1")
        self.threshold = threshold
        self.n_stable = n_stable
        self._stable_count = 0
        self._last_theta: float | None = None
        self._triggered = False

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
            f"Theta stabilized (change <= {self.threshold} "
            f"for {self.n_stable} items)"
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
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.cut_score = cut_score
        self.confidence = confidence
        self._triggered = False
        self._classification: str | None = None

    def should_stop(self, state: CATState) -> bool:
        # Compute z-score for cut score
        z = abs(state.theta - self.cut_score) / state.standard_error

        # Convert to confidence using normal CDF
        from scipy.stats import norm

        conf = norm.cdf(z)

        if conf >= self.confidence:
            self._triggered = True
            self._classification = "above" if state.theta > self.cut_score else "below"
            return True
        return False

    def get_reason(self) -> str:
        direction = self._classification or "undetermined"
        return (
            f"Classification confidence reached ({self.confidence:.0%} "
            f"confident, {direction} cut score {self.cut_score})"
        )


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

        self.rules = rules
        self.operator = operator
        self.min_items = min_items
        self._triggered_rule: StoppingRule | None = None

    def should_stop(self, state: CATState) -> bool:
        # Check minimum items first
        if state.n_items < self.min_items:
            return False

        results = [rule.should_stop(state) for rule in self.rules]

        if self.operator == "or":
            for rule, result in zip(self.rules, results):
                if result:
                    self._triggered_rule = rule
                    return True
            return False
        else:  # "and"
            if all(results):
                self._triggered_rule = self.rules[0]
                return True
            return False

    def get_reason(self) -> str:
        if self._triggered_rule is not None:
            return self._triggered_rule.get_reason()
        return f"Combined rule ({self.operator})"


def create_stopping_rule(
    method: str,
    **kwargs,
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
