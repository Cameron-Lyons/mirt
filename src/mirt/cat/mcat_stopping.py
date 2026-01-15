"""Stopping rules for multidimensional computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from mirt.cat.results import MCATState


class MCATStoppingRule(ABC):
    """Abstract base class for MCAT stopping rules.

    Stopping rules for multidimensional CAT determine when a test session
    should terminate based on precision achieved across all dimensions,
    test length, or other criteria.
    """

    @abstractmethod
    def should_stop(self, state: MCATState) -> bool:
        """Check if the MCAT session should stop.

        Parameters
        ----------
        state : MCATState
            Current state of the MCAT session.

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


class CovarianceTraceStop(MCATStoppingRule):
    """Stop when trace of covariance matrix falls below a threshold.

    The trace represents the sum of variances across all dimensions.
    This stopping rule ensures that the overall precision across all
    dimensions meets a specified criterion.

    Parameters
    ----------
    threshold : float
        Maximum acceptable trace of the covariance matrix.
        Default is 0.5 (equivalent to average SE of ~0.5 per dimension
        for a 2-dimensional model).
    """

    def __init__(self, threshold: float = 0.5):
        if threshold <= 0:
            raise ValueError("Trace threshold must be positive")
        self.threshold = threshold
        self._triggered = False

    def should_stop(self, state: MCATState) -> bool:
        trace = np.trace(state.covariance)
        if trace <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Covariance trace threshold reached (trace <= {self.threshold})"


class CovarianceDeterminantStop(MCATStoppingRule):
    """Stop when determinant of covariance matrix falls below a threshold.

    The determinant represents the volume of the confidence ellipsoid.
    This is consistent with D-optimality selection criteria.

    Parameters
    ----------
    threshold : float
        Maximum acceptable determinant of the covariance matrix.
        Default is 0.01 (approximately 0.1 average SE per dimension
        for a 2-dimensional model).
    """

    def __init__(self, threshold: float = 0.01):
        if threshold <= 0:
            raise ValueError("Determinant threshold must be positive")
        self.threshold = threshold
        self._triggered = False

    def should_stop(self, state: MCATState) -> bool:
        det = np.linalg.det(state.covariance)
        if det <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Covariance determinant threshold reached (det <= {self.threshold})"


class MaxSEStop(MCATStoppingRule):
    """Stop when all standard errors fall below a threshold.

    Ensures that the precision in each individual dimension meets
    a specified criterion. More conservative than trace-based stopping.

    Parameters
    ----------
    threshold : float
        Maximum acceptable standard error for any dimension.
        Default is 0.3.
    """

    def __init__(self, threshold: float = 0.3):
        if threshold <= 0:
            raise ValueError("SE threshold must be positive")
        self.threshold = threshold
        self._triggered = False

    def should_stop(self, state: MCATState) -> bool:
        max_se = np.max(state.standard_error)
        if max_se <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"All SE thresholds reached (max SE <= {self.threshold})"


class AvgSEStop(MCATStoppingRule):
    """Stop when average standard error across dimensions falls below threshold.

    Ensures that the average precision across all dimensions meets
    a specified criterion. Balances between individual and overall precision.

    Parameters
    ----------
    threshold : float
        Maximum acceptable average standard error. Default is 0.3.
    """

    def __init__(self, threshold: float = 0.3):
        if threshold <= 0:
            raise ValueError("Average SE threshold must be positive")
        self.threshold = threshold
        self._triggered = False

    def should_stop(self, state: MCATState) -> bool:
        avg_se = np.mean(state.standard_error)
        if avg_se <= self.threshold:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Average SE threshold reached (mean SE <= {self.threshold})"


class MaxItemsMCATStop(MCATStoppingRule):
    """Stop after a maximum number of items.

    Ensures the test does not exceed a specified length.

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

    def should_stop(self, state: MCATState) -> bool:
        if state.n_items >= self.max_items:
            self._triggered = True
            return True
        return False

    def get_reason(self) -> str:
        return f"Maximum items reached ({self.max_items})"


class ThetaChangeMCATStop(MCATStoppingRule):
    """Stop when theta estimates stabilize across all dimensions.

    Stops when the maximum change in ability estimates between
    consecutive items falls below a threshold for several items.

    Parameters
    ----------
    threshold : float
        Maximum change in any theta to trigger stop. Default is 0.01.
    n_stable : int
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
        self._last_theta: np.ndarray | None = None
        self._triggered = False

    def should_stop(self, state: MCATState) -> bool:
        if self._last_theta is None:
            self._last_theta = state.theta.copy()
            return False

        max_change = np.max(np.abs(state.theta - self._last_theta))
        self._last_theta = state.theta.copy()

        if max_change <= self.threshold:
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
            f"Theta stabilized (max change <= {self.threshold} "
            f"for {self.n_stable} items)"
        )


class CombinedMCATStop(MCATStoppingRule):
    """Combine multiple MCAT stopping rules with logical operators.

    Parameters
    ----------
    rules : list[MCATStoppingRule]
        List of stopping rules to combine.
    operator : {"and", "or"}
        Logical operator for combining rules. Default is "or".
        - "or": Stop when ANY rule is satisfied
        - "and": Stop when ALL rules are satisfied
    min_items : int
        Minimum items before stopping rules are evaluated. Default is 0.
    """

    def __init__(
        self,
        rules: list[MCATStoppingRule],
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
        self._triggered_rule: MCATStoppingRule | None = None

    def should_stop(self, state: MCATState) -> bool:
        if state.n_items < self.min_items:
            return False

        results = [rule.should_stop(state) for rule in self.rules]

        if self.operator == "or":
            for rule, result in zip(self.rules, results):
                if result:
                    self._triggered_rule = rule
                    return True
            return False
        else:
            if all(results):
                self._triggered_rule = self.rules[0]
                return True
            return False

    def get_reason(self) -> str:
        if self._triggered_rule is not None:
            return self._triggered_rule.get_reason()
        return f"Combined rule ({self.operator})"

    def reset(self) -> None:
        """Reset all rules for a new examinee."""
        for rule in self.rules:
            if hasattr(rule, "reset"):
                rule.reset()
        self._triggered_rule = None


def create_mcat_stopping_rule(
    method: str,
    **kwargs: Any,
) -> MCATStoppingRule:
    """Factory function to create MCAT stopping rules.

    Parameters
    ----------
    method : str
        Stopping rule name. One of: "trace", "determinant", "max_se",
        "avg_se", "max_items", "theta_change", "combined".
    **kwargs
        Additional keyword arguments passed to the rule constructor.

    Returns
    -------
    MCATStoppingRule
        The requested stopping rule.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    rules: dict[str, type[MCATStoppingRule]] = {
        "trace": CovarianceTraceStop,
        "determinant": CovarianceDeterminantStop,
        "max_se": MaxSEStop,
        "avg_se": AvgSEStop,
        "max_items": MaxItemsMCATStop,
        "theta_change": ThetaChangeMCATStop,
        "combined": CombinedMCATStop,
    }

    if method not in rules:
        valid = ", ".join(rules.keys())
        raise ValueError(
            f"Unknown MCAT stopping rule '{method}'. Valid options: {valid}"
        )

    return rules[method](**kwargs)
