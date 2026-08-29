"""Stopping rules for multidimensional computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


class CompositeClassificationStop(MCATStoppingRule):
    """Stop after confidently classifying a weighted ability composite.

    The composite estimate is ``weights @ theta`` and its standard error is
    ``sqrt(weights @ covariance @ weights)``. Using the full covariance matrix
    accounts for correlations between latent dimensions.

    Parameters
    ----------
    weights : array-like
        Finite, nonzero factor weights defining the composite and decision
        boundary. The number of weights must match the state dimensions.
    cut_score : float
        Finite cut score on the weighted composite scale.
    confidence : float, default=0.95
        Required one-sided confidence, strictly between 0.5 and 1.
    """

    def __init__(
        self,
        weights: ArrayLike,
        cut_score: float,
        confidence: float = 0.95,
    ) -> None:
        raw_weights = np.asarray(weights)
        if raw_weights.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("weights must be a one-dimensional array of finite values")
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("weights must be a one-dimensional array of finite values")
        if not np.any(values != 0.0):
            raise ValueError("weights must contain at least one nonzero value")
        if isinstance(cut_score, bool) or not isinstance(cut_score, Real):
            raise ValueError("cut_score must be finite")
        cut_value = float(cut_score)
        if not np.isfinite(cut_value):
            raise ValueError("cut_score must be finite")
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("confidence must be strictly between 0.5 and 1")
        confidence_value = float(confidence)
        if not np.isfinite(confidence_value) or not 0.5 < confidence_value < 1.0:
            raise ValueError("confidence must be strictly between 0.5 and 1")

        from scipy.special import ndtri

        self._weights: NDArray[np.float64] = values.copy()
        self.cut_score: float = cut_value
        self.confidence: float = confidence_value
        self._critical_z: float = float(ndtri(confidence_value))
        self._triggered: bool = False
        self._classification: Literal["above", "below"] | None = None

    @property
    def weights(self) -> NDArray[np.float64]:
        """Return a defensive copy of the composite factor weights."""
        return self._weights.copy()

    @property
    def classification(self) -> Literal["above", "below"] | None:
        """Return the reached classification, or ``None`` before a decision."""
        return self._classification

    def should_stop(self, state: MCATState) -> bool:
        theta = np.asarray(state.theta, dtype=np.float64)
        covariance = np.asarray(state.covariance, dtype=np.float64)
        n_factors = self._weights.size
        if theta.shape != (n_factors,):
            raise ValueError(
                f"state.theta must have shape ({n_factors},) to match weights"
            )
        if covariance.shape != (n_factors, n_factors):
            raise ValueError(
                "state.covariance must have shape "
                f"({n_factors}, {n_factors}) to match weights"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("state.theta must contain only finite values")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("state.covariance must contain only finite values")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("state.covariance must be symmetric")

        composite = float(self._weights @ theta)
        variance = float(self._weights @ covariance @ self._weights)
        scale = max(
            1.0,
            float(np.max(np.abs(covariance))) * float(self._weights @ self._weights),
        )
        tolerance = 16.0 * np.finfo(np.float64).eps * n_factors * scale
        if variance < -tolerance:
            raise ValueError(
                "state.covariance must have non-negative projected variance"
            )

        standard_error = float(np.sqrt(max(variance, 0.0)))
        distance = abs(composite - self.cut_score)
        confident = (
            distance > 0.0
            if standard_error == 0.0
            else distance >= self._critical_z * standard_error
        )
        if confident:
            self._triggered = True
            self._classification = "above" if composite > self.cut_score else "below"
            return True
        return False

    def get_reason(self) -> str:
        direction = self._classification or "undetermined"
        return (
            f"Composite classification confidence reached ({self.confidence:.0%} "
            f"confident, {direction} cut score {self.cut_score})"
        )

    def reset(self) -> None:
        """Clear classification details from the prior session."""
        self._triggered = False
        self._classification = None


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
        "avg_se", "max_items", "theta_change", "classification", "combined".
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
        "classification": CompositeClassificationStop,
        "combined": CombinedMCATStop,
    }

    if method not in rules:
        valid = ", ".join(rules.keys())
        raise ValueError(
            f"Unknown MCAT stopping rule '{method}'. Valid options: {valid}"
        )

    return rules[method](**kwargs)
