"""Stopping rules for computerized adaptive testing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from scipy.special import ndtri

from mirt.cat._native import register_native_defaults as _register_native_defaults
from mirt.cat._stopping_common import (
    ChangeTracker,
    combination_operator,
    finite_real,
    integer,
    positive_real,
    stable_count,
    triggered_rule,
)
from mirt.cat.selection import MaxFisherInformation

if TYPE_CHECKING:
    from mirt.cat.results import CATState
    from mirt.models.base import BaseItemModel


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


@_register_native_defaults
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
        self.threshold = positive_real(threshold, "SE threshold")

    def should_stop(self, state: CATState) -> bool:
        return bool(state.standard_error <= self.threshold)

    def get_reason(self) -> str:
        return f"SE threshold reached (SE <= {self.threshold})"


@_register_native_defaults
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
        self.max_items = integer(max_items, "max_items", minimum=1)

    def should_stop(self, state: CATState) -> bool:
        return state.n_items >= self.max_items

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
        self.min_items = integer(min_items, "min_items", minimum=0)

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
        self.threshold: float = positive_real(threshold, "threshold")
        self.n_stable: int = stable_count(n_stable)
        self._changes = ChangeTracker()

    def should_stop(self, state: CATState) -> bool:
        return self._changes.update(state.theta, self.threshold) >= self.n_stable

    def reset(self) -> None:
        """Reset the rule for a new examinee."""
        self._changes.reset()

    def get_reason(self) -> str:
        return (
            f"Theta stabilized (change <= {self.threshold} for {self.n_stable} items)"
        )


class SEChangeStop(StoppingRule):
    """Stop when the standard error stops improving.

    Stops once the absolute change in the standard error between consecutive
    items is at most ``threshold`` for ``n_stable`` items in a row, so further
    items no longer buy appreciable precision. A non-finite standard error
    never counts as stable.

    Parameters
    ----------
    threshold : float, optional
        Maximum change in the standard error that counts as stable.
        Default is 0.01.
    n_stable : int, optional
        Number of consecutive stable changes required. Default is 1.
    """

    def __init__(self, threshold: float = 0.01, n_stable: int = 1):
        self.threshold: float = positive_real(threshold, "threshold")
        self.n_stable: int = stable_count(n_stable)
        self._changes = ChangeTracker()

    def should_stop(self, state: CATState) -> bool:
        changes = self._changes.update(state.standard_error, self.threshold)
        return changes >= self.n_stable

    def reset(self) -> None:
        """Reset the rule for a new examinee."""
        self._changes.reset()

    def get_reason(self) -> str:
        return f"SE stabilized (change <= {self.threshold} for {self.n_stable} items)"


class _RemainingInformationRule(StoppingRule):
    """Base for rules using the best remaining item's Fisher information.

    Every unadministered item of ``model`` counts as remaining; exposure and
    content filters are not applied. ``model`` must be the engine's model.
    """

    def __init__(self, model: BaseItemModel) -> None:
        if getattr(model, "n_factors", None) != 1:
            raise ValueError(
                f"{type(self).__name__} requires a unidimensional item model"
            )
        n_items = getattr(model, "n_items", None)
        integer(n_items, "model.n_items", minimum=1)
        self.model = model

    def _best_remaining_information(self, state: CATState) -> float | None:
        """Return the largest remaining information, or None for an empty pool."""
        remaining = set(range(self.model.n_items)).difference(state.items_administered)
        if not remaining:
            return None
        criteria = MaxFisherInformation().get_item_criteria(
            self.model, float(state.theta), remaining
        )
        return max(criteria.values())


class MinInformationStop(_RemainingInformationRule):
    """Stop when no remaining item is informative at the current estimate.

    This is the ``"minInfo"`` rule of catR: the test ends once the maximum
    Fisher information among unadministered items at the current ability
    estimate falls below ``threshold``. It ends tests early for examinees
    whom the pool cannot measure well, typically at extreme abilities.

    Parameters
    ----------
    model : BaseItemModel
        The engine's unidimensional item model.
    threshold : float, optional
        Minimum useful item information. Default is 0.1.
    """

    def __init__(self, model: BaseItemModel, threshold: float = 0.1) -> None:
        super().__init__(model)
        self.threshold: float = positive_real(threshold, "threshold")

    def should_stop(self, state: CATState) -> bool:
        best = self._best_remaining_information(state)
        return best is not None and best < self.threshold

    def get_reason(self) -> str:
        return f"Remaining item information below threshold ({self.threshold})"


class PredictedSEReductionStop(_RemainingInformationRule):
    """Stop when the next item is predicted to barely reduce the SE.

    The predicted standard error after administering the most informative
    remaining item ``j`` is approximated by adding its Fisher information at
    the current estimate to the current precision,
    ``1 / sqrt(SE**-2 + I_j(theta))``. The test ends
    once the predicted reduction ``SE - predicted`` falls below
    ``min_reduction`` (Choi, Grady, & Dodd, 2011). A non-finite standard
    error never stops the test.

    Parameters
    ----------
    model : BaseItemModel
        The engine's unidimensional item model.
    min_reduction : float, optional
        Smallest worthwhile reduction in the standard error. Default is 0.01.

    References
    ----------
    Choi, S. W., Grady, M. W., & Dodd, B. G. (2011). A new stopping rule for
    computerized adaptive testing. Educational and Psychological
    Measurement, 71(1), 37-53.
    """

    def __init__(self, model: BaseItemModel, min_reduction: float = 0.01) -> None:
        super().__init__(model)
        self.min_reduction: float = positive_real(min_reduction, "min_reduction")

    def should_stop(self, state: CATState) -> bool:
        standard_error = float(state.standard_error)
        if not math.isfinite(standard_error) or standard_error < 0.0:
            return False
        if standard_error == 0.0:
            return True
        best = self._best_remaining_information(state)
        if best is None:
            return False
        predicted = 1.0 / math.sqrt(standard_error**-2 + max(best, 0.0))
        return standard_error - predicted < self.min_reduction

    def get_reason(self) -> str:
        return f"Predicted SE reduction below threshold ({self.min_reduction})"


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
        cut_score_value = finite_real(cut_score, "cut_score")
        confidence_value = finite_real(confidence, "confidence")
        if not 0.0 < confidence_value < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.cut_score = cut_score_value
        self.confidence = confidence_value
        self._critical_z = float(ndtri(confidence_value))
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
        self._classification = None


@_register_native_defaults
class CombinedStop(StoppingRule):
    """Combine multiple stopping rules with logical operators.

    Parameters
    ----------
    rules : list[StoppingRule]
        List of stopping rules to combine.
    operator : {"and", "or"}, optional
        Logical operator for combining rules. Default is "or".
        - "or": Stop when ANY rule is satisfied; later rules are not
          evaluated once one stops.
        - "and": Stop when ALL rules are satisfied; every rule is evaluated.
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
        self.operator = combination_operator(operator)
        self.rules = list(rules)
        self.min_items = integer(min_items, "min_items", minimum=0)
        self._triggered_rule: StoppingRule | None = None

    def should_stop(self, state: CATState) -> bool:
        self._triggered_rule = None
        if state.n_items < self.min_items:
            return False
        self._triggered_rule = triggered_rule(self.rules, self.operator, state)
        return self._triggered_rule is not None

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
        "theta_change", "se_change", "classification", "combined". Rules
        that need the item model, such as :class:`MinInformationStop`, are
        constructed directly.
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
    rules: dict[str, type[StoppingRule]] = {
        "SE": StandardErrorStop,
        "max_items": MaxItemsStop,
        "min_items": MinItemsStop,
        "theta_change": ThetaChangeStop,
        "se_change": SEChangeStop,
        "classification": ClassificationStop,
        "combined": CombinedStop,
    }

    if method not in rules:
        valid = ", ".join(rules.keys())
        raise ValueError(f"Unknown stopping rule '{method}'. Valid options: {valid}")

    return rules[method](**kwargs)
