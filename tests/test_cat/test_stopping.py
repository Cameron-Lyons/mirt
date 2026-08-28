"""Tests for stopping rules."""

import numpy as np
import pytest
from scipy import stats

from mirt.cat.results import CATState
from mirt.cat.stopping import (
    ClassificationStop,
    CombinedStop,
    MaxItemsStop,
    MinItemsStop,
    StandardErrorStop,
    StoppingRule,
    ThetaChangeStop,
    create_stopping_rule,
)


def make_state(
    theta: float = 0.0,
    se: float = 0.5,
    n_items: int = 5,
    is_complete: bool = False,
) -> CATState:
    """Helper to create CATState for testing."""
    return CATState(
        theta=theta,
        standard_error=se,
        items_administered=list(range(n_items)),
        responses=[1] * n_items,
        n_items=n_items,
        is_complete=is_complete,
        next_item=None if is_complete else n_items,
    )


class _RecordingStop(StoppingRule):
    """Small stateful rule used to verify composition semantics."""

    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0
        self.resets = 0

    def should_stop(self, state: CATState) -> bool:
        self.calls += 1
        return self.result

    def get_reason(self) -> str:
        return "recorded"

    def reset(self) -> None:
        self.resets += 1


class TestStandardErrorStop:
    """Tests for StandardErrorStop."""

    def test_initialization(self):
        """Test initialization with threshold."""
        se_stop = StandardErrorStop(threshold=0.3)
        assert se_stop.threshold == 0.3

    def test_initialization_invalid_threshold(self):
        """Test that non-positive threshold raises error."""
        with pytest.raises(ValueError, match="positive"):
            StandardErrorStop(threshold=0)

        with pytest.raises(ValueError, match="positive"):
            StandardErrorStop(threshold=-0.1)

    def test_should_stop_below_threshold(self):
        """Test stop when SE is below threshold."""
        se_stop = StandardErrorStop(threshold=0.3)
        state = make_state(se=0.25)

        assert se_stop.should_stop(state) is True

    def test_should_not_stop_above_threshold(self):
        """Test no stop when SE is above threshold."""
        se_stop = StandardErrorStop(threshold=0.3)
        state = make_state(se=0.5)

        assert se_stop.should_stop(state) is False

    def test_should_stop_at_threshold(self):
        """Test stop when SE equals threshold."""
        se_stop = StandardErrorStop(threshold=0.3)
        state = make_state(se=0.3)

        assert se_stop.should_stop(state) is True

    def test_get_reason(self):
        """Test get_reason returns appropriate string."""
        se_stop = StandardErrorStop(threshold=0.3)
        reason = se_stop.get_reason()

        assert "SE" in reason
        assert "0.3" in reason


class TestMaxItemsStop:
    """Tests for MaxItemsStop."""

    def test_initialization(self):
        """Test initialization with max_items."""
        max_stop = MaxItemsStop(max_items=10)
        assert max_stop.max_items == 10

    def test_initialization_invalid_max_items(self):
        """Test that non-positive max_items raises error."""
        with pytest.raises(ValueError, match="positive"):
            MaxItemsStop(max_items=0)

        with pytest.raises(ValueError, match="positive"):
            MaxItemsStop(max_items=-5)

    def test_should_stop_at_max(self):
        """Test stop when n_items equals max_items."""
        max_stop = MaxItemsStop(max_items=5)
        state = make_state(n_items=5)

        assert max_stop.should_stop(state) is True

    def test_should_stop_above_max(self):
        """Test stop when n_items exceeds max_items."""
        max_stop = MaxItemsStop(max_items=5)
        state = make_state(n_items=7)

        assert max_stop.should_stop(state) is True

    def test_should_not_stop_below_max(self):
        """Test no stop when n_items is below max_items."""
        max_stop = MaxItemsStop(max_items=10)
        state = make_state(n_items=5)

        assert max_stop.should_stop(state) is False

    def test_get_reason(self):
        """Test get_reason returns appropriate string."""
        max_stop = MaxItemsStop(max_items=10)
        reason = max_stop.get_reason()

        assert "Maximum" in reason or "max" in reason.lower()
        assert "10" in reason


class TestMinItemsStop:
    """Tests for MinItemsStop."""

    def test_initialization(self):
        """Test initialization with min_items."""
        min_stop = MinItemsStop(min_items=3)
        assert min_stop.min_items == 3

    def test_initialization_invalid_min_items(self):
        """Test that negative min_items raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            MinItemsStop(min_items=-1)

    def test_should_stop_never_true(self):
        """Test that should_stop is always False."""
        min_stop = MinItemsStop(min_items=3)

        for n in [0, 1, 2, 3, 4, 10]:
            state = make_state(n_items=n)
            assert min_stop.should_stop(state) is False

    def test_is_satisfied_below_min(self):
        """Test is_satisfied when below minimum."""
        min_stop = MinItemsStop(min_items=5)
        state = make_state(n_items=3)

        assert min_stop.is_satisfied(state) is False

    def test_is_satisfied_at_min(self):
        """Test is_satisfied when at minimum."""
        min_stop = MinItemsStop(min_items=5)
        state = make_state(n_items=5)

        assert min_stop.is_satisfied(state) is True

    def test_is_satisfied_above_min(self):
        """Test is_satisfied when above minimum."""
        min_stop = MinItemsStop(min_items=5)
        state = make_state(n_items=10)

        assert min_stop.is_satisfied(state) is True


class TestThetaChangeStop:
    """Tests for ThetaChangeStop."""

    def test_initialization(self):
        """Test initialization with parameters."""
        theta_stop = ThetaChangeStop(threshold=0.05, n_stable=2)
        assert theta_stop.threshold == 0.05
        assert theta_stop.n_stable == 2

    def test_initialization_invalid_threshold(self):
        """Test that non-positive threshold raises error."""
        with pytest.raises(ValueError, match="positive"):
            ThetaChangeStop(threshold=0)

    def test_initialization_invalid_n_stable(self):
        """Test that invalid n_stable raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            ThetaChangeStop(n_stable=0)

    def test_should_not_stop_first_item(self):
        """Test no stop on first item."""
        theta_stop = ThetaChangeStop(threshold=0.01, n_stable=1)
        state = make_state(theta=0.0, n_items=1)

        assert theta_stop.should_stop(state) is False

    def test_should_stop_stable_theta(self):
        """Test stop when theta is stable."""
        theta_stop = ThetaChangeStop(threshold=0.1, n_stable=2)

        theta_stop.should_stop(make_state(theta=0.0, n_items=1))
        theta_stop.should_stop(make_state(theta=0.05, n_items=2))
        result = theta_stop.should_stop(make_state(theta=0.08, n_items=3))

        assert result is True

    def test_should_not_stop_changing_theta(self):
        """Test no stop when theta is changing."""
        theta_stop = ThetaChangeStop(threshold=0.01, n_stable=3)

        theta_stop.should_stop(make_state(theta=0.0, n_items=1))
        theta_stop.should_stop(make_state(theta=0.5, n_items=2))
        result = theta_stop.should_stop(make_state(theta=1.0, n_items=3))

        assert result is False

    def test_reset(self):
        """Test reset clears state."""
        theta_stop = ThetaChangeStop(threshold=0.1, n_stable=1)

        theta_stop.should_stop(make_state(theta=0.0, n_items=1))
        theta_stop.should_stop(make_state(theta=0.05, n_items=2))

        theta_stop.reset()

        assert theta_stop._stable_count == 0
        assert theta_stop._last_theta is None


class TestClassificationStop:
    """Tests for ClassificationStop."""

    def test_initialization(self):
        """Test initialization with parameters."""
        class_stop = ClassificationStop(cut_score=0.5, confidence=0.90)
        assert class_stop.cut_score == 0.5
        assert class_stop.confidence == 0.90

    def test_initialization_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            ClassificationStop(cut_score=0.0, confidence=0.0)

        with pytest.raises(ValueError, match="between 0 and 1"):
            ClassificationStop(cut_score=0.0, confidence=1.0)

    def test_should_stop_confident_above(self):
        """Test stop when confidently above cut score."""
        class_stop = ClassificationStop(cut_score=0.0, confidence=0.95)
        state = make_state(theta=2.0, se=0.1)

        assert class_stop.should_stop(state) is True
        assert class_stop._classification == "above"

    def test_should_stop_confident_below(self):
        """Test stop when confidently below cut score."""
        class_stop = ClassificationStop(cut_score=0.0, confidence=0.95)
        state = make_state(theta=-2.0, se=0.1)

        assert class_stop.should_stop(state) is True
        assert class_stop._classification == "below"

    def test_should_not_stop_uncertain(self):
        """Test no stop when classification is uncertain."""
        class_stop = ClassificationStop(cut_score=0.0, confidence=0.95)
        state = make_state(theta=0.1, se=1.0)

        assert class_stop.should_stop(state) is False

    def test_get_reason(self):
        """Test get_reason returns appropriate string."""
        class_stop = ClassificationStop(cut_score=0.5, confidence=0.95)
        class_stop.should_stop(make_state(theta=2.0, se=0.1))

        reason = class_stop.get_reason()
        assert "Classification" in reason or "classification" in reason.lower()

    @pytest.mark.parametrize(
        ("theta", "se", "confidence"),
        [
            (-2.0, 0.3, 0.99),
            (-0.2, 0.3, 0.95),
            (0.0, 0.3, 0.5),
            (0.2, 0.3, 0.75),
            (2.0, 0.3, 0.999),
        ],
    )
    def test_precomputed_threshold_matches_normal_probability(
        self,
        theta,
        se,
        confidence,
    ):
        rule = ClassificationStop(cut_score=0.0, confidence=confidence)
        expected = stats.norm.cdf(abs(theta) / se) >= confidence

        assert rule.should_stop(make_state(theta=theta, se=se)) is bool(expected)

    def test_zero_standard_error_handles_certain_and_boundary_scores(self):
        rule = ClassificationStop(cut_score=0.0, confidence=0.95)

        assert rule.should_stop(make_state(theta=1.0, se=0.0)) is True
        rule.reset()
        assert rule.should_stop(make_state(theta=0.0, se=0.0)) is False

    def test_infinite_standard_error_is_not_confident(self):
        rule = ClassificationStop(cut_score=0.0, confidence=0.95)

        assert rule.should_stop(make_state(theta=10.0, se=np.inf)) is False

    def test_reset_clears_prior_classification(self):
        rule = ClassificationStop(cut_score=0.0, confidence=0.95)
        rule.should_stop(make_state(theta=2.0, se=0.1))

        rule.reset()

        assert rule._classification is None
        assert rule._triggered is False


class TestCombinedStop:
    """Tests for CombinedStop."""

    def test_initialization(self):
        """Test initialization with rules."""
        rule1 = StandardErrorStop(0.3)
        rule2 = MaxItemsStop(10)

        combined = CombinedStop([rule1, rule2], operator="or")

        assert len(combined.rules) == 2
        assert combined.operator == "or"

    def test_initialization_empty_rules_raises_error(self):
        """Test that empty rules raises error."""
        with pytest.raises(ValueError, match="At least one rule"):
            CombinedStop([], operator="or")

    def test_initialization_invalid_operator_raises_error(self):
        """Test that invalid operator raises error."""
        rule = StandardErrorStop(0.3)

        with pytest.raises(ValueError, match="'and' or 'or'"):
            CombinedStop([rule], operator="xor")

    def test_or_operator_any_triggers(self):
        """Test OR operator stops when any rule triggers."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(10)

        combined = CombinedStop([se_stop, max_stop], operator="or")
        state = make_state(se=0.25, n_items=5)

        assert combined.should_stop(state) is True

    def test_or_operator_none_triggers(self):
        """Test OR operator doesn't stop when no rule triggers."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(10)

        combined = CombinedStop([se_stop, max_stop], operator="or")
        state = make_state(se=0.5, n_items=5)

        assert combined.should_stop(state) is False

    def test_and_operator_all_trigger(self):
        """Test AND operator stops when all rules trigger."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(5)

        combined = CombinedStop([se_stop, max_stop], operator="and")
        state = make_state(se=0.25, n_items=5)

        assert combined.should_stop(state) is True

    def test_and_operator_not_all_trigger(self):
        """Test AND operator doesn't stop when not all rules trigger."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(10)

        combined = CombinedStop([se_stop, max_stop], operator="and")
        state = make_state(se=0.25, n_items=5)

        assert combined.should_stop(state) is False

    def test_min_items_enforced(self):
        """Test that min_items is enforced."""
        se_stop = StandardErrorStop(0.3)
        combined = CombinedStop([se_stop], operator="or", min_items=5)

        state = make_state(se=0.1, n_items=3)
        assert combined.should_stop(state) is False

        state = make_state(se=0.1, n_items=5)
        assert combined.should_stop(state) is True

    def test_get_reason_returns_triggered_rule_reason(self):
        """Test get_reason returns the triggered rule's reason."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(10)

        combined = CombinedStop([se_stop, max_stop], operator="or")
        combined.should_stop(make_state(se=0.25, n_items=5))

        reason = combined.get_reason()
        assert "SE" in reason

    def test_or_short_circuits_without_mutating_later_rules(self):
        first = _RecordingStop(True)
        later = _RecordingStop(False)
        combined = CombinedStop([first, later], operator="or")

        assert combined.should_stop(make_state()) is True
        assert first.calls == 1
        assert later.calls == 0

    def test_or_short_circuit_preserves_theta_tracker_for_next_state(self):
        tracker = ThetaChangeStop(threshold=0.1, n_stable=1)
        combined = CombinedStop(
            [StandardErrorStop(0.3), tracker],
            operator="or",
        )

        assert combined.should_stop(make_state(theta=0.0, se=0.1)) is True
        assert tracker._last_theta is None
        assert tracker._stable_count == 0

    def test_and_evaluates_all_rules_to_preserve_state_tracking(self):
        first = _RecordingStop(False)
        later = _RecordingStop(True)
        combined = CombinedStop([first, later], operator="and")

        assert combined.should_stop(make_state()) is False
        assert first.calls == 1
        assert later.calls == 1

    def test_reset_propagates_to_every_nested_rule(self):
        first = _RecordingStop(True)
        nested_rule = _RecordingStop(False)
        nested = CombinedStop([nested_rule])
        combined = CombinedStop([first, nested], operator="or")
        combined.should_stop(make_state())

        combined.reset()

        assert first.resets == 1
        assert nested_rule.resets == 1
        assert combined._triggered_rule is None

    def test_reset_prevents_theta_state_leaking_between_sessions(self):
        tracker = ThetaChangeStop(threshold=0.1, n_stable=2)
        combined = CombinedStop([tracker])
        combined.should_stop(make_state(theta=0.0, se=1.0, n_items=1))
        combined.should_stop(make_state(theta=0.05, se=1.0, n_items=2))
        assert tracker._stable_count == 1

        combined.reset()

        assert combined.should_stop(make_state(theta=0.0, se=1.0, n_items=1)) is False
        assert tracker._stable_count == 0


class TestCreateStoppingRule:
    """Tests for create_stopping_rule factory."""

    @pytest.mark.parametrize(
        "method,kwargs,expected_class",
        [
            ("SE", {"threshold": 0.3}, StandardErrorStop),
            ("max_items", {"max_items": 10}, MaxItemsStop),
            ("min_items", {"min_items": 3}, MinItemsStop),
            ("theta_change", {"threshold": 0.01}, ThetaChangeStop),
            ("classification", {"cut_score": 0.0}, ClassificationStop),
        ],
    )
    def test_create_valid_rules(self, method, kwargs, expected_class):
        """Test creating valid rules."""
        rule = create_stopping_rule(method, **kwargs)
        assert isinstance(rule, expected_class)

    def test_create_invalid_raises_error(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown stopping rule"):
            create_stopping_rule("invalid_rule")

    def test_create_combined(self):
        """Test creating combined rule."""
        se_stop = StandardErrorStop(0.3)
        max_stop = MaxItemsStop(10)

        combined = create_stopping_rule(
            "combined", rules=[se_stop, max_stop], operator="or"
        )

        assert isinstance(combined, CombinedStop)
        assert len(combined.rules) == 2


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StandardErrorStop(np.nan), "finite"),
        (lambda: StandardErrorStop(np.inf), "finite"),
        (lambda: StandardErrorStop(True), "finite"),
        (lambda: MaxItemsStop(2.5), "integer"),
        (lambda: MaxItemsStop(True), "integer"),
        (lambda: MinItemsStop(1.5), "integer"),
        (lambda: MinItemsStop(True), "integer"),
        (lambda: ThetaChangeStop(np.nan), "finite"),
        (lambda: ThetaChangeStop(n_stable=1.5), "integer"),
        (lambda: ClassificationStop(np.nan), "cut_score"),
        (lambda: ClassificationStop(0.0, np.nan), "confidence"),
        (lambda: CombinedStop([MaxItemsStop(1)], min_items=-1), "non-negative"),
        (lambda: CombinedStop([MaxItemsStop(1)], min_items=1.5), "integer"),
    ],
)
def test_stopping_rules_reject_invalid_configuration(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
