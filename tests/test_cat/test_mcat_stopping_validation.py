"""MCAT stopping rules share validation and composition with CAT rules."""

import numpy as np
import pytest

from mirt.cat.mcat_stopping import (
    AvgSEStop,
    CombinedMCATStop,
    CompositeClassificationStop,
    CovarianceDeterminantStop,
    CovarianceTraceStop,
    MaxItemsMCATStop,
    MaxSEStop,
    MCATStoppingRule,
    ThetaChangeMCATStop,
)
from mirt.cat.results import CATState, MCATState
from mirt.cat.stopping import CombinedStop, StoppingRule, ThetaChangeStop


def _state(theta=(0.0, 0.0), se=(0.5, 0.5), n_items=5) -> MCATState:
    se_values = np.asarray(se, dtype=float)
    return MCATState(
        theta=np.asarray(theta, dtype=float),
        covariance=np.diag(se_values**2),
        standard_error=se_values,
        items_administered=list(range(n_items)),
        responses=[1] * n_items,
        n_items=n_items,
    )


class _Recording(MCATStoppingRule):
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0

    def should_stop(self, state: MCATState) -> bool:
        self.calls += 1
        return self.result

    def get_reason(self) -> str:
        return "recorded"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CovarianceTraceStop(np.nan), "finite"),
        (lambda: CovarianceTraceStop(np.inf), "finite"),
        (lambda: CovarianceTraceStop(True), "finite"),
        (lambda: CovarianceTraceStop(0.0), "positive"),
        (lambda: CovarianceDeterminantStop("0.1"), "finite"),
        (lambda: CovarianceDeterminantStop(-1.0), "positive"),
        (lambda: MaxSEStop(np.inf), "finite"),
        (lambda: MaxSEStop(np.nan), "finite"),
        (lambda: AvgSEStop(True), "finite"),
        (lambda: AvgSEStop(0.0), "positive"),
        (lambda: MaxItemsMCATStop(2.5), "integer"),
        (lambda: MaxItemsMCATStop(True), "integer"),
        (lambda: MaxItemsMCATStop(0), "positive"),
        (lambda: ThetaChangeMCATStop(np.nan), "finite"),
        (lambda: ThetaChangeMCATStop(n_stable=1.5), "integer"),
        (lambda: ThetaChangeMCATStop(n_stable=0), "at least 1"),
        (lambda: ThetaChangeMCATStop(n_stable=True), "integer"),
        (lambda: CombinedMCATStop([MaxItemsMCATStop(1)], min_items=-2), "non-neg"),
        (lambda: CombinedMCATStop([MaxItemsMCATStop(1)], min_items=1.5), "integer"),
        (lambda: CombinedMCATStop([MaxItemsMCATStop(1)], operator="xor"), "'or'"),
        (lambda: CombinedMCATStop([]), "At least one rule"),
        (
            lambda: CompositeClassificationStop([1.0], 0.0, np.nan),
            "confidence must be strictly between 0.5 and 1",
        ),
        (
            lambda: CompositeClassificationStop([1.0], 0.0, "0.95"),
            "confidence must be strictly between 0.5 and 1",
        ),
        (
            lambda: CompositeClassificationStop([1.0], np.inf),
            "cut_score must be finite",
        ),
    ],
)
def test_mcat_rules_reject_invalid_configuration(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_mcat_rules_accept_numpy_scalars():
    assert MaxItemsMCATStop(np.int64(5)).max_items == 5
    assert type(MaxItemsMCATStop(np.int64(5)).max_items) is int
    rule = ThetaChangeMCATStop(np.float64(0.05), n_stable=np.int64(2))
    assert (rule.threshold, rule.n_stable) == (0.05, 2)
    assert CovarianceTraceStop(np.float32(0.5)).threshold == pytest.approx(0.5)
    assert CombinedMCATStop([MaxSEStop()], min_items=np.int64(3)).min_items == 3


def test_threshold_rules_return_python_booleans():
    state = _state(se=(0.2, 0.25))
    for rule in (
        CovarianceTraceStop(0.5),
        CovarianceDeterminantStop(0.01),
        MaxSEStop(0.3),
        AvgSEStop(0.3),
    ):
        assert rule.should_stop(state) is True


def test_base_reset_is_a_no_op_for_custom_rules():
    custom = _Recording(False)
    custom.reset()

    combined = CombinedMCATStop([custom, MaxItemsMCATStop(10)])
    combined.reset()

    assert custom.calls == 0


def test_combined_reason_is_not_stale_after_a_non_stopping_state():
    combined = CombinedMCATStop([MaxSEStop(0.3), MaxItemsMCATStop(10)])
    assert combined.should_stop(_state(se=(0.2, 0.2))) is True
    assert combined.get_reason().startswith("All SE thresholds")

    assert combined.should_stop(_state(se=(0.9, 0.9))) is False
    assert combined.get_reason() == "Combined rule (or)"


def test_combined_or_short_circuits_like_cat_rules():
    first = _Recording(True)
    later = _Recording(False)

    assert CombinedMCATStop([first, later]).should_stop(_state()) is True
    assert (first.calls, later.calls) == (1, 0)


def test_combined_and_evaluates_every_rule_and_reports_the_first():
    first = _Recording(True)
    later = _Recording(True)
    combined = CombinedMCATStop([first, later], operator="and")

    assert combined.should_stop(_state()) is True
    assert (first.calls, later.calls) == (1, 1)
    assert combined.get_reason() == "recorded"


def test_combined_reset_reaches_nested_theta_trackers():
    tracker = ThetaChangeMCATStop(threshold=0.1, n_stable=1)
    combined = CombinedMCATStop([tracker])
    combined.should_stop(_state(theta=(0.0, 0.0)))

    combined.reset()

    assert combined.should_stop(_state(theta=(0.0, 0.0))) is False
    assert combined.should_stop(_state(theta=(0.05, 0.0))) is True


def _reference_theta_change(sequence, threshold, n_stable):
    """Stopping decisions of the original per-family ThetaChange rules."""
    decisions = []
    last = None
    count = 0
    for theta in sequence:
        if last is None:
            last = np.array(theta, dtype=float)
            decisions.append(False)
            continue
        change = np.max(np.abs(np.asarray(theta, dtype=float) - last))
        last = np.array(theta, dtype=float)
        count = count + 1 if change <= threshold else 0
        decisions.append(count >= n_stable)
    return decisions


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_stable", [1, 2, 3])
def test_theta_change_rules_match_original_decisions(seed, n_stable):
    rng = np.random.default_rng(seed)
    steps = rng.choice([0.0, 0.004, 0.02, 0.3], size=(40, 2)) * rng.choice(
        [-1.0, 1.0], size=(40, 2)
    )
    multidimensional = np.cumsum(steps, axis=0)
    multidimensional[17] = np.nan
    scalar = multidimensional[:, 0]

    mcat_rule = ThetaChangeMCATStop(threshold=0.01, n_stable=n_stable)
    cat_rule = ThetaChangeStop(threshold=0.01, n_stable=n_stable)
    mcat = [mcat_rule.should_stop(_state(theta=theta)) for theta in multidimensional]
    cat = [
        cat_rule.should_stop(CATState(theta=float(theta), standard_error=0.5))
        for theta in scalar
    ]

    assert mcat == _reference_theta_change(multidimensional, 0.01, n_stable)
    assert cat == _reference_theta_change(scalar, 0.01, n_stable)


def test_theta_tracker_copies_state_arrays():
    rule = ThetaChangeMCATStop(threshold=0.01, n_stable=1)
    theta = np.array([0.0, 0.0])
    state = _state(theta=theta)
    rule.should_stop(state)

    state.theta += 1.0

    assert rule.should_stop(state) is False


def test_cat_and_mcat_families_stay_separate():
    assert isinstance(CombinedMCATStop([MaxSEStop()]), MCATStoppingRule)
    assert not isinstance(CombinedMCATStop([MaxSEStop()]), StoppingRule)
    assert not isinstance(CombinedStop([ThetaChangeStop()]), MCATStoppingRule)
