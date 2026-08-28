"""Numerical and validation contracts for measurement invariance helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import stats

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.multigroup.invariance import (
    InvarianceSpec,
    compute_delta_fit,
    get_invariance_hierarchy_pairs,
    invariance_lrt,
    parse_invariance,
)
from mirt.multigroup.invariance import (
    test_invariance_step as run_invariance_step,
)
from mirt.multigroup.model import MultigroupModel


def _fit_result(
    log_likelihood: float,
    n_parameters: int,
    *,
    aic: float = 100.0,
    bic: float = 110.0,
) -> Any:
    return SimpleNamespace(
        log_likelihood=log_likelihood,
        n_parameters=n_parameters,
        aic=aic,
        bic=bic,
    )


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("configural", []),
        ("metric", ["discrimination"]),
        ("scalar", ["difficulty", "discrimination"]),
        ("strict", ["difficulty", "guessing", "discrimination"]),
    ],
)
def test_shared_parameters_follow_model_order(level: str, expected: list[str]) -> None:
    model = SimpleNamespace(
        parameter_names=["difficulty", "guessing", "discrimination"]
    )

    shared = InvarianceSpec(level=level).get_shared_parameters(model)

    assert shared == expected


def test_parse_invariance_merges_parameter_aliases_in_appearance_order() -> None:
    specification = parse_invariance(
        "scalar",
        {
            "slopes": [2, 0],
            "discrimination": [0, 1],
            "thresholds": [3],
            "difficulty": [4, 3],
        },
    )

    assert specification.free_discrimination == [2, 0, 1]
    assert specification.free_intercepts == [3, 4]


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        ({"level": "unknown"}, "Unknown invariance level"),
        (
            {"level": "configural", "free_discrimination": [0]},
            "not applicable",
        ),
        (
            {"level": "metric", "free_intercepts": [0]},
            "scalar or strict",
        ),
        (
            {"level": "metric", "free_discrimination": [-1]},
            "non-negative",
        ),
        (
            {"level": "metric", "free_discrimination": [0, 0]},
            "duplicate",
        ),
        (
            {"level": "metric", "free_discrimination": [True]},
            "integer",
        ),
    ],
)
def test_invariance_spec_rejects_invalid_or_ignored_inputs(
    specification: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        InvarianceSpec(**specification)


def test_parse_invariance_rejects_unknown_partial_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown free_items parameter"):
        parse_invariance("scalar", {"not_a_parameter": [0]})


def test_parse_invariance_rejects_second_partial_specification() -> None:
    specification = InvarianceSpec("metric", free_discrimination=[0])

    with pytest.raises(ValueError, match="cannot be combined"):
        parse_invariance(specification, {"discrimination": [1]})


def test_apply_rejects_out_of_range_items_before_mutating_model() -> None:
    model = MultigroupModel(TwoParameterLogistic(2), n_groups=2)
    specification = InvarianceSpec("metric", free_discrimination=[2])

    with pytest.raises(ValueError, match="indices below 2"):
        specification.apply_to_model(model)

    assert all(
        not model.is_parameter_shared(parameter) for parameter in model.parameter_names
    )


def test_apply_preserves_requested_partial_invariance() -> None:
    model = MultigroupModel(TwoParameterLogistic(3), n_groups=2)
    specification = InvarianceSpec("scalar", free_discrimination=[1])

    specification.apply_to_model(model)

    assert model.get_shared_items("discrimination") == [0, 2]
    assert model.get_free_items("discrimination") == [1]
    assert model.get_shared_items("difficulty") == [0, 1, 2]


def test_lrt_preserves_extreme_nonzero_tail_probability() -> None:
    free = _fit_result(0.0, 2)
    constrained = _fit_result(-50.0, 1)

    result = invariance_lrt(constrained, free)

    assert result["chi2"] == 100.0
    assert result["df"] == 1
    assert result["p_value"] == pytest.approx(stats.chi2.sf(100.0, 1))
    assert result["p_value"] > 0.0


def test_lrt_tolerates_small_optimization_roundoff() -> None:
    free = _fit_result(-100.0, 2)
    constrained = _fit_result(-99.9995, 1)

    result = invariance_lrt(constrained, free)

    assert result["chi2"] == 0.0
    assert result["p_value"] == 1.0


@pytest.mark.parametrize(
    ("constrained", "free", "message"),
    [
        (_fit_result(np.nan, 1), _fit_result(-10.0, 2), "finite"),
        (_fit_result(-9.0, 1), _fit_result(-10.0, 2), "not be nested"),
        (_fit_result(-11.0, 2), _fit_result(-10.0, 2), "fewer parameters"),
        (_fit_result(-11.0, 1), _fit_result(-10.0, 2.5), "non-negative integer"),
        (_fit_result(-11.0, 1), _fit_result(-10.0, True), "non-negative integer"),
    ],
)
def test_lrt_rejects_invalid_comparisons(
    constrained: Any,
    free: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        invariance_lrt(constrained, free)


def test_delta_fit_rejects_nonfinite_statistics() -> None:
    constrained = _fit_result(-11.0, 1, aic=np.inf)
    free = _fit_result(-10.0, 2)

    with pytest.raises(ValueError, match="fit statistics must be finite"):
        compute_delta_fit(constrained, free)


@pytest.mark.parametrize(
    ("comparison_name", "alpha", "message"),
    [
        ("", 0.05, "non-empty"),
        ("metric vs scalar", 0.0, "between 0 and 1"),
        ("metric vs scalar", np.nan, "finite"),
        ("metric vs scalar", True, "finite"),
    ],
)
def test_invariance_step_validates_reporting_inputs(
    comparison_name: str,
    alpha: float,
    message: str,
) -> None:
    constrained = _fit_result(-11.0, 1, aic=24.0, bic=26.0)
    free = _fit_result(-10.0, 2, aic=24.0, bic=28.0)

    with pytest.raises(ValueError, match=message):
        run_invariance_step(
            constrained,
            free,
            comparison_name,
            alpha=alpha,
        )


def test_hierarchy_pairs_are_complete_and_ordered() -> None:
    assert get_invariance_hierarchy_pairs() == [
        ("configural", "metric"),
        ("metric", "scalar"),
        ("scalar", "strict"),
    ]
