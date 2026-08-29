"""Tests for batched item-exposure monitoring."""

import numpy as np
import pytest

from mirt.cat.exposure import ExposureReport, SympsonHetter
from mirt.models import TwoParameterLogistic


def test_record_sessions_matches_incremental_accounting() -> None:
    selections = np.array(
        [
            [0, 2, 4, -1],
            [1, 2, -1, -1],
            [0, 3, 4, 5],
        ]
    )
    batched = SympsonHetter()
    incremental = SympsonHetter()

    batched.record_sessions(selections, n_items=6)
    for session in selections:
        for item_idx in session[session >= 0]:
            incremental.update(int(item_idx))
        incremental.reset()

    assert batched.n_examinees == incremental.n_examinees == 3
    assert batched.get_exposure_rates() == incremental.get_exposure_rates()


def test_record_sessions_accepts_one_session_and_custom_padding() -> None:
    control = SympsonHetter()

    control.record_sessions([2, 5, -9], missing_value=-9, n_items=6)

    assert control.n_examinees == 1
    assert control.get_exposure_rates() == {2: 1.0, 5: 1.0}
    np.testing.assert_array_equal(control.exposure_report().item_indices, [2, 5])


@pytest.mark.parametrize(
    ("selections", "kwargs", "error", "message"),
    [
        ([], {}, ValueError, "non-empty"),
        (np.empty((0, 2), dtype=int), {}, ValueError, "non-empty"),
        ([[[0]]], {}, ValueError, "sessions"),
        ([0.0, 1.0], {}, TypeError, "integers"),
        ([True, False], {}, TypeError, "integers"),
        ([0, -2], {}, ValueError, "missing_value"),
        ([0, 0], {}, ValueError, "repeated"),
        ([0, 3], {"n_items": 3}, IndexError, r"\[0, 3\)"),
        ([0], {"n_items": 0}, ValueError, "positive"),
        ([0], {"missing_value": 0}, ValueError, "negative"),
        ([0], {"missing_value": True}, TypeError, "integer"),
    ],
)
def test_record_sessions_validates_before_mutating(
    selections,
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    control = SympsonHetter()

    with pytest.raises(error, match=message):
        control.record_sessions(selections, **kwargs)

    assert control.n_examinees == 0
    assert control.get_exposure_rates() == {}


def test_exposure_report_contains_counts_rates_and_wilson_intervals() -> None:
    control = SympsonHetter(target_rate=0.25)
    selections = np.full((100, 2), -1, dtype=int)
    selections[:50, 0] = 0
    selections[:10, 1] = 1
    control.record_sessions(selections, n_items=3)

    report = control.exposure_report(n_items=3)

    assert isinstance(report, ExposureReport)
    assert report.n_examinees == 100
    np.testing.assert_array_equal(report.item_indices, [0, 1, 2])
    np.testing.assert_array_equal(report.selection_counts, [50, 10, 0])
    np.testing.assert_allclose(report.exposure_rates, [0.5, 0.1, 0.0])
    assert report.confidence_lower[0] == pytest.approx(0.40383153)
    assert report.confidence_upper[0] == pytest.approx(0.59616847)
    np.testing.assert_array_equal(report.above_target, [True, False, False])
    np.testing.assert_array_equal(
        report.significantly_above_target,
        [True, False, False],
    )
    np.testing.assert_array_equal(report.overexposed_items, [0])
    np.testing.assert_array_equal(report.statistically_overexposed_items, [0])
    assert report.max_exposure_rate == pytest.approx(0.5)


def test_exposure_report_infers_pool_and_tracks_eligibility() -> None:
    model = TwoParameterLogistic(n_items=2)
    control = SympsonHetter(exposure_params={0: 1.0, 1: 0.0}, seed=42)

    for _ in range(4):
        eligible = control.filter_items({0, 1}, model, theta=0.0)
        assert eligible == {0}
        control.update(0)
        control.reset()

    report = control.exposure_report()

    np.testing.assert_array_equal(report.opportunity_counts, [4, 4])
    np.testing.assert_array_equal(report.eligibility_counts, [4, 0])
    np.testing.assert_allclose(report.eligibility_rates, [1.0, 0.0])
    np.testing.assert_allclose(report.exposure_rates, [1.0, 0.0])


def test_empty_exposure_report_has_honest_uncertainty() -> None:
    report = SympsonHetter().exposure_report(n_items=2, confidence_level=0.9)

    np.testing.assert_array_equal(report.selection_counts, [0, 0])
    np.testing.assert_allclose(report.exposure_rates, [0.0, 0.0])
    np.testing.assert_allclose(report.confidence_lower, [0.0, 0.0])
    np.testing.assert_allclose(report.confidence_upper, [1.0, 1.0])
    assert report.max_exposure_rate == 0.0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_items": 0}, ValueError, "positive"),
        ({"confidence_level": 0.0}, ValueError, r"\(0, 1\)"),
        ({"confidence_level": 1.0}, ValueError, r"\(0, 1\)"),
        ({"confidence_level": np.nan}, ValueError, "finite"),
        (
            {"confidence_level": np.nextafter(1.0, 0.0)},
            ValueError,
            "too close",
        ),
        ({"confidence_level": True}, TypeError, "real"),
    ],
)
def test_exposure_report_validates_controls(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        SympsonHetter().exposure_report(**kwargs)


def test_exposure_report_rejects_truncated_pool_and_incomplete_sessions() -> None:
    control = SympsonHetter(exposure_params={3: 0.5})
    with pytest.raises(ValueError, match="every recorded item"):
        control.exposure_report(n_items=3)

    control = SympsonHetter()
    control.update(0)
    with pytest.raises(ValueError, match="exceed examinee sessions"):
        control.exposure_report()
