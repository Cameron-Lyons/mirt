"""Behavioral and performance contracts for CAT exposure control."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat import MCATEngine
from mirt.cat.exposure import (
    NoExposureControl,
    ProgressiveRestricted,
    Randomesque,
    SympsonHetter,
    create_exposure_control,
)
from mirt.models.multidimensional import MultidimensionalModel


class _InformationModel:
    def __init__(self, information: list[float], n_factors: int = 1) -> None:
        self.values = np.asarray(information, dtype=np.float64)
        self.n_items = len(information)
        self.n_factors = n_factors
        self.calls = 0
        self.last_theta: np.ndarray | None = None

    def information(
        self,
        theta: np.ndarray,
        item_idx: int | None = None,
    ) -> np.ndarray:
        assert item_idx is None
        self.calls += 1
        self.last_theta = theta.copy()
        return self.values[None, :]


@pytest.mark.parametrize("target_rate", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_sympson_hetter_validates_target_rate(target_rate: float) -> None:
    with pytest.raises(ValueError):
        SympsonHetter(target_rate=target_rate)


@pytest.mark.parametrize("target_rate", [True, "0.25"])
def test_sympson_hetter_requires_numeric_target_rate(target_rate: Any) -> None:
    with pytest.raises(TypeError, match="real number"):
        SympsonHetter(target_rate=target_rate)


@pytest.mark.parametrize(
    "parameters",
    [
        {0: -0.1},
        {0: 1.1},
        {0: np.nan},
        {-1: 0.5},
        {0.5: 0.5},
        np.array([[0.5, 0.8]]),
    ],
)
def test_sympson_hetter_validates_parameters(parameters: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        SympsonHetter(parameters)


def test_sympson_hetter_owns_and_exposes_parameter_copy() -> None:
    parameters = np.array([0.25, 0.75])
    control = SympsonHetter(parameters)
    parameters[:] = 1.0
    returned = control.exposure_parameters
    returned[0] = 1.0

    assert control.exposure_parameters == {0: 0.25, 1: 0.75}


def test_sympson_hetter_tracks_opportunity_and_eligibility_rates() -> None:
    control = SympsonHetter({0: 1.0, 1: 0.0}, seed=12)
    model = _InformationModel([1.0, 1.0])

    for _ in range(20):
        assert control.filter_items({1, 0}, model, 0.0) == {0}

    assert control.get_eligibility_rates() == {0: 1.0, 1: 0.0}


def test_sympson_hetter_records_forced_fallback_as_eligible() -> None:
    control = SympsonHetter({0: 0.0}, seed=4)
    model = _InformationModel([1.0])

    assert control.filter_items({0}, model, 0.0) == {0}
    assert control.get_eligibility_rates() == {0: 1.0}


def test_sympson_hetter_reports_known_zero_exposure_items() -> None:
    control = SympsonHetter({0: 0.8, 1: 0.8})

    assert control.get_exposure_rates() == {0: 0.0, 1: 0.0}
    control.update(0)
    control.reset()
    control.reset()

    assert control.n_examinees == 2
    assert control.get_exposure_rates() == {0: 0.5, 1: 0.0}


def test_sympson_hetter_calibration_adjusts_overused_and_unused_items() -> None:
    control = SympsonHetter({0: 0.8, 1: 0.8}, target_rate=0.25)
    for _ in range(4):
        control.update(0)
        control.reset()

    control.calibrate(2)

    assert_allclose(control.exposure_parameters[0], 0.2)
    assert_allclose(control.exposure_parameters[1], 0.88)


def test_sympson_hetter_calibration_without_sessions_is_noop() -> None:
    control = SympsonHetter({0: 0.8})

    control.calibrate(1)

    assert control.exposure_parameters == {0: 0.8}


@pytest.mark.parametrize("n_items", [0, -1, 1.5, True])
def test_sympson_hetter_calibration_validates_pool_size(n_items: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SympsonHetter().calibrate(n_items)


@pytest.mark.parametrize("selected_item", [-1, 0.5, True])
def test_sympson_hetter_update_validates_selected_item(selected_item: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        SympsonHetter().update(selected_item)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_randomesque_validates_top_k(k: Any) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Randomesque(k=k)


def test_passthrough_controls_return_independent_sets() -> None:
    model = _InformationModel([1.0, 1.0])
    available = {0, 1}

    no_control_result = NoExposureControl().filter_items(available, model, 0.0)
    randomesque_result = Randomesque().filter_items(available, model, 0.0)

    assert no_control_result == available
    assert randomesque_result == available
    assert no_control_result is not available
    assert randomesque_result is not available


@pytest.mark.parametrize("window_size", [-0.1, np.nan, np.inf])
def test_progressive_restricted_validates_window(window_size: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ProgressiveRestricted(window_size)


@pytest.mark.parametrize("window_size", [True, "0.5"])
def test_progressive_restricted_requires_numeric_window(window_size: Any) -> None:
    with pytest.raises(TypeError, match="real number"):
        ProgressiveRestricted(window_size)


def test_progressive_restricted_batches_information_once() -> None:
    model = _InformationModel([0.1, 0.8, 1.0, 0.6, np.nan, 0.95])
    control = ProgressiveRestricted(window_size=0.1)

    eligible = control.filter_items(set(range(model.n_items)), model, 0.25)

    assert eligible == {2, 5}
    assert model.calls == 1
    assert_allclose(model.last_theta, [[0.25]])
    assert control.max_information_seen == {
        0: 0.1,
        1: 0.8,
        2: 1.0,
        3: 0.6,
        5: 0.95,
    }


def test_progressive_restricted_accepts_multidimensional_theta() -> None:
    model = _InformationModel([0.4, 0.7, 0.6], n_factors=2)
    control = ProgressiveRestricted(window_size=0.15)

    assert control.filter_items({0, 1, 2}, model, np.array([0.2, -0.3])) == {
        1,
        2,
    }
    assert_allclose(model.last_theta, [[0.2, -0.3]])

    control.filter_items({0, 1, 2}, model, np.array([[0.1, -0.2]]))
    assert_allclose(model.last_theta, [[0.1, -0.2]])


def test_progressive_restricted_handles_all_nonfinite_information() -> None:
    model = _InformationModel([np.nan, np.inf, -np.inf])
    control = ProgressiveRestricted()

    assert control.filter_items({0, 1, 2}, model, 0.0) == {0, 1, 2}


def test_progressive_restricted_validates_theta_and_item_indices() -> None:
    model = _InformationModel([0.5, 1.0])
    control = ProgressiveRestricted()

    with pytest.raises(ValueError, match="single ability vector"):
        control.filter_items({0}, model, np.zeros((2, 1)))
    with pytest.raises(ValueError, match="theta values must be finite"):
        control.filter_items({0}, model, np.array([np.nan]))
    model.n_factors = 2
    with pytest.raises(ValueError, match="expected 2"):
        control.filter_items({0}, model, 0.0)
    with pytest.raises(IndexError, match="out of range"):
        control.filter_items({2}, model, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        control.filter_items({-1}, model, 0.0)


def test_progressive_restricted_validates_batched_information_shape() -> None:
    model = _InformationModel([0.5, 1.0])
    model.n_items = 3

    with pytest.raises(ValueError, match="one value per item"):
        ProgressiveRestricted().filter_items({0, 1}, model, 0.0)


def test_progressive_restricted_works_in_multidimensional_engine() -> None:
    model = MultidimensionalModel(n_items=8, n_factors=2)
    model._is_fitted = True
    control = ProgressiveRestricted(window_size=0.5)
    engine = MCATEngine(model, exposure_control=control, max_items=2)

    assert engine.select_next_item() in set(range(model.n_items))
    assert len(control.max_information_seen) == model.n_items


def test_factory_accepts_surrounding_whitespace() -> None:
    assert isinstance(
        create_exposure_control("  sympson-hetter  "),
        SympsonHetter,
    )
