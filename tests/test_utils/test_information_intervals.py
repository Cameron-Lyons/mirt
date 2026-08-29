"""Coverage for information-target ability intervals."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils.information import information_intervals


def test_single_item_boundaries_match_logistic_information() -> None:
    model = TwoParameterLogistic(1)
    target = 0.1
    root = np.log(
        (1.0 + np.sqrt(1.0 - 4.0 * target)) / (1.0 - np.sqrt(1.0 - 4.0 * target))
    )

    intervals = information_intervals(model, target, theta_range=(-5.0, 5.0))

    np.testing.assert_allclose(intervals, [[-root, root]], atol=1e-8, rtol=0.0)


def test_disjoint_test_regions_and_item_selection_are_retained() -> None:
    model = TwoParameterLogistic(2).set_parameters(
        discrimination=np.array([4.0, 4.0]),
        difficulty=np.array([-2.0, 2.0]),
    )

    test_intervals = information_intervals(model, 1.0, theta_range=(-4.0, 4.0))
    item_intervals = information_intervals(
        model,
        1.0,
        theta_range=(-4.0, 4.0),
        item_idx=1,
    )

    assert test_intervals.shape == (2, 2)
    np.testing.assert_allclose(test_intervals[0], -test_intervals[1, ::-1], atol=1e-8)
    assert test_intervals[0, 0] < -2.0 < test_intervals[0, 1]
    assert test_intervals[1, 0] < 2.0 < test_intervals[1, 1]
    assert item_intervals.shape == (1, 2)
    assert item_intervals[0, 0] < 2.0 < item_intervals[0, 1]


def test_unmet_target_returns_stable_empty_shape() -> None:
    intervals = information_intervals(TwoParameterLogistic(1), 1.0)

    assert intervals.shape == (0, 2)
    assert intervals.dtype == np.float64


@pytest.mark.parametrize(("item_idx", "target"), [(None, 1.0), (1, 0.3)])
def test_polytomous_test_and_item_boundaries_meet_target(
    item_idx: int | None,
    target: float,
) -> None:
    model = GeneralizedPartialCredit(3, 4)

    intervals = information_intervals(model, target, item_idx=item_idx)
    boundaries = intervals.ravel()[:, None]
    information = model.information(boundaries, item_idx=item_idx)

    assert intervals.shape == (1, 2)
    np.testing.assert_allclose(information, target, atol=1e-8, rtol=0.0)


def test_large_banks_use_bounded_theta_chunks() -> None:
    class TotalInformationModel:
        n_factors = 1
        n_items = 1_000_000
        is_polytomous = False

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def information(self, theta: np.ndarray) -> np.ndarray:
            self.batch_sizes.append(len(theta))
            return np.full(len(theta), 2.0)

    model = TotalInformationModel()

    intervals = information_intervals(
        model,
        1.0,
        theta_range=(-2.0, 2.0),
        n_points=9,
    )

    np.testing.assert_array_equal(intervals, [[-2.0, 2.0]])
    assert model.batch_sizes
    assert max(model.batch_sizes) <= 2


@pytest.mark.parametrize(
    ("target", "theta_range", "n_points", "message"),
    [
        (0.0, (-6.0, 6.0), 101, "min_information"),
        (True, (-6.0, 6.0), 101, "min_information"),
        ([1.0], (-6.0, 6.0), 101, "min_information"),
        (np.inf, (-6.0, 6.0), 101, "min_information"),
        (1.0, (2.0, -2.0), 101, "theta_range"),
        (1.0, (-2.0, 2.0), 1, "n_points"),
        (1.0, (-2.0, 2.0), True, "n_points"),
    ],
)
def test_invalid_search_configuration_is_rejected(
    target: float,
    theta_range: tuple[float, float],
    n_points: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        information_intervals(
            TwoParameterLogistic(1),
            target,
            theta_range=theta_range,
            n_points=n_points,
        )


@pytest.mark.parametrize("item_idx", [-1, 2, 0.5, True])
def test_invalid_item_selection_is_rejected(item_idx: object) -> None:
    with pytest.raises((ValueError, IndexError), match="item_idx"):
        information_intervals(TwoParameterLogistic(2), 0.1, item_idx=item_idx)


def test_multidimensional_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="unidimensional"):
        information_intervals(TwoParameterLogistic(2, n_factors=2), 0.1)


@pytest.mark.parametrize("value", [np.nan, -0.1])
def test_invalid_model_information_is_rejected(value: float) -> None:
    class InvalidInformationModel:
        n_factors = 1
        n_items = 1
        is_polytomous = False

        def information(self, theta: np.ndarray) -> np.ndarray:
            return np.full(len(theta), value)

    with pytest.raises(ValueError, match="information"):
        information_intervals(InvalidInformationModel(), 0.1, n_points=3)


def test_top_level_export_preserves_public_identity() -> None:
    assert mirt.information_intervals is information_intervals
