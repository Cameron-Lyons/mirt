"""Exposure control methods for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class ExposureControl(ABC):
    """Abstract base class for CAT exposure control methods.

    Exposure control ensures that items are not overused in CAT,
    which is important for test security and item pool longevity.
    """

    @abstractmethod
    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float,
    ) -> set[int]:
        """Filter available items based on exposure control.

        Parameters
        ----------
        available_items : set[int]
            Set of item indices that are candidates for selection.
        model : BaseItemModel
            The fitted IRT model.
        theta : float
            Current ability estimate.

        Returns
        -------
        set[int]
            Filtered set of eligible items.
        """
        pass

    def update(self, selected_item: int) -> None:
        """Update exposure control state after item selection.

        Parameters
        ----------
        selected_item : int
            Index of the item that was selected.
        """
        pass

    def reset(self) -> None:
        """Reset exposure control for a new examinee."""
        pass


class NoExposureControl(ExposureControl):
    """No exposure control (all items eligible).

    This is the default when exposure control is not needed.
    """

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float,
    ) -> set[int]:
        return available_items


class SympsonHetter(ExposureControl):
    """Sympson-Hetter probabilistic exposure control.

    Each item has an exposure control parameter that determines
    the probability of being eligible for selection. Parameters
    are typically calibrated through simulation.

    Parameters
    ----------
    exposure_params : NDArray[np.float64] | dict[int, float] | None
        Exposure control parameters for each item (0-1).
        If None, all items start with parameter 1.0 (no control).
    target_rate : float, optional
        Target maximum exposure rate. Default is 0.25.
    seed : int | None, optional
        Random seed for reproducibility.

    References
    ----------
    Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure
    rates in computerized adaptive testing. Proceedings of the 27th
    annual meeting of the Military Testing Association.
    """

    def __init__(
        self,
        exposure_params: NDArray[np.float64] | dict[int, float] | None = None,
        target_rate: float = 0.25,
        seed: int | None = None,
    ):
        self.target_rate = target_rate
        self.rng = np.random.default_rng(seed)

        if exposure_params is None:
            self._params: dict[int, float] = {}
        elif isinstance(exposure_params, dict):
            self._params = dict(exposure_params)
        else:
            self._params = {i: float(p) for i, p in enumerate(exposure_params)}

        self._selection_counts: dict[int, int] = {}
        self._eligibility_counts: dict[int, int] = {}
        self._n_examinees = 0

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float,
    ) -> set[int]:
        eligible = set()

        for item_idx in available_items:
            k = self._params.get(item_idx, 1.0)

            if self.rng.random() <= k:
                eligible.add(item_idx)
                self._eligibility_counts[item_idx] = (
                    self._eligibility_counts.get(item_idx, 0) + 1
                )

        if not eligible and available_items:
            eligible.add(self.rng.choice(list(available_items)))

        return eligible

    def update(self, selected_item: int) -> None:
        self._selection_counts[selected_item] = (
            self._selection_counts.get(selected_item, 0) + 1
        )

    def reset(self) -> None:
        self._n_examinees += 1

    def calibrate(self, n_items: int) -> None:
        """Recalibrate exposure parameters based on observed rates.

        Should be called periodically during operational testing
        to adjust parameters.

        Parameters
        ----------
        n_items : int
            Total number of items in the pool.
        """
        if self._n_examinees == 0:
            return

        for item_idx in range(n_items):
            exposure_rate = self._selection_counts.get(item_idx, 0) / self._n_examinees

            if exposure_rate > self.target_rate:
                current = self._params.get(item_idx, 1.0)
                self._params[item_idx] = current * (self.target_rate / exposure_rate)
            else:
                current = self._params.get(item_idx, 1.0)
                self._params[item_idx] = min(1.0, current * 1.1)

    def get_exposure_rates(self) -> dict[int, float]:
        """Get current exposure rates for all items.

        Returns
        -------
        dict[int, float]
            Dictionary mapping item indices to exposure rates.
        """
        if self._n_examinees == 0:
            return {}

        return {
            item: count / self._n_examinees
            for item, count in self._selection_counts.items()
        }


class Randomesque(ExposureControl):
    """Randomesque exposure control.

    Selects randomly from the top-k items ranked by the selection
    criterion, rather than always choosing the best item.

    Parameters
    ----------
    k : int, optional
        Number of top items to randomize among. Default is 5.
    seed : int | None, optional
        Random seed for reproducibility.

    References
    ----------
    Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting
    items for computerized adaptive tests. Applied Measurement in
    Education, 2(4), 359-375.
    """

    def __init__(self, k: int = 5, seed: int | None = None):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.rng = np.random.default_rng(seed)

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float,
    ) -> set[int]:
        return available_items

    def select_from_ranked(
        self,
        ranked_items: list[tuple[int, float]],
    ) -> int:
        """Select an item from the top-k ranked items.

        Parameters
        ----------
        ranked_items : list[tuple[int, float]]
            List of (item_idx, criterion_value) sorted by criterion
            in descending order.

        Returns
        -------
        int
            Selected item index.
        """
        if not ranked_items:
            raise ValueError("No items to select from")

        k = min(self.k, len(ranked_items))
        top_k = ranked_items[:k]

        idx = self.rng.integers(k)
        return top_k[idx][0]


class ProgressiveRestricted(ExposureControl):
    """Progressive-restricted exposure control.

    Gradually restricts the item pool as the test progresses,
    using item information to determine eligibility windows.

    Parameters
    ----------
    window_size : float, optional
        Size of the eligibility window in information units. Default is 0.5.
    seed : int | None, optional
        Random seed for reproducibility.

    References
    ----------
    Revuelta, J., & Ponsoda, V. (1998). A comparison of item exposure
    control methods in computerized adaptive testing. Journal of
    Educational Measurement, 35(4), 311-327.
    """

    def __init__(self, window_size: float = 0.5, seed: int | None = None):
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self._max_info_seen: dict[int, float] = {}

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float,
    ) -> set[int]:
        theta_arr = np.array([[theta]])
        eligible = set()

        # Compute information for all available items
        item_info: list[tuple[int, float]] = []
        for item_idx in available_items:
            info = model.information(theta_arr, item_idx=item_idx)
            info_val = float(info.sum())
            item_info.append((item_idx, info_val))

        if not item_info:
            return set()

        max_info = max(info for _, info in item_info)

        threshold = max_info - self.window_size
        for item_idx, info in item_info:
            if info >= threshold:
                eligible.add(item_idx)

        return eligible if eligible else available_items

    def reset(self) -> None:
        self._max_info_seen.clear()


def create_exposure_control(
    method: str | None,
    **kwargs: Any,
) -> ExposureControl:
    """Factory function to create exposure control methods.

    Parameters
    ----------
    method : str | None
        Exposure control method name. One of: "sympson-hetter",
        "randomesque", "progressive", None (no control).
    **kwargs
        Additional keyword arguments passed to the constructor.

    Returns
    -------
    ExposureControl
        The requested exposure control method.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    if method is None:
        return NoExposureControl()

    methods = {
        "sympson-hetter": SympsonHetter,
        "randomesque": Randomesque,
        "progressive": ProgressiveRestricted,
        "none": NoExposureControl,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        valid = ", ".join(methods.keys())
        raise ValueError(
            f"Unknown exposure control method '{method}'. Valid options: {valid}"
        )

    return methods[method_lower](**kwargs)
