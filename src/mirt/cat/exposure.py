"""Exposure control methods for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def _validate_item_index(value: Any, *, name: str = "item index") -> int:
    """Return a non-negative integer item index."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


def _validate_probability(value: Any, *, name: str) -> float:
    """Return a finite probability in the unit interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    probability = float(value)
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return probability


def _prepare_available_items(available_items: set[int]) -> NDArray[np.int_]:
    """Validate and sort a candidate item set for deterministic batching."""
    if not available_items:
        return np.empty(0, dtype=np.int_)
    values = np.asarray(list(available_items))
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise TypeError("available item indices must be integers")
    item_indices = values.astype(np.int_, copy=False)
    if np.any(item_indices < 0):
        raise ValueError("available item indices must be non-negative")
    item_indices.sort()
    return item_indices


@dataclass(frozen=True)
class ExposureReport:
    """Item-level exposure monitoring summary.

    Attributes
    ----------
    item_indices : ndarray
        Zero-based item indices represented by each array position.
    selection_counts, opportunity_counts, eligibility_counts : ndarray
        Recorded item counts.
    exposure_rates, eligibility_rates : ndarray
        Selection rates per examinee and eligibility rates per opportunity.
    confidence_lower, confidence_upper : ndarray
        Wilson confidence bounds for exposure rates.
    target_rate : float
        Configured maximum exposure rate.
    above_target : ndarray
        Items whose point exposure estimate exceeds the target.
    significantly_above_target : ndarray
        Items whose lower confidence bound exceeds the target.
    n_examinees : int
        Number of recorded examinee sessions.
    """

    item_indices: NDArray[np.int_]
    selection_counts: NDArray[np.int64]
    opportunity_counts: NDArray[np.int64]
    eligibility_counts: NDArray[np.int64]
    exposure_rates: NDArray[np.float64]
    eligibility_rates: NDArray[np.float64]
    confidence_lower: NDArray[np.float64]
    confidence_upper: NDArray[np.float64]
    target_rate: float
    above_target: NDArray[np.bool_]
    significantly_above_target: NDArray[np.bool_]
    n_examinees: int

    @property
    def overexposed_items(self) -> NDArray[np.int_]:
        """Return item indices whose point estimates exceed the target."""
        return self.item_indices[self.above_target].copy()

    @property
    def statistically_overexposed_items(self) -> NDArray[np.int_]:
        """Return item indices whose confidence intervals exceed the target."""
        return self.item_indices[self.significantly_above_target].copy()

    @property
    def max_exposure_rate(self) -> float:
        """Return the largest item exposure rate, or zero for an empty report."""
        if self.exposure_rates.size == 0:
            return 0.0
        return float(np.max(self.exposure_rates))


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
        theta: float | NDArray[np.float64],
    ) -> set[int]:
        """Filter available items based on exposure control.

        Parameters
        ----------
        available_items : set[int]
            Set of item indices that are candidates for selection.
        model : BaseItemModel
            The fitted IRT model.
        theta : float or ndarray
            Current scalar ability estimate or multidimensional ability vector.

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
        return None

    def reset(self) -> None:
        """Reset exposure control for a new examinee."""
        return None


class NoExposureControl(ExposureControl):
    """No exposure control (all items eligible).

    This is the default when exposure control is not needed.
    """

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float | NDArray[np.float64],
    ) -> set[int]:
        return set(available_items)


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
        Target maximum exposure rate in ``(0, 1]``. Default is 0.25.
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
    ) -> None:
        self.target_rate = _validate_probability(target_rate, name="target_rate")
        if self.target_rate == 0.0:
            raise ValueError("target_rate must be greater than 0")
        self.rng = np.random.default_rng(seed)

        if exposure_params is None:
            self._params: dict[int, float] = {}
        elif isinstance(exposure_params, dict):
            self._params = {
                _validate_item_index(item_idx): _validate_probability(
                    parameter,
                    name=f"exposure parameter for item {item_idx}",
                )
                for item_idx, parameter in exposure_params.items()
            }
        else:
            parameter_array = np.asarray(exposure_params, dtype=np.float64)
            if parameter_array.ndim != 1:
                raise ValueError("exposure_params must be a one-dimensional array")
            self._params = {
                item_idx: _validate_probability(
                    parameter,
                    name=f"exposure parameter for item {item_idx}",
                )
                for item_idx, parameter in enumerate(parameter_array)
            }

        self._selection_counts: Counter[int] = Counter()
        self._eligibility_counts: Counter[int] = Counter()
        self._opportunity_counts: Counter[int] = Counter()
        self._n_examinees = 0

    @property
    def exposure_parameters(self) -> dict[int, float]:
        """Return a copy of item eligibility parameters."""
        return dict(self._params)

    @property
    def n_examinees(self) -> int:
        """Return the number of recorded examinee sessions."""
        return self._n_examinees

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float | NDArray[np.float64],
    ) -> set[int]:
        if not available_items:
            return set()

        item_indices = _prepare_available_items(available_items)
        parameters = np.fromiter(
            (self._params.get(int(item_idx), 1.0) for item_idx in item_indices),
            dtype=np.float64,
            count=len(item_indices),
        )
        accepted = item_indices[self.rng.random(len(item_indices)) <= parameters]

        self._opportunity_counts.update(map(int, item_indices))

        if len(accepted) == 0:
            accepted = np.array(
                [self.rng.choice(item_indices)],
                dtype=np.int_,
            )

        eligible = {int(item_idx) for item_idx in accepted}
        self._eligibility_counts.update(eligible)
        return eligible

    def update(self, selected_item: int) -> None:
        item_idx = _validate_item_index(selected_item, name="selected_item")
        self._selection_counts[item_idx] += 1

    def reset(self) -> None:
        self._n_examinees += 1

    def record_sessions(
        self,
        selections: ArrayLike,
        *,
        missing_value: int = -1,
        n_items: int | None = None,
    ) -> None:
        """Record one or many completed adaptive-test sessions in a batch.

        A one-dimensional input represents one session. A two-dimensional
        input uses one row per examinee and one column per administered
        position. Ragged histories can be padded with ``missing_value``.
        Repeated items within one session are rejected because exposure is an
        examinee-level proportion.

        Parameters
        ----------
        selections : array-like
            Integer selected-item histories.
        missing_value : int, default=-1
            Negative padding value ignored while counting selections.
        n_items : int, optional
            Item-pool size used to validate upper index bounds.
        """
        if isinstance(missing_value, (bool, np.bool_)) or not isinstance(
            missing_value, Integral
        ):
            raise TypeError("missing_value must be an integer")
        missing_value = int(missing_value)
        if missing_value >= 0:
            raise ValueError("missing_value must be negative")

        if n_items is not None:
            if (
                isinstance(n_items, (bool, np.bool_))
                or not isinstance(n_items, Integral)
                or n_items < 1
            ):
                raise ValueError("n_items must be a positive integer")
            n_items = int(n_items)

        values = np.asarray(selections)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("selections must contain one or more non-empty sessions")
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise TypeError("selections must contain integers")

        item_values = values.astype(np.int64, copy=False)
        invalid_negative = (item_values < 0) & (item_values != missing_value)
        if np.any(invalid_negative):
            raise ValueError("negative selections must equal missing_value")
        observed = item_values != missing_value
        if n_items is not None and np.any(item_values[observed] >= n_items):
            raise IndexError(f"selected item must be in [0, {n_items})")

        ordered = np.sort(item_values, axis=1)
        repeated = (ordered[:, 1:] == ordered[:, :-1]) & (
            ordered[:, 1:] != missing_value
        )
        if np.any(repeated):
            raise ValueError("a session cannot contain repeated selected items")

        selected = item_values[observed]
        if selected.size:
            indices, counts = np.unique(selected, return_counts=True)
            self._selection_counts.update(
                {
                    int(item_idx): int(count)
                    for item_idx, count in zip(indices, counts, strict=True)
                }
            )
        self._n_examinees += item_values.shape[0]

    def exposure_report(
        self,
        n_items: int | None = None,
        *,
        confidence_level: float = 0.95,
    ) -> ExposureReport:
        """Summarize item exposure with uncertainty-aware target flags.

        Wilson score intervals remain well behaved for zero and complete
        exposure counts. Items are statistically overexposed only when the
        entire confidence interval lies above ``target_rate``.

        Parameters
        ----------
        n_items : int, optional
            Number of items to include. By default, the report spans every
            item known from parameters or recorded counts.
        confidence_level : float, default=0.95
            Two-sided confidence level strictly between zero and one.
        """
        if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
            confidence_level, Real
        ):
            raise TypeError("confidence_level must be a real number")
        confidence_level = float(confidence_level)
        if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        quantile_probability = 0.5 + confidence_level / 2.0
        if quantile_probability >= 1.0:
            raise ValueError("confidence_level is too close to 1 for finite precision")

        known_items = (
            set(self._params)
            | set(self._selection_counts)
            | set(self._opportunity_counts)
            | set(self._eligibility_counts)
        )
        if n_items is None:
            item_indices = np.fromiter(
                sorted(known_items),
                dtype=np.int_,
                count=len(known_items),
            )
        else:
            if (
                isinstance(n_items, (bool, np.bool_))
                or not isinstance(n_items, Integral)
                or n_items < 1
            ):
                raise ValueError("n_items must be a positive integer")
            item_count = int(n_items)
            if known_items and max(known_items) >= item_count:
                raise ValueError("n_items does not include every recorded item")
            item_indices = np.arange(item_count, dtype=np.int_)

        item_count = item_indices.size

        def counter_values(counter: Counter[int]) -> NDArray[np.int64]:
            return np.fromiter(
                (counter.get(int(item_idx), 0) for item_idx in item_indices),
                dtype=np.int64,
                count=item_count,
            )

        selection_counts = counter_values(self._selection_counts)
        opportunity_counts = counter_values(self._opportunity_counts)
        eligibility_counts = counter_values(self._eligibility_counts)

        if np.any(selection_counts > self._n_examinees):
            raise ValueError(
                "selection counts exceed examinee sessions; record each item once per session"
            )

        exposure_rates = np.zeros(item_count, dtype=np.float64)
        confidence_lower = np.zeros(item_count, dtype=np.float64)
        confidence_upper = np.ones(item_count, dtype=np.float64)
        if self._n_examinees:
            exposure_rates = selection_counts / self._n_examinees
            z_value = NormalDist().inv_cdf(quantile_probability)
            z_squared = z_value**2
            denominator = 1.0 + z_squared / self._n_examinees
            center = (
                exposure_rates + z_squared / (2.0 * self._n_examinees)
            ) / denominator
            half_width = (
                z_value
                * np.sqrt(
                    exposure_rates * (1.0 - exposure_rates) / self._n_examinees
                    + z_squared / (4.0 * self._n_examinees**2)
                )
                / denominator
            )
            confidence_lower = np.maximum(0.0, center - half_width)
            confidence_upper = np.minimum(1.0, center + half_width)

        eligibility_rates = np.divide(
            eligibility_counts,
            opportunity_counts,
            out=np.zeros(item_count, dtype=np.float64),
            where=opportunity_counts > 0,
        )
        above_target = exposure_rates > self.target_rate
        significantly_above_target = confidence_lower > self.target_rate

        return ExposureReport(
            item_indices=item_indices,
            selection_counts=selection_counts,
            opportunity_counts=opportunity_counts,
            eligibility_counts=eligibility_counts,
            exposure_rates=exposure_rates,
            eligibility_rates=eligibility_rates,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            target_rate=self.target_rate,
            above_target=above_target,
            significantly_above_target=significantly_above_target,
            n_examinees=self._n_examinees,
        )

    def calibrate(self, n_items: int) -> None:
        """Recalibrate exposure parameters based on observed rates.

        Should be called periodically during operational testing
        to adjust parameters.

        Parameters
        ----------
        n_items : int
            Total number of items in the pool.
        """
        if (
            isinstance(n_items, (bool, np.bool_))
            or not isinstance(n_items, Integral)
            or n_items < 1
        ):
            raise ValueError("n_items must be a positive integer")
        n_items = int(n_items)

        if self._n_examinees == 0:
            return

        for item_idx in range(n_items):
            exposure_rate = self._selection_counts.get(item_idx, 0) / self._n_examinees

            if exposure_rate > self.target_rate:
                current = self._params.get(item_idx, 1.0)
                updated = current * (self.target_rate / exposure_rate)
            else:
                current = self._params.get(item_idx, 1.0)
                updated = current * 1.1
            self._params[item_idx] = float(np.clip(updated, 0.0, 1.0))

    def get_exposure_rates(self) -> dict[int, float]:
        """Get current exposure rates for all items.

        Returns
        -------
        dict[int, float]
            Dictionary mapping item indices to exposure rates.
        """
        known_items = (
            set(self._params)
            | set(self._selection_counts)
            | set(self._opportunity_counts)
        )
        if self._n_examinees == 0:
            return {item_idx: 0.0 for item_idx in sorted(known_items)}
        return {
            item_idx: self._selection_counts.get(item_idx, 0) / self._n_examinees
            for item_idx in sorted(known_items)
        }

    def get_eligibility_rates(self) -> dict[int, float]:
        """Return empirical eligibility rates by selection opportunity."""
        known_items = set(self._params) | set(self._opportunity_counts)
        return {
            item_idx: (
                self._eligibility_counts.get(item_idx, 0) / opportunities
                if (opportunities := self._opportunity_counts.get(item_idx, 0))
                else 0.0
            )
            for item_idx in sorted(known_items)
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

    def __init__(self, k: int = 5, seed: int | None = None) -> None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be at least 1")
        self.k: int = int(k)
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float | NDArray[np.float64],
    ) -> set[int]:
        return set(available_items)

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
    """Progressive information-window exposure control.

    Restricts the item pool to candidates within a fixed information window
    of the most informative available item. Selection within that window
    blends a random component with item information. Randomness dominates at
    the start of a test, while information dominates near the test horizon.

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

    def __init__(self, window_size: float = 0.5, seed: int | None = None) -> None:
        if isinstance(window_size, (bool, np.bool_)) or not isinstance(
            window_size, Real
        ):
            raise TypeError("window_size must be a real number")
        window_size = float(window_size)
        if not np.isfinite(window_size) or window_size < 0.0:
            raise ValueError("window_size must be finite and non-negative")
        self.window_size: float = window_size
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._max_info_seen: dict[int, float] = {}
        self._latest_item_indices: NDArray[np.int_] = np.empty(0, dtype=np.int_)
        self._latest_information: NDArray[np.float64] = np.empty(0, dtype=np.float64)

    @property
    def max_information_seen(self) -> dict[int, float]:
        """Return the maximum observed information for each evaluated item."""
        return dict(self._max_info_seen)

    def filter_items(
        self,
        available_items: set[int],
        model: BaseItemModel,
        theta: float | NDArray[np.float64],
    ) -> set[int]:
        self._latest_item_indices = np.empty(0, dtype=np.int_)
        self._latest_information = np.empty(0, dtype=np.float64)
        if not available_items:
            return set()

        item_indices = _prepare_available_items(available_items)
        if item_indices[-1] >= model.n_items:
            raise IndexError(
                f"item index {int(item_indices[-1])} out of range [0, {model.n_items})"
            )

        theta_values = np.asarray(theta, dtype=np.float64)
        if theta_values.ndim == 0:
            theta_arr = theta_values.reshape(1, 1)
        elif theta_values.ndim == 1:
            theta_arr = theta_values.reshape(1, -1)
        elif theta_values.ndim == 2 and theta_values.shape[0] == 1:
            theta_arr = theta_values
        else:
            raise ValueError("theta must be a scalar or a single ability vector")
        if theta_arr.shape[1] != model.n_factors:
            raise ValueError(
                f"theta has {theta_arr.shape[1]} factors, expected {model.n_factors}"
            )
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("theta values must be finite")

        all_information = np.asarray(model.information(theta_arr), dtype=np.float64)
        flattened_information = all_information.reshape(-1)
        if flattened_information.size != model.n_items:
            raise ValueError("model.information(theta) must return one value per item")
        candidate_information = flattened_information[item_indices]

        finite = np.isfinite(candidate_information)
        if not np.any(finite):
            self._latest_item_indices = item_indices.copy()
            self._latest_information = candidate_information.copy()
            return {int(item_idx) for item_idx in item_indices}

        for item_idx, information in zip(
            item_indices[finite],
            candidate_information[finite],
            strict=True,
        ):
            item = int(item_idx)
            value = float(information)
            self._max_info_seen[item] = max(
                value,
                self._max_info_seen.get(item, -np.inf),
            )

        threshold = float(np.max(candidate_information[finite])) - self.window_size
        eligible_mask = finite & (candidate_information >= threshold)
        eligible_indices = item_indices[eligible_mask]
        eligible_information = candidate_information[eligible_mask]

        self._latest_item_indices = eligible_indices.copy()
        self._latest_information = eligible_information.copy()
        return {int(item_idx) for item_idx in eligible_indices}

    def select_from_eligible(
        self,
        eligible_items: set[int],
        *,
        n_administered: int,
        max_items: int,
    ) -> int:
        """Select progressively from the most recent eligibility window.

        Parameters
        ----------
        eligible_items : set[int]
            Items returned by the most recent :meth:`filter_items` call.
        n_administered : int
            Number of items already administered in the current test.
        max_items : int
            Maximum possible test length used to scale progress.

        Returns
        -------
        int
            Selected item index.
        """
        item_indices = _prepare_available_items(eligible_items)
        if len(item_indices) == 0:
            raise ValueError("No items to select from")
        if (
            isinstance(n_administered, (bool, np.bool_))
            or not isinstance(n_administered, Integral)
            or n_administered < 0
        ):
            raise ValueError("n_administered must be a non-negative integer")
        if (
            isinstance(max_items, (bool, np.bool_))
            or not isinstance(max_items, Integral)
            or max_items < 1
        ):
            raise ValueError("max_items must be a positive integer")

        if not np.array_equal(item_indices, self._latest_item_indices):
            raise RuntimeError("filter_items must be called before item selection")

        information = self._latest_information
        if not np.all(np.isfinite(information)):
            return int(self.rng.choice(item_indices))

        max_information = float(np.max(information))
        if max_information <= 0.0:
            return int(self.rng.choice(item_indices))

        progress = min(float(n_administered) / float(max_items), 1.0)
        if progress >= 1.0:
            return int(item_indices[int(np.argmax(information))])

        random_component = self.rng.uniform(0.0, max_information, len(item_indices))
        weights = (1.0 - progress) * random_component + progress * information
        return int(item_indices[int(np.argmax(weights))])

    def reset(self) -> None:
        self._max_info_seen.clear()
        self._latest_item_indices = np.empty(0, dtype=np.int_)
        self._latest_information = np.empty(0, dtype=np.float64)


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

    method_lower = method.strip().lower()
    if method_lower not in methods:
        valid = ", ".join(methods.keys())
        raise ValueError(
            f"Unknown exposure control method '{method}'. Valid options: {valid}"
        )

    return methods[method_lower](**kwargs)
