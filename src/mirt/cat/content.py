"""Content balancing for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import fsum, isclose, isfinite
from numbers import Integral, Real
from typing import Any


def _validate_item_index(value: Any, *, name: str = "item index") -> int:
    """Return a non-negative integer item index."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


def _validate_count(value: Any, *, name: str) -> int:
    """Return a non-negative integer count."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _validate_area_name(value: Any, *, name: str = "area name") -> str:
    """Return a normalized non-empty content-area name."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _validate_available_items(available_items: set[int]) -> set[int]:
    """Return an owned, validated candidate-item set."""
    if not isinstance(available_items, set):
        raise TypeError("available_items must be a set")
    if all(type(item_idx) is int for item_idx in available_items):
        if available_items and min(available_items) < 0:
            raise ValueError("available item index must be non-negative")
        return available_items.copy()
    return {
        _validate_item_index(item_idx, name="available item index")
        for item_idx in available_items
    }


def _validate_administered_items(administered_items: list[int]) -> list[int]:
    """Return an owned, validated administration history."""
    if not isinstance(administered_items, list):
        raise TypeError("administered_items must be a list")
    if all(type(item_idx) is int for item_idx in administered_items):
        if administered_items and min(administered_items) < 0:
            raise ValueError("administered item index must be non-negative")
        normalized = administered_items.copy()
    else:
        normalized = [
            _validate_item_index(item_idx, name="administered item index")
            for item_idx in administered_items
        ]
    if len(set(normalized)) != len(normalized):
        raise ValueError("administered_items must not contain duplicates")
    return normalized


def _validate_candidate_history(
    available_items: set[int],
    administered_items: list[int],
) -> tuple[set[int], list[int]]:
    """Validate disjoint available and administered item collections."""
    available = _validate_available_items(available_items)
    administered = _validate_administered_items(administered_items)
    overlap = available & set(administered)
    if overlap:
        raise ValueError("available_items must not include administered items")
    return available, administered


@dataclass
class ContentArea:
    """Specification for a content area in a test blueprint.

    Attributes
    ----------
    name : str
        Name of the content area (e.g., "Algebra", "Geometry").
    items : set[int]
        Set of item indices belonging to this content area.
    min_items : int
        Minimum number of items required from this area.
    max_items : int
        Maximum number of items allowed from this area.
    target_items : int | None
        Target number of items (optional, for soft constraints).
    """

    name: str
    items: set[int] = field(default_factory=set)
    min_items: int = 0
    max_items: int = 999
    target_items: int | None = None

    def __post_init__(self) -> None:
        self.name = _validate_area_name(self.name)
        if not isinstance(self.items, (set, frozenset)):
            raise TypeError("items must be a set")
        self.items = {
            _validate_item_index(item_idx, name="content item index")
            for item_idx in self.items
        }
        self.min_items = _validate_count(self.min_items, name="min_items")
        self.max_items = _validate_count(self.max_items, name="max_items")
        if self.max_items < self.min_items:
            raise ValueError("max_items must be >= min_items")
        if self.min_items > len(self.items):
            raise ValueError("min_items cannot exceed the number of area items")
        if self.target_items is not None:
            self.target_items = _validate_count(
                self.target_items,
                name="target_items",
            )
            if self.target_items < self.min_items:
                raise ValueError("target_items must be >= min_items")
            if self.target_items > self.max_items:
                raise ValueError("target_items must be <= max_items")
            if self.target_items > len(self.items):
                raise ValueError("target_items cannot exceed the number of area items")


class ContentConstraint(ABC):
    """Abstract base class for content balancing constraints."""

    @abstractmethod
    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        """Filter available items based on content constraints.

        Parameters
        ----------
        available_items : set[int]
            Set of item indices that are candidates for selection.
        administered_items : list[int]
            List of already administered item indices.

        Returns
        -------
        set[int]
            Filtered set of items satisfying constraints.
        """
        pass

    def reset(self) -> None:
        """Reset constraint state for a new examinee."""
        return None


class NoContentConstraint(ContentConstraint):
    """No content constraints (all items eligible)."""

    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        available, _ = _validate_candidate_history(
            available_items,
            administered_items,
        )
        return available


class ContentBlueprint(ContentConstraint):
    """Content blueprint for enforcing test specifications.

    Ensures that item selection follows a test blueprint specifying
    minimum and maximum items per content area.

    Parameters
    ----------
    areas : list[ContentArea]
        List of content area specifications.
    strict : bool, optional
        If True, strictly enforce min/max constraints.
        If False, use soft constraints (prefer target). Default is True.

    Examples
    --------
    >>> blueprint = ContentBlueprint([
    ...     ContentArea("Algebra", items={0, 1, 2, 3}, min_items=2, max_items=4),
    ...     ContentArea("Geometry", items={4, 5, 6}, min_items=1, max_items=3),
    ...     ContentArea("Statistics", items={7, 8, 9}, min_items=1, max_items=2),
    ... ])
    """

    def __init__(self, areas: list[ContentArea], strict: bool = True):
        if not isinstance(areas, list):
            raise TypeError("areas must be a list")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")

        self.areas: list[ContentArea] = []
        area_names: set[str] = set()
        for area in areas:
            if not isinstance(area, ContentArea):
                raise TypeError("areas must contain ContentArea instances")
            if area.name in area_names:
                raise ValueError(f"Duplicate content area name: {area.name}")
            area_names.add(area.name)
            self.areas.append(
                ContentArea(
                    name=area.name,
                    items=set(area.items),
                    min_items=area.min_items,
                    max_items=area.max_items,
                    target_items=area.target_items,
                )
            )
        self.strict = strict

        self._item_to_area: dict[int, ContentArea] = {}
        for area in self.areas:
            for item in area.items:
                if item in self._item_to_area:
                    raise ValueError(f"Item {item} belongs to multiple content areas")
                self._item_to_area[item] = area

        self._area_counts: dict[str, int] = {area.name: 0 for area in self.areas}

    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        available, administered = _validate_candidate_history(
            available_items,
            administered_items,
        )
        self._update_counts(administered)
        if not available:
            return set()

        if self.strict:
            infeasible_areas = self._get_infeasible_areas(available)
            if infeasible_areas:
                raise RuntimeError(
                    "Content blueprint constraints cannot be met with the available "
                    f"items: {', '.join(infeasible_areas)}"
                )

        eligible = available.copy()
        for area in self.areas:
            if self._area_counts[area.name] >= area.max_items:
                eligible.difference_update(area.items)
        priority_items = self._get_priority_items(eligible)
        if priority_items:
            return priority_items

        if not self.strict:
            target_items = self._get_target_items(eligible)
            if target_items:
                return target_items

        if eligible:
            return eligible
        raise RuntimeError("No available items satisfy the content blueprint maximums")

    def _update_counts(self, administered_items: list[int]) -> None:
        """Update content area counts."""
        self._area_counts = {area.name: 0 for area in self.areas}
        for item_idx in administered_items:
            if item_idx in self._item_to_area:
                area = self._item_to_area[item_idx]
                self._area_counts[area.name] += 1

    def _get_priority_items(self, available_items: set[int]) -> set[int]:
        """Get items from areas that need more items."""
        priority_items = set()

        for area in self.areas:
            current_count = self._area_counts[area.name]
            if current_count < area.min_items:
                area_available = available_items & area.items
                priority_items.update(area_available)

        return priority_items

    def _get_target_items(self, available_items: set[int]) -> set[int]:
        """Return items in areas that have not reached their soft targets."""
        target_items: set[int] = set()
        for area in self.areas:
            target = area.target_items
            if target is not None and self._area_counts[area.name] < target:
                target_items.update(available_items & area.items)
        return target_items

    def _is_feasible_with_current_counts(self, available_items: set[int]) -> bool:
        """Return whether all remaining blueprint constraints can be met."""
        return not self._get_infeasible_areas(available_items)

    def _get_infeasible_areas(self, available_items: set[int]) -> list[str]:
        """Return areas that have exceeded a max or cannot reach a minimum."""
        infeasible: list[str] = []
        for area in self.areas:
            count = self._area_counts[area.name]
            remaining_min = max(0, area.min_items - count)
            if count > area.max_items or remaining_min > len(
                available_items & area.items
            ):
                infeasible.append(area.name)
        return infeasible

    def is_feasible(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> bool:
        """Return whether the blueprint constraints can still be satisfied."""
        available, administered = _validate_candidate_history(
            available_items,
            administered_items,
        )
        self._update_counts(administered)
        return self._is_feasible_with_current_counts(available)

    def get_unmet_areas(self, administered_items: list[int]) -> list[str]:
        """Return names of content areas whose minimums are not yet met."""
        administered = _validate_administered_items(administered_items)
        self._update_counts(administered)
        return [
            area.name
            for area in self.areas
            if self._area_counts[area.name] < area.min_items
        ]

    def is_blueprint_satisfied(self, administered_items: list[int]) -> bool:
        """Check whether every area is within its minimum and maximum.

        Parameters
        ----------
        administered_items : list[int]
            List of administered item indices.

        Returns
        -------
        bool
            True if all content areas are within their configured bounds.
        """
        administered = _validate_administered_items(administered_items)
        self._update_counts(administered)

        for area in self.areas:
            count = self._area_counts[area.name]
            if not area.min_items <= count <= area.max_items:
                return False
        return True

    def get_area_counts(self, administered_items: list[int]) -> dict[str, int]:
        """Get current counts for each content area.

        Parameters
        ----------
        administered_items : list[int]
            List of administered item indices.

        Returns
        -------
        dict[str, int]
            Dictionary mapping area names to item counts.
        """
        administered = _validate_administered_items(administered_items)
        self._update_counts(administered)
        return dict(self._area_counts)

    def get_remaining_requirements(
        self, administered_items: list[int]
    ) -> dict[str, tuple[int, int]]:
        """Get remaining min/max requirements for each area.

        Parameters
        ----------
        administered_items : list[int]
            List of administered item indices.

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary mapping area names to (remaining_min, remaining_max).
        """
        administered = _validate_administered_items(administered_items)
        self._update_counts(administered)

        remaining = {}
        for area in self.areas:
            count = self._area_counts[area.name]
            remaining_min = max(0, area.min_items - count)
            remaining_max = max(0, area.max_items - count)
            remaining[area.name] = (remaining_min, remaining_max)

        return remaining

    def reset(self) -> None:
        """Reset for a new examinee."""
        self._area_counts = {area.name: 0 for area in self.areas}

    def summary(self) -> str:
        """Return a summary of the content blueprint.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["Content Blueprint:", "-" * 40]
        for area in self.areas:
            target_str = (
                f", target={area.target_items}" if area.target_items is not None else ""
            )
            lines.append(
                f"  {area.name}: {len(area.items)} items, "
                f"min={area.min_items}, max={area.max_items}{target_str}"
            )
        return "\n".join(lines)


class WeightedContent(ContentConstraint):
    """Weighted content balancing based on area priorities.

    Items from underrepresented areas receive higher selection
    priority through weighting.

    Parameters
    ----------
    item_weights : dict[int, float]
        Base weights for each item.
    area_targets : dict[str, float]
        Target proportions for each content area.
    item_areas : dict[int, str]
        Mapping of items to their content areas.
    top_k : int | None, optional
        Number of highest-priority items to keep after the first item. When
        omitted, all items tied for the highest adjusted weight are kept.
    """

    def __init__(
        self,
        item_weights: dict[int, float],
        area_targets: dict[str, float],
        item_areas: dict[int, str],
        top_k: int | None = None,
    ):
        if not isinstance(item_weights, dict):
            raise TypeError("item_weights must be a dictionary")
        if not isinstance(area_targets, dict):
            raise TypeError("area_targets must be a dictionary")
        if not isinstance(item_areas, dict):
            raise TypeError("item_areas must be a dictionary")
        if top_k is not None:
            top_k = _validate_count(top_k, name="top_k")
            if top_k == 0:
                raise ValueError("top_k must be positive")

        normalized_weights: dict[int, float] = {}
        for item_idx, weight in item_weights.items():
            item = _validate_item_index(item_idx, name="item weight index")
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError("item weights must be real numbers")
            numeric_weight = float(weight)
            if not isfinite(numeric_weight) or numeric_weight < 0.0:
                raise ValueError("item weights must be finite and non-negative")
            normalized_weights[item] = numeric_weight

        normalized_targets: dict[str, float] = {}
        for area_name, target in area_targets.items():
            area = _validate_area_name(area_name, name="area target name")
            if area in normalized_targets:
                raise ValueError(f"Duplicate area target name: {area}")
            if isinstance(target, bool) or not isinstance(target, Real):
                raise TypeError("area targets must be real numbers")
            numeric_target = float(target)
            if not isfinite(numeric_target) or numeric_target < 0.0:
                raise ValueError("area targets must be finite and non-negative")
            normalized_targets[area] = numeric_target
        target_total = fsum(normalized_targets.values())
        if not normalized_targets or target_total <= 0.0:
            raise ValueError("area_targets must have a positive total")
        if not isclose(target_total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            normalized_targets = {
                area: target / target_total
                for area, target in normalized_targets.items()
            }

        normalized_areas: dict[int, str] = {}
        for item_idx, area_name in item_areas.items():
            item = _validate_item_index(item_idx, name="item area index")
            area = _validate_area_name(area_name, name="item area name")
            if area not in normalized_targets:
                raise ValueError(f"Missing target for content area: {area}")
            normalized_areas[item] = area
        unmapped_weights = normalized_weights.keys() - normalized_areas.keys()
        if unmapped_weights:
            raise ValueError("Every weighted item must have a content-area mapping")

        self._item_weights = normalized_weights
        self._area_targets = normalized_targets
        self._item_areas = normalized_areas
        self.top_k = top_k

    @property
    def item_weights(self) -> dict[int, float]:
        """Return a copy of the configured base item weights."""
        return dict(self._item_weights)

    @property
    def area_targets(self) -> dict[str, float]:
        """Return a copy of the normalized area targets."""
        return dict(self._area_targets)

    @property
    def item_areas(self) -> dict[int, str]:
        """Return a copy of the item-to-area mapping."""
        return dict(self._item_areas)

    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        available, administered = _validate_candidate_history(
            available_items,
            administered_items,
        )
        if not available or not administered:
            return available

        adjusted = self._get_adjusted_weights(available, administered)
        if self.top_k is not None:
            ranked = sorted(adjusted, key=lambda item: (-adjusted[item], item))
            return set(ranked[: self.top_k])

        highest = max(adjusted.values())
        return {
            item_idx
            for item_idx, weight in adjusted.items()
            if isclose(weight, highest, rel_tol=1e-12, abs_tol=1e-12)
        }

    def get_adjusted_weights(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> dict[int, float]:
        """Get content-adjusted weights for available items.

        Parameters
        ----------
        available_items : set[int]
            Set of available item indices.
        administered_items : list[int]
            List of administered item indices.

        Returns
        -------
        dict[int, float]
            Dictionary mapping item indices to adjusted weights.
        """
        available, administered = _validate_candidate_history(
            available_items,
            administered_items,
        )
        return self._get_adjusted_weights(available, administered)

    def _get_adjusted_weights(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> dict[int, float]:
        """Compute adjusted weights for validated item collections."""
        n_administered = len(administered_items)
        if n_administered == 0:
            return {i: self._item_weights.get(i, 1.0) for i in available_items}

        area_counts: dict[str, int] = {}
        for item_idx in administered_items:
            area = self._item_areas.get(item_idx)
            if area is not None:
                area_counts[area] = area_counts.get(area, 0) + 1

        weights: dict[int, float] = {}
        for item_idx in available_items:
            area = self._item_areas.get(item_idx)
            multiplier = 1.0
            if area is not None:
                desired_count = self._area_targets[area] * (n_administered + 1)
                deficit = max(desired_count - area_counts.get(area, 0), 0.0)
                multiplier += deficit

            base_weight = self._item_weights.get(item_idx, 1.0)
            weights[item_idx] = base_weight * multiplier

        return weights


def create_content_constraint(
    method: str | None,
    **kwargs: Any,
) -> ContentConstraint:
    """Factory function to create content constraints.

    Parameters
    ----------
    method : str | None
        Content constraint method. One of: "blueprint", "weighted", None.
    **kwargs
        Additional keyword arguments passed to the constructor.

    Returns
    -------
    ContentConstraint
        The requested content constraint.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    if method is None:
        return NoContentConstraint()

    if not isinstance(method, str):
        raise TypeError("method must be a string or None")

    methods = {
        "blueprint": ContentBlueprint,
        "weighted": WeightedContent,
        "none": NoContentConstraint,
    }

    method_lower = method.strip().lower()
    if method_lower not in methods:
        valid = ", ".join(methods.keys())
        raise ValueError(
            f"Unknown content constraint method '{method}'. Valid options: {valid}"
        )

    return methods[method_lower](**kwargs)
