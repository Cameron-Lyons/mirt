"""Content balancing for computerized adaptive testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


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

    def __post_init__(self):
        if self.min_items < 0:
            raise ValueError("min_items must be non-negative")
        if self.max_items < self.min_items:
            raise ValueError("max_items must be >= min_items")
        if self.target_items is not None:
            if self.target_items < self.min_items:
                raise ValueError("target_items must be >= min_items")
            if self.target_items > self.max_items:
                raise ValueError("target_items must be <= max_items")


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
        return available_items


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
        self.areas = areas
        self.strict = strict

        self._item_to_area: dict[int, ContentArea] = {}
        for area in areas:
            for item in area.items:
                if item in self._item_to_area:
                    raise ValueError(f"Item {item} belongs to multiple content areas")
                self._item_to_area[item] = area

        self._area_counts: dict[str, int] = {area.name: 0 for area in areas}

    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        self._update_counts(administered_items)

        eligible = set()

        for item_idx in available_items:
            if self._is_item_eligible(item_idx):
                eligible.add(item_idx)

        if not eligible:
            eligible = self._get_priority_items(available_items)

        return eligible if eligible else available_items

    def _update_counts(self, administered_items: list[int]) -> None:
        """Update content area counts."""
        self._area_counts = {area.name: 0 for area in self.areas}
        for item_idx in administered_items:
            if item_idx in self._item_to_area:
                area = self._item_to_area[item_idx]
                self._area_counts[area.name] += 1

    def _is_item_eligible(self, item_idx: int) -> bool:
        """Check if an item is eligible based on content constraints."""
        if item_idx not in self._item_to_area:
            return True

        area = self._item_to_area[item_idx]
        current_count = self._area_counts[area.name]

        if self.strict:
            return current_count < area.max_items
        else:
            target = area.target_items or area.max_items
            return current_count < target

    def _get_priority_items(self, available_items: set[int]) -> set[int]:
        """Get items from areas that need more items."""
        priority_items = set()

        for area in self.areas:
            current_count = self._area_counts[area.name]
            if current_count < area.min_items:
                area_available = available_items & area.items
                priority_items.update(area_available)

        return priority_items

    def is_blueprint_satisfied(self, administered_items: list[int]) -> bool:
        """Check if all minimum requirements are met.

        Parameters
        ----------
        administered_items : list[int]
            List of administered item indices.

        Returns
        -------
        bool
            True if all content areas meet minimum requirements.
        """
        self._update_counts(administered_items)

        for area in self.areas:
            if self._area_counts[area.name] < area.min_items:
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
        self._update_counts(administered_items)
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
        self._update_counts(administered_items)

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
    """

    def __init__(
        self,
        item_weights: dict[int, float],
        area_targets: dict[str, float],
        item_areas: dict[int, str],
    ):
        self.item_weights = item_weights
        self.area_targets = area_targets
        self.item_areas = item_areas

    def filter_items(
        self,
        available_items: set[int],
        administered_items: list[int],
    ) -> set[int]:
        return available_items

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
        n_administered = len(administered_items)
        if n_administered == 0:
            return {i: self.item_weights.get(i, 1.0) for i in available_items}

        area_counts: dict[str, int] = {}
        for item_idx in administered_items:
            area = self.item_areas.get(item_idx, "unknown")
            area_counts[area] = area_counts.get(area, 0) + 1

        weights = {}
        for item_idx in available_items:
            area = self.item_areas.get(item_idx, "unknown")
            current_prop = area_counts.get(area, 0) / n_administered
            target_prop = self.area_targets.get(area, 1.0 / len(self.area_targets))

            if current_prop < target_prop:
                multiplier = target_prop / max(current_prop, 0.01)
            else:
                multiplier = 1.0

            base_weight = self.item_weights.get(item_idx, 1.0)
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

    methods = {
        "blueprint": ContentBlueprint,
        "weighted": WeightedContent,
        "none": NoContentConstraint,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        valid = ", ".join(methods.keys())
        raise ValueError(
            f"Unknown content constraint method '{method}'. Valid options: {valid}"
        )

    return methods[method_lower](**kwargs)
