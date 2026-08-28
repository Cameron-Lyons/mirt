from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ParameterLink:
    """Describes how a parameter is linked across groups.

    Attributes
    ----------
    param_name : str
        Name of the parameter.
    is_shared : bool
        Whether the parameter is shared (constrained equal) across groups.
    shared_items : set[int] | None
        Item indices that are shared. None means all items.
    free_items : set[int]
        Item indices that are free (not constrained).
    """

    param_name: str
    is_shared: bool = False
    shared_items: set[int] | None = None
    free_items: set[int] = field(default_factory=set)

    def is_item_shared(self, item_idx: int) -> bool:
        """Check if a specific item's parameter is shared."""
        if not self.is_shared:
            return False
        if item_idx in self.free_items:
            return False
        if self.shared_items is None:
            return True
        return item_idx in self.shared_items


class MultigroupModel:
    """Container for multiple group-specific IRT models with shared constraints.

    This class manages multiple copies of an IRT model (one per group) and
    tracks which parameters are shared vs. free across groups.

    Parameters
    ----------
    base_model : BaseItemModel
        Template model defining item structure. Will be copied for each group.
    n_groups : int
        Number of groups.
    group_labels : sequence of str, optional
        Human-readable labels for each group.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        n_groups: int,
        group_labels: Sequence[str] | None = None,
    ) -> None:
        if (
            isinstance(n_groups, (bool, np.bool_))
            or not isinstance(n_groups, (int, np.integer))
            or n_groups < 2
        ):
            raise ValueError("n_groups must be an integer greater than or equal to 2")

        self.n_groups = int(n_groups)
        self.n_items = base_model.n_items
        self.n_factors = base_model.n_factors
        self.model_name = base_model.model_name
        self.item_names = base_model.item_names.copy()

        if group_labels is None:
            self.group_labels = [f"Group_{g}" for g in range(self.n_groups)]
        else:
            if isinstance(group_labels, (str, bytes)):
                raise ValueError("group_labels must be a sequence of unique strings")
            labels = list(group_labels)
            if len(labels) != self.n_groups:
                raise ValueError(
                    f"group_labels length ({len(labels)}) must match "
                    f"n_groups ({self.n_groups})"
                )
            if not all(isinstance(label, str) and label for label in labels):
                raise ValueError("group_labels must contain non-empty strings")
            if len(set(labels)) != len(labels):
                raise ValueError("group_labels must be unique")
            self.group_labels = labels

        self._group_models: list[BaseItemModel] = []
        for _ in range(self.n_groups):
            self._group_models.append(base_model.copy())

        self._parameter_links: dict[str, ParameterLink] = {}
        for param_name in base_model.parameters.keys():
            self._parameter_links[param_name] = ParameterLink(param_name=param_name)

        self._base_model_class = base_model.__class__
        self._is_polytomous = base_model.is_polytomous

    @property
    def group_models(self) -> list[BaseItemModel]:
        """Get list of group-specific models."""
        return list(self._group_models)

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return list(self._parameter_links.keys())

    @property
    def is_polytomous(self) -> bool:
        """Whether the model is polytomous."""
        return self._is_polytomous

    @property
    def is_fitted(self) -> bool:
        """Check if all group models are fitted."""
        return all(m._is_fitted for m in self._group_models)

    def get_group_model(self, group_idx: int) -> BaseItemModel:
        """Get model for a specific group.

        Parameters
        ----------
        group_idx : int
            Group index (0-indexed).

        Returns
        -------
        BaseItemModel
            The group's model.
        """
        validated = self._validate_group_index(group_idx, name="group_idx")
        return self._group_models[validated]

    def _validate_group_index(self, group_idx: int, *, name: str) -> int:
        """Return a non-Boolean group index within the configured range."""
        if isinstance(group_idx, (bool, np.bool_)) or not isinstance(
            group_idx, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        validated = int(group_idx)
        if validated < 0 or validated >= self.n_groups:
            raise IndexError(f"{name} {validated} out of range [0, {self.n_groups})")
        return validated

    def _parameter_is_item_major(self, param_name: str) -> bool:
        """Whether a parameter stores one leading block per item."""
        values = self._group_models[0].parameters[param_name]
        return values.ndim > 0 and values.shape[0] == self.n_items

    def _validate_parameter_items(
        self,
        param_name: str,
        item_indices: list[int],
    ) -> set[int]:
        """Validate an item subset for one item-major parameter."""
        if not self._parameter_is_item_major(param_name):
            raise ValueError(
                f"Parameter {param_name} does not contain item-specific values"
            )
        if isinstance(item_indices, (str, bytes)):
            raise TypeError("item_indices must be a sequence of integers")
        try:
            indices = list(item_indices)
        except TypeError as exc:
            raise TypeError("item_indices must be a sequence of integers") from exc
        if not indices:
            raise ValueError("item_indices must contain at least one item")
        if any(
            isinstance(item_idx, (bool, np.bool_))
            or not isinstance(item_idx, (int, np.integer))
            for item_idx in indices
        ):
            raise TypeError("item_indices must contain only integers")
        validated = [int(item_idx) for item_idx in indices]
        if len(set(validated)) != len(validated):
            raise ValueError("item_indices must not contain duplicates")
        if any(item_idx < 0 or item_idx >= self.n_items for item_idx in validated):
            raise IndexError(f"item index out of range [0, {self.n_items})")
        return set(validated)

    def get_group_parameters(self, group_idx: int) -> dict[str, NDArray[np.float64]]:
        """Get all parameters for a specific group.

        Parameters
        ----------
        group_idx : int
            Group index.

        Returns
        -------
        dict
            Dictionary of parameter arrays.
        """
        return self.get_group_model(group_idx).parameters

    def set_group_parameters(
        self,
        group_idx: int,
        **params: NDArray[np.float64],
    ) -> None:
        """Set parameters for a specific group.

        Parameters
        ----------
        group_idx : int
            Group index.
        **params
            Parameter name-value pairs.
        """
        self.get_group_model(group_idx).set_parameters(**params)

    def set_shared_parameter(
        self,
        param_name: str,
        item_indices: list[int] | None = None,
    ) -> None:
        """Mark a parameter as shared (constrained equal) across groups.

        Parameters
        ----------
        param_name : str
            Name of the parameter to share.
        item_indices : list[int], optional
            Specific items to share. If None, all items are shared.
        """
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]

        if item_indices is None:
            link.is_shared = True
            link.shared_items = None
            link.free_items = set()
            return

        validated = self._validate_parameter_items(param_name, item_indices)
        if not link.is_shared:
            link.shared_items = set()
        elif link.shared_items is None:
            # The entire parameter is already shared; this call only restores
            # explicitly freed items in the requested subset.
            link.free_items -= validated
            return

        link.is_shared = True
        link.shared_items.update(validated)
        link.free_items -= validated

    def set_group_specific_parameter(
        self,
        param_name: str,
        item_indices: list[int] | None = None,
    ) -> None:
        """Mark a parameter as group-specific (free across groups).

        Parameters
        ----------
        param_name : str
            Name of the parameter to free.
        item_indices : list[int], optional
            Specific items to free. If None, all items are freed.
        """
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]

        if item_indices is None:
            link.is_shared = False
            link.shared_items = None
            link.free_items = set()
            return

        validated = self._validate_parameter_items(param_name, item_indices)
        link.free_items.update(validated)
        if link.shared_items is not None:
            link.shared_items -= validated
            if not link.shared_items:
                link.is_shared = False
                link.shared_items = None
                link.free_items = set()

    def is_parameter_shared(self, param_name: str) -> bool:
        """Check if a parameter is shared across groups."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self._parameter_links[param_name].is_shared

    def is_item_parameter_shared(self, param_name: str, item_idx: int) -> bool:
        """Check if a specific item's parameter is shared."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")
        validated = self._validate_parameter_items(param_name, [item_idx])
        return self._parameter_links[param_name].is_item_shared(validated.pop())

    def get_shared_items(self, param_name: str) -> list[int]:
        """Get list of items that have shared parameters."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]
        if not link.is_shared or not self._parameter_is_item_major(param_name):
            return []

        if link.shared_items is None:
            return [i for i in range(self.n_items) if i not in link.free_items]
        return [
            i
            for i in range(self.n_items)
            if i in link.shared_items and i not in link.free_items
        ]

    def get_free_items(self, param_name: str) -> list[int]:
        """Get list of items that have group-specific parameters."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]
        if not self._parameter_is_item_major(param_name):
            return []
        if not link.is_shared:
            return list(range(self.n_items))

        if link.shared_items is None:
            return list(link.free_items)
        return [
            i
            for i in range(self.n_items)
            if i not in link.shared_items or i in link.free_items
        ]

    def synchronize_shared_parameters(self) -> None:
        """Synchronize shared parameters across groups.

        Takes the mean of shared parameters across groups and sets all
        groups to that value.
        """
        for param_name, link in self._parameter_links.items():
            if not link.is_shared:
                continue

            stacked_values = np.stack(
                [
                    group_model.parameters[param_name]
                    for group_model in self._group_models
                ]
            )
            stacked_masks = np.stack(
                [
                    np.asarray(
                        group_model.free_parameter_masks[param_name], dtype=np.bool_
                    )
                    for group_model in self._group_models
                ]
            )
            active_counts = np.count_nonzero(stacked_masks, axis=0)
            shared_mask = active_counts > 0
            if self._parameter_is_item_major(param_name):
                item_mask = np.zeros(self.n_items, dtype=np.bool_)
                item_mask[self.get_shared_items(param_name)] = True
                item_mask = item_mask.reshape(
                    (self.n_items,) + (1,) * (shared_mask.ndim - 1)
                )
                shared_mask &= item_mask
            if not np.any(shared_mask):
                continue

            shared_mean = np.divide(
                np.sum(stacked_values, axis=0, where=stacked_masks),
                active_counts,
                out=np.zeros_like(stacked_values[0]),
                where=active_counts > 0,
            )
            for group_model, values in zip(
                self._group_models, stacked_values, strict=True
            ):
                updated = values.copy()
                np.copyto(updated, shared_mean, where=shared_mask)
                group_model.set_parameters(**{param_name: updated})

    def copy_shared_to_all(self, source_group: int = 0) -> None:
        """Copy shared parameters from source group to all groups.

        Parameters
        ----------
        source_group : int
            Group index to copy from.
        """
        source_group = self._validate_group_index(source_group, name="source_group")

        source_params = self._group_models[source_group].parameters

        for param_name, link in self._parameter_links.items():
            if not link.is_shared:
                continue

            source_values = source_params[param_name]
            source_mask = np.asarray(
                self._group_models[source_group].free_parameter_masks[param_name],
                dtype=np.bool_,
            )
            if self._parameter_is_item_major(param_name):
                item_mask = np.zeros(self.n_items, dtype=np.bool_)
                item_mask[self.get_shared_items(param_name)] = True
                item_mask = item_mask.reshape(
                    (self.n_items,) + (1,) * (source_mask.ndim - 1)
                )
                source_mask &= item_mask
            if not np.any(source_mask):
                continue

            for g in range(self.n_groups):
                if g == source_group:
                    continue
                target_values = self._group_models[g].parameters[param_name]
                np.copyto(target_values, source_values, where=source_mask)
                self._group_models[g].set_parameters(**{param_name: target_values})

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters accounting for constraints."""
        n_params = 0

        for param_name in self.parameter_names:
            masks = [
                np.asarray(group_model.free_parameter_masks[param_name], dtype=np.bool_)
                for group_model in self._group_models
            ]
            expected_shape = masks[0].shape
            if any(mask.shape != expected_shape for mask in masks[1:]):
                raise ValueError(
                    f"free-parameter masks for {param_name} must have equal shapes"
                )

            link = self._parameter_links[param_name]
            if self._parameter_is_item_major(param_name):
                for item_idx in range(self.n_items):
                    if link.is_item_shared(item_idx):
                        n_params += int(
                            np.count_nonzero(
                                np.logical_or.reduce([mask[item_idx] for mask in masks])
                            )
                        )
                    else:
                        n_params += sum(
                            int(np.count_nonzero(mask[item_idx])) for mask in masks
                        )
            elif link.is_shared:
                n_params += int(np.count_nonzero(np.logical_or.reduce(masks)))
            else:
                n_params += sum(int(np.count_nonzero(mask)) for mask in masks)

        return n_params

    def __repr__(self) -> str:
        shared = [
            name for name in self.parameter_names if self.is_parameter_shared(name)
        ]
        return (
            f"MultigroupModel(model={self.model_name}, "
            f"n_groups={self.n_groups}, "
            f"n_items={self.n_items}, "
            f"shared={shared})"
        )
