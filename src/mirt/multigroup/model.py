from __future__ import annotations

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
    group_labels : list[str], optional
        Human-readable labels for each group.
    """

    def __init__(
        self,
        base_model: BaseItemModel,
        n_groups: int,
        group_labels: list[str] | None = None,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be at least 2")

        self.n_groups = n_groups
        self.n_items = base_model.n_items
        self.n_factors = base_model.n_factors
        self.model_name = base_model.model_name
        self.item_names = base_model.item_names.copy()

        if group_labels is None:
            self.group_labels = [f"Group_{g}" for g in range(n_groups)]
        else:
            if len(group_labels) != n_groups:
                raise ValueError(
                    f"group_labels length ({len(group_labels)}) must match n_groups ({n_groups})"
                )
            self.group_labels = list(group_labels)

        self._group_models: list[BaseItemModel] = []
        for _ in range(n_groups):
            self._group_models.append(base_model.copy())

        self._parameter_links: dict[str, ParameterLink] = {}
        for param_name in base_model.parameters.keys():
            self._parameter_links[param_name] = ParameterLink(param_name=param_name)

        self._base_model_class = base_model.__class__
        self._is_polytomous = base_model.is_polytomous

    @property
    def group_models(self) -> list[BaseItemModel]:
        """Get list of group-specific models."""
        return self._group_models

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
        if group_idx < 0 or group_idx >= self.n_groups:
            raise IndexError(f"group_idx {group_idx} out of range [0, {self.n_groups})")
        return self._group_models[group_idx]

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
        link.is_shared = True

        if item_indices is None:
            link.shared_items = None
            link.free_items = set()
        else:
            if link.shared_items is None:
                link.shared_items = set(range(self.n_items))
            link.shared_items.update(item_indices)
            link.free_items -= set(item_indices)

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
        else:
            link.free_items.update(item_indices)
            if link.shared_items is not None:
                link.shared_items -= set(item_indices)

    def is_parameter_shared(self, param_name: str) -> bool:
        """Check if a parameter is shared across groups."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self._parameter_links[param_name].is_shared

    def is_item_parameter_shared(self, param_name: str, item_idx: int) -> bool:
        """Check if a specific item's parameter is shared."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self._parameter_links[param_name].is_item_shared(item_idx)

    def get_shared_items(self, param_name: str) -> list[int]:
        """Get list of items that have shared parameters."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]
        if not link.is_shared:
            return []

        if link.shared_items is None:
            return [i for i in range(self.n_items) if i not in link.free_items]
        return [i for i in link.shared_items if i not in link.free_items]

    def get_free_items(self, param_name: str) -> list[int]:
        """Get list of items that have group-specific parameters."""
        if param_name not in self._parameter_links:
            raise ValueError(f"Unknown parameter: {param_name}")

        link = self._parameter_links[param_name]
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

            shared_items = self.get_shared_items(param_name)
            if not shared_items:
                continue

            all_values = [
                self._group_models[g].parameters[param_name]
                for g in range(self.n_groups)
            ]

            for item_idx in shared_items:
                if all_values[0].ndim == 1:
                    mean_val = np.mean([v[item_idx] for v in all_values])
                    for g in range(self.n_groups):
                        self._group_models[g].set_item_parameter(
                            item_idx, param_name, mean_val
                        )
                else:
                    mean_val = np.mean([v[item_idx] for v in all_values], axis=0)
                    for g in range(self.n_groups):
                        self._group_models[g].set_item_parameter(
                            item_idx, param_name, mean_val
                        )

    def copy_shared_to_all(self, source_group: int = 0) -> None:
        """Copy shared parameters from source group to all groups.

        Parameters
        ----------
        source_group : int
            Group index to copy from.
        """
        if source_group < 0 or source_group >= self.n_groups:
            raise IndexError(f"source_group {source_group} out of range")

        source_params = self._group_models[source_group].parameters

        for param_name, link in self._parameter_links.items():
            if not link.is_shared:
                continue

            shared_items = self.get_shared_items(param_name)
            if not shared_items:
                continue

            source_values = source_params[param_name]
            for g in range(self.n_groups):
                if g == source_group:
                    continue
                for item_idx in shared_items:
                    if source_values.ndim == 1:
                        val = source_values[item_idx]
                    else:
                        val = source_values[item_idx]
                    self._group_models[g].set_item_parameter(item_idx, param_name, val)

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters accounting for constraints."""
        n_params = 0

        for param_name in self.parameter_names:
            base_params = self._group_models[0].parameters[param_name]

            shared_items = self.get_shared_items(param_name)
            free_items = self.get_free_items(param_name)

            if base_params.ndim == 1:
                n_params += len(shared_items)
                n_params += len(free_items) * self.n_groups
            else:
                n_per_item = base_params.shape[1]
                n_params += len(shared_items) * n_per_item
                n_params += len(free_items) * n_per_item * self.n_groups

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
