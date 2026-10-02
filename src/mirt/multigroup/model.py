from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
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
        self._fixed_item_parameters: dict[str, dict[int, NDArray[np.float64]]] = {}

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

    def fix_item_parameters(
        self,
        parameters: Mapping[str, Mapping[int, float | NDArray[np.float64]]],
    ) -> None:
        """Fix specified item parameter blocks to the same values in every group.

        The outer keys are stored parameter names; inner keys are item indices.
        A value must have the shape of that parameter's item block (a scalar
        for a one-dimensional parameter array). Values are copied and validated
        before any model is changed. Repeated calls add or replace fixed blocks.
        """
        if not isinstance(parameters, Mapping):
            raise TypeError("fixed parameters must be a mapping")
        candidate = {
            name: {item: value.copy() for item, value in items.items()}
            for name, items in self._fixed_item_parameters.items()
        }
        template = self._group_models[0].parameters
        for name, items in parameters.items():
            if name not in self._parameter_links:
                raise ValueError(f"Unknown parameter: {name}")
            if not isinstance(items, Mapping):
                raise TypeError(f"fixed {name} values must be an item-index mapping")
            for item, value in items.items():
                validated = self._validate_parameter_items(name, [item]).pop()
                row = np.array(value, dtype=np.float64, copy=True)
                expected = template[name][validated].shape
                if row.shape != expected or not np.all(np.isfinite(row)):
                    raise ValueError(
                        f"fixed {name}[{validated}] must be finite with shape {expected}"
                    )
                candidate.setdefault(name, {})[validated] = row

        # Public setters may validate ordering, identification, or model-specific
        # constraints. Check all group contexts before changing the originals.
        validated_parameters = []
        for group_model in self._group_models:
            trial = copy.deepcopy(group_model)
            original = trial.parameters
            updates = {}
            for name, items in candidate.items():
                values = original[name].copy()
                for item, value in items.items():
                    values[item] = value
                if not np.array_equal(values, original[name]):
                    updates[name] = values
            trial.set_parameters(**updates)
            for name, items in candidate.items():
                for item, value in items.items():
                    if not np.array_equal(trial.parameters[name][item], value):
                        raise ValueError(
                            f"fixed {name}[{item}] conflicts with model identification"
                        )
            validated_parameters.append(trial.parameters)
        for group_model, values in zip(
            self._group_models, validated_parameters, strict=True
        ):
            current = group_model.parameters
            group_model.set_parameters(
                **{
                    name: value
                    for name, value in values.items()
                    if not np.array_equal(value, current[name])
                }
            )
        self._fixed_item_parameters = candidate

    @property
    def fixed_item_parameters(self) -> dict[str, dict[int, NDArray[np.float64]]]:
        """Return independent copies of the configured fixed parameter blocks."""
        return {
            name: {item: value.copy() for item, value in items.items()}
            for name, items in self._fixed_item_parameters.items()
        }

    def _raw_free_parameter_masks(self, group_idx: int) -> dict[str, NDArray[np.bool_]]:
        masks = {
            name: np.asarray(mask, dtype=np.bool_).copy()
            for name, mask in self.get_group_model(
                group_idx
            ).free_parameter_masks.items()
        }
        for name, items in self._fixed_item_parameters.items():
            for item in items:
                masks[name][item] = False
        return masks

    def _shared_coordinate_mask(self, param_name: str) -> NDArray[np.bool_]:
        shape = self._group_models[0].parameters[param_name].shape
        shared = np.zeros(shape, dtype=np.bool_)
        if self._parameter_links[param_name].is_shared:
            if self._parameter_is_item_major(param_name):
                shared[self.get_shared_items(param_name)] = True
            else:
                shared[...] = True
        return shared

    def _explicit_fixed_parameter_masks(
        self, group_idx: int
    ) -> dict[str, NDArray[np.bool_]]:
        group = self.get_group_model(group_idx)
        fixed = {
            name: np.zeros(value.shape, dtype=np.bool_)
            for name, value in group.parameters.items()
        }
        if group._free_parameter_restrictions:
            intrinsic = group.copy().set_free_parameter_masks(None).free_parameter_masks
            for name, restriction in group._free_parameter_restrictions.items():
                fixed[name] = intrinsic[name] & ~restriction
        for name, items in self._fixed_item_parameters.items():
            for item in items:
                fixed[name][item] = True
        return fixed

    def effective_free_parameter_masks(
        self, group_idx: int
    ) -> dict[str, NDArray[np.bool_]]:
        """Account for structural, explicit, and shared fixed coordinates.

        A shared coordinate is known when it is fixed in any group. Such a
        coordinate contributes no fitted parameter in the remaining groups.
        Synchronization propagates that known value to all linked groups.
        """
        masks = self._raw_free_parameter_masks(group_idx)
        fixed_masks = [
            self._explicit_fixed_parameter_masks(g) for g in range(self.n_groups)
        ]
        for name in masks:
            if self._parameter_links[name].is_shared:
                known = np.logical_or.reduce([group[name] for group in fixed_masks])
                masks[name] &= ~(known & self._shared_coordinate_mask(name))
        return masks

    def enforce_fixed_parameters(self) -> None:
        """Restore fixed values after initialization or synchronization."""
        if not self._fixed_item_parameters:
            return
        for group_model in self._group_models:
            updates = {}
            current = group_model.parameters
            for name, items in self._fixed_item_parameters.items():
                values = current[name]
                original = values.copy()
                for item, value in items.items():
                    values[item] = value
                if not np.array_equal(values, original):
                    updates[name] = values
            group_model.set_parameters(**updates)

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
        """Synchronize free shared values and propagate known shared values.

        Free coordinates use their group mean. A coordinate fixed in any
        group keeps that exact value in every linked group; inconsistent
        fixed values are rejected before changing shared parameter arrays.
        """
        self.enforce_fixed_parameters()
        group_parameters = [group.parameters for group in self._group_models]
        group_masks = [self._raw_free_parameter_masks(g) for g in range(self.n_groups)]
        fixed_masks = [
            self._explicit_fixed_parameter_masks(g) for g in range(self.n_groups)
        ]
        updates: list[dict[str, NDArray[np.float64]]] = [{} for _ in self._group_models]
        for name, link in self._parameter_links.items():
            if not link.is_shared:
                continue
            shared = self._shared_coordinate_mask(name)
            if not np.any(shared):
                continue
            values = np.stack([parameters[name] for parameters in group_parameters])
            masks = np.stack([mask[name] for mask in group_masks])
            fixed = np.stack([mask[name] for mask in fixed_masks])
            has_fixed = np.any(fixed, axis=0)
            shared &= np.any(masks, axis=0) | has_fixed
            first_fixed = np.argmax(fixed, axis=0)
            fixed_values = np.take_along_axis(values, first_fixed[None, ...], axis=0)[0]
            if np.any(fixed & shared & (values != fixed_values)):
                raise ValueError(f"Shared {name} has incompatible fixed group values")
            target = np.where(has_fixed, fixed_values, values.mean(axis=0))
            for g, current in enumerate(values):
                updated = current.copy()
                np.copyto(updated, target, where=shared)
                if not np.array_equal(updated, current):
                    updates[g][name] = updated
        for group, update in zip(self._group_models, updates, strict=True):
            if update:
                group.set_parameters(**update)

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
                self.effective_free_parameter_masks(source_group)[param_name],
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
        self.synchronize_shared_parameters()

    @property
    def n_parameters(self) -> int:
        """Total number of free parameters accounting for constraints."""
        n_params = 0

        for param_name in self.parameter_names:
            masks = [
                self.effective_free_parameter_masks(group_idx)[param_name]
                for group_idx in range(self.n_groups)
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
