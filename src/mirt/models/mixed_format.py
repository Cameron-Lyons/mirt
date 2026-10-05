"""Tests whose items follow different item-response families.

A :class:`MixedItemModel` combines component models, such as 3PL
multiple-choice items and graded constructed-response items, that measure
one shared latent trait. Every item belongs to exactly one component.
Curves, likelihoods and information are evaluated by the components, so each
keeps its vectorized or native kernels, and the mixed model places their
items in test order.

This is not a latent-class mixture; see :class:`~mirt.models.mixture.MixtureIRT`
for that.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._model_defaults import record_model_base as _record_model_base
from mirt._model_defaults import uses_original_model_hook
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models.base import BaseItemModel, PolytomousItemModel

_SEPARATOR = "."
# Hooks through which the mixed model defers to its components.
_COMPONENT_HOOKS = ("probability", "log_likelihood", "log_likelihood_batch")


def require_single_family(
    model: object, operation: str, hint: str | None = None
) -> None:
    """Raise when ``operation`` cannot handle a mixed-format model.

    Parameters
    ----------
    model : object
        Model passed to ``operation``.
    operation : str
        Name of the calling function, used in the error message.
    hint : str, optional
        What to do instead. By default the components are suggested.

    Raises
    ------
    MirtModelError
        If ``model`` is a :class:`MixedItemModel`.
    """
    if isinstance(model, MixedItemModel):
        advice = hint or "apply it to each component in MixedItemModel.components"
        raise MirtModelError(
            f"{operation} requires a single item family; {advice}",
            model_type=model.model_name,
        )


def uses_component_likelihoods(model: MixedItemModel) -> bool:
    """Whether ``model`` evaluates curves and likelihoods by its components.

    A subclass or instance that replaces these hooks may couple persons or
    items, so shortcuts that rely on the component likelihoods do not apply.
    """
    return all(uses_original_model_hook(model, name) for name in _COMPONENT_HOOKS)


class _ComponentParameters(MutableMapping[str, NDArray[np.float64]]):
    """Live view of the component parameter arrays under qualified names.

    Reads return the stored component arrays and writes replace them, as for
    the plain parameter dictionary of a single-family model, so code that
    perturbs ``model._parameters`` directly reaches the components.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: MixedItemModel) -> None:
        self._owner = owner

    def _locate(self, name: str) -> tuple[BaseItemModel, str]:
        try:
            component, local = self._owner._parameter_index[name]
        except (KeyError, TypeError):
            raise KeyError(name) from None
        return self._owner._models[component], local

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        model, local = self._locate(name)
        return model._parameters[local]

    def __setitem__(self, name: str, value: NDArray[np.float64]) -> None:
        model, local = self._locate(name)
        model._parameters[local] = value

    def __delitem__(self, name: str) -> None:
        raise TypeError("mixed-format model parameters cannot be removed")

    def __iter__(self) -> Iterator[str]:
        return iter(self._owner._parameter_index)

    def __len__(self) -> int:
        return len(self._owner._parameter_index)


def _validate_components(
    components: Sequence[tuple[BaseItemModel, ArrayLike]],
) -> tuple[list[BaseItemModel], list[NDArray[np.intp]]]:
    """Return the component models and their validated test positions."""
    if (
        isinstance(components, (str, bytes))
        or not isinstance(components, Sequence)
        or not components
    ):
        raise MirtValidationError(
            "components must be a non-empty sequence of (model, items) pairs",
            parameter="components",
            value=type(components).__name__,
        )
    models: list[BaseItemModel] = []
    positions: list[NDArray[np.intp]] = []
    for index, entry in enumerate(components):
        try:
            model, items = entry
        except (TypeError, ValueError):
            raise MirtValidationError(
                f"component {index} must be a (model, items) pair",
                parameter="components",
                value=entry,
            ) from None
        if isinstance(model, MixedItemModel):
            raise MirtModelError(
                "mixed-format models cannot be nested; list their components",
                model_type=model.model_name,
            )
        if not isinstance(model, BaseItemModel):
            raise MirtValidationError(
                f"component {index} model must be a BaseItemModel",
                parameter="components",
                value=type(model).__name__,
            )
        values = np.asarray(items)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise MirtValidationError(
                f"component {index} items must be a one-dimensional integer array",
                parameter="components",
                value=items,
            )
        if values.size != model.n_items:
            raise MirtValidationError(
                f"component {index} ({model.model_name}) has {model.n_items} "
                f"items but {values.size} positions",
                parameter="components",
                value=values.size,
                expected=str(model.n_items),
            )
        models.append(model)
        positions.append(values.astype(np.intp))

    n_items = sum(items.size for items in positions)
    covered = np.concatenate(positions)
    if np.any((covered < 0) | (covered >= n_items)) or (
        np.unique(covered).size != n_items
    ):
        raise MirtValidationError(
            "component items must cover positions 0, ..., n_items - 1 exactly once",
            parameter="components",
            expected=f"each of 0..{n_items - 1} once",
        )
    factors = {model.n_factors for model in models}
    if len(factors) != 1:
        raise MirtModelError(
            "all components of a mixed-format model must share n_factors",
            model_type="Mixed",
            value=sorted(factors),
        )
    return models, positions


def _component_prefixes(models: list[BaseItemModel]) -> list[str]:
    """Name components by family, numbering repeated families from 2."""
    prefixes: list[str] = []
    for model in models:
        base = str(model.model_name)
        prefix, count = base, 1
        while prefix in prefixes:
            count += 1
            prefix = f"{base}_{count}"
        prefixes.append(prefix)
    return prefixes


def _component_log_likelihood_batch(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch = getattr(model, "log_likelihood_batch", None)
    if callable(batch):
        return np.asarray(batch(responses, theta), dtype=np.float64)
    return np.column_stack(
        [model.log_likelihood(responses, theta[q : q + 1]) for q in range(len(theta))]
    )


# Recording the authored hooks lets row-aligned scorers batch this model.
@_record_model_base
class MixedItemModel(PolytomousItemModel):
    """Item response model whose items come from different families.

    A mixed-format test, for example 3PL multiple-choice items with graded
    constructed-response items, is described by one component model per
    family and the test positions of its items. All components share the
    latent trait, so a response pattern's likelihood is the product of the
    component likelihoods. Each item reports ``n_categories``, with two
    categories for dichotomous items, and ``probability`` returns padded
    category probabilities whose binary items read ``[1 - p, p]``. Single-item
    curves, information and ``get_item_parameters`` come from the owning
    component.

    Parameters are stored by the components and exposed under qualified
    names ``"<component>.<parameter>"`` such as ``"3PL.guessing"`` or
    ``"GRM.thresholds"``; their arrays are indexed by the component's own
    item order. Components are named by ``model_name``, and a repeated family
    is numbered from two (``"2PL_2"``). ``n_parameters`` is the sum over
    components.

    Fit with :class:`~mirt.estimation.mixed_format_em.MixedFormatEMEstimator`
    or ``fit_mirt(data, model=["3PL", ..., "GRM"])``. Fitted components can
    also be combined to score, simulate or administer an adaptive test from a
    pre-calibrated mixed item pool.

    Parameters
    ----------
    components : sequence of (BaseItemModel, array_like of int)
        Component models with the test positions of their items. Every
        position ``0, ..., n_items - 1`` must be covered exactly once, and all
        components must share ``n_factors``. The models are copied, with
        their parameters, fixed coordinates and fitted state.
    item_names : list of str, optional
        Names of all items in test order. By default the component item
        names are used when they are unique, and ``Item_<j>`` otherwise.

    Raises
    ------
    MirtValidationError
        If the components or their item positions are malformed.
    MirtModelError
        If the components differ in ``n_factors`` or are themselves mixed.

    Examples
    --------
    >>> from mirt import GradedResponseModel, MixedItemModel, ThreeParameterLogistic
    >>> model = MixedItemModel(
    ...     [
    ...         (ThreeParameterLogistic(20), range(20)),
    ...         (GradedResponseModel(5, n_categories=4), range(20, 25)),
    ...     ]
    ... )
    >>> model.item_types[18:22]
    ['3PL', '3PL', 'GRM', 'GRM']
    """

    model_name = "Mixed"
    supports_multidimensional = True

    def __init__(
        self,
        components: Sequence[tuple[BaseItemModel, ArrayLike]],
        item_names: list[str] | None = None,
    ) -> None:
        # The base initializers would reset the parameters the components
        # already hold, so only the shared structure is set up here.
        models, positions = _validate_components(components)
        self._models = [model.copy() for model in models]
        self._items = positions
        self._prefixes = _component_prefixes(self._models)
        self.n_items = int(sum(items.size for items in positions))
        self.n_factors = int(self._models[0].n_factors)

        self._item_component = np.empty(self.n_items, dtype=np.intp)
        self._item_local = np.empty(self.n_items, dtype=np.intp)
        counts = np.empty(self.n_items, dtype=np.intp)
        component_names: list[str] = [""] * self.n_items
        for index, (model, items) in enumerate(
            zip(self._models, positions, strict=True)
        ):
            self._item_component[items] = index
            self._item_local[items] = np.arange(items.size)
            counts[items] = (
                np.broadcast_to(np.asarray(model.n_categories), (model.n_items,))
                if model.is_polytomous
                else 2
            )
            for local, item in enumerate(items):
                component_names[item] = str(model.item_names[local])
        self._n_categories = [int(count) for count in counts]

        if item_names is None:
            unique = len(set(component_names)) == self.n_items
            item_names = (
                component_names
                if unique
                else [f"Item_{index}" for index in range(self.n_items)]
            )
        elif len(item_names) != self.n_items:
            raise MirtValidationError(
                f"Length of item_names ({len(item_names)}) must match n_items "
                f"({self.n_items})",
                parameter="item_names",
                value=len(item_names),
                expected=str(self.n_items),
            )
        self.item_names = list(item_names)
        for model, items in zip(self._models, positions, strict=True):
            model.item_names = [self.item_names[item] for item in items]

        self._parameter_index: dict[str, tuple[int, str]] = {
            f"{prefix}{_SEPARATOR}{name}": (index, name)
            for index, (prefix, model) in enumerate(
                zip(self._prefixes, self._models, strict=True)
            )
            for name in model._parameters
        }

    @classmethod
    def from_itemtypes(
        cls,
        itemtypes: Sequence[str],
        *,
        n_categories: int | Sequence[int] | None = None,
        n_factors: int = 1,
        item_names: list[str] | None = None,
        responses: NDArray[np.int_] | None = None,
    ) -> MixedItemModel:
        """Build an unfitted model from one built-in family name per item.

        Parameters
        ----------
        itemtypes : sequence of str
            Family of each item: "1PL", "2PL", "3PL", "4PL", "GRM", "GPCM",
            "PCM" or "NRM". Items of one family form one component, in the
            order families first appear.
        n_categories : int or sequence of int, optional
            Category count of every polytomous item, or one count per item
            with 2 for dichotomous items. Inferred from ``responses`` when
            omitted.
        n_factors : int, default=1
            Number of latent factors; every family must support it.
        item_names : list of str, optional
            Item names in test order.
        responses : ndarray of shape (n_persons, n_items), optional
            Validated responses used to infer and check category counts.

        Returns
        -------
        MixedItemModel
            Model with family default starting values.
        """
        from mirt.models._factory import build_mixed_item_model

        return build_mixed_item_model(
            itemtypes,
            n_factors=n_factors,
            n_categories=n_categories,
            item_names=item_names,
            responses=responses,
        )

    @property
    def components(self) -> tuple[tuple[BaseItemModel, NDArray[np.intp]], ...]:
        """Component models with the test positions of their items.

        The models are the live components: fitting the mixed model updates
        them, and changing them changes the mixed model.
        """
        return tuple(
            (model, items.copy())
            for model, items in zip(self._models, self._items, strict=True)
        )

    @property
    def component_models(self) -> tuple[BaseItemModel, ...]:
        """The live component models in component order."""
        return tuple(self._models)

    @property
    def component_names(self) -> list[str]:
        """Prefixes that qualify each component's parameter names."""
        return list(self._prefixes)

    @property
    def item_types(self) -> list[str]:
        """Family name (``model_name``) of every item in test order."""
        return [
            str(self._models[component].model_name)
            for component in self._item_component
        ]

    def locate_item(self, item_idx: int) -> tuple[int, int]:
        """Return the component index and the item's index within it."""
        item = self._validate_item_index(item_idx)
        return int(self._item_component[item]), int(self._item_local[item])

    def parameter_component(self, name: str) -> tuple[int, str]:
        """Return the component index and local name of a qualified parameter.

        Raises
        ------
        MirtValidationError
            If ``name`` is not a parameter of this model.
        """
        try:
            return self._parameter_index[name]
        except (KeyError, TypeError):
            valid = ", ".join(self._parameter_index)
            raise MirtValidationError(
                f"Unknown parameter: {name}. Valid parameters: {valid}",
                parameter=str(name),
                expected=valid,
            ) from None

    @property
    def _shared_parameters(self) -> frozenset[str]:
        """Qualified parameters that a component shares across its items."""
        return frozenset(
            f"{prefix}{_SEPARATOR}{name}"
            for prefix, model in zip(self._prefixes, self._models, strict=True)
            for name in model._shared_parameters
        )

    def _item_indexed(self, name: str) -> bool:
        """Whether a qualified parameter has one row per item of its component."""
        component, local = self.parameter_component(name)
        return self._models[component]._item_indexed(local)

    def parameter_items(self, name: str) -> NDArray[np.intp]:
        """Return the test positions of a qualified parameter's component items.

        Row ``r`` of a per-item array ``parameters[name]`` belongs to test
        item ``parameter_items(name)[r]``. The result is read-only.

        Raises
        ------
        MirtValidationError
            If ``name`` is not a parameter of this model.
        """
        component, _ = self.parameter_component(name)
        items = self._items[component].view()
        items.flags.writeable = False
        return items

    @property
    def _parameters(self) -> MutableMapping[str, NDArray[np.float64]]:
        return _ComponentParameters(self)

    @_parameters.setter
    def _parameters(self, values: Mapping[str, NDArray[np.float64]]) -> None:
        view = _ComponentParameters(self)
        for name, value in values.items():
            view[name] = value

    @property
    def _free_parameter_restrictions(self) -> dict[str, NDArray[np.bool_]]:
        return {
            f"{prefix}{_SEPARATOR}{name}": mask
            for prefix, model in zip(self._prefixes, self._models, strict=True)
            for name, mask in model._free_parameter_restrictions.items()
        }

    @_free_parameter_restrictions.setter
    def _free_parameter_restrictions(
        self, masks: Mapping[str, NDArray[np.bool_]]
    ) -> None:
        grouped = self._group_by_component(masks)
        for index, model in enumerate(self._models):
            model._free_parameter_restrictions = grouped.get(index, {})

    @property
    def _is_fitted(self) -> bool:
        return all(model._is_fitted for model in self._models)

    @_is_fitted.setter
    def _is_fitted(self, value: bool) -> None:
        for model in self._models:
            model._is_fitted = bool(value)

    def _group_by_component(self, values: Mapping[str, Any]) -> dict[int, dict]:
        grouped: dict[int, dict] = {}
        for name, value in values.items():
            component, local = self.parameter_component(name)
            grouped.setdefault(component, {})[local] = value
        return grouped

    def _initialize_parameters(self) -> None:
        for model in self._models:
            model._initialize_parameters()

    def item_parameter_arrays(
        self, values: Mapping[str, ArrayLike] | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Arrange component parameters with one row per test item.

        Parameters
        ----------
        values : mapping of str to array_like, optional
            Arrays under qualified names, such as a fit's standard errors.
            Defaults to the parameters; missing names are NaN.

        Returns
        -------
        dict of str to ndarray
            One array per unqualified parameter name, in order of first
            appearance, with ``n_items`` rows. A parameter that is a matrix in
            any component has as many columns as its widest component.
            Items whose family lacks a parameter, and columns beyond an
            item's own, are NaN.

        Raises
        ------
        MirtModelError
            If a component shares a parameter across its items, such as the
            thresholds of a rating-scale component; use
            ``FitResult.parameter_statistics()`` or the component's own
            parameters instead.
        """
        source = self.parameters if values is None else values
        blocks: dict[str, list[tuple[NDArray[np.intp], NDArray[np.float64]]]] = {}
        for qualified, (component, local) in self._parameter_index.items():
            model, items = self._models[component], self._items[component]
            stored = model._parameters[local]
            if stored.ndim not in (1, 2) or not model._item_indexed(local):
                raise MirtModelError(
                    f"{qualified} is shared by the items of its component and "
                    "has no per-item value",
                    model_type=model.model_name,
                )
            raw = source.get(qualified)
            block = (
                np.full(stored.shape, np.nan)
                if raw is None
                else np.asarray(raw, dtype=np.float64)
            )
            if block.shape != stored.shape:
                raise MirtValidationError(
                    f"{qualified} must have shape {stored.shape}",
                    parameter=qualified,
                    value=block.shape,
                    expected=str(stored.shape),
                )
            blocks.setdefault(local, []).append((items, block))

        arrays: dict[str, NDArray[np.float64]] = {}
        for name, parts in blocks.items():
            if all(block.ndim == 1 for _, block in parts):
                arrays[name] = np.full(self.n_items, np.nan)
                for items, block in parts:
                    arrays[name][items] = block
                continue
            columns = [block.reshape(block.shape[0], -1) for _, block in parts]
            width = max(column.shape[1] for column in columns)
            arrays[name] = np.full((self.n_items, width), np.nan)
            for (items, _), column in zip(parts, columns, strict=True):
                arrays[name][items, : column.shape[1]] = column
        return arrays

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Component free-parameter masks under qualified names."""
        return {
            f"{prefix}{_SEPARATOR}{name}": mask
            for prefix, model in zip(self._prefixes, self._models, strict=True)
            for name, mask in model.free_parameter_masks.items()
        }

    def set_free_parameter_masks(
        self, masks: Mapping[str, NDArray[np.bool_]] | None
    ) -> Self:
        """Restrict free coordinates by qualified parameter name.

        Each component validates its masks as
        :meth:`BaseItemModel.set_free_parameter_masks` does. Components
        without an entry keep only their family masks; ``None`` clears every
        restriction.
        """
        if masks is not None and not isinstance(masks, Mapping):
            raise MirtValidationError("masks must be a mapping", parameter="masks")
        grouped = {} if masks is None else self._group_by_component(masks)
        previous = [model._free_parameter_restrictions for model in self._models]
        try:
            for index, model in enumerate(self._models):
                model.set_free_parameter_masks(grouped.get(index, {}))
        except Exception:
            for model, restrictions in zip(self._models, previous, strict=True):
                model._free_parameter_restrictions = restrictions
            raise
        return self

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set parameters by qualified name; each component validates its values.

        The update is atomic: if any component rejects its values, no
        component changes.
        """
        grouped = self._group_by_component(params)
        previous = {index: dict(self._models[index]._parameters) for index in grouped}
        try:
            for index, updates in grouped.items():
                self._models[index].set_parameters(**updates)
        except Exception:
            for index, stored in previous.items():
                self._models[index]._parameters = stored
            raise
        return self

    def set_item_parameter(
        self,
        item_idx: int,
        param_name: str,
        value: float | NDArray[np.float64],
    ) -> None:
        """Set one item's parameter, named locally or with its component prefix."""
        component, local = self.locate_item(item_idx)
        prefix = self._prefixes[component] + _SEPARATOR
        name = param_name.removeprefix(prefix)
        self._models[component].set_item_parameter(local, name, value)

    def get_item_parameters(
        self, item_idx: int
    ) -> dict[str, float | NDArray[np.float64]]:
        """Return the owning component's parameters for one item."""
        component, local = self.locate_item(item_idx)
        return self._models[component].get_item_parameters(local)

    def _canonical_parameter_values(
        self,
        name: str,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        component, local = self.parameter_component(name)
        return self._models[component]._canonical_parameter_values(local, values)

    def _expand_parameter_standard_errors(
        self,
        name: str,
        errors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        component, local = self.parameter_component(name)
        return self._models[component]._expand_parameter_standard_errors(local, errors)

    def _item_probability(
        self, theta: NDArray[np.float64], item_idx: int
    ) -> NDArray[np.float64]:
        component, local = self.locate_item(item_idx)
        model = self._models[component]
        values = np.asarray(model.probability(theta, local), dtype=np.float64)
        if model.is_polytomous:
            return values
        values = values.reshape(theta.shape[0])
        return np.column_stack((1.0 - values, values))

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return category probabilities with binary items as ``[1 - p, p]``.

        Without ``item_idx`` the result has shape
        ``(n_theta, n_items, max_categories)`` with zero padding; one item
        returns ``(n_theta, n_categories[item_idx])``.
        """
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._item_probability(theta, item_idx)
        result = np.zeros((theta.shape[0], self.n_items, self.max_categories))
        for model, items in zip(self._models, self._items, strict=True):
            values = np.asarray(model.probability(theta), dtype=np.float64)
            if values.ndim == 2:
                result[:, items, 0] = 1.0 - values
                result[:, items, 1] = values
            else:
                result[:, items, : values.shape[2]] = values
        return result

    def _category_probabilities(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        return self._item_probability(self._ensure_theta_2d(theta), item_idx)

    def category_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
        category: int,
    ) -> NDArray[np.float64]:
        """Return one category's probability for one item."""
        n_categories = self._n_categories[self._validate_item_index(item_idx)]
        if category < 0 or category >= n_categories:
            raise ValueError(f"Category {category} out of range [0, {n_categories})")
        return self._category_probabilities(theta, item_idx)[:, category]

    def probability_pairs(
        self,
        theta: NDArray[np.float64],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned respondent-item pairs with each component's kernel.

        Returns a ``(n_pairs, max_categories)`` matrix padded with zeros, with
        binary items as ``[1 - p, p]``.
        """
        theta_2d, indices = self._prepare_probability_pairs(theta, item_indices)
        result = np.zeros((indices.size, self.max_categories))
        owners = self._item_component[indices]
        for component, model in enumerate(self._models):
            selected = np.flatnonzero(owners == component)
            if not selected.size:
                continue
            values = np.asarray(
                model.probability_pairs(
                    theta_2d[selected], self._item_local[indices[selected]]
                ),
                dtype=np.float64,
            )
            if model.is_polytomous:
                result[selected, : values.shape[1]] = values
            else:
                result[selected, 0] = 1.0 - values
                result[selected, 1] = values
        return result

    def _component_information(
        self, component: int, theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return ``(n_theta, n_component_items)`` item information."""
        model = self._models[component]
        if isinstance(model, PolytomousItemModel):
            return model._information_by_item(theta)
        if not model.is_polytomous:
            values = np.asarray(model.information(theta), dtype=np.float64)
            if values.shape == (theta.shape[0], model.n_items):
                return values
        return np.column_stack(
            [
                np.asarray(model.information(theta, item), dtype=np.float64).reshape(
                    theta.shape[0]
                )
                for item in range(model.n_items)
            ]
        )

    def _information_by_item(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = self._ensure_theta_2d(theta)
        information = np.empty((theta.shape[0], self.n_items), dtype=np.float64)
        for component, items in enumerate(self._items):
            information[:, items] = self._component_information(component, theta)
        return information

    def _item_information(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        component, local = self.locate_item(item_idx)
        return np.asarray(
            self._models[component].information(theta, local), dtype=np.float64
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return one item's information, or the test information.

        Test information has shape ``(n_theta,)`` as for other polytomous
        models; :meth:`_information_by_item` keeps the item columns.
        """
        theta = self._ensure_theta_2d(theta)
        if item_idx is not None:
            return self._item_information(theta, item_idx)
        return self._information_by_item(theta).sum(axis=1)

    def item_information_matrix(
        self,
        theta: NDArray[np.float64],
        item_idx: int,
    ) -> NDArray[np.float64]:
        """Return one item's Fisher information matrices ``(n_theta, F, F)``.

        Components that define ``item_information_matrix`` supply it. For a
        unidimensional model it is the item information; multidimensional
        binary items use the compensatory logistic form ``p q a a'``.

        Raises
        ------
        MirtModelError
            If a multidimensional polytomous component lacks the method.
        """
        theta = self._ensure_theta_2d(theta)
        component, local = self.locate_item(item_idx)
        model = self._models[component]
        native = getattr(model, "item_information_matrix", None)
        if callable(native):
            return np.asarray(native(theta, local), dtype=np.float64)
        if self.n_factors == 1:
            return self._item_information(theta, item_idx).reshape(-1, 1, 1)
        slopes = model.parameters.get("discrimination")
        if model.is_polytomous or slopes is None:
            raise MirtModelError(
                f"{model.model_name} items do not define item_information_matrix",
                model_type=model.model_name,
            )
        a = np.asarray(slopes[local], dtype=np.float64).reshape(self.n_factors)
        p = np.asarray(model.probability(theta, local), dtype=np.float64).ravel()
        p = np.clip(p, PROB_EPSILON, 1.0 - PROB_EPSILON)
        return (p * (1.0 - p))[:, None, None] * np.outer(a, a)[None, :, :]

    def expected_score(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return one item's expected score, or the expected test score."""
        if item_idx is not None:
            return super().expected_score(theta, item_idx)
        theta = self._ensure_theta_2d(theta)
        total = np.zeros(theta.shape[0], dtype=np.float64)
        for model in self._models:
            total += np.asarray(model.expected_score(theta), dtype=np.float64)
        return total

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum the component log-likelihoods of row-aligned responses."""
        responses = self._validate_polytomous_responses(responses)
        theta = self._ensure_theta_2d(theta)
        n_rows = {responses.shape[0], theta.shape[0]}
        if len(n_rows - {1}) > 1:
            raise MirtDataError(
                "responses and theta must have matching row counts or a single row"
            )
        total = np.zeros(max(n_rows), dtype=np.float64)
        for model, items in zip(self._models, self._items, strict=True):
            total += model.log_likelihood(responses[:, items], theta)
        return total

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Sum the component log-likelihoods at every ability point.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Category codes, with negative values for missing responses.
        theta : ndarray of shape (n_theta, n_factors)
            Ability values.

        Returns
        -------
        ndarray of shape (n_persons, n_theta)
            Each component's ``log_likelihood_batch`` on its own items,
            summed.
        """
        responses = self._validate_polytomous_responses(responses)
        theta = self._ensure_theta_2d(theta)
        total = np.zeros((responses.shape[0], theta.shape[0]), dtype=np.float64)
        for model, items in zip(self._models, self._items, strict=True):
            total += _component_log_likelihood_batch(model, responses[:, items], theta)
        return total

    def copy(self) -> Self:
        return type(self)(
            list(zip(self._models, self._items, strict=True)),
            item_names=list(self.item_names),
        )

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{prefix}: {items.size} items"
            for prefix, items in zip(self._prefixes, self._items, strict=True)
        )
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{type(self).__name__}({parts}, n_factors={self.n_factors}, {status})"
