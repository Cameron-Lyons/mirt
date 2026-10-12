"""Conventional latent-scale identification for prospective group constraints.

Configural fits standardize every group; metric fits may estimate covariance
after fixing means; scalar fits can estimate means after location anchors are
present. These conventions follow the primary multiple-group mirt documentation:
https://philchalmers.github.io/mirt/reference/multipleGroup.html

Fixed calibration blocks can instead establish an external scale and origin.
The checks deliberately use sufficient whole-block anchors rather than infer
identification from a numerical Hessian or from the quadrature grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.multigroup.invariance import InvarianceSpec
    from mirt.multigroup.model import MultigroupModel


_SCALE_PARAMETERS = frozenset(
    {"discrimination", "slopes", "loadings", "general_loadings", "specific_loadings"}
)
_LOCATION_PARAMETERS = frozenset(
    {"difficulty", "thresholds", "steps", "intercepts", "location"}
)


def _item_can_depend_on_theta(
    parameters: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    item: int | None,
    n_items: int,
) -> bool:
    """Exclude location anchors on known zero-loading, flat response curves."""
    names = _SCALE_PARAMETERS & parameters.keys()
    if not names:
        return True
    for name in names:
        row, mask = parameters[name], masks[name]
        if item is not None and row.ndim > 0 and row.shape[0] == n_items:
            row, mask = row[item], mask[item]
        if np.any(row != 0) or np.any(mask):
            return True
    return False


def _has_fixed_family(
    group: BaseItemModel,
    registry: dict[str, dict[int, np.ndarray]],
    family: frozenset[str],
    *,
    scale: bool,
    state: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> bool:
    """Recognize known whole rows, including structural fixed unit slopes."""
    parameters, masks = state
    restrictions = getattr(group, "_free_parameter_restrictions", {})
    for name in family & parameters.keys():
        values = parameters[name]
        if values.ndim == 0 or values.shape[0] != group.n_items:
            continue
        for item, row in enumerate(values):
            if not scale and not _item_can_depend_on_theta(
                parameters, masks, item, group.n_items
            ):
                continue
            registered = item in registry.get(name, {})
            restricted = name in restrictions and not np.any(restrictions[name][item])
            structurally_fixed_scale = scale and not np.any(masks[name][item])
            if not (registered or restricted or structurally_fixed_scale):
                continue
            if scale and not np.any(np.asarray(row) != 0):
                continue
            return True
    return False


def _has_shared_family(
    model: MultigroupModel,
    spec: InvarianceSpec,
    family: frozenset[str],
    *,
    scale: bool,
    reference_group: int,
    group_index: int,
    states: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> bool:
    """Inspect prospective shared rows without changing parameter links."""
    shared = set(spec.get_shared_parameters(model))
    parameters, masks = states[reference_group]
    group_parameters, group_masks = states[group_index]
    for name in family & shared:
        if model._parameter_is_item_major(name):
            freed = set(spec.get_free_items(name) or [])
            rows = [item for item in range(model.n_items) if item not in freed]
        else:
            rows = [None]
        for item in rows:
            if not scale and not _item_can_depend_on_theta(
                parameters, masks, item, model.n_items
            ):
                continue
            if not scale and not _item_can_depend_on_theta(
                group_parameters, group_masks, item, model.n_items
            ):
                continue
            row = parameters[name] if item is None else parameters[name][item]
            mask = masks[name] if item is None else masks[name][item]
            # A free shared loading can become informative during estimation;
            # a known zero loading supplies no latent-scale information.
            if scale and not (np.any(mask) or np.any(np.asarray(row) != 0)):
                continue
            return True
    return False


def _prospective_shared_states(
    model: MultigroupModel, spec: InvarianceSpec, reference_group: int
) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Account for a shared coordinate frozen by any group, without mutation."""
    states = [
        (
            group.parameters,
            {name: mask.copy() for name, mask in group.free_parameter_masks.items()},
        )
        for group in model.group_models
    ]
    for name in spec.get_shared_parameters(model):
        template = states[reference_group][0][name]
        selected = np.ones(template.shape, dtype=bool)
        if model._parameter_is_item_major(name):
            selected[spec.get_free_items(name) or []] = False
        common_mask = np.logical_and.reduce([masks[name] for _, masks in states])
        common_value = template.copy()
        for parameters, masks in states:
            np.copyto(common_value, parameters[name], where=selected & ~masks[name])
        for parameters, masks in states:
            np.copyto(parameters[name], common_value, where=selected)
            np.copyto(masks[name], common_mask, where=selected)
    return states


def _known_zero_loadings(
    group: BaseItemModel, masks: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Return the ``(n_items, n_factors)`` loadings fixed at zero, or None.

    ``None`` means the family has no rotatable slopes. A loading is a known
    zero when every stored coordinate of it is fixed and zero.
    """
    name = getattr(group, "_rotatable_slopes", None)
    n_factors = group.n_factors
    if n_factors < 2 or name is None or name not in masks:
        return None
    values = np.asarray(group.parameters[name])
    if values.ndim < 2 or values.shape[-1] != n_factors:
        return None
    values = values.reshape(group.n_items, -1, n_factors)
    free = np.asarray(masks[name], dtype=np.bool_).reshape(values.shape)
    return np.all(~free & (values == 0.0), axis=1)


def factor_correlations_identified(model: MultigroupModel) -> bool:
    """Whether every group's loading pattern identifies factor correlations.

    With unit variances, correlated factors are identified up to their
    signs when, for every factor, the items that do not load on it have
    loadings of full rank on the remaining factors, the usual sufficient
    rank condition for confirmatory factor models. The rank is checked
    generically, with random values for the loadings that are free or fixed
    at nonzero values. An exploratory pattern, in which every item loads on
    every factor, fails the check: its rotation is identified only by
    uncorrelated factors.
    """
    n_factors = model.n_factors
    if n_factors < 2:
        return False
    generator = np.random.default_rng(0)
    for index, group in enumerate(model.group_models):
        zero = _known_zero_loadings(group, model.effective_free_parameter_masks(index))
        if zero is None:
            return False
        generic = np.where(zero, 0.0, generator.uniform(0.5, 1.5, zero.shape))
        for factor in range(n_factors):
            rows = generic[zero[:, factor]][:, np.arange(n_factors) != factor]
            if len(rows) < n_factors - 1 or np.linalg.matrix_rank(rows) < n_factors - 1:
                return False
    return True


def _apply_latent_options(
    flags: list[tuple[bool, bool]],
    identified: list[tuple[bool, bool]],
    invariance: InvarianceSpec,
    reference_group: int,
    labels: Sequence[str],
) -> list[tuple[bool, bool]]:
    """Apply ``free_means`` and ``free_variances`` to the default flags."""
    from mirt.exceptions import MirtValidationError

    adjusted = [list(flag) for flag in flags]
    options = (
        ("free_means", invariance.free_means, "means", "intercepts (scalar)"),
        ("free_variances", invariance.free_variances, "variances", "slopes (metric)"),
    )
    for position, (name, option, moment, anchors) in enumerate(options):
        if option is None:
            continue
        missing = [
            labels[index]
            for index, flag in enumerate(identified)
            if index != reference_group and not flag[position]
        ]
        if option and missing:
            raise MirtValidationError(
                f"{name}=True, but the latent {moment} of groups {missing} are not "
                f"identified at {invariance.level} invariance; constrain the "
                f"{anchors} of some items across groups, or fix calibration "
                "blocks with known values",
                parameter=name,
                value=option,
            )
        for index, flag in enumerate(adjusted):
            flag[position] = bool(option) and index != reference_group
    return [(mean, cov) for mean, cov in adjusted]


def infer_latent_identification(
    model: MultigroupModel,
    invariance: InvarianceSpec,
    reference_group: int = 0,
    *,
    mean_order: Sequence[int] | None = None,
) -> tuple[tuple[bool, bool], ...]:
    """Return ``(estimate_mean, estimate_cov)`` for every group.

    The reference distribution stays fixed. Configural fits standardize all
    distributions unless known scale and location rows provide external
    calibration. Metric fits estimate covariance when scale anchors remain,
    with means fixed. Scalar/strict fits estimate each latent component only
    when its respective anchor family remains after partial-invariance frees.
    External whole-row promotion is currently limited to unidimensional
    models; a single loading row cannot span a multidimensional latent space.

    These are sufficient conventional constraints, not a rank analysis of
    arbitrary custom model families. ``mean_order`` requires an identified
    free mean in every nonreference group and a unidimensional model.

    The ``free_means`` and ``free_variances`` options of ``invariance``
    override these defaults: ``False`` fixes the moment in every group, and
    ``True`` estimates it in every nonreference group, which requires a
    location (mean) or scale (variance) anchor in each, shared or fixed.
    """
    if (
        isinstance(reference_group, (bool, np.bool_))
        or not isinstance(reference_group, (int, np.integer))
        or not 0 <= reference_group < model.n_groups
    ):
        raise ValueError("reference_group must be a valid integer group index")
    registry = model.fixed_item_parameters
    states = _prospective_shared_states(model, invariance, reference_group)
    flags = []
    identified = []
    for index, group in enumerate(model.group_models):
        if index == reference_group:
            flags.append((False, False))
            identified.append((True, True))
            continue
        shared_scale = _has_shared_family(
            model,
            invariance,
            _SCALE_PARAMETERS,
            scale=True,
            reference_group=reference_group,
            group_index=index,
            states=states,
        )
        shared_location = _has_shared_family(
            model,
            invariance,
            _LOCATION_PARAMETERS,
            scale=False,
            reference_group=reference_group,
            group_index=index,
            states=states,
        )
        fixed_scale = _has_fixed_family(
            group, registry, _SCALE_PARAMETERS, scale=True, state=states[index]
        )
        fixed_location = _has_fixed_family(
            group, registry, _LOCATION_PARAMETERS, scale=False, state=states[index]
        )
        identified.append(
            (shared_location or fixed_location, shared_scale or fixed_scale)
        )
        if model.n_factors == 1 and fixed_scale and fixed_location:
            flags.append((True, True))
        elif invariance.level == "configural":
            flags.append((False, False))
        elif invariance.level == "metric":
            flags.append((False, shared_scale or fixed_scale))
        else:
            flags.append(
                (shared_location or fixed_location, shared_scale or fixed_scale)
            )
    flags = _apply_latent_options(
        flags, identified, invariance, reference_group, model.group_labels
    )
    if mean_order is not None and (
        model.n_factors != 1
        or any(
            not mean
            for index, (mean, _) in enumerate(flags)
            if index != reference_group
        )
    ):
        raise ValueError(
            "mean_order requires identified free means in every nonreference group; "
            "supply location and scale anchors or fixed calibration blocks"
        )
    return tuple(flags)
