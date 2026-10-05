"""Item-parameter priors for Bayes modal (MAP) EM M-steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.priors import Prior, PriorSpecification
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

ItemPriors: TypeAlias = PriorSpecification | Mapping[str, Prior]
_Objective: TypeAlias = Callable[
    [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
]

_SPECIFICATION_FIELDS = ("discrimination", "difficulty", "guessing", "upper")
# Fraction of an optimizer box by which a bound moves inward when a prior has
# zero density there, as a Beta prior on guessing does at zero.
_BOUND_SHRINK = 1e-6


def validate_item_priors(
    priors: ItemPriors | None,
) -> PriorSpecification | dict[str, Prior] | None:
    """Return validated item priors; an empty mapping means no priors."""
    if priors is None or isinstance(priors, PriorSpecification):
        return priors
    if not isinstance(priors, Mapping):
        raise MirtValidationError(
            "item_priors must be a PriorSpecification or a mapping of parameter "
            "names to Prior objects",
            parameter="item_priors",
            value=type(priors).__name__,
            expected="PriorSpecification or Mapping[str, Prior]",
        )
    validated: dict[str, Prior] = {}
    for name, prior in priors.items():
        if not isinstance(name, str) or not isinstance(prior, Prior):
            raise MirtValidationError(
                "item_priors must map parameter names to Prior objects",
                parameter="item_priors",
                value=name,
                expected="Mapping[str, Prior]",
            )
        validated[name] = prior
    return validated or None


def resolve_item_priors(
    priors: PriorSpecification | Mapping[str, Prior] | None,
    model: BaseItemModel,
) -> dict[str, Prior]:
    """Map priors to the model's stored item parameters.

    A ``PriorSpecification`` contributes its discrimination, difficulty,
    guessing and upper priors to the parameters of those names that the model
    has; its ``theta`` prior does not apply to item parameters. Mapping keys
    must name per-item parameters of the model.
    """
    if priors is None:
        return {}
    item_parameters = [
        name
        for name, values in model._parameters.items()
        if values.ndim and values.shape[0] == model.n_items
    ]
    if isinstance(priors, PriorSpecification):
        resolved = {}
        for name in _SPECIFICATION_FIELDS:
            prior = getattr(priors, name)
            if prior is not None and name in item_parameters:
                resolved[name] = prior
        return resolved
    for name in priors:
        if name not in item_parameters:
            raise MirtValidationError(
                f"item_priors refers to {name!r}, which is not an item parameter "
                f"of {model.model_name}",
                parameter="item_priors",
                value=name,
                expected=", ".join(item_parameters),
            )
    return dict(priors)


class ItemPriorPenalty:
    """Log-prior over the free item coordinates that EM M-steps optimize.

    Coordinates follow the layout of
    :meth:`~mirt.estimation.base.BaseEstimator._get_item_params_and_bounds`:
    per-item parameter arrays in storage order, canonical values, and free
    coordinates only, so fixed coordinates never contribute.
    """

    def __init__(self, priors: Mapping[str, Prior]) -> None:
        self.priors = dict(priors)

    def item_terms(
        self, model: BaseItemModel, item_idx: int
    ) -> list[tuple[slice, str, Prior]]:
        """Return the free-coordinate slices of one item that carry priors."""
        masks = model.free_parameter_masks
        terms = []
        offset = 0
        for name, values in model._parameters.items():
            if values.ndim == 0 or values.shape[0] != model.n_items:
                continue
            n_free = int(np.count_nonzero(masks[name][item_idx]))
            prior = self.priors.get(name)
            if prior is not None and n_free:
                terms.append((slice(offset, offset + n_free), name, prior))
            offset += n_free
        return terms

    def log_prior(self, model: BaseItemModel) -> float:
        """Return the summed log-prior of every free item coordinate."""
        masks = model.free_parameter_masks
        total = 0.0
        for name, prior in self.priors.items():
            values = model._parameters[name]
            canonical = model._canonical_parameter_values(name, values)
            free = canonical[masks[name]]
            if free.size:
                total += float(np.sum(prior.log_pdf(free)))
        return total

    def penalize(
        self,
        model: BaseItemModel,
        item_idx: int,
        objective: _Objective,
        bounds: list[tuple[float, float]],
        *,
        analytic: bool,
    ) -> tuple[_Objective, list[tuple[float, float]]]:
        """Subtract the item log-prior from a negative expected log-likelihood.

        Returns the penalized objective and the optimizer box, moved inward
        wherever a prior has zero density at a bound.
        """
        terms = self.item_terms(model, item_idx)
        if not terms:
            return objective, bounds
        bounds = list(bounds)
        for segment, name, prior in terms:
            for index in range(segment.start, segment.stop):
                bounds[index] = _supported_bounds(bounds[index], name, prior)

        if analytic:

            def penalized_with_gradient(
                params: NDArray[np.float64],
            ) -> tuple[float, NDArray[np.float64]]:
                value, gradient = cast(
                    tuple[float, NDArray[np.float64]], objective(params)
                )
                gradient = np.array(gradient, dtype=np.float64)
                log_prior = 0.0
                for segment, _, prior in terms:
                    coordinates = params[segment]
                    log_prior += float(np.sum(prior.log_pdf(coordinates)))
                    gradient[segment] -= prior.grad_log_pdf(coordinates)
                return float(value) - log_prior, gradient

            return penalized_with_gradient, bounds

        def penalized(params: NDArray[np.float64]) -> float:
            value = float(cast(float, objective(params)))
            for segment, _, prior in terms:
                value -= float(np.sum(prior.log_pdf(params[segment])))
            return value

        return penalized, bounds


def _supported_bounds(
    bound: tuple[float, float], name: str, prior: Prior
) -> tuple[float, float]:
    """Move box ends with zero prior density slightly into the box."""
    lower, upper = bound
    if lower is None or upper is None:
        return bound
    shift = _BOUND_SHRINK * (upper - lower)
    ends = np.array([lower, upper], dtype=np.float64)
    density = np.asarray(prior.log_pdf(ends), dtype=np.float64)
    if not np.isfinite(density[0]):
        ends[0] += shift
    if not np.isfinite(density[1]):
        ends[1] -= shift
    if not np.all(np.isfinite(prior.log_pdf(ends))):
        raise MirtValidationError(
            f"the prior for {name!r} must have positive density on the "
            f"optimizer box [{lower}, {upper}]",
            parameter="item_priors",
            value=name,
        )
    return float(ends[0]), float(ends[1])
