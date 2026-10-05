"""Item-parameter priors for Bayes modal (MAP) EM M-steps."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.priors import _SPECIFICATION_DEFAULTS, Prior, PriorSpecification
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


def _item_parameters(model: BaseItemModel) -> list[str]:
    return [name for name in model._parameters if model._item_indexed(name)]


def _is_default_prior(name: str, prior: Prior) -> bool:
    """Whether ``prior`` is the one ``PriorSpecification`` fills in for ``name``."""
    default = _SPECIFICATION_DEFAULTS.get(name)
    if default is None:
        return False
    family, mu, sigma = default
    return type(prior) is family and (
        getattr(prior, "mu", None),
        getattr(prior, "sigma", None),
    ) == (mu, sigma)


def check_prior_specification(
    priors: PriorSpecification | Mapping[str, Prior] | None,
    models: Sequence[BaseItemModel],
    model_name: str,
) -> None:
    """Require a ``PriorSpecification`` to reach the item parameters of a fit.

    Parameters
    ----------
    priors : PriorSpecification, mapping or None
        Item priors. Only a ``PriorSpecification`` is checked; mapping keys
        are validated by :func:`resolve_item_priors`.
    models : sequence of BaseItemModel
        The fitted model, or every component of a mixed-format model.
    model_name : str
        Model name used in messages.

    Raises
    ------
    MirtValidationError
        If none of the specification's discrimination, difficulty, guessing
        and upper priors names a per-item parameter of ``models``.

    Warns
    -----
    UserWarning
        If a discrimination or difficulty prior other than the
        specification's default names a parameter that none of ``models``
        has, such as ``difficulty`` for a graded model, which stores
        ``thresholds``. That prior is ignored. Guessing and upper priors
        apply only to models with those parameters, as documented.
    """
    if not isinstance(priors, PriorSpecification):
        return
    available = sorted({name for model in models for name in _item_parameters(model)})
    fields = [
        name for name in _SPECIFICATION_FIELDS if getattr(priors, name) is not None
    ]
    if fields and not set(fields).intersection(available):
        parameters = sorted({name for model in models for name in model.parameters})
        raise MirtValidationError(
            "the PriorSpecification sets no prior on the parameters of the "
            f"{model_name} model ({', '.join(parameters)}); pass priors as a "
            "mapping of parameter names to priors",
            parameter="priors",
            expected=", ".join(available),
        )
    ignored = [
        name
        for name in _SPECIFICATION_DEFAULTS
        if name in fields
        and name not in available
        and not _is_default_prior(name, getattr(priors, name))
    ]
    if ignored:
        warnings.warn(
            f"the PriorSpecification's {' and '.join(ignored)} prior is ignored "
            f"because the {model_name} model has no such item parameter (it has "
            f"{', '.join(available)}); pass priors as a mapping of parameter "
            "names to priors",
            UserWarning,
            stacklevel=4,
        )


def resolve_item_priors(
    priors: PriorSpecification | Mapping[str, Prior] | None,
    model: BaseItemModel,
    *,
    check_specification: bool = True,
) -> dict[str, Prior]:
    """Map priors to the model's stored item parameters.

    A ``PriorSpecification`` contributes its discrimination, difficulty,
    guessing and upper priors to the parameters of those names that the model
    has; its ``theta`` prior does not apply to item parameters. Unless
    ``check_specification`` is false, :func:`check_prior_specification`
    first rejects a specification that reaches none of them and warns about
    non-default discrimination or difficulty priors it drops. Mapping keys
    must name per-item parameters of the model.
    """
    if priors is None:
        return {}
    item_parameters = _item_parameters(model)
    if isinstance(priors, PriorSpecification):
        if check_specification:
            check_prior_specification(priors, [model], model.model_name)
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
        for name in model._parameters:
            if not model._item_indexed(name):
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

    def information(self, model: BaseItemModel) -> dict[str, NDArray[np.float64]]:
        """Return the negative second derivative of the log-prior.

        Item priors are independent across coordinates, so the log-prior
        Hessian is diagonal. Arrays have the stored parameter shapes, with the
        curvature at free coordinates and zeros elsewhere.
        """
        masks = model.free_parameter_masks
        result = {}
        for name, prior in self.priors.items():
            canonical = model._canonical_parameter_values(name, model._parameters[name])
            curvature = np.zeros_like(canonical)
            free = masks[name]
            if np.any(free):
                curvature[free] = -np.asarray(
                    prior.hess_log_pdf(canonical[free]), dtype=np.float64
                )
            result[name] = curvature
        return result

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
