"""EM estimation for mixed-format tests.

The marginal likelihood of a :class:`~mirt.models.mixed_format.MixedItemModel`
is one E-step over all items, after which every component is updated by the
M-step of its own family: native polytomous optimization, the batched Newton
step for 1PL and 2PL items, or analytic itemwise objectives.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt._backend_config import should_use_rust
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._item_priors import ItemPriorPenalty, resolve_item_priors
from mirt.estimation.base import StartValues
from mirt.estimation.em import EMEstimator
from mirt.estimation.priors import Prior, PriorSpecification
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.models.mixed_format import MixedItemModel
    from mirt.results.fit_result import FitResult


@dataclass(frozen=True)
class _ComponentFit:
    """A component with its response columns and item priors for one fit."""

    model: BaseItemModel
    context: EMFitContext
    penalty: ItemPriorPenalty | None


class _ComponentPriorPenalty(ItemPriorPenalty):
    """Sum of the component log-priors of a mixed-format model."""

    def __init__(
        self, components: list[tuple[BaseItemModel, ItemPriorPenalty]]
    ) -> None:
        super().__init__({})
        self.components = components

    def log_prior(self, model: BaseItemModel) -> float:
        return float(
            sum(penalty.log_prior(component) for component, penalty in self.components)
        )


def _component_priors(
    priors: PriorSpecification | Mapping[str, Prior] | None,
    model: MixedItemModel,
) -> list[dict[str, Prior]]:
    """Resolve item priors for every component.

    A ``PriorSpecification`` applies to the matching parameters of every
    component. Mapping keys either qualify a parameter of one component, such
    as ``"3PL.guessing"``, or name a parameter, such as ``"guessing"``, for
    every component that has it; a qualified key takes precedence.
    """
    components = model.component_models
    if priors is None or isinstance(priors, PriorSpecification):
        return [resolve_item_priors(priors, component) for component in components]
    resolved: list[dict[str, Prior]] = [{} for _ in components]
    qualified = {name: prior for name, prior in priors.items() if "." in name}
    for name, prior in priors.items():
        if name in qualified:
            continue
        owners = [
            index
            for index, component in enumerate(components)
            if name in component._parameters
        ]
        if not owners:
            raise MirtValidationError(
                f"item_priors refers to {name!r}, which no component has",
                parameter="item_priors",
                value=name,
                expected=", ".join(model.parameters),
            )
        for index in owners:
            resolved[index][name] = prior
    for name, prior in qualified.items():
        index, local = model.parameter_component(name)
        resolved[index][local] = prior
    return [
        resolve_item_priors(component_priors or None, component)
        for component_priors, component in zip(resolved, components, strict=True)
    ]


class MixedFormatEMEstimator(EMEstimator):
    """Marginal maximum likelihood EM for mixed-format tests.

    Takes the arguments of :class:`~mirt.estimation.em.EMEstimator`. Each
    E-step evaluates the whole test; each M-step then updates every
    component of a :class:`~mirt.models.mixed_format.MixedItemModel` with the
    shared posterior through the M-step of its own family, so graded and
    partial-credit components keep the native optimizer and 1PL and 2PL
    components the batched Newton step. Other models are fitted by the
    inherited EM algorithm.

    Notes
    -----
    ``item_priors`` given as a ``PriorSpecification`` apply to every
    component. A mapping names a component parameter with its prefix, for
    example ``{"3PL.guessing": BetaPrior(5, 17)}``, or without one to apply
    to every component that has the parameter.

    With ``se_method="auto"``, standard errors come from the observed
    information (``"oakes"``) when every component is a unidimensional
    built-in 1PL-4PL, GRM, GPCM or PCM model, and from itemwise
    complete-data curvature otherwise. The observed information, score
    cross-product and sandwich estimators treat all components jointly, so
    ``FitResult.vcov`` includes covariances between components.

    SQUAREM acceleration is not available for mixed-format models, which
    fall back to plain EM with a warning.

    Examples
    --------
    >>> from mirt import MixedFormatEMEstimator, MixedItemModel
    >>> model = MixedItemModel.from_itemtypes(
    ...     ["3PL"] * 20 + ["GRM"] * 5, n_categories=4
    ... )
    >>> result = MixedFormatEMEstimator().fit(model, responses)  # doctest: +SKIP
    """

    _component_fits: list[_ComponentFit] | None = None
    _component_penalties: list[ItemPriorPenalty | None] | None = None

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit a mixed-format model; see :meth:`EMEstimator.fit`.

        ``start`` mappings and ``set_free_parameter_masks`` use qualified
        parameter names such as ``"GRM.thresholds"``.
        """
        from mirt.models.mixed_format import MixedItemModel

        if not isinstance(model, MixedItemModel):
            return super().fit(model, responses, prior_mean, prior_cov, start=start)
        responses = model._validate_polytomous_responses(
            self._validate_responses(responses, model.n_items)
        )
        penalties = [
            ItemPriorPenalty(priors) if priors else None
            for priors in _component_priors(self.item_priors, model)
        ]
        # The components own every item parameter, so their priors replace
        # the whole-model resolution of the parent fit.
        item_priors, self.item_priors = self.item_priors, None
        self._component_penalties = penalties
        try:
            return super().fit(model, responses, prior_mean, prior_cov, start=start)
        finally:
            self.item_priors = item_priors
            self._component_penalties = None

    def _fit_prepared(self, model: BaseItemModel, context: EMFitContext) -> FitResult:
        from mirt.estimation._patterns import supports_pattern_compression
        from mirt.models.mixed_format import MixedItemModel, uses_component_likelihoods

        if not isinstance(model, MixedItemModel):
            return super()._fit_prepared(model, context)
        components = model.components
        penalties = self._component_penalties or [None] * len(components)
        with ExitStack() as stack:
            # Identical rows share one likelihood only for built-in curves.
            if uses_component_likelihoods(model) and all(
                supports_pattern_compression(part) for part, _ in components
            ):
                context = stack.enter_context(
                    EMFitContext(
                        context.responses,
                        compress=True,
                        native=should_use_rust(self.use_rust),
                    )
                )
                self._fit_context = context
                self._pattern_frequencies = context.frequencies
            self._component_fits = [
                _ComponentFit(
                    component,
                    stack.enter_context(
                        EMFitContext(np.ascontiguousarray(context.responses[:, items]))
                    ),
                    penalty,
                )
                for (component, items), penalty in zip(
                    components, penalties, strict=True
                )
            ]
            priors = [
                (fit.model, fit.penalty)
                for fit in self._component_fits
                if fit.penalty is not None
            ]
            self._prior_penalty = _ComponentPriorPenalty(priors) if priors else None
            try:
                return super()._fit_prepared(model, context)
            finally:
                self._component_fits = None

    def _uses_squarem(self) -> bool:
        if self._component_fits is not None and self.accelerate == "squarem":
            warnings.warn(
                "accelerate='squarem' is not available for mixed-format models; "
                "using plain EM",
                UserWarning,
                stacklevel=6,
            )
            return False
        return super()._uses_squarem()

    def _component_parts(
        self, model: MixedItemModel, responses: NDArray[np.int_], stack: ExitStack
    ) -> list[_ComponentFit]:
        """Return the fit's components, or temporary ones for other responses."""
        fits = self._component_fits
        context = self._fit_context
        if fits is not None and context is not None and context.responses is responses:
            return fits
        return [
            _ComponentFit(
                component,
                stack.enter_context(
                    EMFitContext(np.ascontiguousarray(responses[:, items]))
                ),
                None,
            )
            for component, items in model.components
        ]

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> None:
        from mirt.models.mixed_format import MixedItemModel

        if not isinstance(model, MixedItemModel):
            super()._m_step(model, responses, posterior_weights)
            return
        context, penalty = self._fit_context, self._prior_penalty
        with ExitStack() as stack:
            try:
                for fit in self._component_parts(model, responses, stack):
                    self._fit_context, self._prior_penalty = fit.context, fit.penalty
                    super()._m_step(fit.model, fit.context.responses, posterior_weights)
            finally:
                self._fit_context, self._prior_penalty = context, penalty

    def _resolve_se_method(
        self,
        model: BaseItemModel,
        person_weights: NDArray[np.float64] | None,
    ) -> str:
        from mirt.estimation._louis_information import has_analytic_item_derivatives
        from mirt.models.mixed_format import MixedItemModel, uses_component_likelihoods

        if not isinstance(model, MixedItemModel):
            return super()._resolve_se_method(model, person_weights)
        if person_weights is not None or self.se_method == "complete_data":
            return "complete_data"
        if self.se_method != "auto":
            return self.se_method
        analytic = uses_component_likelihoods(model) and all(
            has_analytic_item_derivatives(component)
            for component in model.component_models
        )
        return "oakes" if analytic else "complete_data"

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        *,
        person_weights: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        from mirt.models.mixed_format import MixedItemModel

        if (
            not isinstance(model, MixedItemModel)
            or self._resolve_se_method(model, person_weights) != "complete_data"
        ):
            return super()._compute_standard_errors(
                model, responses, posterior_weights, person_weights=person_weights
            )
        # Complete-data curvature is itemwise, so each component supplies its
        # own errors from its response columns.
        errors: dict[str, NDArray[np.float64]] = {}
        context = self._fit_context
        with ExitStack() as stack:
            try:
                fits = self._component_parts(model, responses, stack)
                for prefix, fit in zip(model.component_names, fits, strict=True):
                    self._fit_context = fit.context
                    component_errors = self._complete_data_standard_errors(
                        fit.model,
                        fit.context.responses,
                        posterior_weights,
                        person_weights=person_weights,
                    )
                    errors.update(
                        (f"{prefix}.{name}", values)
                        for name, values in component_errors.items()
                    )
            finally:
                self._fit_context = context
        self._se_details = ("complete_data", None)
        return errors


def em_estimator_for(model: BaseItemModel, **options: Any) -> EMEstimator:
    """Return an EM estimator for ``model`` built with ``options``.

    Mixed-format models get :class:`MixedFormatEMEstimator`; other models
    get :class:`~mirt.estimation.em.EMEstimator`.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        return MixedFormatEMEstimator(**options)
    return EMEstimator(**options)
