"""Prepared marginal likelihood and item gradients for joint BL fitting."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.estimation._affine_objective import prepare_affine_objective
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._polytomous_objective import prepare_polytomous_objective
from mirt.estimation._posterior import normalize_log_posterior
from mirt.models.base import BaseItemModel

_MAX_LIKELIHOOD_ENTRIES = 131_072
_Setter = Callable[[BaseItemModel, NDArray[np.float64], dict[str, Any]], None]


class PreparedBLObjective:
    """Own trial model state while sharing response preparation across calls."""

    def __init__(
        self,
        model: BaseItemModel,
        context: EMFitContext,
        nodes: NDArray[np.float64],
        log_prior_mass: NDArray[np.float64],
        structure: dict[str, Any],
        bounds: list[tuple[float, float]],
        setter: _Setter,
    ) -> None:
        self.model = deepcopy(model)
        self.context = context
        self.nodes = nodes
        self.log_prior_mass = log_prior_mass
        self.structure = structure
        self.setter = setter
        self.indices = []
        self.bounds = []
        parameter_indices = []
        for info in structure.values():
            indices = np.full(info["shape"], -1, dtype=np.intp)
            indices.ravel()[info["free_indices"]] = np.arange(
                info["start_idx"], info["end_idx"]
            )
            parameter_indices.append(indices)
        for item in range(model.n_items):
            indices = np.concatenate(
                [np.asarray(field[item]).reshape(-1) for field in parameter_indices]
            )
            indices = indices[indices >= 0]
            self.indices.append(indices)
            self.bounds.append([bounds[index] for index in indices])
        if model.is_polytomous:
            model._validate_polytomous_responses(context.responses)

    def _binary_objective(self, item, observed, correct, params=None):
        bounds = self.bounds[item]
        if params is not None:
            # Unconstrained methods can leave the original optimizer box.
            # Keep each kernel's overflow bound valid for the current trial.
            bounds = [
                (min(low, float(value)), max(high, float(value)))
                for (low, high), value in zip(bounds, params, strict=True)
            ]
        objective = prepare_dichotomous_objective(
            self.model,
            item,
            self.nodes,
            observed,
            correct,
            PROB_EPSILON,
            bounds,
        )
        if objective is None:
            objective = prepare_affine_objective(
                self.model,
                item,
                self.nodes,
                observed,
                correct,
                PROB_EPSILON,
                bounds,
            )
        return objective

    def supports_gradients(self) -> bool:
        """Require every item's curve and parameter layout to have a kernel."""
        empty = np.zeros(len(self.nodes))
        for item in range(self.model.n_items):
            if self.model.is_polytomous:
                objective = prepare_polytomous_objective(
                    self.model,
                    item,
                    self.nodes,
                    np.zeros((len(self.nodes), self.model.n_categories[item])),
                    PROB_EPSILON,
                )
            else:
                objective = self._binary_objective(item, empty, empty)
            if objective is None:
                return False
        return True

    def _log_likelihoods(self) -> NDArray[np.float64]:
        """Build an owned likelihood buffer with bounded response scratch."""
        responses = self.context.responses
        n_persons, n_items = responses.shape
        n_points = len(self.nodes)
        likelihood = np.zeros((n_persons, n_points))
        chunk_size = max(1, _MAX_LIKELIHOOD_ENTRIES // max(n_points, 2 * n_items))
        if self.model.is_polytomous:
            for item in range(n_items):
                log_probabilities = np.log(
                    np.clip(
                        self.model.probability(self.nodes, item),
                        PROB_EPSILON,
                        1.0 - PROB_EPSILON,
                    )
                )
                for start in range(0, n_persons, chunk_size):
                    stop = min(start + chunk_size, n_persons)
                    values = responses[start:stop, item]
                    np.add(
                        likelihood[start:stop],
                        log_probabilities[:, np.maximum(values, 0)].T,
                        out=likelihood[start:stop],
                        where=values[:, None] >= 0,
                    )
        else:
            probabilities = np.clip(
                self.model.probability(self.nodes),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            log_failure = np.log1p(-probabilities)
            log_odds = np.log(probabilities) - log_failure
            for start in range(0, n_persons, chunk_size):
                stop = min(start + chunk_size, n_persons)
                correct, observed = self.context.response_components(start, stop)
                block = likelihood[start:stop]
                np.matmul(correct, log_odds.T, out=block)
                block += observed @ log_failure.T
        return likelihood

    def value(self, params: NDArray[np.float64]) -> float:
        """Evaluate the same marginal target without computing its gradient."""
        self.setter(self.model, params, self.structure)
        _, marginal = normalize_log_posterior(
            self._log_likelihoods(), self.log_prior_mass
        )
        return -float(marginal.sum())

    def __call__(
        self, params: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        self.setter(self.model, params, self.structure)
        posterior, marginal = normalize_log_posterior(
            self._log_likelihoods(), self.log_prior_mass
        )
        gradient = np.zeros_like(params)
        correct = observed = None
        if not self.model.is_polytomous:
            correct, observed = self.context.expected_counts(posterior)
        for item, indices in enumerate(self.indices):
            if not indices.size:
                continue
            if self.model.is_polytomous:
                counts = self.context.expected_category_counts(
                    item, self.model.n_categories[item], posterior
                )
                objective = prepare_polytomous_objective(
                    self.model, item, self.nodes, counts, PROB_EPSILON
                )
            else:
                objective = self._binary_objective(
                    item, observed[item], correct[item], params[indices]
                )
            if objective is None:
                raise RuntimeError("Prepared BL item no longer supports gradients")
            _, item_gradient = objective(params[indices])
            gradient[indices] = item_gradient
        return -float(marginal.sum()), gradient


def prepare_bl_objective(
    model: BaseItemModel,
    context: EMFitContext,
    nodes: NDArray[np.float64],
    log_prior_mass: NDArray[np.float64],
    structure: dict[str, Any],
    bounds: list[tuple[float, float]],
    setter: _Setter,
) -> PreparedBLObjective | None:
    """Keep custom likelihoods, curves, and parameter layouts on their own path."""
    from mirt.models.bifactor import BifactorModel
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
        PartialCreditModel,
    )

    if type(model) not in (
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
        MultidimensionalModel,
        BifactorModel,
    ) or any(
        name in vars(model)
        for name in (
            "probability",
            "log_likelihood",
            "log_likelihood_batch",
            "_category_probabilities",
            "_validate_polytomous_responses",
            "_evaluate_logistic",
            "_logits",
            "_curve_parameters",
            "_ensure_theta_2d",
            "set_parameters",
            "set_item_parameter",
            "_canonical_parameter_values",
            "free_parameter_masks",
        )
    ):
        return None
    if any(
        not info["shape"] or info["shape"][0] != model.n_items
        for info in structure.values()
    ):
        return None
    objective = PreparedBLObjective(
        model, context, nodes, log_prior_mass, structure, bounds, setter
    )
    return objective if objective.supports_gradients() else None
