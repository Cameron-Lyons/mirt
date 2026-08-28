"""Advanced Cognitive Diagnosis Models.

This module provides:
- GDINA (Generalized DINA with reduced models)
- HigherOrderCDM (Higher-order model with attribute hierarchy)
- AttributeHierarchy (DAG of prerequisite attributes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.models.cdm import BaseCDM

ReducedModelType = Literal["DINA", "DINO", "ACDM", "LLM", "RRUM", "saturated"]

_REDUCED_MODEL_TO_CODE: dict[ReducedModelType, int] = {
    "DINA": 0,
    "DINO": 1,
    "ACDM": 2,
    "LLM": 3,
    "RRUM": 4,
    "saturated": 5,
}
_REDUCED_CODE_TO_MODEL: dict[int, ReducedModelType] = {
    code: name for name, code in _REDUCED_MODEL_TO_CODE.items()
}
_DEFAULT_REDUCED_MODELS: tuple[ReducedModelType, ...] = (
    "DINA",
    "DINO",
    "ACDM",
    "LLM",
    "RRUM",
    "saturated",
)


@dataclass
class AttributeHierarchy:
    """Describes an attribute hierarchy (directed acyclic graph).

    Attributes
    ----------
    adjacency : NDArray
        Adjacency matrix where adjacency[i, j] = 1 means attribute i
        is a prerequisite for attribute j.
    attribute_names : list of str, optional
        Names for attributes.
    """

    adjacency: NDArray[np.int_]
    attribute_names: list[str] | None = None

    def __post_init__(self):
        self.adjacency = np.asarray(self.adjacency, dtype=np.int_)
        n_attrs = self.adjacency.shape[0]

        if self.adjacency.shape != (n_attrs, n_attrs):
            raise ValueError("Adjacency matrix must be square")

        if self.attribute_names is None:
            self.attribute_names = [f"A{i}" for i in range(n_attrs)]
        elif len(self.attribute_names) != n_attrs:
            raise ValueError(
                f"attribute_names length ({len(self.attribute_names)}) "
                f"must match n_attributes ({n_attrs})"
            )

    @property
    def n_attributes(self) -> int:
        return self.adjacency.shape[0]

    def prerequisites(self, attribute: int) -> list[int]:
        """Get direct prerequisites for an attribute."""
        return list(np.where(self.adjacency[:, attribute] == 1)[0])

    def all_prerequisites(self, attribute: int) -> set[int]:
        """Get all prerequisites (direct and indirect) for an attribute."""
        visited = set()
        stack = self.prerequisites(attribute)

        while stack:
            prereq = stack.pop()
            if prereq not in visited:
                visited.add(prereq)
                stack.extend(self.prerequisites(prereq))

        return visited

    def is_valid_pattern(self, pattern: NDArray[np.int_]) -> bool:
        """Check if an attribute pattern is valid under the hierarchy.

        A pattern is valid if all prerequisites of mastered attributes
        are also mastered.
        """
        pattern = np.asarray(pattern).ravel()
        for k in range(self.n_attributes):
            if pattern[k] == 1:
                for prereq in self.prerequisites(k):
                    if pattern[prereq] == 0:
                        return False
        return True

    def valid_patterns(self) -> NDArray[np.int_]:
        """Generate all valid attribute patterns under the hierarchy."""
        n_attrs = self.n_attributes
        n_all = 2**n_attrs

        valid = []
        for i in range(n_all):
            pattern = np.array([(i >> k) & 1 for k in range(n_attrs)])
            if self.is_valid_pattern(pattern):
                valid.append(pattern)

        return np.array(valid, dtype=np.int_)

    def topological_order(self) -> list[int]:
        """Return attributes in topological order (prerequisites first)."""
        n_attrs = self.n_attributes
        in_degree = np.sum(self.adjacency, axis=0)
        order = []
        remaining = set(range(n_attrs))

        while remaining:
            for attr in remaining:
                if in_degree[attr] == 0:
                    order.append(attr)
                    remaining.remove(attr)
                    for j in range(n_attrs):
                        if self.adjacency[attr, j] == 1:
                            in_degree[j] -= 1
                    break
            else:
                raise ValueError("Adjacency matrix contains a cycle")

        return order


class GDINA(BaseCDM):
    """Generalized DINA model with item-specific reduced models.

    The G-DINA model provides a general framework for CDMs where
    item response probability is modeled using main effects and
    interactions of required attributes.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_attributes : int
        Number of attributes.
    q_matrix : NDArray
        Q-matrix (n_items x n_attributes).
    reduced_models : list of str, optional
        Reduced model type for each item. Options: 'DINA', 'DINO',
        'ACDM', 'LLM', 'RRUM', 'saturated'. Default is 'saturated'.
    item_names : list of str, optional
        Names for items.

    Notes
    -----
    Reduced models:
    - saturated: All main effects and interactions (default)
    - DINA: Only intercept and highest-order interaction
    - DINO: Compensatory (OR gate)
    - ACDM: Only main effects (additive CDM)
    - LLM: Linear logistic model
    - RRUM: Reduced reparameterized unified model
    """

    model_name = "GDINA"

    def __init__(
        self,
        n_items: int,
        n_attributes: int,
        q_matrix: NDArray[np.int_],
        reduced_models: list[ReducedModelType] | None = None,
        item_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            item_names=item_names,
        )

        if reduced_models is None:
            self._reduced_models: list[ReducedModelType] = ["saturated"] * n_items
        else:
            if len(reduced_models) != n_items:
                raise ValueError(
                    f"reduced_models length ({len(reduced_models)}) "
                    f"must match n_items ({n_items})"
                )
            invalid_models = [
                model for model in reduced_models if model not in _REDUCED_MODEL_TO_CODE
            ]
            if invalid_models:
                valid = ", ".join(_DEFAULT_REDUCED_MODELS)
                raise ValueError(
                    f"Unknown reduced model: {invalid_models[0]!r}. "
                    f"Valid models: {valid}"
                )
            self._reduced_models = list(reduced_models)

        self._delta_params: list[NDArray[np.float64]] = []
        self._latent_groups: list[NDArray[np.int_]] = []
        self._initialize_gdina_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize model parameters and latent grouping structures."""
        if not hasattr(self, "_reduced_models"):
            self._reduced_models = ["saturated"] * self.n_items
        self._initialize_gdina_parameters()

    def _initialize_gdina_parameters(self) -> None:
        """Initialize delta parameters for each item."""
        self._delta_params = []
        self._latent_groups = []

        for j in range(self.n_items):
            q_j = self._q_matrix[j]
            k_j = np.sum(q_j)
            n_groups = 2**k_j

            latent_groups = np.zeros((n_groups, self._n_attributes), dtype=np.int_)
            required_attrs = np.where(q_j == 1)[0]

            for g in range(n_groups):
                for idx, attr in enumerate(required_attrs):
                    latent_groups[g, attr] = (g >> idx) & 1

            self._latent_groups.append(latent_groups)

            model_type = self._reduced_models[j]
            if model_type == "saturated":
                n_params = n_groups
            elif model_type in ("DINA", "DINO"):
                n_params = 2
            elif model_type in ("ACDM", "LLM", "RRUM"):
                n_params = int(k_j) + 1
            else:
                n_params = n_groups

            delta = np.zeros(n_params)
            delta[0] = 0.2
            if n_params > 1:
                delta[-1] = 0.6
                if n_params > 2:
                    remaining = 0.3 / (n_params - 2)
                    delta[1:-1] = remaining

            self._delta_params.append(delta)

        self._sync_parameter_cache()

    def _sync_parameter_cache(self) -> None:
        """Synchronize generic parameter cache with item-specific deltas."""
        max_params = max((len(delta) for delta in self._delta_params), default=1)
        delta_matrix = np.full((self.n_items, max_params), np.nan, dtype=np.float64)
        n_delta_params = np.zeros(self.n_items, dtype=np.float64)

        for j, delta in enumerate(self._delta_params):
            n_params = len(delta)
            delta_matrix[j, :n_params] = delta
            n_delta_params[j] = float(n_params)

        reduced_model_code = np.array(
            [_REDUCED_MODEL_TO_CODE[model] for model in self._reduced_models],
            dtype=np.float64,
        )

        self._parameters["delta"] = delta_matrix
        self._parameters["delta_n_params"] = n_delta_params
        self._parameters["reduced_model_code"] = reduced_model_code

    @property
    def reduced_models(self) -> list[ReducedModelType]:
        """Reduced model type for each item."""
        return list(self._reduced_models)

    @property
    def delta_parameters(self) -> list[NDArray[np.float64]]:
        """Delta parameters for each item."""
        return [d.copy() for d in self._delta_params]

    def set_delta_parameters(self, item_idx: int, delta: NDArray[np.float64]) -> Self:
        """Set delta parameters for an item."""
        delta = np.asarray(delta, dtype=np.float64)
        expected_len = len(self._delta_params[item_idx])

        if len(delta) != expected_len:
            raise ValueError(
                f"delta length ({len(delta)}) must be {expected_len} for item {item_idx}"
            )

        self._delta_params[item_idx] = delta
        self._sync_parameter_cache()
        return self

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set model parameters using the generic BaseItemModel interface."""
        allowed = {"delta", "delta_n_params", "reduced_model_code"}
        unknown = set(params) - allowed
        if unknown:
            valid = ", ".join(sorted(allowed))
            unknown_s = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown parameter(s): {unknown_s}. Valid parameters: {valid}"
            )

        if "reduced_model_code" in params:
            reduced_codes = np.asarray(params["reduced_model_code"], dtype=np.float64)
            if reduced_codes.shape != (self.n_items,):
                raise ValueError(
                    f"Shape mismatch for reduced_model_code: expected ({self.n_items},), "
                    f"got {reduced_codes.shape}"
                )
            self._reduced_models = []
            for code in reduced_codes:
                rounded = int(np.rint(code))
                if rounded not in _REDUCED_CODE_TO_MODEL:
                    raise ValueError(f"Unknown reduced model code: {code}")
                self._reduced_models.append(_REDUCED_CODE_TO_MODEL[rounded])

        if "delta" in params:
            delta_matrix = np.asarray(params["delta"], dtype=np.float64)
            if delta_matrix.ndim != 2 or delta_matrix.shape[0] != self.n_items:
                raise ValueError(
                    f"Shape mismatch for delta: expected ({self.n_items}, n_params), "
                    f"got {delta_matrix.shape}"
                )
        else:
            delta_matrix = self._parameters["delta"]

        if "delta_n_params" in params:
            n_delta_params = np.asarray(params["delta_n_params"], dtype=np.float64)
            if n_delta_params.shape != (self.n_items,):
                raise ValueError(
                    f"Shape mismatch for delta_n_params: expected ({self.n_items},), "
                    f"got {n_delta_params.shape}"
                )
        else:
            n_delta_params = self._parameters["delta_n_params"]

        max_params = delta_matrix.shape[1]
        updated_delta_params: list[NDArray[np.float64]] = []
        for j in range(self.n_items):
            n_params = int(np.rint(n_delta_params[j]))
            if n_params < 1 or n_params > max_params:
                raise ValueError(
                    f"Invalid number of delta parameters for item {j}: {n_params}"
                )
            updated_delta_params.append(
                delta_matrix[j, :n_params].astype(np.float64).copy()
            )

        self._delta_params = updated_delta_params
        self._sync_parameter_cache()
        return self

    def _latent_group_idx(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.int_]:
        """Map attribute patterns to latent group indices for an item."""
        alpha = self._ensure_alpha_2d(alpha)
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]

        idx = np.zeros(alpha.shape[0], dtype=np.int_)
        for i, attr in enumerate(required_attrs):
            idx += alpha[:, attr] * (2**i)

        return idx

    def _compute_prob_saturated(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for saturated model."""
        group_idx = self._latent_group_idx(alpha, item_idx)
        delta = self._delta_params[item_idx]
        return delta[group_idx]

    def _compute_prob_dina(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for DINA reduced model."""
        q_j = self._q_matrix[item_idx]
        eta = np.all(alpha >= q_j, axis=1).astype(np.float64)
        delta = self._delta_params[item_idx]
        return delta[0] * (1 - eta) + delta[1] * eta

    def _compute_prob_dino(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for DINO reduced model."""
        q_j = self._q_matrix[item_idx]
        required = q_j == 1
        eta = np.any(alpha[:, required], axis=1).astype(np.float64)
        delta = self._delta_params[item_idx]
        return delta[0] * (1 - eta) + delta[1] * eta

    def _compute_prob_acdm(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for additive CDM."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        prob = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            prob += delta[i + 1] * alpha[:, attr]

        return np.clip(prob, 0, 1)

    def _compute_prob_llm(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for linear logistic model."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        logit = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            logit += delta[i + 1] * alpha[:, attr]

        return 1 / (1 + np.exp(-logit))

    def _compute_prob_rrum(
        self, alpha: NDArray[np.int_], item_idx: int
    ) -> NDArray[np.float64]:
        """Compute probability for RRUM."""
        q_j = self._q_matrix[item_idx]
        required_attrs = np.where(q_j == 1)[0]
        delta = self._delta_params[item_idx]

        prob = np.full(alpha.shape[0], delta[0], dtype=np.float64)
        for i, attr in enumerate(required_attrs):
            penalty = delta[i + 1]
            prob *= penalty ** (1 - alpha[:, attr])

        return np.clip(prob, 0, 1)

    def probability(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        Parameters
        ----------
        alpha : NDArray
            Attribute patterns (n_patterns,) or (n_patterns, n_attributes)
        item_idx : int, optional
            Item index. If None, compute for all items.

        Returns
        -------
        NDArray
            Probabilities (n_patterns,) or (n_patterns, n_items)
        """
        alpha = self._ensure_alpha_2d(alpha)
        n_patterns = alpha.shape[0]

        if item_idx is not None:
            model_type = self._reduced_models[item_idx]

            if model_type == "saturated":
                return self._compute_prob_saturated(alpha, item_idx)
            elif model_type == "DINA":
                return self._compute_prob_dina(alpha, item_idx)
            elif model_type == "DINO":
                return self._compute_prob_dino(alpha, item_idx)
            elif model_type == "ACDM":
                return self._compute_prob_acdm(alpha, item_idx)
            elif model_type == "LLM":
                return self._compute_prob_llm(alpha, item_idx)
            elif model_type == "RRUM":
                return self._compute_prob_rrum(alpha, item_idx)
            else:
                return self._compute_prob_saturated(alpha, item_idx)

        probs = np.zeros((n_patterns, self.n_items))
        for j in range(self.n_items):
            probs[:, j] = self.probability(alpha, j)

        return probs

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta for G-DINA (returns latent group index)."""
        return self._latent_group_idx(alpha, item_idx)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        alpha: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for each person."""
        responses = np.asarray(responses)
        alpha = self._ensure_alpha_2d(alpha)

        probs = self.probability(alpha)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(probs) + (1 - responses) * np.log(1 - probs),
            0.0,
        )

        return ll.sum(axis=1)

    def information(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        alpha = self._ensure_alpha_2d(alpha)
        probs = self.probability(alpha, item_idx)
        return probs * (1 - probs)

    @staticmethod
    def _selection_log_likelihood(
        probabilities: NDArray[np.float64],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> float:
        probabilities = np.clip(
            np.asarray(probabilities, dtype=np.float64),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        return float(
            np.sum(
                successes * np.log(probabilities)
                + (totals - successes) * np.log1p(-probabilities)
            )
        )

    @staticmethod
    def _selection_group_rates(
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        total = float(np.sum(totals))
        global_rate = float(
            np.clip(
                np.sum(successes) / total,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
        )
        rates = np.full(totals.shape, global_rate, dtype=np.float64)
        np.divide(successes, totals, out=rates, where=totals > PROB_EPSILON)
        return (
            np.clip(rates, PROB_EPSILON, 1.0 - PROB_EPSILON),
            global_rate,
        )

    def _fit_gate_candidate(
        self,
        mastered: NDArray[np.bool_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, int]:
        categories = mastered.astype(np.int_)
        category_successes = np.bincount(
            categories,
            weights=successes,
            minlength=2,
        )
        category_totals = np.bincount(
            categories,
            weights=totals,
            minlength=2,
        )
        rates, global_rate = self._selection_group_rates(
            category_successes,
            category_totals,
        )
        if rates[1] < rates[0]:
            rates[:] = global_rate
        probabilities = rates[categories]
        n_parameters = int(np.unique(categories).size)
        return (
            rates,
            self._selection_log_likelihood(
                probabilities,
                successes,
                totals,
            ),
            n_parameters,
        )

    def _fit_acdm_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        global_rate: float,
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (np.ones(group_patterns.shape[0], dtype=np.float64), group_patterns)
        )
        initial = np.zeros(design.shape[1], dtype=np.float64)
        initial[0] = global_rate

        def objective(parameters: NDArray[np.float64]) -> float:
            probabilities = design @ parameters
            return -self._selection_log_likelihood(
                probabilities,
                successes,
                totals,
            )

        def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.clip(
                design @ parameters,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            derivative = (totals * probabilities - successes) / (
                probabilities * (1.0 - probabilities)
            )
            return design.T @ derivative

        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=gradient,
            bounds=[(0.0, 1.0 - PROB_EPSILON)] * design.shape[1],
            constraints=LinearConstraint(
                design,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            ),
            options={"ftol": 1e-10, "maxiter": 200},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        probabilities = design @ parameters
        feasible = bool(
            np.all(np.isfinite(parameters))
            and np.all(probabilities >= PROB_EPSILON - 1e-12)
            and np.all(probabilities <= 1.0 - PROB_EPSILON + 1e-12)
            and float(result.fun) <= initial_objective + 1e-8
        )
        if not feasible:
            parameters = initial
        return parameters, self._selection_log_likelihood(
            design @ parameters, successes, totals
        )

    def _fit_llm_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        global_rate: float,
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (np.ones(group_patterns.shape[0], dtype=np.float64), group_patterns)
        )
        initial = np.zeros(design.shape[1], dtype=np.float64)
        initial[0] = np.log(global_rate / (1.0 - global_rate))

        def objective(parameters: NDArray[np.float64]) -> float:
            linear = design @ parameters
            return float(
                np.sum(totals * np.logaddexp(0.0, linear) - successes * linear)
            )

        def gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.asarray(sigmoid(design @ parameters), dtype=np.float64)
            return design.T @ (totals * probabilities - successes)

        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(-23.0, 23.0)] + [(0.0, 23.0)] * group_patterns.shape[1],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        if (
            not np.all(np.isfinite(parameters))
            or float(result.fun) > initial_objective + 1e-8
        ):
            parameters = initial
        probabilities = np.asarray(sigmoid(design @ parameters), dtype=np.float64)
        return parameters, self._selection_log_likelihood(
            probabilities,
            successes,
            totals,
        )

    def _fit_rrum_candidate(
        self,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
        rates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        design = np.column_stack(
            (
                np.ones(group_patterns.shape[0], dtype=np.float64),
                1.0 - group_patterns,
            )
        )
        initial = np.full(design.shape[1], np.log(0.8), dtype=np.float64)
        initial[0] = np.log(np.clip(np.max(rates), PROB_EPSILON, 1.0 - PROB_EPSILON))

        def objective(log_parameters: NDArray[np.float64]) -> float:
            log_probabilities = design @ log_parameters
            probabilities = np.exp(log_probabilities)
            return float(
                -np.sum(
                    successes * log_probabilities
                    + (totals - successes) * np.log1p(-probabilities)
                )
            )

        def gradient(log_parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            probabilities = np.exp(design @ log_parameters)
            derivative = (totals * probabilities - successes) / (1.0 - probabilities)
            return design.T @ derivative

        lower = float(np.log(PROB_EPSILON))
        initial_objective = objective(initial)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=[(lower, float(np.log1p(-PROB_EPSILON)))]
            + [(lower, 0.0)] * group_patterns.shape[1],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        log_parameters = np.asarray(result.x, dtype=np.float64)
        if (
            not np.all(np.isfinite(log_parameters))
            or float(result.fun) > initial_objective + 1e-8
        ):
            log_parameters = initial
        probabilities = np.exp(design @ log_parameters)
        return np.exp(log_parameters), self._selection_log_likelihood(
            probabilities,
            successes,
            totals,
        )

    def _fit_reduced_parameters(
        self,
        model_type: ReducedModelType,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, int]:
        rates, global_rate = self._selection_group_rates(successes, totals)
        n_attributes = group_patterns.shape[1]

        if model_type == "saturated":
            return (
                rates,
                self._selection_log_likelihood(
                    rates,
                    successes,
                    totals,
                ),
                group_patterns.shape[0],
            )
        if model_type == "DINA":
            return self._fit_gate_candidate(
                np.all(group_patterns == 1, axis=1),
                successes,
                totals,
            )
        if model_type == "DINO":
            return self._fit_gate_candidate(
                np.any(group_patterns == 1, axis=1),
                successes,
                totals,
            )
        if model_type == "ACDM":
            parameters, log_likelihood = self._fit_acdm_candidate(
                group_patterns,
                successes,
                totals,
                global_rate,
            )
        elif model_type == "LLM":
            parameters, log_likelihood = self._fit_llm_candidate(
                group_patterns,
                successes,
                totals,
                global_rate,
            )
        else:
            parameters, log_likelihood = self._fit_rrum_candidate(
                group_patterns,
                successes,
                totals,
                rates,
            )
        return parameters, log_likelihood, n_attributes + 1

    def _fit_selection_candidate(
        self,
        model_type: ReducedModelType,
        group_patterns: NDArray[np.int_],
        successes: NDArray[np.float64],
        totals: NDArray[np.float64],
    ) -> tuple[float, int]:
        _, log_likelihood, n_parameters = self._fit_reduced_parameters(
            model_type,
            group_patterns,
            successes,
            totals,
        )
        return log_likelihood, n_parameters

    @staticmethod
    def _validate_selection_candidates(
        candidate_models: list[ReducedModelType] | None,
    ) -> list[ReducedModelType]:
        if candidate_models is None:
            return list(_DEFAULT_REDUCED_MODELS)
        if len(candidate_models) == 0:
            raise ValueError("candidate_models must contain at least one model")

        validated: list[ReducedModelType] = []
        for model_type in candidate_models:
            if (
                not isinstance(model_type, str)
                or model_type not in _REDUCED_MODEL_TO_CODE
            ):
                valid = ", ".join(_DEFAULT_REDUCED_MODELS)
                raise ValueError(
                    f"Unknown candidate model: {model_type!r}. Valid models: {valid}"
                )
            if model_type in validated:
                raise ValueError(f"Duplicate candidate model: {model_type}")
            validated.append(model_type)
        return validated

    def _validate_responses(
        self,
        responses: NDArray[np.int_],
    ) -> NDArray[np.int_]:
        raw_responses = np.asarray(responses)
        if raw_responses.ndim != 2:
            raise ValueError("responses must be a two-dimensional matrix")
        if raw_responses.shape[0] == 0:
            raise ValueError("responses must contain at least one person")
        if raw_responses.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {raw_responses.shape[1]} items, expected {self.n_items}"
            )
        if raw_responses.dtype.kind not in "biuf":
            raise ValueError("responses must contain numeric values")
        values = np.asarray(raw_responses, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("responses must contain only finite values")
        observed = values >= 0.0
        if np.any(observed & (values != 0.0) & (values != 1.0)):
            raise ValueError(
                "responses must contain only 0, 1, or negative missing values"
            )
        if np.any(np.sum(observed, axis=0) == 0):
            raise ValueError("every item must have at least one observed response")
        return np.where(observed, values, -1.0).astype(np.int_, copy=False)

    def model_selection(
        self,
        responses: NDArray[np.int_],
        candidate_models: list[ReducedModelType] | None = None,
    ) -> list[ReducedModelType]:
        """Select the best fitted reduced model for each item using BIC.

        Candidate parameters are fitted to expected item-by-pattern counts. Class
        posteriors for each item are computed from all other items, preventing the
        response being scored from also determining its own latent-class weights.
        The selection is read-only and does not alter model parameters.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        candidate_models : list of str, optional
            Models to consider. Default is all available models.

        Returns
        -------
        list
            Best model type for each item.
        """
        candidates = self._validate_selection_candidates(candidate_models)
        response_values = self._validate_responses(responses)
        patterns = self._attribute_patterns
        probabilities = np.asarray(self.probability(patterns), dtype=np.float64)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("model probabilities must contain only finite values")
        probabilities = np.clip(probabilities, PROB_EPSILON, 1.0 - PROB_EPSILON)

        log_correct = np.log(probabilities)
        log_incorrect = np.log1p(-probabilities)
        observed = response_values >= 0
        response_float = np.where(observed, response_values, 0).astype(
            np.float64,
            copy=False,
        )
        observed_float = observed.astype(np.float64)
        total_log_likelihood = (
            response_float @ log_correct.T
            + (observed_float - response_float) @ log_incorrect.T
        )

        best_models: list[ReducedModelType] = []

        for j in range(self.n_items):
            item_observed = observed[:, j]
            item_responses = response_float[:, j]
            item_contribution = observed_float[:, j, None] * (
                item_responses[:, None] * log_correct[None, :, j]
                + (1.0 - item_responses[:, None]) * log_incorrect[None, :, j]
            )
            leave_one_out = total_log_likelihood - item_contribution
            leave_one_out -= np.logaddexp.reduce(
                leave_one_out,
                axis=1,
                keepdims=True,
            )
            posterior = np.exp(leave_one_out)

            observed_posterior = posterior[item_observed]
            pattern_totals = np.sum(observed_posterior, axis=0)
            pattern_successes = np.sum(
                observed_posterior * item_responses[item_observed, None],
                axis=0,
            )

            group_indices = self._latent_group_idx(patterns, j)
            required_attributes = np.flatnonzero(self._q_matrix[j] == 1)
            n_groups = 2 ** len(required_attributes)
            group_totals = np.bincount(
                group_indices,
                weights=pattern_totals,
                minlength=n_groups,
            )
            group_successes = np.bincount(
                group_indices,
                weights=pattern_successes,
                minlength=n_groups,
            )
            group_patterns = self._latent_groups[j][:, required_attributes]
            n_observed = int(np.sum(item_observed))

            best_bic = np.inf
            best_model = candidates[0]
            for model_type in candidates:
                item_log_likelihood, n_parameters = self._fit_selection_candidate(
                    model_type,
                    group_patterns,
                    group_successes,
                    group_totals,
                )
                bic = -2.0 * item_log_likelihood + n_parameters * np.log(n_observed)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model_type

            best_models.append(best_model)

        return best_models

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = GDINA(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            reduced_models=list(self._reduced_models),
            item_names=self.item_names.copy() if self.item_names else None,
        )

        new_model._delta_params = [d.copy() for d in self._delta_params]
        new_model._latent_groups = [g.copy() for g in self._latent_groups]
        new_model._is_fitted = self._is_fitted
        new_model._sync_parameter_cache()

        return new_model


class HigherOrderCDM(BaseCDM):
    """Higher-order CDM with attributes loading on a general factor.

    This model assumes that attribute mastery is determined by an
    underlying continuous latent trait (general factor) through
    a latent regression.

    Parameters
    ----------
    n_items : int
        Number of items.
    n_attributes : int
        Number of attributes.
    q_matrix : NDArray
        Q-matrix (n_items x n_attributes).
    hierarchy : AttributeHierarchy, optional
        Attribute hierarchy for constrained patterns.
    item_names : list of str, optional
        Names for items.

    Notes
    -----
    The model is:
        P(alpha_k = 1 | theta) = Phi(lambda_k * (theta - tau_k))

    where theta is the higher-order trait, lambda_k is the loading,
    and tau_k is the threshold for attribute k.
    """

    model_name = "HigherOrderCDM"

    def __init__(
        self,
        n_items: int,
        n_attributes: int,
        q_matrix: NDArray[np.int_],
        hierarchy: AttributeHierarchy | None = None,
        item_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            item_names=item_names,
        )

        self._hierarchy = hierarchy
        self._loadings = np.ones(n_attributes, dtype=np.float64)
        self._thresholds = np.zeros(n_attributes, dtype=np.float64)

        self._base_cdm = GDINA(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            item_names=item_names,
        )

        if hierarchy is not None:
            self._valid_patterns = hierarchy.valid_patterns()
        else:
            self._valid_patterns = self._attribute_patterns

        self._sync_parameter_cache()

    def _initialize_parameters(self) -> None:
        """Initialize higher-order parameters."""
        self._loadings = np.ones(self._n_attributes, dtype=np.float64)
        self._thresholds = np.zeros(self._n_attributes, dtype=np.float64)
        self._sync_parameter_cache()

    def _sync_parameter_cache(self) -> None:
        """Synchronize generic parameter cache with higher-order parameters."""
        self._parameters["loadings"] = self._loadings.copy()
        self._parameters["thresholds"] = self._thresholds.copy()

    @property
    def hierarchy(self) -> AttributeHierarchy | None:
        return self._hierarchy

    @property
    def loadings(self) -> NDArray[np.float64]:
        """Higher-order loadings for each attribute."""
        return self._loadings.copy()

    @property
    def thresholds(self) -> NDArray[np.float64]:
        """Thresholds for each attribute."""
        return self._thresholds.copy()

    @property
    def valid_patterns(self) -> NDArray[np.int_]:
        """Valid attribute patterns under hierarchy."""
        return self._valid_patterns.copy()

    def set_higher_order_params(
        self,
        loadings: NDArray[np.float64],
        thresholds: NDArray[np.float64],
    ) -> Self:
        """Set higher-order parameters."""
        loadings = np.asarray(loadings, dtype=np.float64)
        thresholds = np.asarray(thresholds, dtype=np.float64)

        if loadings.shape != (self._n_attributes,):
            raise ValueError(
                f"loadings shape {loadings.shape} != ({self._n_attributes},)"
            )
        if thresholds.shape != (self._n_attributes,):
            raise ValueError(
                f"thresholds shape {thresholds.shape} != ({self._n_attributes},)"
            )

        self._loadings = loadings
        self._thresholds = thresholds
        self._sync_parameter_cache()
        return self

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set higher-order parameters using the generic BaseItemModel interface."""
        allowed = {"loadings", "thresholds"}
        unknown = set(params) - allowed
        if unknown:
            valid = ", ".join(sorted(allowed))
            unknown_s = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown parameter(s): {unknown_s}. Valid parameters: {valid}"
            )

        if "loadings" in params:
            loadings = np.asarray(params["loadings"], dtype=np.float64)
            if loadings.shape != (self._n_attributes,):
                raise ValueError(
                    f"Shape mismatch for loadings: expected ({self._n_attributes},), "
                    f"got {loadings.shape}"
                )
            self._loadings = loadings

        if "thresholds" in params:
            thresholds = np.asarray(params["thresholds"], dtype=np.float64)
            if thresholds.shape != (self._n_attributes,):
                raise ValueError(
                    f"Shape mismatch for thresholds: expected ({self._n_attributes},), "
                    f"got {thresholds.shape}"
                )
            self._thresholds = thresholds

        self._sync_parameter_cache()
        return self

    def attribute_probability(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute attribute mastery probabilities given theta.

        P(alpha_k = 1 | theta) = Phi(lambda_k * (theta - tau_k))

        Parameters
        ----------
        theta : NDArray
            Higher-order ability (n_persons,) or (n_persons, 1).

        Returns
        -------
        NDArray
            Probabilities (n_persons, n_attributes).
        """
        theta = np.asarray(theta).ravel()

        z = self._loadings * (theta[:, np.newaxis] - self._thresholds)
        prob = 1 / (1 + np.exp(-z))

        return prob

    def pattern_probability(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute probability of each valid attribute pattern given theta.

        Parameters
        ----------
        theta : NDArray
            Higher-order ability (n_persons,).

        Returns
        -------
        NDArray
            Pattern probabilities (n_persons, n_valid_patterns).
        """
        theta = np.asarray(theta).ravel()
        n_persons = len(theta)

        attr_prob = self.attribute_probability(theta)
        patterns = self._valid_patterns
        n_patterns = len(patterns)

        pattern_prob = np.zeros((n_persons, n_patterns))

        for p_idx, pattern in enumerate(patterns):
            prob = np.ones(n_persons)
            for k in range(self._n_attributes):
                if pattern[k] == 1:
                    prob *= attr_prob[:, k]
                else:
                    prob *= 1 - attr_prob[:, k]
            pattern_prob[:, p_idx] = prob

        pattern_prob = pattern_prob / (
            pattern_prob.sum(axis=1, keepdims=True) + PROB_EPSILON
        )

        return pattern_prob

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal response probabilities.

        Parameters
        ----------
        theta : NDArray
            Higher-order ability.
        item_idx : int, optional
            Item index.

        Returns
        -------
        NDArray
            Response probabilities.
        """
        theta = np.asarray(theta).ravel()
        n_persons = len(theta)

        pattern_prob = self.pattern_probability(theta)
        patterns = self._valid_patterns

        if item_idx is not None:
            probs = np.zeros(n_persons)
            for p_idx, pattern in enumerate(patterns):
                alpha = pattern.reshape(1, -1)
                cond_prob = self._base_cdm.probability(alpha, item_idx)[0]
                probs += pattern_prob[:, p_idx] * cond_prob
            return probs

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            for p_idx, pattern in enumerate(patterns):
                alpha = pattern.reshape(1, -1)
                cond_prob = self._base_cdm.probability(alpha, j)[0]
                probs[:, j] += pattern_prob[:, p_idx] * cond_prob

        return probs

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta using base CDM."""
        return self._base_cdm.eta(alpha, item_idx)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for each person."""
        responses = np.asarray(responses)
        theta = np.asarray(theta).ravel()

        probs = self.probability(theta)
        probs = np.clip(probs, PROB_EPSILON, 1 - PROB_EPSILON)

        valid = responses >= 0
        ll = np.where(
            valid,
            responses * np.log(probs) + (1 - responses) * np.log(1 - probs),
            0.0,
        )

        return ll.sum(axis=1)

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information."""
        probs = self.probability(theta, item_idx)
        return probs * (1 - probs)

    def estimate_theta(
        self,
        responses: NDArray[np.int_],
        method: Literal["EAP", "MLE"] = "EAP",
        n_quad: int = 31,
    ) -> NDArray[np.float64]:
        """Estimate higher-order ability.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items).
        method : str
            Estimation method ('EAP' or 'MLE').
        n_quad : int
            Number of quadrature points for EAP.

        Returns
        -------
        NDArray
            Estimated theta values (n_persons,).
        """
        responses = np.asarray(responses)
        n_persons = responses.shape[0]

        quad_points = np.linspace(-4, 4, n_quad)

        log_likes = np.zeros((n_persons, n_quad))
        for q, theta_q in enumerate(quad_points):
            theta_arr = np.full(n_persons, theta_q)
            log_likes[:, q] = self.log_likelihood(responses, theta_arr)

        if method == "MLE":
            best_idx = np.argmax(log_likes, axis=1)
            return quad_points[best_idx]

        log_prior = -0.5 * quad_points**2
        log_posterior = log_likes + log_prior

        log_sum = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_sum)

        theta_eap = np.sum(posterior * quad_points, axis=1)

        return theta_eap

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = HigherOrderCDM(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            hierarchy=self._hierarchy,
            item_names=self.item_names.copy() if self.item_names else None,
        )

        new_model._loadings = self._loadings.copy()
        new_model._thresholds = self._thresholds.copy()
        new_model._base_cdm = self._base_cdm.copy()
        new_model._valid_patterns = self._valid_patterns.copy()
        new_model._is_fitted = self._is_fitted
        new_model._sync_parameter_cache()

        return new_model


def fit_gdina(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    reduced_models: list[ReducedModelType] | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[GDINA, NDArray[np.float64]]:
    """Fit G-DINA model using EM algorithm.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items).
    q_matrix : NDArray
        Q-matrix (n_items, n_attributes).
    reduced_models : list of str, optional
        Reduced model for each item.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple
        (fitted_model, class_probabilities)
    """
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise ValueError("responses must be a two-dimensional matrix")
    if raw_responses.shape[0] == 0 or raw_responses.shape[1] == 0:
        raise ValueError("responses must contain at least one person and one item")
    raw_q_matrix = np.asarray(q_matrix)
    if raw_q_matrix.ndim != 2:
        raise ValueError("q_matrix must be a two-dimensional matrix")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0.0
    ):
        raise ValueError("tol must be a positive finite number")

    n_items = raw_responses.shape[1]
    n_attributes = raw_q_matrix.shape[1]

    model = GDINA(
        n_items=n_items,
        n_attributes=n_attributes,
        q_matrix=raw_q_matrix,
        reduced_models=reduced_models,
    )
    response_values = model._validate_responses(raw_responses)

    patterns = model.attribute_patterns
    n_patterns = len(patterns)
    observed = response_values >= 0
    response_float = np.where(observed, response_values, 0).astype(
        np.float64,
        copy=False,
    )
    observed_float = observed.astype(np.float64)
    group_indices = [
        model._latent_group_idx(patterns, item_idx) for item_idx in range(n_items)
    ]
    required_attributes = [
        np.flatnonzero(model._q_matrix[item_idx] == 1) for item_idx in range(n_items)
    ]
    group_patterns = [
        model._latent_groups[item_idx][:, required_attributes[item_idx]]
        for item_idx in range(n_items)
    ]

    class_probs = np.full(n_patterns, 1.0 / n_patterns, dtype=np.float64)
    prev_ll = -np.inf

    for iteration in range(max_iter):
        probabilities = np.asarray(model.probability(patterns), dtype=np.float64)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("model probabilities must contain only finite values")
        probabilities = np.clip(
            probabilities,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        log_correct = np.log(probabilities)
        log_incorrect = np.log1p(-probabilities)
        log_like_matrix = (
            response_float @ log_correct.T
            + (observed_float - response_float) @ log_incorrect.T
        )
        log_posterior = log_like_matrix + np.log(class_probs)
        log_sum = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_sum)
        current_ll = float(np.sum(log_sum))

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break

        class_probs = np.maximum(np.mean(posterior, axis=0), PROB_EPSILON)
        class_probs /= np.sum(class_probs)

        for item_idx in range(n_items):
            item_observed = observed[:, item_idx]
            observed_posterior = posterior[item_observed]
            pattern_totals = np.sum(observed_posterior, axis=0)
            pattern_successes = np.sum(
                observed_posterior
                * response_float[item_observed, item_idx, np.newaxis],
                axis=0,
            )
            n_groups = group_patterns[item_idx].shape[0]
            item_totals = np.bincount(
                group_indices[item_idx],
                weights=pattern_totals,
                minlength=n_groups,
            )
            item_successes = np.bincount(
                group_indices[item_idx],
                weights=pattern_successes,
                minlength=n_groups,
            )
            parameters, _, _ = model._fit_reduced_parameters(
                model._reduced_models[item_idx],
                group_patterns[item_idx],
                item_successes,
                item_totals,
            )
            model._delta_params[item_idx] = parameters

        prev_ll = current_ll

    model._sync_parameter_cache()
    model._is_fitted = True
    return model, class_probs
