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

from mirt.models.cdm import BaseCDM

ReducedModelType = Literal["DINA", "DINO", "ACDM", "LLM", "RRUM", "saturated"]


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
            self._reduced_models = list(reduced_models)

        self._delta_params: list[NDArray[np.float64]] = []
        self._latent_groups: list[NDArray[np.int_]] = []
        self._initialize_gdina_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters (called by parent class)."""
        pass

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
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

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

    def model_selection(
        self,
        responses: NDArray[np.int_],
        candidate_models: list[ReducedModelType] | None = None,
    ) -> list[ReducedModelType]:
        """Select best reduced model for each item using Wald test.

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
        if candidate_models is None:
            candidate_models = ["DINA", "DINO", "ACDM", "LLM", "RRUM", "saturated"]

        responses = np.asarray(responses)
        n_persons = responses.shape[0]
        patterns = self._attribute_patterns
        n_patterns = len(patterns)

        log_like_matrix = np.zeros((n_persons, n_patterns))
        for p_idx, pattern in enumerate(patterns):
            alpha = np.tile(pattern, (n_persons, 1))
            log_like_matrix[:, p_idx] = self.log_likelihood(responses, alpha)

        log_posterior = log_like_matrix - np.logaddexp.reduce(
            log_like_matrix, axis=1, keepdims=True
        )
        posterior = np.exp(log_posterior)

        best_models: list[ReducedModelType] = []

        for j in range(self.n_items):
            best_ll = -np.inf
            best_model: ReducedModelType = "saturated"

            original_model = self._reduced_models[j]
            original_delta = self._delta_params[j].copy()

            for model_type in candidate_models:
                self._reduced_models[j] = model_type
                self._initialize_delta_for_item(j)

                item_ll = 0.0
                for p_idx in range(n_patterns):
                    alpha = patterns[p_idx].reshape(1, -1)
                    prob = self.probability(alpha, j)[0]
                    prob = np.clip(prob, 1e-10, 1 - 1e-10)

                    weight = posterior[:, p_idx]
                    valid = responses[:, j] >= 0

                    item_ll += np.sum(
                        weight[valid]
                        * (
                            responses[valid, j] * np.log(prob)
                            + (1 - responses[valid, j]) * np.log(1 - prob)
                        )
                    )

                q_j = self._q_matrix[j]
                k_j = np.sum(q_j)
                if model_type == "saturated":
                    n_params = 2**k_j
                elif model_type in ("DINA", "DINO"):
                    n_params = 2
                else:
                    n_params = k_j + 1

                bic = -2 * item_ll + n_params * np.log(n_persons)

                if -bic > best_ll:
                    best_ll = -bic
                    best_model = model_type

            self._reduced_models[j] = original_model
            self._delta_params[j] = original_delta

            best_models.append(best_model)

        return best_models

    def _initialize_delta_for_item(self, item_idx: int) -> None:
        """Initialize delta parameters for a single item."""
        q_j = self._q_matrix[item_idx]
        k_j = np.sum(q_j)
        n_groups = 2**k_j

        model_type = self._reduced_models[item_idx]
        if model_type == "saturated":
            n_params = n_groups
        elif model_type in ("DINA", "DINO"):
            n_params = 2
        else:
            n_params = int(k_j) + 1

        delta = np.zeros(n_params)
        delta[0] = 0.2
        if n_params > 1:
            delta[-1] = 0.6

        self._delta_params[item_idx] = delta

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

    def _initialize_parameters(self) -> None:
        """Initialize parameters (called by parent class)."""
        pass

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

        pattern_prob = pattern_prob / (pattern_prob.sum(axis=1, keepdims=True) + 1e-10)

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
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

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
    responses = np.asarray(responses)
    q_matrix = np.asarray(q_matrix)

    n_persons, n_items = responses.shape
    n_attributes = q_matrix.shape[1]

    model = GDINA(
        n_items=n_items,
        n_attributes=n_attributes,
        q_matrix=q_matrix,
        reduced_models=reduced_models,
    )

    patterns = model.attribute_patterns
    n_patterns = len(patterns)

    class_probs = np.ones(n_patterns) / n_patterns
    prev_ll = -np.inf

    for iteration in range(max_iter):
        log_like_matrix = np.zeros((n_persons, n_patterns))
        for p_idx, pattern in enumerate(patterns):
            alpha = np.tile(pattern, (n_persons, 1))
            log_like_matrix[:, p_idx] = model.log_likelihood(responses, alpha)

        log_posterior = log_like_matrix + np.log(class_probs + 1e-10)
        log_sum = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_sum)

        class_probs = posterior.mean(axis=0)

        for j in range(n_items):
            model_type = model._reduced_models[j]
            delta = model._delta_params[j]
            n_params = len(delta)

            if model_type == "saturated":
                n_groups = n_params
                for g in range(n_groups):
                    group_idx = model._latent_group_idx(patterns, j)
                    in_group = group_idx == g

                    num = 0.0
                    denom = 0.0

                    for p_idx in range(n_patterns):
                        if in_group[p_idx]:
                            weight = posterior[:, p_idx]
                            valid = responses[:, j] >= 0
                            num += np.sum(weight[valid] * responses[valid, j])
                            denom += np.sum(weight[valid])

                    if denom > 1e-10:
                        model._delta_params[j][g] = np.clip(num / denom, 0.01, 0.99)

        current_ll = np.sum(log_sum)

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if abs(current_ll - prev_ll) < tol:
            break

        prev_ll = current_ll

    model._is_fitted = True
    return model, class_probs
