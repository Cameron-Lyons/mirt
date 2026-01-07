"""Cognitive Diagnosis Models (CDM).

This module implements cognitive diagnosis models including:
- DINA (Deterministic Input, Noisy AND gate)
- DINO (Deterministic Input, Noisy OR gate)
- GDINA (Generalized DINA)
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from mirt.models.base import BaseItemModel


class BaseCDM(BaseItemModel):
    """Base class for Cognitive Diagnosis Models.

    CDMs model the relationship between discrete latent attributes
    (skills) and item responses using a Q-matrix that specifies
    which attributes are required for each item.
    """

    model_name = "BaseCDM"
    n_params_per_item = 2
    supports_multidimensional = True

    def __init__(
        self,
        n_items: int,
        n_attributes: int,
        q_matrix: NDArray[np.int_],
        item_names: list[str] | None = None,
    ) -> None:
        """Initialize CDM.

        Parameters
        ----------
        n_items : int
            Number of items
        n_attributes : int
            Number of latent attributes (skills)
        q_matrix : NDArray
            Binary Q-matrix (n_items x n_attributes) indicating which
            attributes are required for each item
        item_names : list of str, optional
            Names for items
        """
        self._n_attributes = n_attributes
        self._q_matrix = np.asarray(q_matrix, dtype=np.int_)

        if self._q_matrix.shape != (n_items, n_attributes):
            raise ValueError(
                f"Q-matrix shape {self._q_matrix.shape} does not match "
                f"({n_items}, {n_attributes})"
            )

        super().__init__(n_items=n_items, n_factors=n_attributes, item_names=item_names)

        self._attribute_patterns = self._generate_attribute_patterns()

    @property
    def n_attributes(self) -> int:
        """Number of latent attributes."""
        return self._n_attributes

    @property
    def q_matrix(self) -> NDArray[np.int_]:
        """Q-matrix specifying item-attribute relationships."""
        return self._q_matrix

    @property
    def attribute_patterns(self) -> NDArray[np.int_]:
        """All possible attribute mastery patterns."""
        return self._attribute_patterns

    def _generate_attribute_patterns(self) -> NDArray[np.int_]:
        """Generate all 2^K attribute mastery patterns."""
        n_patterns = 2**self._n_attributes
        patterns = np.zeros((n_patterns, self._n_attributes), dtype=np.int_)

        for i in range(n_patterns):
            for k in range(self._n_attributes):
                patterns[i, k] = (i >> k) & 1

        return patterns

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta (ideal response) for given attribute patterns.

        Must be implemented by subclasses (DINA uses AND, DINO uses OR).

        Parameters
        ----------
        alpha : NDArray
            Attribute patterns (n_patterns x n_attributes)
        item_idx : int
            Item index

        Returns
        -------
        NDArray
            Eta values (0 or 1) for each pattern
        """
        raise NotImplementedError("Subclasses must implement eta()")

    def _ensure_alpha_2d(self, alpha: NDArray) -> NDArray[np.int_]:
        """Ensure alpha is 2D (n_patterns x n_attributes)."""
        alpha = np.asarray(alpha, dtype=np.int_)
        if alpha.ndim == 1:
            alpha = alpha.reshape(1, -1)
        return alpha


class DINA(BaseCDM):
    """Deterministic Input, Noisy AND gate model.

    The DINA model assumes that a correct response requires mastery
    of ALL attributes specified in the Q-matrix (conjunctive).

    P(X_ij = 1 | alpha) = (1 - s_j)^eta_ij * g_j^(1 - eta_ij)

    where:
    - s_j = slip parameter (probability of incorrect response given mastery)
    - g_j = guess parameter (probability of correct response without mastery)
    - eta_ij = 1 if respondent masters ALL required attributes, 0 otherwise
    """

    model_name = "DINA"

    def _initialize_parameters(self) -> None:
        """Initialize slip and guess parameters."""
        self._parameters["slip"] = np.full(self.n_items, 0.1, dtype=np.float64)
        self._parameters["guess"] = np.full(self.n_items, 0.2, dtype=np.float64)

    @property
    def slip(self) -> NDArray[np.float64]:
        """Slip parameters."""
        return self._parameters["slip"]

    @property
    def guess(self) -> NDArray[np.float64]:
        """Guess parameters."""
        return self._parameters["guess"]

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta using AND rule (conjunctive).

        eta_ij = prod_k(alpha_ik^q_jk)

        Returns 1 if ALL required attributes are mastered.
        """
        alpha = self._ensure_alpha_2d(alpha)
        q_j = self._q_matrix[item_idx]

        return np.all(alpha >= q_j, axis=1).astype(np.int_)

    def probability(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        P(X = 1 | alpha) = (1 - s)^eta * g^(1 - eta)

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

        s = self._parameters["slip"]
        g = self._parameters["guess"]

        if item_idx is not None:
            eta_j = self.eta(alpha, item_idx)
            prob = ((1 - s[item_idx]) ** eta_j) * (g[item_idx] ** (1 - eta_j))
            return prob

        probs = np.zeros((n_patterns, self.n_items))
        for j in range(self.n_items):
            eta_j = self.eta(alpha, j)
            probs[:, j] = ((1 - s[j]) ** eta_j) * (g[j] ** (1 - eta_j))

        return probs

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        alpha: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute log-likelihood for each person.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        alpha : NDArray
            Attribute patterns (n_persons, n_attributes)

        Returns
        -------
        NDArray
            Log-likelihood for each person (n_persons,)
        """
        responses = np.asarray(responses)
        alpha = self._ensure_alpha_2d(alpha)
        responses.shape[0]

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
        """Compute Fisher information.

        For CDMs, information is P(1-P) / (s*g + (1-s)*(1-g))^2

        Parameters
        ----------
        alpha : NDArray
            Attribute patterns
        item_idx : int, optional
            Item index

        Returns
        -------
        NDArray
            Information values
        """
        alpha = self._ensure_alpha_2d(alpha)

        probs = self.probability(alpha, item_idx)
        info = probs * (1 - probs)

        return info

    def classify_respondents(
        self,
        responses: NDArray[np.int_],
        method: str = "MLE",
    ) -> NDArray[np.int_]:
        """Classify respondents into attribute patterns.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_items)
        method : str
            Classification method ('MLE' or 'MAP')

        Returns
        -------
        NDArray
            Estimated attribute patterns (n_persons, n_attributes)
        """
        responses = np.asarray(responses)
        n_persons = responses.shape[0]

        patterns = self._attribute_patterns
        n_patterns = len(patterns)

        log_likes = np.zeros((n_persons, n_patterns))
        for p_idx, pattern in enumerate(patterns):
            alpha_broadcast = np.tile(pattern, (n_persons, 1))
            log_likes[:, p_idx] = self.log_likelihood(responses, alpha_broadcast)

        if method == "MAP":
            prior = np.log(1 / n_patterns)
            log_likes = log_likes + prior

        best_idx = np.argmax(log_likes, axis=1)

        return patterns[best_idx]

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = DINA(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


class DINO(BaseCDM):
    """Deterministic Input, Noisy OR gate model.

    The DINO model assumes that a correct response requires mastery
    of ANY attribute specified in the Q-matrix (disjunctive).

    P(X_ij = 1 | alpha) = (1 - s_j)^eta_ij * g_j^(1 - eta_ij)

    where:
    - s_j = slip parameter
    - g_j = guess parameter
    - eta_ij = 1 if respondent masters ANY required attribute, 0 otherwise
    """

    model_name = "DINO"

    def _initialize_parameters(self) -> None:
        """Initialize slip and guess parameters."""
        self._parameters["slip"] = np.full(self.n_items, 0.1, dtype=np.float64)
        self._parameters["guess"] = np.full(self.n_items, 0.2, dtype=np.float64)

    @property
    def slip(self) -> NDArray[np.float64]:
        """Slip parameters."""
        return self._parameters["slip"]

    @property
    def guess(self) -> NDArray[np.float64]:
        """Guess parameters."""
        return self._parameters["guess"]

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute eta using OR rule (disjunctive).

        eta_ij = 1 - prod_k(1 - alpha_ik)^q_jk

        Returns 1 if ANY required attribute is mastered.
        """
        alpha = self._ensure_alpha_2d(alpha)
        q_j = self._q_matrix[item_idx]

        required_attrs = q_j == 1
        if not required_attrs.any():
            return np.ones(alpha.shape[0], dtype=np.int_)

        mastered_required = alpha[:, required_attrs]
        return np.any(mastered_required, axis=1).astype(np.int_)

    def probability(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities.

        Same formula as DINA but with OR-based eta.
        """
        alpha = self._ensure_alpha_2d(alpha)
        n_patterns = alpha.shape[0]

        s = self._parameters["slip"]
        g = self._parameters["guess"]

        if item_idx is not None:
            eta_j = self.eta(alpha, item_idx)
            prob = ((1 - s[item_idx]) ** eta_j) * (g[item_idx] ** (1 - eta_j))
            return prob

        probs = np.zeros((n_patterns, self.n_items))
        for j in range(self.n_items):
            eta_j = self.eta(alpha, j)
            probs[:, j] = ((1 - s[j]) ** eta_j) * (g[j] ** (1 - eta_j))

        return probs

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

    def classify_respondents(
        self,
        responses: NDArray[np.int_],
        method: str = "MLE",
    ) -> NDArray[np.int_]:
        """Classify respondents into attribute patterns."""
        responses = np.asarray(responses)
        n_persons = responses.shape[0]

        patterns = self._attribute_patterns
        n_patterns = len(patterns)

        log_likes = np.zeros((n_persons, n_patterns))
        for p_idx, pattern in enumerate(patterns):
            alpha_broadcast = np.tile(pattern, (n_persons, 1))
            log_likes[:, p_idx] = self.log_likelihood(responses, alpha_broadcast)

        if method == "MAP":
            prior = np.log(1 / n_patterns)
            log_likes = log_likes + prior

        best_idx = np.argmax(log_likes, axis=1)
        return patterns[best_idx]

    def copy(self) -> Self:
        """Create a deep copy of this model."""
        new_model = DINO(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            item_names=self.item_names.copy() if self.item_names else None,
        )

        if self._parameters:
            for name, values in self._parameters.items():
                new_model._parameters[name] = values.copy()
            new_model._is_fitted = self._is_fitted

        return new_model


def fit_cdm(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    model: str = "DINA",
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[BaseCDM, NDArray[np.float64]]:
    """Fit a cognitive diagnosis model using EM algorithm.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    q_matrix : NDArray
        Q-matrix (n_items, n_attributes)
    model : str
        Model type ('DINA' or 'DINO')
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple
        (fitted_model, class_probabilities)
    """
    responses = np.asarray(responses)
    q_matrix = np.asarray(q_matrix)

    n_persons, n_items = responses.shape
    n_attributes = q_matrix.shape[1]

    if model.upper() == "DINA":
        cdm = DINA(n_items=n_items, n_attributes=n_attributes, q_matrix=q_matrix)
    elif model.upper() == "DINO":
        cdm = DINO(n_items=n_items, n_attributes=n_attributes, q_matrix=q_matrix)
    else:
        raise ValueError(f"Unknown CDM model: {model}")

    cdm._initialize_parameters()
    patterns = cdm.attribute_patterns
    n_patterns = len(patterns)

    class_probs = np.ones(n_patterns) / n_patterns

    prev_ll = -np.inf

    for iteration in range(max_iter):
        log_like_matrix = np.zeros((n_persons, n_patterns))

        for p_idx, pattern in enumerate(patterns):
            alpha = np.tile(pattern, (n_persons, 1))
            log_like_matrix[:, p_idx] = cdm.log_likelihood(responses, alpha)

        log_posterior = log_like_matrix + np.log(class_probs + 1e-10)

        log_sum = np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_sum)

        class_probs = posterior.mean(axis=0)

        for j in range(n_items):
            eta_patterns = cdm.eta(patterns, j)

            n_1_eta_1 = 0
            n_0_eta_1 = 0
            n_1_eta_0 = 0
            n_0_eta_0 = 0

            for p_idx in range(n_patterns):
                weight = posterior[:, p_idx]
                valid = responses[:, j] >= 0

                if eta_patterns[p_idx] == 1:
                    n_1_eta_1 += np.sum(weight[valid] * responses[valid, j])
                    n_0_eta_1 += np.sum(weight[valid] * (1 - responses[valid, j]))
                else:
                    n_1_eta_0 += np.sum(weight[valid] * responses[valid, j])
                    n_0_eta_0 += np.sum(weight[valid] * (1 - responses[valid, j]))

            denom_s = n_1_eta_1 + n_0_eta_1
            if denom_s > 1e-10:
                cdm._parameters["slip"][j] = np.clip(n_0_eta_1 / denom_s, 0.001, 0.999)

            denom_g = n_1_eta_0 + n_0_eta_0
            if denom_g > 1e-10:
                cdm._parameters["guess"][j] = np.clip(n_1_eta_0 / denom_g, 0.001, 0.999)

        current_ll = np.sum(log_sum)

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_ll:.4f}")

        if abs(current_ll - prev_ll) < tol:
            break

        prev_ll = current_ll

    cdm._is_fitted = True

    return cdm, class_probs
