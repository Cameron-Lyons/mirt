"""Cognitive diagnosis models."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON
from mirt.models.base import BaseItemModel
from mirt.utils.numeric import logsumexp

ClassificationMethod = Literal["MLE", "MAP"]


class BaseCDM(BaseItemModel):
    """Base class for cognitive diagnosis models.

    CDMs model discrete latent attributes using a binary Q-matrix whose rows
    identify the attributes required by each item.
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
        if (
            isinstance(n_attributes, (bool, np.bool_))
            or not isinstance(n_attributes, (int, np.integer))
            or n_attributes <= 0
        ):
            raise ValueError("n_attributes must be a positive integer")

        raw_q = np.asarray(q_matrix)
        if raw_q.shape != (n_items, int(n_attributes)):
            raise ValueError(
                f"Q-matrix shape {raw_q.shape} does not match "
                f"({n_items}, {n_attributes})"
            )
        if raw_q.dtype.kind not in "biuf":
            raise ValueError("Q-matrix must contain numeric binary values")
        q_values = np.asarray(raw_q, dtype=np.float64)
        if not np.all(np.isfinite(q_values)) or np.any(
            (q_values != 0.0) & (q_values != 1.0)
        ):
            raise ValueError("Q-matrix must contain only 0 and 1")

        self._n_attributes = int(n_attributes)
        self._q_matrix = q_values.astype(np.int_, copy=True)
        super().__init__(
            n_items=n_items,
            n_factors=self._n_attributes,
            item_names=item_names,
        )
        self._attribute_patterns = self._generate_attribute_patterns()

    @property
    def n_attributes(self) -> int:
        """Number of latent attributes."""
        return self._n_attributes

    @property
    def q_matrix(self) -> NDArray[np.int_]:
        """Return an owned copy of the item-by-attribute Q-matrix."""
        return self._q_matrix.copy()

    @property
    def attribute_patterns(self) -> NDArray[np.int_]:
        """Return an owned copy of all ``2**n_attributes`` mastery patterns."""
        return self._attribute_patterns.copy()

    def _generate_attribute_patterns(self) -> NDArray[np.int_]:
        """Generate all binary mastery patterns in little-endian order."""
        pattern_indices = np.arange(2**self._n_attributes, dtype=np.uint64)[:, None]
        bit_indices = np.arange(self._n_attributes, dtype=np.uint64)[None, :]
        return ((pattern_indices >> bit_indices) & 1).astype(np.int_)

    def _validate_item_index(self, item_idx: int) -> int:
        if (
            isinstance(item_idx, (bool, np.bool_))
            or not isinstance(item_idx, (int, np.integer))
            or item_idx < 0
            or item_idx >= self.n_items
        ):
            raise IndexError(f"item_idx must be in [0, {self.n_items})")
        return int(item_idx)

    def _ensure_alpha_2d(self, alpha: ArrayLike) -> NDArray[np.int_]:
        """Validate and normalize one or more binary mastery patterns."""
        raw = np.asarray(alpha)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2:
            raise ValueError("alpha must be a one- or two-dimensional array")
        if raw.shape[1] != self._n_attributes:
            raise ValueError(
                f"alpha has {raw.shape[1]} attributes, expected {self._n_attributes}"
            )
        if raw.dtype.kind not in "biuf":
            raise ValueError("alpha must contain numeric binary values")
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
            raise ValueError("alpha must contain only 0 and 1")
        return values.astype(np.int_, copy=False)

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute ideal responses for one item."""
        raise NotImplementedError("Subclasses must implement eta()")


class _NoisyGateCDM(BaseCDM):
    """Shared implementation for two-parameter conjunctive/disjunctive CDMs."""

    def _initialize_parameters(self) -> None:
        self._parameters["slip"] = np.full(self.n_items, 0.1, dtype=np.float64)
        self._parameters["guess"] = np.full(self.n_items, 0.2, dtype=np.float64)

    @property
    def slip(self) -> NDArray[np.float64]:
        """Slip probabilities for mastered item profiles."""
        return self._parameters["slip"]

    @property
    def guess(self) -> NDArray[np.float64]:
        """Correct-response probabilities for nonmastered item profiles."""
        return self._parameters["guess"]

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        slip = np.asarray(params.get("slip", self.slip), dtype=np.float64)
        guess = np.asarray(params.get("guess", self.guess), dtype=np.float64)
        for name, values in (("slip", slip), ("guess", guess)):
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
        if slip.shape == self.slip.shape and guess.shape == self.guess.shape:
            if np.any(slip + guess >= 1.0):
                raise ValueError("slip + guess must be less than 1 for every item")
        return super().set_parameters(**params)

    def _ideal_responses(self, alpha: NDArray[np.int_]) -> NDArray[np.bool_]:
        raise NotImplementedError

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        alpha_values = self._ensure_alpha_2d(alpha)
        index = self._validate_item_index(item_idx)
        return self._ideal_responses(alpha_values)[:, index].astype(np.int_)

    def probability(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute correct-response probabilities for mastery patterns."""
        alpha_values = self._ensure_alpha_2d(alpha)
        ideal = self._ideal_responses(alpha_values)
        probabilities = np.where(
            ideal,
            1.0 - self._parameters["slip"][None, :],
            self._parameters["guess"][None, :],
        )
        if item_idx is None:
            return probabilities
        return probabilities[:, self._validate_item_index(item_idx)]

    def _validate_responses(self, responses: ArrayLike) -> NDArray[np.int_]:
        raw = np.asarray(responses)
        if raw.ndim != 2:
            raise ValueError(f"responses must be 2D, got {raw.ndim}D")
        if raw.shape[1] != self.n_items:
            raise ValueError(
                f"responses has {raw.shape[1]} items, expected {self.n_items}"
            )
        if raw.dtype.kind not in "biuf":
            raise ValueError("responses must contain numeric values")
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("responses must contain only finite values")
        observed = values >= 0.0
        if np.any(observed & (values != 0.0) & (values != 1.0)):
            raise ValueError(
                "responses must contain only 0, 1, or negative missing values"
            )
        return np.where(observed, values, -1.0).astype(np.int_, copy=False)

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        alpha: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute paired or broadcast log likelihoods."""
        response_values = self._validate_responses(responses)
        alpha_values = self._ensure_alpha_2d(alpha)
        if (
            response_values.shape[0] != alpha_values.shape[0]
            and response_values.shape[0] != 1
            and alpha_values.shape[0] != 1
        ):
            raise ValueError("responses and alpha must have compatible row counts")

        probabilities = np.clip(
            self.probability(alpha_values),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        observed = response_values >= 0
        values = np.where(observed, response_values, 0)
        contributions = np.where(
            observed,
            values * np.log(probabilities) + (1 - values) * np.log1p(-probabilities),
            0.0,
        )
        return np.sum(contributions, axis=1)

    def information(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return Bernoulli response variance for each pattern and item."""
        probabilities = self.probability(alpha, item_idx)
        return probabilities * (1.0 - probabilities)

    def pattern_log_likelihoods(
        self,
        responses: ArrayLike,
    ) -> NDArray[np.float64]:
        """Evaluate every respondent against every mastery pattern.

        Returns an array with shape ``(n_persons, 2**n_attributes)``.
        Negative response values are treated as missing.
        """
        response_values = self._validate_responses(responses)
        probabilities = np.clip(
            self.probability(self._attribute_patterns),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        log_correct = np.log(probabilities)
        log_incorrect = np.log1p(-probabilities)
        observed = response_values >= 0
        values = np.where(observed, response_values, 0).astype(np.float64, copy=False)
        observed_values = observed.astype(np.float64)
        return values @ log_correct.T + (observed_values - values) @ log_incorrect.T

    def _normalize_pattern_prior(
        self,
        pattern_prior: ArrayLike | None,
    ) -> NDArray[np.float64]:
        n_patterns = self._attribute_patterns.shape[0]
        if pattern_prior is None:
            return np.full(n_patterns, 1.0 / n_patterns, dtype=np.float64)
        prior = np.asarray(pattern_prior, dtype=np.float64)
        if prior.shape != (n_patterns,):
            raise ValueError(f"pattern_prior must have shape ({n_patterns},)")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise ValueError("pattern_prior must contain finite non-negative values")
        total = float(np.sum(prior))
        if total <= 0.0:
            raise ValueError("pattern_prior must have a positive sum")
        return prior / total

    def attribute_posteriors(
        self,
        responses: ArrayLike,
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return posterior probabilities for every mastery pattern."""
        log_likelihoods = self.pattern_log_likelihoods(responses)
        prior = self._normalize_pattern_prior(pattern_prior)
        log_prior = np.full(prior.shape, -np.inf, dtype=np.float64)
        positive = prior > 0.0
        log_prior[positive] = np.log(prior[positive])
        log_posterior = log_likelihoods + log_prior[None, :]
        log_normalizer = logsumexp(log_posterior, axis=1, keepdims=True)
        return np.exp(log_posterior - log_normalizer)

    def attribute_marginals(
        self,
        responses: ArrayLike,
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return posterior mastery probability for each attribute."""
        posterior = self.attribute_posteriors(responses, pattern_prior)
        return posterior @ self._attribute_patterns

    def classify_respondents(
        self,
        responses: NDArray[np.int_],
        method: ClassificationMethod = "MLE",
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.int_]:
        """Classify respondents by maximum likelihood or posterior mass."""
        if not isinstance(method, str) or method.upper() not in {"MLE", "MAP"}:
            raise ValueError("method must be 'MLE' or 'MAP'")
        normalized_method = method.upper()
        if normalized_method == "MLE":
            if pattern_prior is not None:
                raise ValueError("pattern_prior can only be used with method='MAP'")
            criterion = self.pattern_log_likelihoods(responses)
        else:
            criterion = self.attribute_posteriors(responses, pattern_prior)
        best_indices = np.argmax(criterion, axis=1)
        return self._attribute_patterns[best_indices].copy()

    def copy(self) -> Self:
        """Create a structure-preserving deep copy."""
        new_model = self.__class__(
            n_items=self.n_items,
            n_attributes=self._n_attributes,
            q_matrix=self._q_matrix.copy(),
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        return new_model


class DINA(_NoisyGateCDM):
    """Deterministic Input, Noisy AND gate model."""

    model_name = "DINA"

    def _ideal_responses(self, alpha: NDArray[np.int_]) -> NDArray[np.bool_]:
        return np.all(alpha[:, None, :] >= self._q_matrix[None, :, :], axis=2)


class DINO(_NoisyGateCDM):
    """Deterministic Input, Noisy OR gate model."""

    model_name = "DINO"

    def _ideal_responses(self, alpha: NDArray[np.int_]) -> NDArray[np.bool_]:
        required_mastered = np.any(
            (alpha[:, None, :] == 1) & (self._q_matrix[None, :, :] == 1),
            axis=2,
        )
        return required_mastered | ~np.any(self._q_matrix, axis=1)[None, :]


def _validate_fit_configuration(
    max_iter: int,
    tol: float,
    verbose: bool,
) -> tuple[int, float]:
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("verbose must be boolean")
    return int(max_iter), float(tol)


def fit_cdm(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    model: str = "DINA",
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[BaseCDM, NDArray[np.float64]]:
    """Fit a DINA or DINO model with vectorized expectation maximization."""
    max_iter, tol = _validate_fit_configuration(max_iter, tol, verbose)
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise ValueError(f"responses must be 2D, got {raw_responses.ndim}D")
    if raw_responses.shape[0] == 0 or raw_responses.shape[1] == 0:
        raise ValueError("responses must contain at least one person and one item")

    raw_q = np.asarray(q_matrix)
    if raw_q.ndim != 2 or raw_q.shape[1] == 0:
        raise ValueError("Q-matrix must be a non-empty two-dimensional array")

    if not isinstance(model, str):
        raise ValueError("model must be 'DINA' or 'DINO'")
    normalized_model = model.strip().upper()
    model_types: dict[str, type[_NoisyGateCDM]] = {"DINA": DINA, "DINO": DINO}
    if normalized_model not in model_types:
        raise ValueError(f"Unknown CDM model: {model}")

    _, n_items = raw_responses.shape
    cdm = model_types[normalized_model](
        n_items=n_items,
        n_attributes=raw_q.shape[1],
        q_matrix=raw_q,
    )
    response_values = cdm._validate_responses(raw_responses)
    patterns = cdm._attribute_patterns
    n_patterns = patterns.shape[0]
    class_probabilities = np.full(n_patterns, 1.0 / n_patterns)
    ideal_patterns = cdm._ideal_responses(patterns).astype(np.float64)
    observed = response_values >= 0
    observed_values = observed.astype(np.float64)
    correct = np.where(observed, response_values, 0).astype(np.float64, copy=False)
    incorrect = observed_values - correct
    previous_log_likelihood = -np.inf

    for iteration in range(max_iter):
        log_likelihoods = cdm.pattern_log_likelihoods(response_values)
        log_prior = np.full(n_patterns, -np.inf, dtype=np.float64)
        positive_prior = class_probabilities > 0.0
        log_prior[positive_prior] = np.log(class_probabilities[positive_prior])
        log_posterior = log_likelihoods + log_prior[None, :]
        log_normalizer = logsumexp(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_normalizer)
        class_probabilities = np.mean(posterior, axis=0)

        mastery_mass = posterior @ ideal_patterns
        nonmastery_mass = 1.0 - mastery_mass
        mastered_total = np.sum(observed_values * mastery_mass, axis=0)
        nonmastered_total = np.sum(observed_values * nonmastery_mass, axis=0)
        slip_numerator = np.sum(incorrect * mastery_mass, axis=0)
        guess_numerator = np.sum(correct * nonmastery_mass, axis=0)

        slip = cdm._parameters["slip"].copy()
        guess = cdm._parameters["guess"].copy()
        np.divide(
            slip_numerator,
            mastered_total,
            out=slip,
            where=mastered_total > PROB_EPSILON,
        )
        np.divide(
            guess_numerator,
            nonmastered_total,
            out=guess,
            where=nonmastered_total > PROB_EPSILON,
        )
        np.clip(slip, 0.001, 0.999, out=slip)
        np.clip(guess, 0.001, 0.999, out=guess)

        # Enforce the identifying monotonicity constraint P(correct | ideal)
        # > P(correct | nonideal). When unconstrained group estimates cross,
        # their pooled correct rate is the boundary optimum; use a small gap
        # around that rate to keep both latent groups distinguishable.
        violations = 1.0 - slip <= guess
        if np.any(violations):
            pooled_correct = np.divide(
                mastered_total - slip_numerator + guess_numerator,
                mastered_total + nonmastered_total,
                out=np.full(n_items, 0.5, dtype=np.float64),
                where=(mastered_total + nonmastered_total) > PROB_EPSILON,
            )
            nonideal_correct = np.clip(pooled_correct - 0.0005, 0.001, 0.998)
            ideal_correct = np.clip(pooled_correct + 0.0005, 0.002, 0.999)
            guess[violations] = nonideal_correct[violations]
            slip[violations] = 1.0 - ideal_correct[violations]

        cdm._parameters["slip"] = slip
        cdm._parameters["guess"] = guess

        current_log_likelihood = float(np.sum(log_normalizer))
        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_log_likelihood:.4f}")
        if abs(current_log_likelihood - previous_log_likelihood) < tol:
            break
        previous_log_likelihood = current_log_likelihood

    cdm._is_fitted = True
    return cdm, class_probabilities.copy()
