"""Cognitive diagnosis models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.base import BaseItemModel

if TYPE_CHECKING:
    from mirt.models.cdm_advanced import AttributeHierarchy

ClassificationMethod = Literal["MLE", "MAP", "EAP"]


class BaseCDM(BaseItemModel):
    """Base class for cognitive diagnosis models.

    CDMs model discrete latent attributes using a binary Q-matrix whose rows
    identify the attributes required by each item. Respondents belong to one
    latent class per mastery pattern (attribute profile). The scoring methods
    :meth:`pattern_log_likelihoods`, :meth:`attribute_posteriors`,
    :meth:`attribute_marginals` and :meth:`classify_respondents` evaluate
    every respondent against every pattern.
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
            raise MirtValidationError(
                "n_attributes must be a positive integer", parameter="n_attributes"
            )

        raw_q = np.asarray(q_matrix)
        if raw_q.shape != (n_items, int(n_attributes)):
            raise MirtValidationError(
                f"Q-matrix shape {raw_q.shape} does not match "
                f"({n_items}, {n_attributes})"
            )
        if raw_q.dtype.kind not in "biuf":
            raise MirtValidationError("Q-matrix must contain numeric binary values")
        q_values = np.asarray(raw_q, dtype=np.float64)
        if not np.all(np.isfinite(q_values)) or np.any(
            (q_values != 0.0) & (q_values != 1.0)
        ):
            raise MirtValidationError("Q-matrix must contain only 0 and 1")

        self._n_attributes = int(n_attributes)
        self._q_matrix = q_values.astype(np.int_, copy=True)
        # Estimated latent-class distribution and, for a model fitted under an
        # attribute hierarchy, the patterns that hierarchy permits.
        self._class_probabilities: NDArray[np.float64] | None = None
        self._permissible_patterns: NDArray[np.bool_] | None = None
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

    @property
    def class_probabilities(self) -> NDArray[np.float64] | None:
        """Estimated probabilities of the latent classes, or ``None``.

        :func:`fit_cdm` and :func:`~mirt.models.cdm_advanced.fit_gdina` store
        the estimated probability of each row of :attr:`attribute_patterns`
        here. Scoring methods use it as the default pattern prior. An unfitted
        model has no estimate and returns ``None``; its scoring methods then
        use a uniform prior.
        """
        if self._class_probabilities is None:
            return None
        return self._class_probabilities.copy()

    def _generate_attribute_patterns(self) -> NDArray[np.int_]:
        """Generate all binary mastery patterns in little-endian order."""
        pattern_indices = np.arange(2**self._n_attributes, dtype=np.uint64)[:, None]
        bit_indices = np.arange(self._n_attributes, dtype=np.uint64)[None, :]
        return ((pattern_indices >> bit_indices) & 1).astype(np.int_)

    def _ensure_alpha_2d(self, alpha: ArrayLike) -> NDArray[np.int_]:
        """Validate and normalize one or more binary mastery patterns."""
        raw = np.asarray(alpha)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2:
            raise MirtValidationError("alpha must be a one- or two-dimensional array")
        if raw.shape[1] != self._n_attributes:
            raise MirtValidationError(
                f"alpha has {raw.shape[1]} attributes, expected {self._n_attributes}"
            )
        if raw.dtype.kind not in "biuf":
            raise MirtValidationError("alpha must contain numeric binary values")
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any((values != 0.0) & (values != 1.0)):
            raise MirtValidationError("alpha must contain only 0 and 1")
        return values.astype(np.int_, copy=False)

    def _validate_responses(
        self,
        responses: ArrayLike,
        *,
        require_observed: bool = False,
    ) -> NDArray[np.int_]:
        """Return binary responses with every missing value coded as ``-1``.

        Negative values mark missing responses. With ``require_observed``,
        as estimation needs, the matrix must also contain at least one person
        and an observed response to every item.
        """
        raw = np.asarray(responses)
        if raw.ndim != 2:
            raise MirtDataError(f"responses must be 2D, got {raw.ndim}D")
        if raw.shape[1] != self.n_items:
            raise MirtDataError(
                f"responses has {raw.shape[1]} items, expected {self.n_items}"
            )
        if raw.dtype.kind not in "biuf":
            raise MirtDataError("responses must contain numeric values")
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise MirtDataError("responses must contain only finite values")
        observed = values >= 0.0
        if np.any(observed & (values != 0.0) & (values != 1.0)):
            raise MirtDataError(
                "responses must contain only 0, 1, or negative missing values"
            )
        if require_observed:
            if raw.shape[0] == 0:
                raise MirtDataError("responses must contain at least one person")
            if np.any(~np.any(observed, axis=0)):
                raise MirtDataError(
                    "every item must have at least one observed response"
                )
        return np.where(observed, values, -1.0).astype(np.int_, copy=False)

    def eta(
        self,
        alpha: NDArray[np.int_],
        item_idx: int,
    ) -> NDArray[np.int_]:
        """Compute ideal responses for one item."""
        raise NotImplementedError("Subclasses must implement eta()")

    def _profile_patterns(self) -> NDArray[np.int_]:
        """Mastery patterns that the scoring methods evaluate."""
        return self._attribute_patterns

    def _pattern_response_probabilities(
        self,
        alpha: NDArray[np.int_] | None = None,
    ) -> NDArray[np.float64]:
        """Item success probabilities given validated mastery patterns."""
        patterns = self._profile_patterns() if alpha is None else alpha
        return np.asarray(self.probability(patterns), dtype=np.float64)

    def _default_pattern_prior(self) -> NDArray[np.float64]:
        """Fitted class probabilities, or a uniform prior for unfitted models."""
        if self._class_probabilities is not None:
            return self._class_probabilities.copy()
        n_patterns = self._profile_patterns().shape[0]
        return np.full(n_patterns, 1.0 / n_patterns, dtype=np.float64)

    def _normalize_pattern_prior(
        self,
        pattern_prior: ArrayLike | None,
    ) -> NDArray[np.float64]:
        if pattern_prior is None:
            return self._default_pattern_prior()
        n_patterns = self._profile_patterns().shape[0]
        prior = np.asarray(pattern_prior, dtype=np.float64)
        if prior.shape != (n_patterns,):
            raise MirtValidationError(
                f"pattern_prior must have shape ({n_patterns},)",
                parameter="pattern_prior",
            )
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise MirtValidationError(
                "pattern_prior must contain finite non-negative values",
                parameter="pattern_prior",
            )
        total = float(np.sum(prior))
        if total <= 0.0:
            raise MirtValidationError(
                "pattern_prior must have a positive sum", parameter="pattern_prior"
            )
        return prior / total

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        alpha: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute paired or broadcast log likelihoods.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Binary responses; negative values are missing.
        alpha : ndarray of shape (n_persons, n_attributes)
            Mastery pattern of each respondent. A single row of either
            argument is broadcast against the other.

        Returns
        -------
        ndarray
            Log-likelihood of each response row given its mastery pattern.
        """
        response_values = self._validate_responses(responses)
        alpha_values = self._ensure_alpha_2d(alpha)
        if (
            response_values.shape[0] != alpha_values.shape[0]
            and response_values.shape[0] != 1
            and alpha_values.shape[0] != 1
        ):
            raise MirtValidationError(
                "responses and alpha must have compatible row counts"
            )

        probabilities = np.clip(
            self._pattern_response_probabilities(alpha_values),
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

    def pattern_log_likelihoods(
        self,
        responses: ArrayLike,
    ) -> NDArray[np.float64]:
        """Evaluate every respondent against every mastery pattern.

        Parameters
        ----------
        responses : array_like of shape (n_persons, n_items)
            Binary responses; negative values are missing.

        Returns
        -------
        ndarray of shape (n_persons, n_patterns)
            Log-likelihood of each response row for each row of
            :attr:`attribute_patterns` (``2**n_attributes`` patterns).
        """
        response_values = self._validate_responses(responses)
        correct, incorrect, _ = _response_indicators(response_values)
        return _pattern_log_likelihood_matrix(
            correct,
            incorrect,
            self._pattern_response_probabilities(),
        )

    def attribute_posteriors(
        self,
        responses: ArrayLike,
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return posterior probabilities for every mastery pattern.

        Parameters
        ----------
        responses : array_like of shape (n_persons, n_items)
            Binary responses; negative values are missing.
        pattern_prior : array_like of shape (n_patterns,), optional
            Non-negative prior weights of the patterns, normalized to sum to
            one. By default, a fitted model uses its estimated
            :attr:`class_probabilities` and an unfitted model a uniform prior.

        Returns
        -------
        ndarray of shape (n_persons, n_patterns)
            Posterior probability of each pattern for each respondent.
        """
        log_likelihoods = self.pattern_log_likelihoods(responses)
        prior = self._normalize_pattern_prior(pattern_prior)
        posterior, _ = _class_posterior(log_likelihoods, prior)
        return posterior

    def attribute_marginals(
        self,
        responses: ArrayLike,
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return posterior mastery probability for each attribute.

        Parameters
        ----------
        responses : array_like of shape (n_persons, n_items)
            Binary responses; negative values are missing.
        pattern_prior : array_like of shape (n_patterns,), optional
            Pattern prior, as in :meth:`attribute_posteriors`.

        Returns
        -------
        ndarray of shape (n_persons, n_attributes)
            Marginal posterior probability that each attribute is mastered.
        """
        posterior = self.attribute_posteriors(responses, pattern_prior)
        return posterior @ self._profile_patterns()

    def classify_respondents(
        self,
        responses: NDArray[np.int_],
        method: ClassificationMethod = "MLE",
        pattern_prior: ArrayLike | None = None,
    ) -> NDArray[np.int_]:
        """Classify respondents into mastery patterns.

        Parameters
        ----------
        responses : ndarray of shape (n_persons, n_items)
            Binary responses; negative values are missing.
        method : {"MLE", "MAP", "EAP"}, default="MLE"
            ``"MLE"`` picks the pattern with the largest likelihood,
            ``"MAP"`` the pattern with the largest posterior probability and
            ``"EAP"`` masters every attribute whose marginal posterior
            probability exceeds 0.5. Case is ignored.
        pattern_prior : array_like of shape (n_patterns,), optional
            Pattern prior for ``"MAP"`` and ``"EAP"``, as in
            :meth:`attribute_posteriors`. By default, a fitted model uses its
            estimated :attr:`class_probabilities`.

        Returns
        -------
        ndarray of shape (n_persons, n_attributes)
            Classified mastery pattern of each respondent.

        Notes
        -----
        For a model fitted under an attribute hierarchy, ``"MLE"`` only
        considers the patterns that hierarchy permits. ``"EAP"`` classifies
        attributes one at a time and may return a pattern outside it.
        """
        if not isinstance(method, str) or method.upper() not in {"MLE", "MAP", "EAP"}:
            raise MirtValidationError(
                "method must be 'MLE', 'MAP' or 'EAP'", parameter="method"
            )
        normalized_method = method.upper()
        if normalized_method == "EAP":
            marginals = self.attribute_marginals(responses, pattern_prior)
            return (marginals > 0.5).astype(np.int_)
        if normalized_method == "MLE":
            if pattern_prior is not None:
                raise MirtValidationError(
                    "pattern_prior can only be used with method='MAP' or 'EAP'",
                    parameter="pattern_prior",
                )
            criterion = self.pattern_log_likelihoods(responses)
            if self._permissible_patterns is not None:
                criterion[:, ~self._permissible_patterns] = -np.inf
        else:
            criterion = self.attribute_posteriors(responses, pattern_prior)
        best_indices = np.argmax(criterion, axis=1)
        return self._profile_patterns()[best_indices].copy()

    def sample_attribute_patterns(
        self,
        n_persons: int,
        seed: int | None = None,
        class_probabilities: ArrayLike | None = None,
    ) -> NDArray[np.int_]:
        """Draw mastery patterns from a latent-class distribution.

        Parameters
        ----------
        n_persons : int
            Number of patterns to draw.
        seed : int, optional
            Random seed for reproducible draws.
        class_probabilities : array_like of shape (n_patterns,), optional
            Probabilities of the patterns. By default, a fitted model uses its
            estimated :attr:`class_probabilities` and an unfitted model a
            uniform distribution.

        Returns
        -------
        ndarray of shape (n_persons, n_attributes)
            Drawn mastery patterns, suitable for :meth:`simulate`.
        """
        if (
            isinstance(n_persons, (bool, np.bool_))
            or not isinstance(n_persons, (int, np.integer))
            or n_persons < 0
        ):
            raise MirtValidationError(
                "n_persons must be a non-negative integer", parameter="n_persons"
            )
        probabilities = self._normalize_pattern_prior(class_probabilities)
        rng = np.random.default_rng(seed)
        indices = rng.choice(probabilities.size, size=int(n_persons), p=probabilities)
        return self._profile_patterns()[indices].copy()

    def simulate(
        self,
        alpha: ArrayLike,
        seed: int | None = None,
    ) -> NDArray[np.int32]:
        """Simulate binary responses conditional on mastery patterns.

        Parameters
        ----------
        alpha : array_like of shape (n_persons, n_attributes)
            Mastery pattern of each simulated respondent, for example from
            :meth:`sample_attribute_patterns`.
        seed : int, optional
            Random seed for reproducible response draws.

        Returns
        -------
        ndarray of shape (n_persons, n_items)
            Binary response matrix.
        """
        alpha_values = self._ensure_alpha_2d(alpha)
        probabilities = self._pattern_response_probabilities(alpha_values)
        rng = np.random.default_rng(seed)
        return (rng.random(probabilities.shape) < probabilities).astype(np.int32)

    def _copy_cdm_state_to(self, model: BaseCDM) -> None:
        """Give a copy owned versions of the estimated latent-class state."""
        model._class_probabilities = (
            None
            if self._class_probabilities is None
            else self._class_probabilities.copy()
        )
        model._permissible_patterns = (
            None
            if self._permissible_patterns is None
            else self._permissible_patterns.copy()
        )


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

    @property
    def free_parameter_masks(self) -> dict[str, NDArray[np.bool_]]:
        masks = super().free_parameter_masks
        # With no required attributes both noisy gates always classify an
        # item as mastered, so its nonmastered probability is unused.
        masks["guess"] &= np.any(self._q_matrix, axis=1)
        return self._apply_free_parameter_restrictions(masks)

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        slip = np.asarray(params.get("slip", self.slip), dtype=np.float64)
        guess = np.asarray(params.get("guess", self.guess), dtype=np.float64)
        for name, values in (("slip", slip), ("guess", guess)):
            if not np.all(np.isfinite(values)) or np.any(
                (values < 0.0) | (values > 1.0)
            ):
                raise MirtValidationError(
                    f"{name} must contain finite probabilities in [0, 1]",
                    parameter=name,
                )
        if slip.shape == self.slip.shape and guess.shape == self.guess.shape:
            if np.any(slip + guess >= 1.0):
                raise MirtValidationError(
                    "slip + guess must be less than 1 for every item"
                )
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

    def probability_pairs(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Evaluate aligned mastery-pattern and item pairs."""
        alpha_values = self._ensure_alpha_2d(alpha)
        indices = self._prepare_item_indices(item_indices, alpha_values.shape[0])
        ideal = self._paired_ideal_responses(alpha_values, indices)
        return np.where(
            ideal,
            1.0 - self._parameters["slip"][indices],
            self._parameters["guess"][indices],
        )

    def _paired_ideal_responses(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        raise NotImplementedError

    def information(
        self,
        alpha: NDArray[np.int_],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Return Bernoulli response variance for each pattern and item."""
        probabilities = self.probability(alpha, item_idx)
        return probabilities * (1.0 - probabilities)

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
        self._copy_parameter_restrictions_to(new_model)
        self._copy_cdm_state_to(new_model)
        return new_model


class DINA(_NoisyGateCDM):
    """Deterministic Input, Noisy AND gate model."""

    model_name = "DINA"

    def _ideal_responses(self, alpha: NDArray[np.int_]) -> NDArray[np.bool_]:
        return np.all(alpha[:, None, :] >= self._q_matrix[None, :, :], axis=2)

    def _paired_ideal_responses(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        return np.all(alpha >= self._q_matrix[item_indices], axis=1)


class DINO(_NoisyGateCDM):
    """Deterministic Input, Noisy OR gate model."""

    model_name = "DINO"

    def _ideal_responses(self, alpha: NDArray[np.int_]) -> NDArray[np.bool_]:
        required_mastered = np.any(
            (alpha[:, None, :] == 1) & (self._q_matrix[None, :, :] == 1),
            axis=2,
        )
        return required_mastered | ~np.any(self._q_matrix, axis=1)[None, :]

    def _paired_ideal_responses(
        self,
        alpha: NDArray[np.int_],
        item_indices: NDArray[np.intp],
    ) -> NDArray[np.bool_]:
        q_matrix = self._q_matrix[item_indices]
        required_mastered = np.any((alpha == 1) & (q_matrix == 1), axis=1)
        return required_mastered | ~np.any(q_matrix, axis=1)


CDMModelT = TypeVar("CDMModelT", bound=BaseCDM)


@dataclass
class CDMResult(Generic[CDMModelT]):
    """Result of fitting a cognitive diagnosis model by EM.

    The result also unpacks as the pair returned before it existed, so
    ``model, class_probabilities = fit_cdm(...)`` keeps working, and
    ``result[0]`` and ``result[1]`` index that pair.

    Attributes
    ----------
    model : BaseCDM
        Fitted model. It stores the estimated ``class_probabilities`` and
        uses them as the default prior of its scoring methods.
    class_probabilities : ndarray of shape (n_patterns,)
        Estimated probability of each mastery pattern, in the order of the
        model's ``attribute_patterns`` (``valid_patterns`` for
        :class:`~mirt.models.cdm_advanced.HigherOrderCDM`).
    log_likelihood : float
        Marginal log-likelihood at the returned estimates.
    n_parameters : int
        Free item parameters plus the structural parameters of the
        latent-class distribution: one fewer than the number of permitted
        patterns, or the higher-order loadings and thresholds.
    n_observations : int
        Number of respondents, the sample size of :attr:`bic`.
    n_iterations : int
        Number of EM iterations performed.
    converged : bool
        Whether the log-likelihood change fell below ``tol`` before
        ``max_iter`` iterations.
    """

    model: CDMModelT
    class_probabilities: NDArray[np.float64]
    log_likelihood: float
    n_parameters: int
    n_observations: int
    n_iterations: int
    converged: bool

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    @property
    def bic(self) -> float:
        """Bayesian information criterion with the number of respondents as ``n``."""
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(
            max(self.n_observations, 1)
        )

    def __iter__(self) -> Iterator[Any]:
        yield self.model
        yield self.class_probabilities

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.model, self.class_probabilities)[index]

    def summary(self) -> str:
        """Return a short text summary of the fit."""
        width = 60
        lines = [
            "=" * width,
            f"{self.model.model_name + ' Estimation Results':^{width}}",
            "=" * width,
            f"Respondents:        {self.n_observations}",
            f"Parameters:         {self.n_parameters}",
            f"Log-Likelihood:     {self.log_likelihood:.4f}",
            f"AIC:                {self.aic:.4f}",
            f"BIC:                {self.bic:.4f}",
            f"Iterations:         {self.n_iterations}",
            f"Converged:          {self.converged}",
            "-" * width,
            "Attribute prevalence:",
        ]
        patterns = self.model._profile_patterns()
        prevalence = self.class_probabilities @ patterns
        lines.extend(
            f"  A{attribute}: {value:.3f}" for attribute, value in enumerate(prevalence)
        )
        lines.append("=" * width)
        return "\n".join(lines)


def _response_indicators(
    response_values: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return correct, incorrect and observed indicators of coded responses."""
    observed = (response_values >= 0).astype(np.float64)
    correct = np.where(response_values > 0, 1.0, 0.0)
    return correct, observed - correct, observed


def _pattern_log_likelihood_matrix(
    correct: NDArray[np.float64],
    incorrect: NDArray[np.float64],
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Log-likelihood of each response row for each pattern's item curves."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise MirtValidationError("model probabilities must contain only finite values")
    clipped = np.clip(values, PROB_EPSILON, 1.0 - PROB_EPSILON)
    return correct @ np.log(clipped).T + incorrect @ np.log1p(-clipped).T


def _class_posterior(
    log_likelihoods: NDArray[np.float64],
    class_probabilities: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return pattern posteriors and the marginal log-likelihood.

    Patterns with zero prior probability keep zero posterior probability.
    """
    log_prior = np.full(class_probabilities.shape, -np.inf, dtype=np.float64)
    positive = class_probabilities > 0.0
    log_prior[positive] = np.log(class_probabilities[positive])
    posterior = log_likelihoods + log_prior[None, :]
    shift = np.max(posterior, axis=1, keepdims=True)
    posterior -= shift
    np.exp(posterior, out=posterior)
    totals = np.sum(posterior, axis=1, keepdims=True)
    posterior /= totals
    return posterior, float(np.sum(shift) + np.sum(np.log(totals)))


def _hierarchy_pattern_mask(
    hierarchy: AttributeHierarchy | None,
    n_attributes: int,
    patterns: NDArray[np.int_],
) -> NDArray[np.bool_] | None:
    """Mark the patterns that satisfy every prerequisite of ``hierarchy``."""
    if hierarchy is None:
        return None
    from mirt.models.cdm_advanced import AttributeHierarchy

    if not isinstance(hierarchy, AttributeHierarchy):
        raise MirtValidationError(
            "hierarchy must be an AttributeHierarchy", parameter="hierarchy"
        )
    if hierarchy.n_attributes != n_attributes:
        raise MirtValidationError(
            f"hierarchy has {hierarchy.n_attributes} attributes, "
            f"expected {n_attributes}",
            parameter="hierarchy",
        )
    prerequisites, attributes = np.nonzero(hierarchy.adjacency == 1)
    return np.all(patterns[:, attributes] <= patterns[:, prerequisites], axis=1)


def _validate_fit_configuration(
    max_iter: int,
    tol: float,
    verbose: bool,
) -> tuple[int, float]:
    """Validate the EM controls shared by every CDM fitting function."""
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise MirtValidationError(
            "max_iter must be a positive integer", parameter="max_iter"
        )
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0.0
    ):
        raise MirtValidationError(
            "tol must be a positive finite number", parameter="tol"
        )
    if not isinstance(verbose, (bool, np.bool_)):
        raise MirtValidationError("verbose must be boolean", parameter="verbose")
    return int(max_iter), float(tol)


def _validate_fit_inputs(
    responses: ArrayLike,
    q_matrix: ArrayLike,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Check the response and Q-matrix shapes shared by the CDM fitters."""
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise MirtDataError(f"responses must be 2D, got {raw_responses.ndim}D")
    if raw_responses.shape[0] == 0 or raw_responses.shape[1] == 0:
        raise MirtDataError("responses must contain at least one person and one item")
    raw_q = np.asarray(q_matrix)
    if raw_q.ndim != 2 or raw_q.shape[1] == 0:
        raise MirtValidationError(
            "Q-matrix must be a non-empty two-dimensional array",
            parameter="q_matrix",
        )
    return raw_responses, raw_q


_GDINA_FIT_MODELS = {
    "GDINA": None,
    "ACDM": "ACDM",
    "LLM": "LLM",
    "RRUM": "RRUM",
}


def fit_cdm(
    responses: NDArray[np.int_],
    q_matrix: NDArray[np.int_],
    model: str = "DINA",
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    *,
    hierarchy: AttributeHierarchy | None = None,
) -> CDMResult[Any]:
    """Fit a cognitive diagnosis model by expectation maximization.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Binary responses; negative values are missing. Every item needs at
        least one observed response.
    q_matrix : ndarray of shape (n_items, n_attributes)
        Binary Q-matrix.
    model : {"DINA", "DINO", "GDINA", "ACDM", "LLM", "RRUM"}, default="DINA"
        Model to fit; case and surrounding whitespace are ignored. ``"DINA"``
        and ``"DINO"`` fit :class:`DINA` and :class:`DINO`. ``"GDINA"`` fits
        the saturated G-DINA model, and ``"ACDM"``, ``"LLM"`` and ``"RRUM"``
        fit that reduced G-DINA model to every item, through
        :func:`~mirt.models.cdm_advanced.fit_gdina`.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance on the change in marginal log-likelihood.
    verbose : bool, default=False
        Print the log-likelihood at each iteration.
    hierarchy : AttributeHierarchy, optional
        Attribute hierarchy restricting the latent classes. Patterns in which
        an attribute is mastered without all of its prerequisites keep zero
        class probability.

    Returns
    -------
    CDMResult
        Fitted model, estimated class probabilities, log-likelihood,
        information criteria and convergence information. The fitted model
        stores the class probabilities, so its ``classify_respondents``,
        ``attribute_posteriors`` and ``attribute_marginals`` use them as
        their default prior. The result unpacks as
        ``model, class_probabilities``.
    """
    max_iter, tol = _validate_fit_configuration(max_iter, tol, verbose)
    raw_responses, raw_q = _validate_fit_inputs(responses, q_matrix)

    valid_models = "'DINA', 'DINO', 'GDINA', 'ACDM', 'LLM' or 'RRUM'"
    if not isinstance(model, str):
        raise MirtValidationError(f"model must be {valid_models}", parameter="model")
    normalized_model = model.strip().upper()
    if normalized_model in _GDINA_FIT_MODELS:
        from mirt.models.cdm_advanced import fit_gdina

        reduced_model = _GDINA_FIT_MODELS[normalized_model]
        return fit_gdina(
            raw_responses,
            raw_q,
            reduced_models=(
                None
                if reduced_model is None
                else [reduced_model] * raw_responses.shape[1]
            ),
            max_iter=max_iter,
            tol=tol,
            verbose=bool(verbose),
            hierarchy=hierarchy,
        )
    model_types: dict[str, type[_NoisyGateCDM]] = {"DINA": DINA, "DINO": DINO}
    if normalized_model not in model_types:
        raise MirtValidationError(
            f"Unknown CDM model: {model}. Valid models: {valid_models}",
            parameter="model",
        )

    n_persons, n_items = raw_responses.shape
    cdm = model_types[normalized_model](
        n_items=n_items,
        n_attributes=raw_q.shape[1],
        q_matrix=raw_q,
    )
    response_values = cdm._validate_responses(raw_responses, require_observed=True)
    patterns = cdm._attribute_patterns
    permissible = _hierarchy_pattern_mask(hierarchy, cdm.n_attributes, patterns)
    n_patterns = patterns.shape[0]
    if permissible is None:
        class_probabilities = np.full(n_patterns, 1.0 / n_patterns)
    else:
        class_probabilities = permissible / np.sum(permissible)
    ideal_patterns = cdm._ideal_responses(patterns).astype(np.float64)
    correct, incorrect, observed_values = _response_indicators(response_values)
    previous_log_likelihood = -np.inf
    converged = False
    n_iterations = max_iter

    for iteration in range(max_iter):
        log_likelihoods = _pattern_log_likelihood_matrix(
            correct, incorrect, cdm._pattern_response_probabilities()
        )
        posterior, current_log_likelihood = _class_posterior(
            log_likelihoods, class_probabilities
        )
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

        if verbose:
            print(f"Iteration {iteration + 1}: LL = {current_log_likelihood:.4f}")
        if abs(current_log_likelihood - previous_log_likelihood) < tol:
            converged = True
            n_iterations = iteration + 1
            break
        previous_log_likelihood = current_log_likelihood

    # Each iteration's log-likelihood precedes its M-step; report the value
    # at the returned estimates.
    _, final_log_likelihood = _class_posterior(
        _pattern_log_likelihood_matrix(
            correct, incorrect, cdm._pattern_response_probabilities()
        ),
        class_probabilities,
    )
    cdm._class_probabilities = class_probabilities.copy()
    cdm._permissible_patterns = None if permissible is None else permissible.copy()
    cdm._is_fitted = True
    n_classes = n_patterns if permissible is None else int(np.sum(permissible))
    return CDMResult(
        model=cdm,
        class_probabilities=class_probabilities.copy(),
        log_likelihood=final_log_likelihood,
        n_parameters=cdm.n_parameters + n_classes - 1,
        n_observations=n_persons,
        n_iterations=n_iterations,
        converged=converged,
    )
