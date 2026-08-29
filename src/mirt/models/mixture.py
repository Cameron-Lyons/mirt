"""Finite-mixture IRT models and marginal maximum-likelihood fitting."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import logsumexp, roots_hermite

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtDataError, MirtEstimationError, MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.utils.data import validate_responses

_BASE_MODELS = frozenset({"1PL", "2PL", "3PL"})
_LOG_DISCRIMINATION_BOUNDS = (float(np.log(0.01)), float(np.log(10.0)))
_DIFFICULTY_BOUNDS = (-8.0, 8.0)
_GUESSING_BOUNDS = (0.0, 0.5)
_MIXTURE_MAX_SIMULATION_VALUES = 1_000_000
_MIXTURE_MAX_LIKELIHOOD_VALUES = 1_000_000


def _validate_chunk_size(value: int | None, name: str) -> int | None:
    """Validate an optional positive chunk control."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class MixtureIRT(BaseItemModel):
    """Unidimensional IRT model with respondent-level latent classes.

    Each respondent belongs to one class for the complete response pattern.
    Classes have distinct 1PL, 2PL, or 3PL item parameters and share a
    standard-normal latent ability distribution during marginal fitting.
    """

    model_name = "MixtureIRT"
    supports_multidimensional = False

    def __init__(
        self,
        n_items: int,
        n_classes: int = 2,
        base_model: Literal["1PL", "2PL", "3PL"] = "2PL",
        item_names: list[str] | None = None,
    ) -> None:
        """Initialize a finite-mixture IRT model."""
        if isinstance(n_classes, (bool, np.bool_)) or not isinstance(
            n_classes, (int, np.integer)
        ):
            raise MirtValidationError(
                "n_classes must be an integer",
                parameter="n_classes",
                value=n_classes,
                expected="integer >= 2",
            )
        if n_classes < 2:
            raise MirtValidationError(
                "n_classes must be at least 2",
                parameter="n_classes",
                value=n_classes,
                expected=">= 2",
            )
        if base_model not in _BASE_MODELS:
            raise MirtValidationError(
                f"Unknown base_model: {base_model}",
                parameter="base_model",
                value=base_model,
                expected="'1PL', '2PL', or '3PL'",
            )

        self._n_classes = int(n_classes)
        self._base_model = base_model
        self._convergence_info: dict[str, object] | None = None
        super().__init__(n_items=n_items, n_factors=1, item_names=item_names)

    @property
    def n_classes(self) -> int:
        """Number of latent classes."""
        return self._n_classes

    @property
    def base_model(self) -> str:
        """Class-specific item model family."""
        return self._base_model

    @property
    def class_proportions(self) -> NDArray[np.float64]:
        """Return an independent copy of the class mixing proportions."""
        return self._parameters["class_proportions"].copy()

    @property
    def convergence_info(self) -> dict[str, object] | None:
        """Return fitting diagnostics, including the likelihood history."""
        if self._convergence_info is None:
            return None
        result = self._convergence_info.copy()
        history = result.get("log_likelihood_history")
        if isinstance(history, np.ndarray):
            result["log_likelihood_history"] = history.copy()
        return result

    def _initialize_parameters(self) -> None:
        """Initialize separated class locations and equal proportions."""
        self._parameters["class_proportions"] = np.full(
            self._n_classes, 1.0 / self._n_classes, dtype=np.float64
        )
        for class_idx in range(self._n_classes):
            if self._base_model != "1PL":
                self._parameters[f"discrimination_class{class_idx}"] = np.ones(
                    self.n_items, dtype=np.float64
                )
            offset = (class_idx - (self._n_classes - 1) / 2.0) * 0.5
            self._parameters[f"difficulty_class{class_idx}"] = np.full(
                self.n_items, offset, dtype=np.float64
            )
            if self._base_model == "3PL":
                self._parameters[f"guessing_class{class_idx}"] = np.full(
                    self.n_items, 0.2, dtype=np.float64
                )

    def set_parameters(self, **params: NDArray[np.float64]) -> Self:
        """Set class parameters after validating their statistical domain."""
        for name, value in params.items():
            values = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must be finite", parameter=name, value=value
                )
            if name == "class_proportions":
                if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
                    raise MirtValidationError(
                        "class_proportions must be non-negative and sum to 1",
                        parameter=name,
                        value=value,
                    )
            elif name.startswith("discrimination_class") and np.any(values <= 0.0):
                raise MirtValidationError(
                    "Discriminations must be positive",
                    parameter=name,
                    value=value,
                )
            elif name.startswith("guessing_class") and (
                np.any(values < 0.0) or np.any(values >= 1.0)
            ):
                raise MirtValidationError(
                    "Guessing parameters must lie in [0, 1)",
                    parameter=name,
                    value=value,
                )
        return super().set_parameters(**params)

    def _validate_class_index(self, class_idx: int) -> int:
        if isinstance(class_idx, (bool, np.bool_)) or not isinstance(
            class_idx, (int, np.integer)
        ):
            raise IndexError("class_idx must be an integer")
        index = int(class_idx)
        if index < 0 or index >= self._n_classes:
            raise IndexError(f"class_idx {index} out of range [0, {self._n_classes})")
        return index

    def _validate_item_index(self, item_idx: int) -> int:
        if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
            item_idx, (int, np.integer)
        ):
            raise IndexError("item_idx must be an integer")
        index = int(item_idx)
        if index < 0 or index >= self.n_items:
            raise IndexError(f"item_idx {index} out of range [0, {self.n_items})")
        return index

    def _stack_class_parameters(
        self,
        item_idx: int | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if item_idx is None:
            item_selection: int | slice = slice(None)
        else:
            item_selection = self._validate_item_index(item_idx)

        if self._base_model == "1PL":
            shape = (self._n_classes, self.n_items)
            discrimination = np.ones(shape, dtype=np.float64)
        else:
            discrimination = np.stack(
                [
                    self._parameters[f"discrimination_class{class_idx}"]
                    for class_idx in range(self._n_classes)
                ]
            )
        difficulty = np.stack(
            [
                self._parameters[f"difficulty_class{class_idx}"]
                for class_idx in range(self._n_classes)
            ]
        )
        if self._base_model == "3PL":
            guessing = np.stack(
                [
                    self._parameters[f"guessing_class{class_idx}"]
                    for class_idx in range(self._n_classes)
                ]
            )
        else:
            guessing = np.zeros_like(difficulty)

        discrimination = discrimination[:, item_selection]
        difficulty = difficulty[:, item_selection]
        guessing = guessing[:, item_selection]
        proportions = self._parameters["class_proportions"]
        if (
            proportions.shape != (self._n_classes,)
            or not np.all(np.isfinite(proportions))
            or np.any(proportions < 0.0)
            or not np.isclose(proportions.sum(), 1.0)
        ):
            raise MirtValidationError(
                "Stored class proportions must be non-negative and sum to 1"
            )
        if (
            not np.all(np.isfinite(discrimination))
            or np.any(discrimination <= 0.0)
            or not np.all(np.isfinite(difficulty))
            or not np.all(np.isfinite(guessing))
            or np.any(guessing < 0.0)
            or np.any(guessing >= 1.0)
        ):
            raise MirtValidationError("Stored class item parameters are invalid")
        return discrimination, difficulty, guessing

    def _class_parameter_arrays(
        self, class_idx: int
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if self._base_model == "1PL":
            discrimination = np.ones(self.n_items, dtype=np.float64)
        else:
            discrimination = self._parameters[f"discrimination_class{class_idx}"]
        difficulty = self._parameters[f"difficulty_class{class_idx}"]
        if self._base_model == "3PL":
            guessing = self._parameters[f"guessing_class{class_idx}"]
        else:
            guessing = np.zeros(self.n_items, dtype=np.float64)
        return discrimination, difficulty, guessing

    def get_class_parameters(self, class_idx: int) -> dict[str, NDArray[np.float64]]:
        """Return independent item parameter arrays for one class."""
        index = self._validate_class_index(class_idx)
        discrimination, difficulty, guessing = self._class_parameter_arrays(index)
        return {
            "discrimination": discrimination.copy(),
            "difficulty": difficulty.copy(),
            "guessing": guessing.copy(),
        }

    def _class_curves(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        theta_values = self._ensure_theta_2d(theta).ravel()
        discrimination, difficulty, guessing = self._stack_class_parameters(item_idx)
        if item_idx is None:
            z = discrimination[None, :, :] * (
                theta_values[:, None, None] - difficulty[None, :, :]
            )
            logistic = sigmoid(z)
            probability = guessing[None, :, :] + (1.0 - guessing[None, :, :]) * logistic
            derivative = (
                (1.0 - guessing[None, :, :])
                * discrimination[None, :, :]
                * logistic
                * (1.0 - logistic)
            )
        else:
            z = discrimination[None, :] * (theta_values[:, None] - difficulty[None, :])
            logistic = sigmoid(z)
            probability = guessing[None, :] + (1.0 - guessing[None, :]) * logistic
            derivative = (
                (1.0 - guessing[None, :])
                * discrimination[None, :]
                * logistic
                * (1.0 - logistic)
            )
        return probability, derivative

    def class_probability(
        self,
        theta: NDArray[np.float64],
        class_idx: int,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probabilities conditional on one class."""
        index = self._validate_class_index(class_idx)
        theta_values = self._ensure_theta_2d(theta).ravel()
        discrimination, difficulty, guessing = self._class_parameter_arrays(index)
        if (
            not np.all(np.isfinite(discrimination))
            or np.any(discrimination <= 0.0)
            or not np.all(np.isfinite(difficulty))
            or not np.all(np.isfinite(guessing))
            or np.any(guessing < 0.0)
            or np.any(guessing >= 1.0)
        ):
            raise MirtValidationError(
                f"Stored item parameters for class {index} are invalid"
            )
        if item_idx is None:
            z = discrimination[None, :] * (theta_values[:, None] - difficulty[None, :])
            logistic = sigmoid(z)
            return guessing[None, :] + (1.0 - guessing[None, :]) * logistic

        item = self._validate_item_index(item_idx)
        item_discrimination = discrimination[item]
        item_difficulty = difficulty[item]
        item_guessing = guessing[item]
        logistic = sigmoid(item_discrimination * (theta_values - item_difficulty))
        return item_guessing + (1.0 - item_guessing) * logistic

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute marginal item probabilities averaged over classes."""
        curves, _ = self._class_curves(theta, item_idx)
        return np.einsum(
            "k,tk...->t...",
            self._parameters["class_proportions"],
            curves,
            optimize=True,
        )

    def simulate(
        self,
        theta: NDArray[np.float64],
        seed: int | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Simulate response patterns with one latent class per person.

        Parameters
        ----------
        theta : ndarray
            Person abilities. A one-dimensional array represents people in
            this unidimensional model.
        seed : int, optional
            Random seed for reproducible class and response draws.
        chunk_size : int, optional
            Maximum number of persons evaluated at once. By default, a
            memory-bounded chunk size is selected from the item count.

        Returns
        -------
        tuple
            ``(responses, true_classes)`` where responses has shape
            ``(n_persons, n_items)`` and true_classes has shape
            ``(n_persons,)``.

        Notes
        -----
        Each complete response pattern uses one shared class, matching the
        model likelihood. A fixed seed produces identical results for every
        valid chunk size.
        """
        theta_values = self._ensure_theta_2d(theta).ravel()
        discrimination, difficulty, guessing = self._stack_class_parameters()
        n_persons = theta_values.size
        if chunk_size is None:
            working_values_per_person = 5 * self.n_items
            chunk_size = max(
                1,
                min(
                    n_persons,
                    _MIXTURE_MAX_SIMULATION_VALUES // working_values_per_person,
                ),
            )
        elif isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
            chunk_size, (int, np.integer)
        ):
            raise ValueError("chunk_size must be a positive integer")
        elif chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        rng = np.random.default_rng(seed)
        true_classes = rng.choice(
            self._n_classes,
            size=n_persons,
            p=self._parameters["class_proportions"],
        )
        responses = np.empty((n_persons, self.n_items), dtype=np.int32)
        for start in range(0, n_persons, int(chunk_size)):
            stop = min(start + int(chunk_size), n_persons)
            selected_classes = true_classes[start:stop]
            selected_discrimination = discrimination[selected_classes]
            selected_difficulty = difficulty[selected_classes]
            selected_guessing = guessing[selected_classes]
            logistic = sigmoid(
                selected_discrimination
                * (theta_values[start:stop, None] - selected_difficulty)
            )
            probabilities = selected_guessing + (1.0 - selected_guessing) * logistic
            responses[start:stop] = rng.random(probabilities.shape) < probabilities
        return responses, true_classes

    def _validate_responses(self, responses: NDArray[np.int_]) -> NDArray[np.int_]:
        values = validate_responses(responses, n_items=self.n_items)
        observed = values >= 0
        if np.any(values[observed] > 1):
            raise MirtDataError(
                "MixtureIRT requires binary responses coded 0 or 1",
                n_persons=values.shape[0],
                n_items=values.shape[1],
            )
        return values

    @staticmethod
    def _response_components(
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        observed = responses >= 0
        correct = np.where(observed, responses, 0).astype(np.float64, copy=False)
        incorrect = observed.astype(np.float64) - correct
        return correct, incorrect

    def _class_log_curves(
        self,
        theta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return class-specific log probabilities for both responses."""
        curves, _ = self._class_curves(theta)
        np.clip(
            curves,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
            out=curves,
        )
        log_incorrect = np.log1p(-curves)
        np.log(curves, out=curves)
        return curves, log_incorrect

    def _paired_class_log_joint_chunk(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one aligned chunk after input validation."""
        correct, incorrect = self._response_components(responses)
        log_correct, log_incorrect = self._class_log_curves(theta)
        class_log_likelihood = np.einsum(
            "pj,pkj->pk", correct, log_correct, optimize=True
        )
        class_log_likelihood += np.einsum(
            "pj,pkj->pk", incorrect, log_incorrect, optimize=True
        )
        return class_log_likelihood + self._log_class_proportions()[None, :]

    def _automatic_paired_chunk_size(self, n_output: int) -> int:
        working_values_per_person = max(
            1,
            2 * self._n_classes * self.n_items + 2 * self.n_items,
        )
        return max(
            1,
            min(
                n_output,
                _MIXTURE_MAX_LIKELIHOOD_VALUES // working_values_per_person,
            ),
        )

    def _paired_class_log_joint(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        values = self._validate_responses(responses)
        theta_values = self._ensure_theta_2d(theta)
        n_responses = values.shape[0]
        n_theta = theta_values.shape[0]
        if n_responses != n_theta and n_responses != 1 and n_theta != 1:
            raise MirtDataError(
                "responses and theta must have equal row counts or one row to broadcast",
                n_persons=n_responses,
                n_theta=n_theta,
            )
        n_output = max(n_responses, n_theta)
        resolved_chunk_size = _validate_chunk_size(chunk_size, "chunk_size")
        if resolved_chunk_size is None:
            resolved_chunk_size = self._automatic_paired_chunk_size(n_output)

        result = np.empty((n_output, self._n_classes), dtype=np.float64)
        for start in range(0, n_output, resolved_chunk_size):
            stop = min(start + resolved_chunk_size, n_output)
            if n_responses == 1:
                response_chunk = np.broadcast_to(
                    values,
                    (stop - start, self.n_items),
                )
            else:
                response_chunk = values[start:stop]
            if n_theta == 1:
                theta_chunk = np.broadcast_to(theta_values, (stop - start, 1))
            else:
                theta_chunk = theta_values[start:stop]
            result[start:stop] = self._paired_class_log_joint_chunk(
                response_chunk,
                theta_chunk,
            )
        return result

    def _log_class_proportions(self) -> NDArray[np.float64]:
        proportions = self._parameters["class_proportions"]
        log_proportions = np.full(self._n_classes, -np.inf, dtype=np.float64)
        np.log(proportions, out=log_proportions, where=proportions > 0.0)
        return log_proportions

    def log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response-pattern likelihoods with one shared class per row.

        Parameters
        ----------
        responses : ndarray
            Binary response matrix. Negative values are treated as missing.
        theta : ndarray
            Aligned ability values, or one value to broadcast to every row.
        chunk_size : int, optional
            Maximum aligned rows evaluated at once. By default, the chunk is
            selected automatically from the item and class counts.
        """
        return logsumexp(
            self._paired_class_log_joint(
                responses,
                theta,
                chunk_size=chunk_size,
            ),
            axis=1,
        )

    def _automatic_grid_chunk_sizes(
        self,
        n_patterns: int,
        n_theta: int,
        pattern_chunk_size: int | None,
        theta_chunk_size: int | None,
    ) -> tuple[int, int]:
        if theta_chunk_size is None:
            curve_values_per_theta = max(1, 2 * self._n_classes * self.n_items)
            theta_chunk_size = max(
                1,
                min(
                    n_theta,
                    _MIXTURE_MAX_LIKELIHOOD_VALUES // curve_values_per_theta,
                ),
            )
        if pattern_chunk_size is None:
            working_values_per_pattern = max(
                1,
                2 * self.n_items,
                2 * self._n_classes * theta_chunk_size,
            )
            pattern_chunk_size = max(
                1,
                min(
                    n_patterns,
                    _MIXTURE_MAX_LIKELIHOOD_VALUES // working_values_per_pattern,
                ),
            )
        return pattern_chunk_size, theta_chunk_size

    def _grid_log_likelihood_chunk(
        self,
        responses: NDArray[np.int_],
        log_correct: NDArray[np.float64],
        log_incorrect: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate one response-pattern by ability block."""
        correct, incorrect = self._response_components(responses)
        class_log_likelihood = np.einsum(
            "pj,qkj->pkq", correct, log_correct, optimize=True
        )
        class_log_likelihood += np.einsum(
            "pj,qkj->pkq", incorrect, log_incorrect, optimize=True
        )
        class_log_likelihood += self._log_class_proportions()[None, :, None]
        return logsumexp(class_log_likelihood, axis=1)

    def log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        pattern_chunk_size: int | None = None,
        theta_chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate every response pattern at every supplied ability point.

        ``pattern_chunk_size`` and ``theta_chunk_size`` bound the two working
        dimensions independently. Omitted limits are selected automatically.
        """
        values = self._validate_responses(responses)
        theta_values = self._ensure_theta_2d(theta)
        pattern_chunk_size = _validate_chunk_size(
            pattern_chunk_size,
            "pattern_chunk_size",
        )
        theta_chunk_size = _validate_chunk_size(
            theta_chunk_size,
            "theta_chunk_size",
        )
        pattern_chunk_size, theta_chunk_size = self._automatic_grid_chunk_sizes(
            values.shape[0],
            theta_values.shape[0],
            pattern_chunk_size,
            theta_chunk_size,
        )

        result = np.empty(
            (values.shape[0], theta_values.shape[0]),
            dtype=np.float64,
        )
        for theta_start in range(0, theta_values.shape[0], theta_chunk_size):
            theta_stop = min(theta_start + theta_chunk_size, theta_values.shape[0])
            log_correct, log_incorrect = self._class_log_curves(
                theta_values[theta_start:theta_stop]
            )
            for pattern_start in range(0, values.shape[0], pattern_chunk_size):
                pattern_stop = min(
                    pattern_start + pattern_chunk_size,
                    values.shape[0],
                )
                result[
                    pattern_start:pattern_stop,
                    theta_start:theta_stop,
                ] = self._grid_log_likelihood_chunk(
                    values[pattern_start:pattern_stop],
                    log_correct,
                    log_incorrect,
                )
        return result

    def class_posterior(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute posterior class probabilities conditional on ability.

        ``chunk_size`` bounds the aligned rows evaluated at once. An automatic
        memory-bounded limit is used when it is omitted.
        """
        log_joint = self._paired_class_log_joint(
            responses,
            theta,
            chunk_size=chunk_size,
        )
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def classify_persons(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        *,
        chunk_size: int | None = None,
    ) -> NDArray[np.int_]:
        """Assign each response pattern to its maximum-posterior class.

        Classification uses the class log joint directly because posterior
        normalization cannot change its maximum.
        """
        return np.argmax(
            self._paired_class_log_joint(
                responses,
                theta,
                chunk_size=chunk_size,
            ),
            axis=1,
        )

    def information(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute Fisher information from the marginal curve derivative."""
        curves, derivatives = self._class_curves(theta, item_idx)
        proportions = self._parameters["class_proportions"]
        probability = np.einsum("k,tk...->t...", proportions, curves, optimize=True)
        derivative = np.einsum("k,tk...->t...", proportions, derivatives, optimize=True)
        denominator = np.clip(probability * (1.0 - probability), PROB_EPSILON, None)
        return derivative**2 / denominator

    def copy(self) -> Self:
        """Create a deep copy of the model and its fit diagnostics."""
        new_model = MixtureIRT(
            n_items=self.n_items,
            n_classes=self._n_classes,
            base_model=self._base_model,
            item_names=self.item_names.copy(),
        )
        new_model._parameters = {
            name: values.copy() for name, values in self._parameters.items()
        }
        new_model._is_fitted = self._is_fitted
        if self._convergence_info is not None:
            new_model._convergence_info = self.convergence_info
        return new_model


def fit_mixture_irt(
    responses: NDArray[np.int_],
    n_classes: int = 2,
    base_model: Literal["1PL", "2PL", "3PL"] = "2PL",
    max_iter: int = 100,
    tol: float = 1e-4,
    n_quadpts: int = 21,
    verbose: bool = False,
) -> tuple[MixtureIRT, NDArray[np.float64]]:
    """Fit a finite-mixture IRT model by marginal maximum-likelihood EM.

    The E-step computes the joint posterior over respondent class and
    quadrature point in log space. The M-step updates class proportions and
    every free item parameter with aggregated expected response counts.
    """
    _validate_fit_options(max_iter, tol, n_quadpts)
    raw_responses = np.asarray(responses)
    if raw_responses.ndim != 2:
        raise MirtDataError(f"responses must be 2D array, got {raw_responses.ndim}D")
    n_items = raw_responses.shape[1] if raw_responses.ndim == 2 else 0
    model = MixtureIRT(
        n_items=n_items,
        n_classes=n_classes,
        base_model=base_model,
    )
    values = model._validate_responses(raw_responses)

    nodes, quadrature_weights = roots_hermite(n_quadpts)
    nodes = nodes * np.sqrt(2.0)
    quadrature_weights = quadrature_weights / np.sqrt(np.pi)
    log_quadrature_weights = np.log(quadrature_weights)

    joint_posterior, class_posteriors, current_ll = _mixture_e_step(
        model, values, nodes, log_quadrature_weights
    )
    history = [current_ll]
    converged = False
    n_iterations = 0

    for iteration in range(1, max_iter + 1):
        _mixture_m_step(model, values, nodes, joint_posterior)
        next_joint, next_class, next_ll = _mixture_e_step(
            model, values, nodes, log_quadrature_weights
        )
        history.append(next_ll)
        improvement = next_ll - current_ll
        scale = 1.0 + abs(current_ll)
        if improvement < -1e-7 * scale:
            raise MirtEstimationError(
                "Mixture EM decreased the marginal log likelihood",
                iteration=iteration,
                log_likelihood=next_ll,
                previous_log_likelihood=current_ll,
            )

        joint_posterior = next_joint
        class_posteriors = next_class
        current_ll = next_ll
        n_iterations = iteration
        if verbose:
            print(
                f"Iteration {iteration}: LL = {current_ll:.6f}, "
                f"change = {improvement:.6g}"
            )
        if abs(improvement) <= tol * scale:
            converged = True
            break

    model._is_fitted = True
    model._convergence_info = {
        "converged": converged,
        "n_iterations": n_iterations,
        "log_likelihood": current_ll,
        "log_likelihood_history": np.asarray(history, dtype=np.float64),
    }
    return model, class_posteriors


def _validate_fit_options(max_iter: int, tol: float, n_quadpts: int) -> None:
    """Validate EM controls before constructing work arrays."""
    for name, value, minimum in (
        ("max_iter", max_iter, 1),
        ("n_quadpts", n_quadpts, 3),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise MirtValidationError(
                f"{name} must be an integer", parameter=name, value=value
            )
        if value < minimum:
            raise MirtValidationError(
                f"{name} must be at least {minimum}",
                parameter=name,
                value=value,
                expected=f">= {minimum}",
            )
    if not np.isfinite(tol) or tol <= 0.0:
        raise MirtValidationError(
            "tol must be finite and positive",
            parameter="tol",
            value=tol,
            expected="> 0",
        )


def _mixture_e_step(
    model: MixtureIRT,
    responses: NDArray[np.int_],
    nodes: NDArray[np.float64],
    log_quadrature_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute stable joint class-and-ability posterior weights."""
    correct, incorrect = model._response_components(responses)
    curves, _ = model._class_curves(nodes)
    curves = np.clip(curves, PROB_EPSILON, 1.0 - PROB_EPSILON)
    class_log_likelihood = np.einsum(
        "pj,qkj->pkq", correct, np.log(curves), optimize=True
    )
    class_log_likelihood += np.einsum(
        "pj,qkj->pkq", incorrect, np.log1p(-curves), optimize=True
    )
    log_joint = (
        class_log_likelihood
        + model._log_class_proportions()[None, :, None]
        + log_quadrature_weights[None, None, :]
    )
    log_normalizer = logsumexp(log_joint, axis=(1, 2))
    if not np.all(np.isfinite(log_normalizer)):
        raise MirtEstimationError("Mixture E-step produced a non-finite likelihood")
    joint_posterior = np.exp(log_joint - log_normalizer[:, None, None])
    class_posteriors = joint_posterior.sum(axis=2)
    return joint_posterior, class_posteriors, float(np.sum(log_normalizer))


def _mixture_m_step(
    model: MixtureIRT,
    responses: NDArray[np.int_],
    nodes: NDArray[np.float64],
    joint_posterior: NDArray[np.float64],
) -> None:
    """Update proportions and all free class-specific item parameters."""
    class_proportions = joint_posterior.sum(axis=(0, 2))
    class_proportions = np.maximum(class_proportions, PROB_EPSILON)
    class_proportions /= class_proportions.sum()
    model._parameters["class_proportions"] = class_proportions

    correct, _ = model._response_components(responses)
    observed = (responses >= 0).astype(np.float64)
    for class_idx in range(model.n_classes):
        posterior = joint_posterior[:, class_idx, :]
        expected_correct = posterior.T @ correct
        expected_total = posterior.T @ observed
        for item_idx in range(model.n_items):
            _update_mixture_item(
                model,
                class_idx,
                item_idx,
                nodes,
                expected_correct[:, item_idx],
                expected_total[:, item_idx],
            )


def _update_mixture_item(
    model: MixtureIRT,
    class_idx: int,
    item_idx: int,
    nodes: NDArray[np.float64],
    expected_correct: NDArray[np.float64],
    expected_total: NDArray[np.float64],
) -> None:
    """Maximize one item's expected complete-data log likelihood."""
    if float(np.sum(expected_total)) <= PROB_EPSILON:
        return
    expected_correct = np.clip(expected_correct, 0.0, expected_total)
    difficulty = float(model._parameters[f"difficulty_class{class_idx}"][item_idx])
    if model.base_model == "1PL":
        discrimination = 1.0
    else:
        discrimination = float(
            model._parameters[f"discrimination_class{class_idx}"][item_idx]
        )
    if model.base_model == "3PL":
        guessing = float(model._parameters[f"guessing_class{class_idx}"][item_idx])
    else:
        guessing = 0.0

    if model.base_model == "1PL":
        initial = np.array([difficulty], dtype=np.float64)
        bounds = [_DIFFICULTY_BOUNDS]
    elif model.base_model == "2PL":
        initial = np.array([np.log(discrimination), difficulty], dtype=np.float64)
        bounds = [_LOG_DISCRIMINATION_BOUNDS, _DIFFICULTY_BOUNDS]
    else:
        initial = np.array(
            [np.log(discrimination), difficulty, guessing], dtype=np.float64
        )
        bounds = [
            _LOG_DISCRIMINATION_BOUNDS,
            _DIFFICULTY_BOUNDS,
            _GUESSING_BOUNDS,
        ]

    initial_value, _ = _item_expected_objective(
        initial, model.base_model, nodes, expected_correct, expected_total
    )
    result = optimize.minimize(
        _item_expected_objective,
        initial,
        args=(model.base_model, nodes, expected_correct, expected_total),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-11, "gtol": 1e-7},
    )
    if (
        not np.all(np.isfinite(result.x))
        or not np.isfinite(result.fun)
        or float(result.fun) > initial_value + 1e-8
    ):
        return

    if model.base_model == "1PL":
        model._parameters[f"difficulty_class{class_idx}"][item_idx] = result.x[0]
    else:
        model._parameters[f"discrimination_class{class_idx}"][item_idx] = np.exp(
            result.x[0]
        )
        model._parameters[f"difficulty_class{class_idx}"][item_idx] = result.x[1]
        if model.base_model == "3PL":
            model._parameters[f"guessing_class{class_idx}"][item_idx] = result.x[2]


def _item_expected_objective(
    values: NDArray[np.float64],
    base_model: str,
    nodes: NDArray[np.float64],
    expected_correct: NDArray[np.float64],
    expected_total: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return an item-level negative expected log likelihood and gradient."""
    if base_model == "1PL":
        discrimination = 1.0
        difficulty = float(values[0])
    else:
        discrimination = float(np.exp(values[0]))
        difficulty = float(values[1])
    z = discrimination * (nodes - difficulty)
    logistic = sigmoid(z)

    if base_model != "3PL":
        incorrect = expected_total - expected_correct
        objective = float(
            np.sum(
                expected_correct * np.logaddexp(0.0, -z)
                + incorrect * np.logaddexp(0.0, z)
            )
        )
        derivative_z = expected_total * logistic - expected_correct
    else:
        guessing = float(values[2])
        probability = guessing + (1.0 - guessing) * logistic
        probability = np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)
        incorrect = expected_total - expected_correct
        objective = float(
            -np.sum(
                expected_correct * np.log(probability)
                + incorrect * np.log1p(-probability)
            )
        )
        derivative_probability = (expected_total * probability - expected_correct) / (
            probability * (1.0 - probability)
        )
        derivative_z = (
            derivative_probability * (1.0 - guessing) * logistic * (1.0 - logistic)
        )

    gradient_difficulty = float(np.sum(-discrimination * derivative_z))
    if base_model == "1PL":
        return objective, np.array([gradient_difficulty], dtype=np.float64)

    gradient_log_discrimination = float(np.sum(z * derivative_z))
    if base_model == "2PL":
        return objective, np.array(
            [gradient_log_discrimination, gradient_difficulty],
            dtype=np.float64,
        )

    gradient_guessing = float(np.sum(derivative_probability * (1.0 - logistic)))
    return objective, np.array(
        [
            gradient_log_discrimination,
            gradient_difficulty,
            gradient_guessing,
        ],
        dtype=np.float64,
    )
