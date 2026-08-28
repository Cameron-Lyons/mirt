"""State-space IRT for continuous latent trait evolution."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.utils.numeric import standard_normal_quadrature

_STATE_SPACE_MAX_PROBABILITY_VALUES = 1_000_000
_STANDARD_NORMAL = NormalDist()


def _state_interval(
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    confidence: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a validated central Gaussian interval for state moments."""
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(
        confidence,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("confidence must be a finite value in (0, 1)")
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be a finite value in (0, 1)")
    critical_value = _STANDARD_NORMAL.inv_cdf((1.0 + confidence_value) / 2.0)
    radius = critical_value * np.sqrt(variances)
    return means - radius, means + radius


@dataclass(frozen=True, slots=True)
class StateSpaceStepResult:
    """Prediction and state update from one response occasion.

    The response log likelihood is joint over observed items. Item log
    likelihoods and residuals use the marginal item probabilities and contain
    ``numpy.nan`` at missing responses.
    """

    response_probabilities: NDArray[np.float64]
    response_log_likelihood: float
    item_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    updated_mean: float
    updated_variance: float
    next_mean: float
    next_variance: float


@dataclass(frozen=True, slots=True)
class StateSpaceBatchStepResult:
    """Vectorized predictions and state updates from one occasion.

    Item log likelihoods and residuals have shape
    ``(n_persons, n_items)`` and contain ``numpy.nan`` at missing responses.
    """

    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    item_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    updated_means: NDArray[np.float64]
    updated_variances: NDArray[np.float64]
    next_means: NDArray[np.float64]
    next_variances: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of state distributions represented by the result."""
        return int(self.updated_means.size)


@dataclass(frozen=True, slots=True)
class StateSpacePredictiveResult:
    """Causal predictions, diagnostics, and states for one response history.

    Predicted state moments condition only on earlier occasions. Filtered
    moments additionally condition on responses from the corresponding
    occasion. Item diagnostics contain ``numpy.nan`` at missing responses.
    """

    predicted_means: NDArray[np.float64]
    predicted_variances: NDArray[np.float64]
    filtered_means: NDArray[np.float64]
    filtered_variances: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    item_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]

    @property
    def total_log_likelihood(self) -> float:
        """Sum of joint predictive log likelihoods across occasions."""
        return float(np.sum(self.response_log_likelihoods))


@dataclass(frozen=True, slots=True)
class StateSpaceBatchPredictiveResult:
    """Causal predictions, diagnostics, and states for multiple histories.

    Predicted state moments condition only on earlier occasions. Filtered
    moments additionally condition on the corresponding responses. Item
    diagnostics contain ``numpy.nan`` at missing responses.
    """

    predicted_means: NDArray[np.float64]
    predicted_variances: NDArray[np.float64]
    filtered_means: NDArray[np.float64]
    filtered_variances: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    item_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of response histories represented by the result."""
        return int(self.predicted_means.shape[0])

    @property
    def total_log_likelihoods(self) -> NDArray[np.float64]:
        """Joint predictive log-likelihood totals for each person."""
        return np.sum(self.response_log_likelihoods, axis=1)


@dataclass(frozen=True, slots=True)
class StateSpaceForecastResult:
    """Latent-state and response forecasts for one person."""

    state_means: NDArray[np.float64]
    state_variances: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]

    @property
    def n_steps(self) -> int:
        """Number of future occasions represented by the result."""
        return int(self.state_means.size)

    @property
    def state_standard_deviations(self) -> NDArray[np.float64]:
        """Standard deviations of the Gaussian state forecasts."""
        return np.sqrt(self.state_variances)

    def state_interval(
        self,
        confidence: float = 0.95,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a central Gaussian interval for every forecast state."""
        return _state_interval(self.state_means, self.state_variances, confidence)


@dataclass(frozen=True, slots=True)
class StateSpaceBatchForecastResult:
    """Latent-state and response forecasts for multiple people."""

    state_means: NDArray[np.float64]
    state_variances: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of people represented by the result."""
        return int(self.state_means.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of future occasions represented by the result."""
        return int(self.state_means.shape[1])

    @property
    def state_standard_deviations(self) -> NDArray[np.float64]:
        """Standard deviations of the Gaussian state forecasts."""
        return np.sqrt(self.state_variances)

    def state_interval(
        self,
        confidence: float = 0.95,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return central Gaussian intervals for every forecast state."""
        return _state_interval(self.state_means, self.state_variances, confidence)


@dataclass
class StateSpaceIRT:
    """State-space formulation for continuous latent trait evolution.

    State equation: θ_t = A·θ_{t-1} + w_t, w_t ~ N(0, Q)
    Observation: P(X_t = 1 | θ_t) = IRT model

    Parameters
    ----------
    n_items : int
        Number of items per time point
    n_timepoints : int
        Number of time points
    transition_matrix : NDArray, optional
        State transition matrix A (default: identity = random walk)
    process_noise : NDArray, optional
        Process noise covariance Q
    observation_noise : float
        Additional non-negative response-scale variance used by the
        linearized observation update
    base_model : str
        IRT model for observations, either ``"2PL"`` or ``"3PL"``
    discrimination : NDArray, optional
        Positive item discrimination parameters
    difficulty : NDArray, optional
        Item difficulty parameters
    guessing : NDArray, optional
        Item guessing parameters for a 3PL observation model
    initial_mean : float
        Mean of the initial latent-state distribution
    initial_var : float
        Positive variance of the initial latent-state distribution
    """

    n_items: int
    n_timepoints: int
    transition_matrix: NDArray[np.float64] | None = None
    process_noise: NDArray[np.float64] | None = None
    observation_noise: float = 0.0
    base_model: Literal["2PL", "3PL"] = "2PL"

    discrimination: NDArray[np.float64] | None = None
    difficulty: NDArray[np.float64] | None = None
    guessing: NDArray[np.float64] | None = None

    initial_mean: float = 0.0
    initial_var: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_items, bool)
            or not isinstance(self.n_items, (int, np.integer))
            or self.n_items < 1
        ):
            raise ValueError("n_items must be a positive integer")
        if (
            isinstance(self.n_timepoints, bool)
            or not isinstance(self.n_timepoints, (int, np.integer))
            or self.n_timepoints < 1
        ):
            raise ValueError("n_timepoints must be a positive integer")
        self.n_items = int(self.n_items)
        self.n_timepoints = int(self.n_timepoints)
        if self.base_model not in ("2PL", "3PL"):
            raise ValueError("base_model must be '2PL' or '3PL'")

        if self.transition_matrix is None:
            self.transition_matrix = np.array([[1.0]])
        else:
            self.transition_matrix = np.asarray(
                self.transition_matrix,
                dtype=np.float64,
            ).copy()
        if self.transition_matrix.shape != (1, 1) or not np.all(
            np.isfinite(self.transition_matrix)
        ):
            raise ValueError("transition_matrix must be a finite 1x1 matrix")

        if self.process_noise is None:
            self.process_noise = np.array([[0.1]])
        else:
            self.process_noise = np.asarray(
                self.process_noise,
                dtype=np.float64,
            ).copy()
        if (
            self.process_noise.shape != (1, 1)
            or not np.all(np.isfinite(self.process_noise))
            or self.process_noise[0, 0] < 0.0
        ):
            raise ValueError("process_noise must be a finite non-negative 1x1 matrix")
        if (
            self.n_timepoints > 1
            and self.transition_matrix[0, 0] == 0.0
            and self.process_noise[0, 0] == 0.0
        ):
            raise ValueError(
                "transition_matrix and process_noise cannot both have zero variance "
                "propagation"
            )

        if self.discrimination is None:
            self.discrimination = np.ones(self.n_items)
        else:
            self.discrimination = np.asarray(
                self.discrimination,
                dtype=np.float64,
            ).copy()
        if (
            self.discrimination.shape != (self.n_items,)
            or not np.all(np.isfinite(self.discrimination))
            or np.any(self.discrimination <= 0.0)
        ):
            raise ValueError(
                f"discrimination must contain {self.n_items} finite positive values"
            )

        if self.difficulty is None:
            self.difficulty = np.zeros(self.n_items)
        else:
            self.difficulty = np.asarray(
                self.difficulty,
                dtype=np.float64,
            ).copy()
        if self.difficulty.shape != (self.n_items,) or not np.all(
            np.isfinite(self.difficulty)
        ):
            raise ValueError(f"difficulty must contain {self.n_items} finite values")

        if self.base_model == "3PL":
            if self.guessing is None:
                self.guessing = np.full(self.n_items, 0.2)
            else:
                self.guessing = np.asarray(
                    self.guessing,
                    dtype=np.float64,
                ).copy()
            if (
                self.guessing.shape != (self.n_items,)
                or not np.all(np.isfinite(self.guessing))
                or np.any((self.guessing < 0.0) | (self.guessing >= 1.0))
            ):
                raise ValueError(
                    f"guessing must contain {self.n_items} finite values in [0, 1)"
                )
        elif self.guessing is not None:
            raise ValueError("guessing is only supported for base_model='3PL'")

        for parameter_name in ("observation_noise", "initial_mean", "initial_var"):
            if isinstance(getattr(self, parameter_name), bool):
                raise ValueError(f"{parameter_name} must be finite")
            try:
                value = float(getattr(self, parameter_name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{parameter_name} must be finite") from exc
            if not np.isfinite(value):
                raise ValueError(f"{parameter_name} must be finite")
            setattr(self, parameter_name, value)
        if self.observation_noise < 0.0:
            raise ValueError("observation_noise must be non-negative")
        if self.initial_var <= 0.0:
            raise ValueError("initial_var must be positive")

    def _validated_filter_responses(
        self,
        responses: NDArray[np.int_],
        *,
        batch: bool,
    ) -> NDArray[np.int_]:
        """Validate and normalize state-space response arrays."""
        response_values = np.asarray(responses)
        expected_shape = (
            (self.n_timepoints, self.n_items)
            if not batch
            else (None, self.n_timepoints, self.n_items)
        )
        valid_shape = (
            response_values.shape == expected_shape
            if not batch
            else response_values.ndim == 3
            and response_values.shape[0] > 0
            and response_values.shape[1:] == expected_shape[1:]
        )
        if not valid_shape:
            if batch:
                raise ValueError(
                    "responses must have shape "
                    f"(n_persons, {self.n_timepoints}, {self.n_items})"
                )
            raise ValueError(
                f"responses must have shape ({self.n_timepoints}, {self.n_items})"
            )
        if not np.issubdtype(response_values.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if not np.all(np.isin(response_values, (-1, 0, 1))):
            raise ValueError("responses must contain only -1, 0, or 1")
        return response_values.astype(np.int32, copy=False)

    def _validated_update_responses(
        self,
        responses: NDArray[np.int_],
        *,
        batch: bool,
    ) -> NDArray[np.int_]:
        """Validate responses from one modeled occasion."""
        response_values = np.asarray(responses)
        valid_shape = (
            response_values.ndim == 2
            and response_values.shape[0] > 0
            and response_values.shape[1] == self.n_items
            if batch
            else response_values.shape == (self.n_items,)
        )
        if not valid_shape:
            expected_shape = (
                f"(n_persons, {self.n_items})" if batch else f"({self.n_items},)"
            )
            raise ValueError(f"responses must have shape {expected_shape}")
        if not np.issubdtype(response_values.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if not np.all(np.isin(response_values, (-1, 0, 1))):
            raise ValueError("responses must contain only -1, 0, or 1")
        return response_values.astype(np.int32, copy=False)

    @staticmethod
    def _validated_state_vector(
        values: float | NDArray[np.float64] | None,
        n_persons: int,
        name: str,
        *,
        default: float,
        constraint: Literal["finite", "nonnegative", "positive"] = "finite",
    ) -> NDArray[np.float64]:
        """Return finite state values broadcast across a batch."""
        raw_values = np.asarray(default if values is None else values)
        if (
            np.issubdtype(raw_values.dtype, np.bool_)
            or np.issubdtype(raw_values.dtype, np.complexfloating)
            or not np.issubdtype(raw_values.dtype, np.number)
        ):
            raise ValueError(f"{name} must contain {constraint} values")
        if raw_values.ndim == 0:
            state_values = np.full(
                n_persons,
                raw_values.item(),
                dtype=np.float64,
            )
        elif raw_values.shape == (n_persons,):
            state_values = raw_values.astype(np.float64, copy=True)
        else:
            raise ValueError(f"{name} must be scalar or have shape ({n_persons},)")
        violates_constraint = (
            constraint == "positive" and np.any(state_values <= 0.0)
        ) or (constraint == "nonnegative" and np.any(state_values < 0.0))
        if not np.all(np.isfinite(state_values)) or violates_constraint:
            raise ValueError(f"{name} must contain {constraint} values")
        return state_values

    def _validated_state_moments_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate a non-empty batch of Gaussian state moments."""
        raw_means = np.asarray(state_means)
        if raw_means.ndim != 1 or raw_means.size < 1:
            raise ValueError("state_means must have shape (n_persons,)")
        n_persons = raw_means.size
        mean_values = self._validated_state_vector(
            raw_means,
            n_persons,
            "state_means",
            default=self.initial_mean,
        )
        variance_values = self._validated_state_vector(
            state_variances,
            n_persons,
            "state_variances",
            default=self.initial_var,
            constraint="nonnegative",
        )
        return mean_values, variance_values

    @staticmethod
    def _validated_positive_integer(value: int, name: str) -> int:
        """Return a validated positive integer parameter."""
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _base_observation_probability(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return logistic response probabilities before 3PL scaling."""
        logits = self.discrimination[None, :] * (
            theta[:, None] - self.difficulty[None, :]
        )
        return np.asarray(sigmoid(logits), dtype=np.float64)

    def _observation_probability(
        self,
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return item response probabilities for one state per row."""
        base_probability = self._base_observation_probability(theta)
        if self.base_model == "3PL":
            probability = (
                self.guessing[None, :]
                + (1.0 - self.guessing[None, :]) * base_probability
            )
        else:
            probability = base_probability
        return np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON)

    def _observation_probability_and_derivative(
        self,
        theta: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return response probabilities and derivatives by person and item."""
        base_probability = self._base_observation_probability(theta)
        if self.base_model == "3PL":
            guessing_scale = 1.0 - self.guessing[None, :]
            probability = self.guessing[None, :] + guessing_scale * base_probability
            derivative = (
                self.discrimination[None, :]
                * guessing_scale
                * base_probability
                * (1.0 - base_probability)
            )
        else:
            probability = base_probability
            derivative = (
                self.discrimination[None, :]
                * base_probability
                * (1.0 - base_probability)
            )
        return np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON), derivative

    def _extended_kalman_update_batch(
        self,
        responses: NDArray[np.int_],
        prior_means: NDArray[np.float64],
        prior_variances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update validated state priors from one response occasion."""
        observed = responses >= 0
        has_observations = np.any(observed, axis=1)
        updated_means = prior_means.copy()
        updated_variances = prior_variances.copy()

        if not np.any(has_observations):
            return updated_means, updated_variances

        candidate_means = prior_means.copy()
        for _ in range(5):
            probability, derivative = self._observation_probability_and_derivative(
                candidate_means
            )
            response_variance = (
                probability * (1.0 - probability) + self.observation_noise
            )
            score = (
                np.sum(
                    np.where(
                        observed,
                        derivative * (responses - probability) / response_variance,
                        0.0,
                    ),
                    axis=1,
                )
                - (candidate_means - prior_means) / prior_variances
            )
            information = 1.0 / prior_variances + np.sum(
                np.where(
                    observed,
                    derivative**2 / response_variance,
                    0.0,
                ),
                axis=1,
            )
            candidate_means[has_observations] += (
                score[has_observations] / information[has_observations]
            )

        probability, derivative = self._observation_probability_and_derivative(
            candidate_means
        )
        response_variance = probability * (1.0 - probability) + self.observation_noise
        final_information = 1.0 / prior_variances + np.sum(
            np.where(
                observed,
                derivative**2 / response_variance,
                0.0,
            ),
            axis=1,
        )
        updated_means[has_observations] = candidate_means[has_observations]
        updated_variances[has_observations] = 1.0 / final_information[has_observations]
        return updated_means, updated_variances

    def extended_kalman_update(
        self,
        responses: NDArray[np.int_],
        *,
        prior_mean: float | None = None,
        prior_variance: float | None = None,
    ) -> tuple[float, float]:
        """Update one state prior from one response occasion.

        Omitted prior moments use the model's initial state distribution.
        Use :meth:`propagate_state` to obtain the next occasion's prior.

        Parameters
        ----------
        responses : NDArray
            Integer response vector with shape ``(n_items,)``. Use ``-1`` for
            missing item responses.
        prior_mean : float, optional
            Finite predicted state mean for this occasion.
        prior_variance : float, optional
            Finite positive predicted state variance for this occasion.

        Returns
        -------
        tuple
            Updated state mean and variance.
        """
        response_values = self._validated_update_responses(responses, batch=False)
        prior_means = self._validated_state_vector(
            prior_mean,
            1,
            "prior_mean",
            default=self.initial_mean,
        )
        prior_variances = self._validated_state_vector(
            prior_variance,
            1,
            "prior_variance",
            default=self.initial_var,
            constraint="positive",
        )
        updated_means, updated_variances = self._extended_kalman_update_batch(
            response_values[None, :],
            prior_means,
            prior_variances,
        )
        return float(updated_means[0]), float(updated_variances[0])

    def extended_kalman_update_batch(
        self,
        responses: NDArray[np.int_],
        *,
        prior_means: float | NDArray[np.float64] | None = None,
        prior_variances: float | NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Update multiple state priors from one response occasion.

        Scalar prior moments are broadcast across people. Omitted moments use
        the model's initial state distribution.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_items)``. Use
            ``-1`` for missing item responses.
        prior_means : float or NDArray, optional
            Finite predicted state means, either scalar or shape
            ``(n_persons,)``.
        prior_variances : float or NDArray, optional
            Finite positive predicted state variances, either scalar or shape
            ``(n_persons,)``.

        Returns
        -------
        tuple
            Updated means and variances, each with shape ``(n_persons,)``.
        """
        response_values = self._validated_update_responses(responses, batch=True)
        n_persons = response_values.shape[0]
        mean_values = self._validated_state_vector(
            prior_means,
            n_persons,
            "prior_means",
            default=self.initial_mean,
        )
        variance_values = self._validated_state_vector(
            prior_variances,
            n_persons,
            "prior_variances",
            default=self.initial_var,
            constraint="positive",
        )
        return self._extended_kalman_update_batch(
            response_values,
            mean_values,
            variance_values,
        )

    def _propagate_state_moments(
        self,
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate validated state moments without observation updates."""
        propagated_means = state_means.copy()
        propagated_variances = state_variances.copy()
        transition = float(self.transition_matrix[0, 0])
        transition_squared = transition**2
        process_variance = float(self.process_noise[0, 0])
        for _ in range(n_steps):
            propagated_means *= transition
            propagated_variances = (
                transition_squared * propagated_variances + process_variance
            )
        return propagated_means, propagated_variances

    def propagate_state(
        self,
        state_mean: float,
        state_variance: float,
        n_steps: int = 1,
    ) -> tuple[float, float]:
        """Propagate one state distribution forward without observations.

        Parameters
        ----------
        state_mean : float
            Finite mean of the current state distribution.
        state_variance : float
            Finite non-negative variance of the current state distribution.
        n_steps : int, default=1
            Number of state transitions to apply.

        Returns
        -------
        tuple
            Propagated state mean and variance.
        """
        n_steps = self._validated_positive_integer(n_steps, "n_steps")
        state_means = self._validated_state_vector(
            state_mean,
            1,
            "state_mean",
            default=self.initial_mean,
        )
        state_variances = self._validated_state_vector(
            state_variance,
            1,
            "state_variance",
            default=self.initial_var,
            constraint="nonnegative",
        )
        propagated_means, propagated_variances = self._propagate_state_moments(
            state_means,
            state_variances,
            n_steps,
        )
        return float(propagated_means[0]), float(propagated_variances[0])

    def propagate_state_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        n_steps: int = 1,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate multiple state distributions without observations.

        Parameters
        ----------
        state_means : NDArray
            Finite current state means with shape ``(n_persons,)``.
        state_variances : float or NDArray
            Finite non-negative current state variances, either scalar or shape
            ``(n_persons,)``.
        n_steps : int, default=1
            Number of state transitions to apply.

        Returns
        -------
        tuple
            Propagated means and variances, each with shape ``(n_persons,)``.
        """
        n_steps = self._validated_positive_integer(n_steps, "n_steps")
        mean_values, variance_values = self._validated_state_moments_batch(
            state_means,
            state_variances,
        )
        return self._propagate_state_moments(
            mean_values,
            variance_values,
            n_steps,
        )

    def extended_kalman_filter(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extended Kalman filter for a single person.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_timepoints, n_items)``.
            Use ``-1`` for missing item responses.

        Returns
        -------
        tuple
            Filtered means and variances, each with shape ``(n_timepoints,)``.
        """
        response_values = self._validated_filter_responses(responses, batch=False)
        filtered_means, filtered_variances = self.extended_kalman_filter_batch(
            response_values[None, :, :]
        )
        return filtered_means[0].copy(), filtered_variances[0].copy()

    def _filter_state_moments_batch(
        self,
        responses: NDArray[np.int_],
        *,
        predicted_means: NDArray[np.float64] | None = None,
        predicted_variances: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Filter validated histories, optionally recording state priors."""
        n_persons = responses.shape[0]
        shape = (n_persons, self.n_timepoints)
        filtered_means = np.empty(shape, dtype=np.float64)
        filtered_variances = np.empty(shape, dtype=np.float64)
        predicted_mean = np.full(n_persons, self.initial_mean, dtype=np.float64)
        predicted_variance = np.full(n_persons, self.initial_var, dtype=np.float64)

        for time_index in range(self.n_timepoints):
            if predicted_means is not None and predicted_variances is not None:
                predicted_means[:, time_index] = predicted_mean
                predicted_variances[:, time_index] = predicted_variance
            updated_mean, updated_variance = self._extended_kalman_update_batch(
                responses[:, time_index],
                predicted_mean,
                predicted_variance,
            )
            filtered_means[:, time_index] = updated_mean
            filtered_variances[:, time_index] = updated_variance
            if time_index < self.n_timepoints - 1:
                predicted_mean, predicted_variance = self._propagate_state_moments(
                    updated_mean,
                    updated_variance,
                    1,
                )

        return filtered_means, filtered_variances

    def _predictive_state_moments_batch(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return predicted and filtered moments for validated histories."""
        shape = (responses.shape[0], self.n_timepoints)
        predicted_means = np.empty(shape, dtype=np.float64)
        predicted_variances = np.empty(shape, dtype=np.float64)
        filtered_means, filtered_variances = self._filter_state_moments_batch(
            responses,
            predicted_means=predicted_means,
            predicted_variances=predicted_variances,
        )
        return (
            predicted_means,
            predicted_variances,
            filtered_means,
            filtered_variances,
        )

    def extended_kalman_filter_batch(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Filter multiple response histories in one vectorized pass.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.

        Returns
        -------
        tuple
            Filtered means and variances, each with shape
            ``(n_persons, n_timepoints)``.
        """
        response_values = self._validated_filter_responses(responses, batch=True)
        return self._filter_state_moments_batch(response_values)

    def extended_kalman_smoother(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Smooth one response history using evidence from every occasion.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_timepoints, n_items)``.
            Use ``-1`` for missing item responses.

        Returns
        -------
        tuple
            Smoothed means and variances, each with shape ``(n_timepoints,)``.
        """
        response_values = self._validated_filter_responses(responses, batch=False)
        smoothed_means, smoothed_variances = self.extended_kalman_smoother_batch(
            response_values[None, :, :]
        )
        return smoothed_means[0].copy(), smoothed_variances[0].copy()

    def extended_kalman_smoother_batch(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Smooth multiple response histories in one vectorized pass.

        The forward pass uses the extended Kalman filter. A
        Rauch--Tung--Striebel backward pass then conditions each state on all
        response occasions.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.

        Returns
        -------
        tuple
            Smoothed means and variances, each with shape
            ``(n_persons, n_timepoints)``.
        """
        filtered_means, filtered_variances = self.extended_kalman_filter_batch(
            responses
        )
        return self._smooth_filtered_batch(filtered_means, filtered_variances)

    def _smooth_filtered_batch(
        self,
        filtered_means: NDArray[np.float64],
        filtered_variances: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply a vectorized RTS pass to validated filter output."""
        smoothed_means = filtered_means.copy()
        smoothed_variances = filtered_variances.copy()
        transition = float(self.transition_matrix[0, 0])

        for time_index in range(self.n_timepoints - 2, -1, -1):
            predicted_mean, predicted_variance = self._propagate_state_moments(
                filtered_means[:, time_index],
                filtered_variances[:, time_index],
                1,
            )
            smoothing_gain = (
                filtered_variances[:, time_index] * transition / predicted_variance
            )
            smoothed_means[:, time_index] = filtered_means[:, time_index] + (
                smoothing_gain * (smoothed_means[:, time_index + 1] - predicted_mean)
            )
            smoothed_variances[:, time_index] = np.maximum(
                filtered_variances[:, time_index]
                + smoothing_gain**2
                * (smoothed_variances[:, time_index + 1] - predicted_variance),
                0.0,
            )

        return smoothed_means, smoothed_variances

    def _forecast_state_moments_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast from validated current state moments."""
        n_persons = state_means.size
        forecast_means = np.empty((n_persons, n_steps), dtype=np.float64)
        forecast_variances = np.empty_like(forecast_means)
        current_mean = state_means
        current_variance = state_variances

        for step_index in range(n_steps):
            current_mean, current_variance = self._propagate_state_moments(
                current_mean,
                current_variance,
                1,
            )
            forecast_means[:, step_index] = current_mean
            forecast_variances[:, step_index] = current_variance

        return forecast_means, forecast_variances

    def forecast(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast latent-state moments after one response history.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_timepoints, n_items)``.
            Use ``-1`` for missing item responses.
        n_steps : int
            Number of future occasions to forecast.

        Returns
        -------
        tuple
            Forecast means and variances, each with shape ``(n_steps,)``.
        """
        response_values = self._validated_filter_responses(responses, batch=False)
        forecast_means, forecast_variances = self.forecast_batch(
            response_values[None, :, :],
            n_steps,
        )
        return forecast_means[0].copy(), forecast_variances[0].copy()

    def forecast_batch(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast latent-state moments for multiple response histories.

        Forecasts begin one step after the last modeled response occasion.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.
        n_steps : int
            Number of future occasions to forecast.

        Returns
        -------
        tuple
            Forecast means and variances, each with shape
            ``(n_persons, n_steps)``.
        """
        n_steps = self._validated_positive_integer(n_steps, "n_steps")
        filtered_means, filtered_variances = self.extended_kalman_filter_batch(
            responses
        )
        return self._forecast_state_moments_batch(
            filtered_means[:, -1],
            filtered_variances[:, -1],
            n_steps,
        )

    def forecast_from_state(
        self,
        state_mean: float,
        state_variance: float,
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast latent-state moments from one current posterior state.

        Forecasts begin one transition after the supplied state, making this
        method suitable for moments retained from :meth:`online_step`.

        Parameters
        ----------
        state_mean : float
            Finite current posterior state mean.
        state_variance : float
            Finite non-negative current posterior state variance.
        n_steps : int
            Number of future occasions to forecast.

        Returns
        -------
        tuple
            Forecast means and variances, each with shape ``(n_steps,)``.
        """
        n_steps = self._validated_positive_integer(n_steps, "n_steps")
        state_means = self._validated_state_vector(
            state_mean,
            1,
            "state_mean",
            default=self.initial_mean,
        )
        state_variances = self._validated_state_vector(
            state_variance,
            1,
            "state_variance",
            default=self.initial_var,
            constraint="nonnegative",
        )
        forecast_means, forecast_variances = self._forecast_state_moments_batch(
            state_means,
            state_variances,
            n_steps,
        )
        return forecast_means[0].copy(), forecast_variances[0].copy()

    def forecast_from_state_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        n_steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast latent-state moments from multiple posterior states.

        Parameters
        ----------
        state_means : NDArray
            Finite current posterior means with shape ``(n_persons,)``.
        state_variances : float or NDArray
            Finite non-negative current posterior variances, either scalar or
            shape ``(n_persons,)``.
        n_steps : int
            Number of future occasions to forecast.

        Returns
        -------
        tuple
            Forecast means and variances, each with shape
            ``(n_persons, n_steps)``.
        """
        n_steps = self._validated_positive_integer(n_steps, "n_steps")
        mean_values, variance_values = self._validated_state_moments_batch(
            state_means,
            state_variances,
        )
        return self._forecast_state_moments_batch(
            mean_values,
            variance_values,
            n_steps,
        )

    def forecast_response_probabilities(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Forecast marginal item-success probabilities for one person."""
        response_values = self._validated_filter_responses(responses, batch=False)
        probabilities = self.forecast_response_probabilities_batch(
            response_values[None, :, :],
            n_steps,
            n_quadpts=n_quadpts,
        )
        return probabilities[0].copy()

    def forecast_response_probabilities_batch(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Forecast marginal item-success probabilities for multiple people.

        Item probabilities are integrated over each Gaussian forecast-state
        distribution with standard-normal Gauss--Hermite quadrature.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.
        n_steps : int
            Number of future occasions to forecast.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Marginal success probabilities with shape
            ``(n_persons, n_steps, n_items)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        forecast_means, forecast_variances = self.forecast_batch(
            responses,
            n_steps,
        )
        return self._integrated_observation_probabilities(
            forecast_means,
            forecast_variances,
            n_quadpts,
        )

    def forecast_response_probabilities_from_state(
        self,
        state_mean: float,
        state_variance: float,
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Forecast item probabilities from one current posterior state.

        Parameters
        ----------
        state_mean : float
            Finite current posterior state mean.
        state_variance : float
            Finite non-negative current posterior state variance.
        n_steps : int
            Number of future occasions to forecast.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Marginal success probabilities with shape
            ``(n_steps, n_items)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        forecast_means, forecast_variances = self.forecast_from_state(
            state_mean,
            state_variance,
            n_steps,
        )
        probabilities = self._integrated_observation_probabilities(
            forecast_means[None, :],
            forecast_variances[None, :],
            n_quadpts,
        )
        return probabilities[0].copy()

    def forecast_response_probabilities_from_state_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Forecast item probabilities from multiple posterior states.

        Parameters
        ----------
        state_means : NDArray
            Finite current posterior means with shape ``(n_persons,)``.
        state_variances : float or NDArray
            Finite non-negative current posterior variances, either scalar or
            shape ``(n_persons,)``.
        n_steps : int
            Number of future occasions to forecast.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Marginal success probabilities with shape
            ``(n_persons, n_steps, n_items)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        forecast_means, forecast_variances = self.forecast_from_state_batch(
            state_means,
            state_variances,
            n_steps,
        )
        return self._integrated_observation_probabilities(
            forecast_means,
            forecast_variances,
            n_quadpts,
        )

    def _build_forecast_summary(
        self,
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> StateSpaceBatchForecastResult:
        """Combine validated forecast moments and response probabilities."""
        response_probabilities = self._integrated_observation_probabilities(
            state_means,
            state_variances,
            n_quadpts,
        )
        return StateSpaceBatchForecastResult(
            state_means=state_means,
            state_variances=state_variances,
            response_probabilities=response_probabilities,
        )

    @staticmethod
    def _single_forecast_summary(
        result: StateSpaceBatchForecastResult,
    ) -> StateSpaceForecastResult:
        """Copy one row from a batched forecast summary."""
        return StateSpaceForecastResult(
            state_means=result.state_means[0].copy(),
            state_variances=result.state_variances[0].copy(),
            response_probabilities=result.response_probabilities[0].copy(),
        )

    def forecast_summary(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> StateSpaceForecastResult:
        """Forecast latent states and item probabilities for one history."""
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_filter_responses(responses, batch=False)
        result = self.forecast_summary_batch(
            response_values[None, :, :],
            n_steps,
            n_quadpts=n_quadpts,
        )
        return self._single_forecast_summary(result)

    def forecast_summary_batch(
        self,
        responses: NDArray[np.int_],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> StateSpaceBatchForecastResult:
        """Forecast latent states and item probabilities for many histories.

        Filtering and state propagation are shared across all returned fields.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        state_means, state_variances = self.forecast_batch(responses, n_steps)
        return self._build_forecast_summary(
            state_means,
            state_variances,
            n_quadpts,
        )

    def forecast_summary_from_state(
        self,
        state_mean: float,
        state_variance: float,
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> StateSpaceForecastResult:
        """Forecast states and item probabilities from one posterior state."""
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        state_means, state_variances = self.forecast_from_state(
            state_mean,
            state_variance,
            n_steps,
        )
        result = self._build_forecast_summary(
            state_means[None, :],
            state_variances[None, :],
            n_quadpts,
        )
        return self._single_forecast_summary(result)

    def forecast_summary_from_state_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        n_steps: int,
        *,
        n_quadpts: int = 21,
    ) -> StateSpaceBatchForecastResult:
        """Forecast states and item probabilities from many posterior states.

        State propagation is shared across all returned fields.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        forecast_means, forecast_variances = self.forecast_from_state_batch(
            state_means,
            state_variances,
            n_steps,
        )
        return self._build_forecast_summary(
            forecast_means,
            forecast_variances,
            n_quadpts,
        )

    def _integrated_observation_probabilities(
        self,
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> NDArray[np.float64]:
        """Integrate item probabilities over Gaussian state distributions."""
        marginal = np.zeros(
            (state_means.size, self.n_items),
            dtype=np.float64,
        )

        for weight, probabilities in self._state_observation_quadrature(
            state_means,
            state_variances,
            n_quadpts,
        ):
            marginal += weight * probabilities

        probabilities = marginal.reshape(state_means.shape + (self.n_items,))
        return np.clip(probabilities, 0.0, 1.0)

    def _state_observation_quadrature(
        self,
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> Iterator[tuple[float, NDArray[np.float64]]]:
        """Yield weighted conditional probabilities over Gaussian states."""
        nodes, weights = standard_normal_quadrature(n_quadpts)
        flat_means = state_means.ravel()
        flat_scales = np.sqrt(state_variances).ravel()

        for node, weight in zip(nodes, weights, strict=True):
            states = flat_means + flat_scales * node
            yield float(weight), self._observation_probability(states)

    def state_response_probabilities(
        self,
        state_mean: float,
        state_variance: float,
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Return marginal item probabilities from one state distribution.

        Parameters
        ----------
        state_mean : float
            Finite mean of the Gaussian state distribution.
        state_variance : float
            Finite non-negative variance of the Gaussian state distribution.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Item-success probabilities with shape ``(n_items,)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        state_means = self._validated_state_vector(
            state_mean,
            1,
            "state_mean",
            default=self.initial_mean,
        )
        state_variances = self._validated_state_vector(
            state_variance,
            1,
            "state_variance",
            default=self.initial_var,
            constraint="nonnegative",
        )
        probabilities = self._integrated_observation_probabilities(
            state_means,
            state_variances,
            n_quadpts,
        )
        return probabilities[0].copy()

    def state_response_probabilities_batch(
        self,
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Return marginal item probabilities from multiple state distributions.

        Parameters
        ----------
        state_means : NDArray
            Finite Gaussian state means with shape ``(n_persons,)``.
        state_variances : float or NDArray
            Finite non-negative state variances, either scalar or shape
            ``(n_persons,)``.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Item-success probabilities with shape ``(n_persons, n_items)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        mean_values, variance_values = self._validated_state_moments_batch(
            state_means,
            state_variances,
        )
        return self._integrated_observation_probabilities(
            mean_values,
            variance_values,
            n_quadpts,
        )

    def _integrated_response_log_likelihoods(
        self,
        responses: NDArray[np.int_],
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> NDArray[np.float64]:
        """Integrate joint response-pattern likelihoods over Gaussian states."""
        flat_responses = responses.reshape(-1, self.n_items)
        observed = flat_responses >= 0
        correct = flat_responses == 1
        incorrect = flat_responses == 0
        scores = np.full(state_means.size, -np.inf, dtype=np.float64)

        for weight, probabilities in self._state_observation_quadrature(
            state_means,
            state_variances,
            n_quadpts,
        ):
            conditional_score = np.sum(
                np.where(
                    correct,
                    np.log(probabilities),
                    np.where(incorrect, np.log1p(-probabilities), 0.0),
                ),
                axis=1,
            )
            scores = np.logaddexp(
                scores,
                np.log(weight) + conditional_score,
            )

        scores[~np.any(observed, axis=1)] = 0.0
        return scores.reshape(state_means.shape)

    def _integrated_response_diagnostics(
        self,
        responses: NDArray[np.int_],
        state_means: NDArray[np.float64],
        state_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate item probabilities and joint scores in one quadrature pass."""
        flat_responses = responses.reshape(-1, self.n_items)
        observed = flat_responses >= 0
        correct = flat_responses == 1
        incorrect = flat_responses == 0
        marginal = np.zeros(
            (state_means.size, self.n_items),
            dtype=np.float64,
        )
        scores = np.full(state_means.size, -np.inf, dtype=np.float64)

        for weight, probabilities in self._state_observation_quadrature(
            state_means,
            state_variances,
            n_quadpts,
        ):
            marginal += weight * probabilities
            conditional_score = np.sum(
                np.where(
                    correct,
                    np.log(probabilities),
                    np.where(incorrect, np.log1p(-probabilities), 0.0),
                ),
                axis=1,
            )
            scores = np.logaddexp(
                scores,
                np.log(weight) + conditional_score,
            )

        scores[~np.any(observed, axis=1)] = 0.0
        response_probabilities = np.clip(
            marginal.reshape(state_means.shape + (self.n_items,)),
            0.0,
            1.0,
        )
        return response_probabilities, scores.reshape(state_means.shape)

    @staticmethod
    def _item_response_diagnostics(
        responses: NDArray[np.int_],
        probabilities: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return missing-aware item log scores and predictive residuals."""
        observed = responses >= 0
        correct = responses == 1
        incorrect = responses == 0
        safe_probabilities = np.clip(
            probabilities,
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        item_log_likelihoods = np.where(
            correct,
            np.log(safe_probabilities),
            np.where(incorrect, np.log1p(-safe_probabilities), np.nan),
        )
        residuals = np.where(
            observed,
            responses - probabilities,
            np.nan,
        )
        response_variances = np.maximum(
            probabilities * (1.0 - probabilities),
            PROB_EPSILON,
        )
        standardized_residuals = residuals / np.sqrt(response_variances)
        return item_log_likelihoods, residuals, standardized_residuals

    def state_response_log_likelihood(
        self,
        responses: NDArray[np.int_],
        state_mean: float,
        state_variance: float,
        *,
        n_quadpts: int = 21,
    ) -> float:
        """Score one response pattern against a Gaussian state distribution.

        The joint likelihood integrates over the shared latent state. Missing
        items are omitted, and a fully missing pattern returns zero.

        Parameters
        ----------
        responses : NDArray
            Integer response vector with shape ``(n_items,)``. Use ``-1`` for
            missing item responses.
        state_mean : float
            Finite mean of the Gaussian state distribution.
        state_variance : float
            Finite non-negative variance of the Gaussian state distribution.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        float
            Joint response-pattern log likelihood.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_update_responses(responses, batch=False)
        state_means = self._validated_state_vector(
            state_mean,
            1,
            "state_mean",
            default=self.initial_mean,
        )
        state_variances = self._validated_state_vector(
            state_variance,
            1,
            "state_variance",
            default=self.initial_var,
            constraint="nonnegative",
        )
        scores = self._integrated_response_log_likelihoods(
            response_values[None, :],
            state_means,
            state_variances,
            n_quadpts,
        )
        return float(scores[0])

    def state_response_log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        state_means: NDArray[np.float64],
        state_variances: float | NDArray[np.float64],
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Score response patterns against multiple state distributions.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_items)``. Use
            ``-1`` for missing item responses.
        state_means : NDArray
            Finite Gaussian state means with shape ``(n_persons,)``.
        state_variances : float or NDArray
            Finite non-negative state variances, either scalar or shape
            ``(n_persons,)``.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Joint response-pattern log likelihoods with shape
            ``(n_persons,)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_update_responses(responses, batch=True)
        mean_values, variance_values = self._validated_state_moments_batch(
            state_means,
            state_variances,
        )
        if len(response_values) != len(mean_values):
            raise ValueError(
                "responses and state moments must contain the same number of people"
            )
        return self._integrated_response_log_likelihoods(
            response_values,
            mean_values,
            variance_values,
            n_quadpts,
        )

    def _online_step_batch(
        self,
        responses: NDArray[np.int_],
        prior_means: NDArray[np.float64],
        prior_variances: NDArray[np.float64],
        n_quadpts: int,
    ) -> StateSpaceBatchStepResult:
        """Process validated response rows and state priors in one pass."""
        response_probabilities, response_log_likelihoods = (
            self._integrated_response_diagnostics(
                responses,
                prior_means,
                prior_variances,
                n_quadpts,
            )
        )
        item_log_likelihoods, residuals, standardized_residuals = (
            self._item_response_diagnostics(
                responses,
                response_probabilities,
            )
        )
        updated_means, updated_variances = self._extended_kalman_update_batch(
            responses,
            prior_means,
            prior_variances,
        )
        next_means, next_variances = self._propagate_state_moments(
            updated_means,
            updated_variances,
            1,
        )
        return StateSpaceBatchStepResult(
            response_probabilities=response_probabilities,
            response_log_likelihoods=response_log_likelihoods,
            item_log_likelihoods=item_log_likelihoods,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            updated_means=updated_means,
            updated_variances=updated_variances,
            next_means=next_means,
            next_variances=next_variances,
        )

    def online_step(
        self,
        responses: NDArray[np.int_],
        *,
        prior_mean: float | None = None,
        prior_variance: float | None = None,
        n_quadpts: int = 21,
    ) -> StateSpaceStepResult:
        """Predict, score, update, and propagate one response occasion.

        Omitted prior moments use the model's initial state distribution. The
        returned next moments are ready for the following occasion.

        Parameters
        ----------
        responses : NDArray
            Integer response vector with shape ``(n_items,)``. Use ``-1`` for
            missing item responses.
        prior_mean : float, optional
            Finite predicted state mean for this occasion.
        prior_variance : float, optional
            Finite positive predicted state variance for this occasion.
        n_quadpts : int, default=21
            Number of quadrature points used for prediction and scoring.

        Returns
        -------
        StateSpaceStepResult
            Response predictions, updated state, and next state prior.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_update_responses(responses, batch=False)
        prior_means = self._validated_state_vector(
            prior_mean,
            1,
            "prior_mean",
            default=self.initial_mean,
        )
        prior_variances = self._validated_state_vector(
            prior_variance,
            1,
            "prior_variance",
            default=self.initial_var,
            constraint="positive",
        )
        result = self._online_step_batch(
            response_values[None, :],
            prior_means,
            prior_variances,
            n_quadpts,
        )
        return StateSpaceStepResult(
            response_probabilities=result.response_probabilities[0].copy(),
            response_log_likelihood=float(result.response_log_likelihoods[0]),
            item_log_likelihoods=result.item_log_likelihoods[0].copy(),
            residuals=result.residuals[0].copy(),
            standardized_residuals=result.standardized_residuals[0].copy(),
            updated_mean=float(result.updated_means[0]),
            updated_variance=float(result.updated_variances[0]),
            next_mean=float(result.next_means[0]),
            next_variance=float(result.next_variances[0]),
        )

    def online_step_batch(
        self,
        responses: NDArray[np.int_],
        *,
        prior_means: float | NDArray[np.float64] | None = None,
        prior_variances: float | NDArray[np.float64] | None = None,
        n_quadpts: int = 21,
    ) -> StateSpaceBatchStepResult:
        """Predict, score, update, and propagate multiple response rows.

        Scalar prior moments are broadcast across people. Omitted moments use
        the model's initial state distribution.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_items)``. Use
            ``-1`` for missing item responses.
        prior_means : float or NDArray, optional
            Finite predicted state means, either scalar or shape
            ``(n_persons,)``.
        prior_variances : float or NDArray, optional
            Finite positive predicted state variances, either scalar or shape
            ``(n_persons,)``.
        n_quadpts : int, default=21
            Number of quadrature points used for prediction and scoring.

        Returns
        -------
        StateSpaceBatchStepResult
            Response predictions, updated states, and next state priors.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_update_responses(responses, batch=True)
        n_persons = len(response_values)
        mean_values = self._validated_state_vector(
            prior_means,
            n_persons,
            "prior_means",
            default=self.initial_mean,
        )
        variance_values = self._validated_state_vector(
            prior_variances,
            n_persons,
            "prior_variances",
            default=self.initial_var,
            constraint="positive",
        )
        return self._online_step_batch(
            response_values,
            mean_values,
            variance_values,
            n_quadpts,
        )

    def _predicted_state_moments_batch(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return causal state predictions for validated response histories."""
        predicted_means, predicted_variances, _, _ = (
            self._predictive_state_moments_batch(responses)
        )
        return predicted_means, predicted_variances

    def _predictive_diagnostics_batch(
        self,
        responses: NDArray[np.int_],
        n_quadpts: int,
    ) -> StateSpaceBatchPredictiveResult:
        """Build complete causal diagnostics from validated histories."""
        (
            predicted_means,
            predicted_variances,
            filtered_means,
            filtered_variances,
        ) = self._predictive_state_moments_batch(responses)
        response_probabilities, response_log_likelihoods = (
            self._integrated_response_diagnostics(
                responses,
                predicted_means,
                predicted_variances,
                n_quadpts,
            )
        )
        item_log_likelihoods, residuals, standardized_residuals = (
            self._item_response_diagnostics(
                responses,
                response_probabilities,
            )
        )
        return StateSpaceBatchPredictiveResult(
            predicted_means=predicted_means,
            predicted_variances=predicted_variances,
            filtered_means=filtered_means,
            filtered_variances=filtered_variances,
            response_probabilities=response_probabilities,
            response_log_likelihoods=response_log_likelihoods,
            item_log_likelihoods=item_log_likelihoods,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
        )

    def predictive_diagnostics(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
    ) -> StateSpacePredictiveResult:
        """Return complete causal diagnostics for one response history.

        The result combines state predictions, filtered states, response
        probabilities, joint and item log scores, and residuals in one pass.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_timepoints, n_items)``.
            Use ``-1`` for missing item responses.
        n_quadpts : int, default=21
            Number of quadrature points used for prediction and scoring.

        Returns
        -------
        StateSpacePredictiveResult
            Causal state trajectories and response diagnostics.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_filter_responses(responses, batch=False)
        result = self._predictive_diagnostics_batch(
            response_values[None, :, :],
            n_quadpts,
        )
        return StateSpacePredictiveResult(
            predicted_means=result.predicted_means[0].copy(),
            predicted_variances=result.predicted_variances[0].copy(),
            filtered_means=result.filtered_means[0].copy(),
            filtered_variances=result.filtered_variances[0].copy(),
            response_probabilities=result.response_probabilities[0].copy(),
            response_log_likelihoods=result.response_log_likelihoods[0].copy(),
            item_log_likelihoods=result.item_log_likelihoods[0].copy(),
            residuals=result.residuals[0].copy(),
            standardized_residuals=result.standardized_residuals[0].copy(),
        )

    def predictive_diagnostics_batch(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
    ) -> StateSpaceBatchPredictiveResult:
        """Return complete causal diagnostics for multiple histories.

        All outputs are computed from one vectorized filtering pass and one
        shared quadrature pass. State arrays have shape
        ``(n_persons, n_timepoints)``; response arrays additionally have an
        item dimension.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.
        n_quadpts : int, default=21
            Number of quadrature points used for prediction and scoring.

        Returns
        -------
        StateSpaceBatchPredictiveResult
            Batched causal state trajectories and response diagnostics.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_filter_responses(responses, batch=True)
        return self._predictive_diagnostics_batch(response_values, n_quadpts)

    def predictive_response_probabilities(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Return one-step-ahead item probabilities for one response history."""
        response_values = self._validated_filter_responses(responses, batch=False)
        probabilities = self.predictive_response_probabilities_batch(
            response_values[None, :, :],
            n_quadpts=n_quadpts,
        )
        return probabilities[0].copy()

    def predictive_response_probabilities_batch(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
    ) -> NDArray[np.float64]:
        """Return causal item probabilities for multiple response histories.

        Each occasion is predicted only from the initial state and responses
        observed at earlier occasions. Probabilities are integrated over the
        Gaussian state prediction with standard-normal Gauss--Hermite
        quadrature.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.
        n_quadpts : int, default=21
            Number of quadrature points used for predictive integration.

        Returns
        -------
        NDArray
            Success probabilities with shape
            ``(n_persons, n_timepoints, n_items)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        response_values = self._validated_filter_responses(responses, batch=True)
        predicted_means, predicted_variances = self._predicted_state_moments_batch(
            response_values
        )
        return self._integrated_observation_probabilities(
            predicted_means,
            predicted_variances,
            n_quadpts,
        )

    def predictive_residuals(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
        standardized: bool = False,
    ) -> NDArray[np.float64]:
        """Return one-step-ahead residuals for one response history."""
        response_values = self._validated_filter_responses(responses, batch=False)
        residuals = self.predictive_residuals_batch(
            response_values[None, :, :],
            n_quadpts=n_quadpts,
            standardized=standardized,
        )
        return residuals[0].copy()

    def predictive_residuals_batch(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
        standardized: bool = False,
    ) -> NDArray[np.float64]:
        """Return missing-aware predictive residuals for multiple people.

        Raw residuals are observed minus predicted success probabilities.
        ``standardized=True`` returns Pearson residuals. Missing responses are
        represented by ``numpy.nan`` in either form.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        if not isinstance(standardized, (bool, np.bool_)):
            raise ValueError("standardized must be boolean")
        response_values = self._validated_filter_responses(responses, batch=True)
        predicted_means, predicted_variances = self._predicted_state_moments_batch(
            response_values
        )
        probabilities = self._integrated_observation_probabilities(
            predicted_means,
            predicted_variances,
            n_quadpts,
        )
        _, residuals, standardized_residuals = self._item_response_diagnostics(
            response_values,
            probabilities,
        )
        if standardized:
            return standardized_residuals
        return residuals

    def predictive_log_likelihood(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
        pointwise: bool = False,
    ) -> float | NDArray[np.float64]:
        """Return the prequential log score for one response history.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_timepoints, n_items)``.
            Use ``-1`` for missing item responses.
        n_quadpts : int, default=21
            Number of quadrature points used at each occasion.
        pointwise : bool, default=False
            Return one score per occasion instead of their sum.

        Returns
        -------
        float or NDArray
            Total predictive log likelihood, or an array with shape
            ``(n_timepoints,)`` when ``pointwise=True``.
        """
        if not isinstance(pointwise, (bool, np.bool_)):
            raise ValueError("pointwise must be boolean")
        response_values = self._validated_filter_responses(responses, batch=False)
        scores = self.predictive_log_likelihood_batch(
            response_values[None, :, :],
            n_quadpts=n_quadpts,
            pointwise=True,
        )[0]
        if pointwise:
            return scores.copy()
        return float(np.sum(scores))

    def predictive_log_likelihood_batch(
        self,
        responses: NDArray[np.int_],
        *,
        n_quadpts: int = 21,
        pointwise: bool = False,
    ) -> NDArray[np.float64]:
        """Return prequential log scores for multiple response histories.

        At each occasion, the joint observed item pattern is integrated over
        the Gaussian state prediction from earlier occasions. Fully missing
        occasions contribute zero. The filtering distribution is an extended
        Kalman approximation.

        Parameters
        ----------
        responses : NDArray
            Integer response array with shape
            ``(n_persons, n_timepoints, n_items)``. Use ``-1`` for missing
            item responses.
        n_quadpts : int, default=21
            Number of quadrature points used at each occasion.
        pointwise : bool, default=False
            Return one score per person and occasion instead of per-person
            totals.

        Returns
        -------
        NDArray
            Per-person totals with shape ``(n_persons,)``, or pointwise scores
            with shape ``(n_persons, n_timepoints)``.
        """
        n_quadpts = self._validated_positive_integer(n_quadpts, "n_quadpts")
        if not isinstance(pointwise, (bool, np.bool_)):
            raise ValueError("pointwise must be boolean")
        response_values = self._validated_filter_responses(responses, batch=True)
        predicted_means, predicted_variances = self._predicted_state_moments_batch(
            response_values
        )
        pointwise_scores = self._integrated_response_log_likelihoods(
            response_values,
            predicted_means,
            predicted_variances,
            n_quadpts,
        )
        if pointwise:
            return pointwise_scores
        return np.sum(pointwise_scores, axis=1)

    def simulate(
        self,
        n_persons: int,
        seed: int | None = None,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Simulate response data.

        Parameters
        ----------
        n_persons : int
            Number of persons
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            (responses, theta_trajectories)
        """
        if (
            isinstance(n_persons, bool)
            or not isinstance(n_persons, (int, np.integer))
            or n_persons < 1
        ):
            raise ValueError("n_persons must be a positive integer")
        n_persons = int(n_persons)
        rng = np.random.default_rng(seed)

        A = self.transition_matrix[0, 0]
        Q = self.process_noise[0, 0]

        theta = np.zeros((n_persons, self.n_timepoints))
        theta[:, 0] = rng.normal(
            self.initial_mean, np.sqrt(self.initial_var), n_persons
        )

        for t in range(1, self.n_timepoints):
            theta[:, t] = A * theta[:, t - 1] + rng.normal(0, np.sqrt(Q), n_persons)

        flat_theta = theta.reshape(-1)
        flat_responses = np.empty((flat_theta.size, self.n_items), dtype=np.int32)
        rows_per_chunk = max(
            1,
            _STATE_SPACE_MAX_PROBABILITY_VALUES // self.n_items,
        )
        for start in range(0, flat_theta.size, rows_per_chunk):
            stop = min(start + rows_per_chunk, flat_theta.size)
            logits = self.discrimination[None, :] * (
                flat_theta[start:stop, None] - self.difficulty[None, :]
            )
            probabilities = sigmoid(logits)
            if self.base_model == "3PL":
                probabilities = (
                    self.guessing[None, :]
                    + (1.0 - self.guessing[None, :]) * probabilities
                )
            flat_responses[start:stop] = rng.random(probabilities.shape) < probabilities

        responses = flat_responses.reshape(
            n_persons,
            self.n_timepoints,
            self.n_items,
        )

        return responses, theta

    def summary(self) -> str:
        """Generate model summary."""
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'State-Space IRT Model Summary':^{width}}")
        lines.append("=" * width)

        lines.append(f"Base Model:         {self.base_model}")
        lines.append(f"Number of Items:    {self.n_items}")
        lines.append(f"Number of Times:    {self.n_timepoints}")
        lines.append(f"Transition (A):     {self.transition_matrix[0, 0]:.4f}")
        lines.append(f"Process Noise (Q):  {self.process_noise[0, 0]:.4f}")
        lines.append(f"Observation Noise:  {self.observation_noise:.4f}")
        lines.append(f"Initial Mean:       {self.initial_mean:.4f}")
        lines.append(f"Initial Variance:   {self.initial_var:.4f}")

        lines.append("=" * width)
        return "\n".join(lines)
