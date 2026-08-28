"""State-space IRT for continuous latent trait evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.utils.numeric import standard_normal_quadrature

_STATE_SPACE_MAX_PROBABILITY_VALUES = 1_000_000


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
        n_persons = response_values.shape[0]
        filtered_means = np.empty(
            (n_persons, self.n_timepoints),
            dtype=np.float64,
        )
        filtered_variances = np.empty_like(filtered_means)
        transition = float(self.transition_matrix[0, 0])
        process_variance = float(self.process_noise[0, 0])
        predicted_mean = np.full(n_persons, self.initial_mean, dtype=np.float64)
        predicted_variance = np.full(n_persons, self.initial_var, dtype=np.float64)

        for time_index in range(self.n_timepoints):
            time_responses = response_values[:, time_index]
            observed = time_responses >= 0
            has_observations = np.any(observed, axis=1)
            updated_mean = predicted_mean.copy()
            updated_variance = predicted_variance.copy()

            if np.any(has_observations):
                candidate_mean = predicted_mean.copy()
                for _ in range(5):
                    probability, derivative = (
                        self._observation_probability_and_derivative(candidate_mean)
                    )
                    response_variance = (
                        probability * (1.0 - probability) + self.observation_noise
                    )
                    score = (
                        np.sum(
                            np.where(
                                observed,
                                derivative
                                * (time_responses - probability)
                                / response_variance,
                                0.0,
                            ),
                            axis=1,
                        )
                        - (candidate_mean - predicted_mean) / predicted_variance
                    )
                    information = 1.0 / predicted_variance + np.sum(
                        np.where(
                            observed,
                            derivative**2 / response_variance,
                            0.0,
                        ),
                        axis=1,
                    )
                    candidate_mean[has_observations] += (
                        score[has_observations] / information[has_observations]
                    )

                probability, derivative = self._observation_probability_and_derivative(
                    candidate_mean
                )
                response_variance = (
                    probability * (1.0 - probability) + self.observation_noise
                )
                final_information = 1.0 / predicted_variance + np.sum(
                    np.where(
                        observed,
                        derivative**2 / response_variance,
                        0.0,
                    ),
                    axis=1,
                )
                updated_mean[has_observations] = candidate_mean[has_observations]
                updated_variance[has_observations] = (
                    1.0 / final_information[has_observations]
                )

            filtered_means[:, time_index] = updated_mean
            filtered_variances[:, time_index] = updated_variance
            if time_index < self.n_timepoints - 1:
                predicted_mean = transition * updated_mean
                predicted_variance = transition**2 * updated_variance + process_variance

        return filtered_means, filtered_variances

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
        transition_squared = transition**2
        process_variance = float(self.process_noise[0, 0])

        for time_index in range(self.n_timepoints - 2, -1, -1):
            predicted_mean = transition * filtered_means[:, time_index]
            predicted_variance = (
                transition_squared * filtered_variances[:, time_index]
                + process_variance
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
        n_persons = filtered_means.shape[0]
        forecast_means = np.empty((n_persons, n_steps), dtype=np.float64)
        forecast_variances = np.empty_like(forecast_means)
        transition = float(self.transition_matrix[0, 0])
        transition_squared = transition**2
        process_variance = float(self.process_noise[0, 0])
        current_mean = filtered_means[:, -1]
        current_variance = filtered_variances[:, -1]

        for step_index in range(n_steps):
            current_mean = transition * current_mean
            current_variance = transition_squared * current_variance + process_variance
            forecast_means[:, step_index] = current_mean
            forecast_variances[:, step_index] = current_variance

        return forecast_means, forecast_variances

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
        nodes, weights = standard_normal_quadrature(n_quadpts)
        flat_means = forecast_means.ravel()
        flat_scales = np.sqrt(forecast_variances).ravel()
        marginal = np.zeros((flat_means.size, self.n_items), dtype=np.float64)

        for node, weight in zip(nodes, weights, strict=True):
            states = flat_means + flat_scales * node
            marginal += weight * self._observation_probability(states)

        probabilities = marginal.reshape(
            forecast_means.shape[0],
            forecast_means.shape[1],
            self.n_items,
        )
        return np.clip(probabilities, 0.0, 1.0)

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
        filtered_means, filtered_variances = self.extended_kalman_filter_batch(
            response_values
        )
        predicted_means = np.empty_like(filtered_means)
        predicted_variances = np.empty_like(filtered_variances)
        predicted_means[:, 0] = self.initial_mean
        predicted_variances[:, 0] = self.initial_var
        transition = float(self.transition_matrix[0, 0])
        transition_squared = transition**2
        process_variance = float(self.process_noise[0, 0])
        predicted_means[:, 1:] = transition * filtered_means[:, :-1]
        predicted_variances[:, 1:] = (
            transition_squared * filtered_variances[:, :-1] + process_variance
        )

        observed = response_values >= 0
        correct = response_values == 1
        incorrect = response_values == 0
        nodes, weights = standard_normal_quadrature(n_quadpts)
        flat_means = predicted_means.ravel()
        flat_scales = np.sqrt(predicted_variances).ravel()
        pointwise_scores = np.full(predicted_means.shape, -np.inf, dtype=np.float64)

        for node, weight in zip(nodes, weights, strict=True):
            states = flat_means + flat_scales * node
            probabilities = self._observation_probability(states).reshape(
                response_values.shape
            )
            conditional_score = np.sum(
                np.where(
                    correct,
                    np.log(probabilities),
                    np.where(incorrect, np.log1p(-probabilities), 0.0),
                ),
                axis=2,
            )
            pointwise_scores = np.logaddexp(
                pointwise_scores,
                np.log(weight) + conditional_score,
            )

        pointwise_scores[~np.any(observed, axis=2)] = 0.0
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
