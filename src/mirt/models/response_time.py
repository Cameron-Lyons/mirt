"""Joint response-accuracy and response-time models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Literal, Self

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError

AccuracyModel = Literal["2PL", "3PL"]
_LOG_2PI = float(np.log(2.0 * np.pi))


def _safe_sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the shared sigmoid without harmless tail warnings."""
    with np.errstate(over="ignore", invalid="ignore"):
        return sigmoid(values)


@dataclass
class ResponseTimeModel:
    """Van der Linden hierarchical response-time model.

    Accuracy follows a 2PL or 3PL model and log response time follows
    ``log(T) | tau ~ N(beta - tau, 1 / alpha**2)``.  Population ability and
    speed are jointly normal, which also defines conditional simulation when
    only one person parameter is supplied.
    """

    n_items: int
    accuracy_model: AccuracyModel = "2PL"
    item_names: list[str] | None = None

    discrimination: NDArray[np.float64] | None = None
    difficulty: NDArray[np.float64] | None = None
    guessing: NDArray[np.float64] | None = None
    time_intensity: NDArray[np.float64] | None = None
    time_discrimination: NDArray[np.float64] | None = None
    ability_speed_mean: NDArray[np.float64] | None = None
    ability_speed_cov: NDArray[np.float64] | None = None

    _PARAMETER_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "discrimination",
            "difficulty",
            "guessing",
            "time_intensity",
            "time_discrimination",
            "ability_speed_mean",
            "ability_speed_cov",
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.n_items, bool) or not isinstance(
            self.n_items, (int, np.integer)
        ):
            raise MirtValidationError(
                "n_items must be an integer",
                parameter="n_items",
                value=self.n_items,
                expected="positive integer",
            )
        self.n_items = int(self.n_items)
        if self.n_items <= 0:
            raise MirtValidationError(
                "n_items must be positive",
                parameter="n_items",
                value=self.n_items,
                expected="> 0",
            )
        if self.accuracy_model not in ("2PL", "3PL"):
            raise MirtValidationError(
                "accuracy_model must be '2PL' or '3PL'",
                parameter="accuracy_model",
                value=self.accuracy_model,
                expected="'2PL' or '3PL'",
            )

        if self.item_names is None:
            self.item_names = [f"Item_{i}" for i in range(self.n_items)]
        else:
            if len(self.item_names) != self.n_items:
                raise MirtValidationError(
                    "item_names length must match n_items",
                    parameter="item_names",
                    value=len(self.item_names),
                    expected=str(self.n_items),
                )
            if not all(isinstance(name, str) and name for name in self.item_names):
                raise MirtValidationError(
                    "item_names must contain non-empty strings",
                    parameter="item_names",
                    value=self.item_names,
                    expected="non-empty strings",
                )
            self.item_names = self.item_names.copy()

        self.discrimination = self._item_parameter(
            "discrimination", self.discrimination, default=1.0
        )
        self.difficulty = self._item_parameter(
            "difficulty", self.difficulty, default=0.0
        )
        self.time_intensity = self._item_parameter(
            "time_intensity", self.time_intensity, default=0.0
        )
        self.time_discrimination = self._item_parameter(
            "time_discrimination", self.time_discrimination, default=1.0
        )

        if self.accuracy_model == "3PL":
            self.guessing = self._item_parameter("guessing", self.guessing, default=0.2)
        elif self.guessing is not None:
            raise MirtValidationError(
                "guessing is only valid for the 3PL accuracy model",
                parameter="guessing",
                value=self.guessing,
                expected="None for 2PL",
            )

        self.ability_speed_mean = self._fixed_parameter(
            "ability_speed_mean", self.ability_speed_mean, (2,), np.zeros(2)
        )
        self.ability_speed_cov = self._fixed_parameter(
            "ability_speed_cov", self.ability_speed_cov, (2, 2), np.eye(2)
        )
        self._validated_state()

    def _item_parameter(
        self,
        name: str,
        values: NDArray[np.float64] | None,
        *,
        default: float,
    ) -> NDArray[np.float64]:
        if values is None:
            return np.full(self.n_items, default, dtype=np.float64)
        return self._as_float_array(name, values, (self.n_items,)).copy()

    @staticmethod
    def _fixed_parameter(
        name: str,
        values: NDArray[np.float64] | None,
        shape: tuple[int, ...],
        default: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if values is None:
            return default.astype(np.float64, copy=True)
        return ResponseTimeModel._as_float_array(name, values, shape).copy()

    @staticmethod
    def _as_float_array(
        name: str,
        values: object,
        shape: tuple[int, ...],
    ) -> NDArray[np.float64]:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=values,
                expected=f"numeric array with shape {shape}",
            ) from exc
        if array.shape != shape:
            raise MirtValidationError(
                f"{name} must have shape {shape}",
                parameter=name,
                value=array.shape,
                expected=str(shape),
            )
        return array

    def _validated_state(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if self.accuracy_model not in ("2PL", "3PL"):
            raise MirtValidationError(
                "accuracy_model must be '2PL' or '3PL'",
                parameter="accuracy_model",
                value=self.accuracy_model,
                expected="'2PL' or '3PL'",
            )
        discrimination = self._as_float_array(
            "discrimination", self.discrimination, (self.n_items,)
        )
        difficulty = self._as_float_array(
            "difficulty", self.difficulty, (self.n_items,)
        )
        time_intensity = self._as_float_array(
            "time_intensity", self.time_intensity, (self.n_items,)
        )
        time_discrimination = self._as_float_array(
            "time_discrimination", self.time_discrimination, (self.n_items,)
        )
        mean = self._as_float_array("ability_speed_mean", self.ability_speed_mean, (2,))
        covariance = self._as_float_array(
            "ability_speed_cov", self.ability_speed_cov, (2, 2)
        )

        guessing = None
        if self.accuracy_model == "3PL":
            guessing = self._as_float_array("guessing", self.guessing, (self.n_items,))
        elif self.guessing is not None:
            raise MirtValidationError(
                "guessing is only valid for the 3PL accuracy model",
                parameter="guessing",
                value=self.guessing,
                expected="None for 2PL",
            )

        arrays = {
            "discrimination": discrimination,
            "difficulty": difficulty,
            "time_intensity": time_intensity,
            "time_discrimination": time_discrimination,
            "ability_speed_mean": mean,
            "ability_speed_cov": covariance,
        }
        if guessing is not None:
            arrays["guessing"] = guessing
        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise MirtValidationError(
                    f"{name} must contain only finite values",
                    parameter=name,
                    value=values,
                    expected="finite values",
                )

        if np.any(discrimination <= 0.0):
            raise MirtValidationError(
                "discrimination must be strictly positive",
                parameter="discrimination",
                value=discrimination,
                expected="> 0",
            )
        if np.any(time_discrimination <= 0.0):
            raise MirtValidationError(
                "time_discrimination must be strictly positive",
                parameter="time_discrimination",
                value=time_discrimination,
                expected="> 0",
            )
        if guessing is not None and np.any((guessing < 0.0) | (guessing >= 1.0)):
            raise MirtValidationError(
                "guessing must be in [0, 1)",
                parameter="guessing",
                value=guessing,
                expected="[0, 1)",
            )
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise MirtValidationError(
                "ability_speed_cov must be symmetric",
                parameter="ability_speed_cov",
                value=covariance,
                expected="symmetric positive-definite matrix",
            )
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise MirtValidationError(
                "ability_speed_cov must be positive definite",
                parameter="ability_speed_cov",
                value=covariance,
                expected="symmetric positive-definite matrix",
            ) from exc

        return (
            discrimination,
            difficulty,
            guessing,
            time_intensity,
            time_discrimination,
            mean,
            covariance,
        )

    def set_parameters(self, **parameters: object) -> Self:
        """Update numeric parameters atomically after validating the full state."""
        unknown = set(parameters) - self._PARAMETER_NAMES
        if unknown:
            name = sorted(unknown)[0]
            valid = ", ".join(sorted(self._PARAMETER_NAMES))
            raise MirtValidationError(
                f"Unknown parameter: {name}. Valid parameters: {valid}",
                parameter=name,
                expected=valid,
            )

        candidate = replace(self, **parameters)
        for name in self._PARAMETER_NAMES:
            value = getattr(candidate, name)
            setattr(self, name, None if value is None else value.copy())
        return self

    def _validate_item_idx(self, item_idx: int) -> int:
        if isinstance(item_idx, bool) or not isinstance(item_idx, (int, np.integer)):
            raise MirtValidationError(
                "item_idx must be an integer",
                parameter="item_idx",
                value=item_idx,
                expected="integer",
            )
        item_idx = int(item_idx)
        if item_idx < 0 or item_idx >= self.n_items:
            raise IndexError(f"Item index {item_idx} out of range [0, {self.n_items})")
        return item_idx

    @staticmethod
    def _person_vector(
        values: NDArray[np.float64] | float,
        name: str,
        expected_length: int | None = None,
    ) -> NDArray[np.float64]:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=values,
                expected="one-dimensional finite values",
            ) from exc
        if vector.ndim == 0:
            vector = vector.reshape(1)
        if vector.ndim != 1 or vector.size == 0:
            raise MirtValidationError(
                f"{name} must be a non-empty one-dimensional array",
                parameter=name,
                value=vector.shape,
                expected="non-empty one-dimensional array",
            )
        if expected_length is not None and vector.size != expected_length:
            raise MirtValidationError(
                f"{name} length must be {expected_length}",
                parameter=name,
                value=vector.size,
                expected=str(expected_length),
            )
        if not np.all(np.isfinite(vector)):
            raise MirtValidationError(
                f"{name} must contain only finite values",
                parameter=name,
                value=vector,
                expected="finite values",
            )
        return vector

    @property
    def ability_speed_corr(self) -> float:
        """Correlation between population ability and speed."""
        *_, covariance = self._validated_state()
        return float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))

    def _accuracy_probability_and_derivative(
        self,
        theta: NDArray[np.float64] | float,
        item_idx: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        discrimination, difficulty, guessing, *_ = self._validated_state()
        theta_values = self._person_vector(theta, "theta")
        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
            a = discrimination[item_idx]
            b = difficulty[item_idx]
            logistic_probability = _safe_sigmoid(a * (theta_values - b))
            derivative = a * logistic_probability * (1.0 - logistic_probability)
            if guessing is None:
                return logistic_probability, derivative
            c = guessing[item_idx]
            return c + (1.0 - c) * logistic_probability, (1.0 - c) * derivative

        a = discrimination[None, :]
        b = difficulty[None, :]
        logistic_probability = _safe_sigmoid(a * (theta_values[:, None] - b))
        derivative = a * logistic_probability * (1.0 - logistic_probability)
        if guessing is None:
            return logistic_probability, derivative
        c = guessing[None, :]
        return c + (1.0 - c) * logistic_probability, (1.0 - c) * derivative

    def accuracy_probability(
        self,
        theta: NDArray[np.float64] | float,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute ``P(X=1 | theta)`` for one item or all items."""
        probability, _ = self._accuracy_probability_and_derivative(theta, item_idx)
        return probability

    def accuracy_information(
        self,
        theta: NDArray[np.float64] | float,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute exact Fisher information about ability from accuracy."""
        probability, derivative = self._accuracy_probability_and_derivative(
            theta, item_idx
        )
        denominator = probability * (1.0 - probability)
        return np.divide(
            derivative**2,
            denominator,
            out=np.zeros_like(probability),
            where=denominator > 0.0,
        )

    def rt_log_density(
        self,
        log_rt: NDArray[np.float64] | float,
        tau: NDArray[np.float64] | float,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute the Gaussian log-density of log response times."""
        *_, time_intensity, time_discrimination, _, _ = self._validated_state()
        tau_values = self._person_vector(tau, "tau")

        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
            log_rt_values = np.asarray(log_rt, dtype=np.float64)
            if log_rt_values.ndim == 0:
                log_rt_values = np.full(tau_values.size, float(log_rt_values))
            log_rt_values = self._person_vector(
                log_rt_values, "log_rt", tau_values.size
            )
            alpha = time_discrimination[item_idx]
            residual = log_rt_values - (time_intensity[item_idx] - tau_values)
            return np.log(alpha) - 0.5 * _LOG_2PI - 0.5 * (alpha * residual) ** 2

        log_rt_values = np.asarray(log_rt, dtype=np.float64)
        if log_rt_values.ndim == 1 and self.n_items == 1:
            log_rt_values = log_rt_values[:, None]
        expected_shape = (tau_values.size, self.n_items)
        if log_rt_values.shape != expected_shape:
            raise MirtValidationError(
                f"log_rt must have shape {expected_shape}",
                parameter="log_rt",
                value=log_rt_values.shape,
                expected=str(expected_shape),
            )
        if not np.all(np.isfinite(log_rt_values)):
            raise MirtValidationError(
                "log_rt must contain only finite values",
                parameter="log_rt",
                value=log_rt_values,
                expected="finite values",
            )
        alpha = time_discrimination[None, :]
        residual = log_rt_values - (time_intensity[None, :] - tau_values[:, None])
        return np.log(alpha) - 0.5 * _LOG_2PI - 0.5 * (alpha * residual) ** 2

    def response_time_information(
        self,
        item_idx: int | None = None,
    ) -> NDArray[np.float64] | np.float64:
        """Return Fisher information about speed from each timing observation."""
        *_, time_discrimination, _, _ = self._validated_state()
        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
            return np.float64(time_discrimination[item_idx] ** 2)
        return time_discrimination**2

    def expected_response_time(
        self,
        tau: NDArray[np.float64] | float,
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute mean response time on the original time scale."""
        *_, time_intensity, time_discrimination, _, _ = self._validated_state()
        tau_values = self._person_vector(tau, "tau")
        if item_idx is not None:
            item_idx = self._validate_item_idx(item_idx)
            log_mean = (
                time_intensity[item_idx]
                - tau_values
                + 0.5 / time_discrimination[item_idx] ** 2
            )
            return np.exp(log_mean)
        log_mean = (
            time_intensity[None, :]
            - tau_values[:, None]
            + 0.5 / time_discrimination[None, :] ** 2
        )
        return np.exp(log_mean)

    def joint_log_likelihood(
        self,
        responses: NDArray[np.int_] | NDArray[np.float64],
        log_rt: NDArray[np.float64],
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute conditional accuracy-plus-time log likelihood per person.

        Negative or ``NaN`` responses are treated as missing.  ``NaN`` log
        response times are missing independently, so a timing observation still
        contributes when its corresponding accuracy response is unavailable.
        """
        response_values = np.asarray(responses)
        log_rt_values = np.asarray(log_rt, dtype=np.float64)
        if response_values.ndim != 2 or response_values.shape[1:] != (self.n_items,):
            raise MirtValidationError(
                f"responses must have shape (n_persons, {self.n_items})",
                parameter="responses",
                value=response_values.shape,
                expected=f"(n_persons, {self.n_items})",
            )
        if response_values.shape[0] == 0:
            raise MirtValidationError(
                "responses must contain at least one person",
                parameter="responses",
                value=response_values.shape,
                expected="at least one row",
            )
        if log_rt_values.shape != response_values.shape:
            raise MirtValidationError(
                "log_rt must have the same shape as responses",
                parameter="log_rt",
                value=log_rt_values.shape,
                expected=str(response_values.shape),
            )

        n_persons = response_values.shape[0]
        theta_values = self._person_vector(theta, "theta", n_persons)
        tau_values = self._person_vector(tau, "tau", n_persons)

        try:
            response_numeric = response_values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "responses must be numeric",
                parameter="responses",
                value=response_values,
                expected="0, 1, or a missing value",
            ) from exc
        response_finite = np.isfinite(response_numeric)
        response_missing = np.isnan(response_numeric) | (
            response_finite & (response_numeric < 0.0)
        )
        response_observed = response_finite & (response_numeric >= 0.0)
        invalid_response = (~response_missing & ~response_observed) | (
            response_observed & (response_numeric != 0.0) & (response_numeric != 1.0)
        )
        if np.any(invalid_response):
            raise MirtValidationError(
                "observed responses must be 0 or 1",
                parameter="responses",
                value=response_numeric[invalid_response],
                expected="0, 1, or a negative/NaN missing value",
            )

        if np.any(np.isinf(log_rt_values)):
            raise MirtValidationError(
                "log_rt may contain finite values or NaN for missingness",
                parameter="log_rt",
                value=log_rt_values,
                expected="finite values or NaN",
            )

        probability = np.clip(
            self.accuracy_probability(theta_values),
            PROB_EPSILON,
            1.0 - PROB_EPSILON,
        )
        observed_values = np.where(response_observed, response_numeric, 0.0)
        accuracy_log_likelihood = np.where(
            response_observed,
            observed_values * np.log(probability)
            + (1.0 - observed_values) * np.log1p(-probability),
            0.0,
        )

        timing_observed = ~np.isnan(log_rt_values)
        timing_values = np.where(timing_observed, log_rt_values, 0.0)
        timing_log_likelihood = self.rt_log_density(timing_values, tau_values)
        timing_log_likelihood = np.where(timing_observed, timing_log_likelihood, 0.0)
        return np.sum(accuracy_log_likelihood + timing_log_likelihood, axis=1)

    def simulate(
        self,
        n_persons: int,
        theta: NDArray[np.float64] | None = None,
        tau: NDArray[np.float64] | None = None,
        seed: int | None = None,
    ) -> tuple[
        NDArray[np.int_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Simulate responses, response times, ability, and speed.

        If exactly one person parameter is supplied, the other is drawn from
        its conditional normal distribution under ``ability_speed_cov``.
        """
        if isinstance(n_persons, bool) or not isinstance(n_persons, (int, np.integer)):
            raise MirtValidationError(
                "n_persons must be an integer",
                parameter="n_persons",
                value=n_persons,
                expected="positive integer",
            )
        n_persons = int(n_persons)
        if n_persons <= 0:
            raise MirtValidationError(
                "n_persons must be positive",
                parameter="n_persons",
                value=n_persons,
                expected="> 0",
            )

        (
            _,
            _,
            _,
            time_intensity,
            time_discrimination,
            mean,
            covariance,
        ) = self._validated_state()
        theta_values = (
            None
            if theta is None
            else self._person_vector(theta, "theta", n_persons).copy()
        )
        tau_values = (
            None if tau is None else self._person_vector(tau, "tau", n_persons).copy()
        )
        rng = np.random.default_rng(seed)

        if theta_values is None and tau_values is None:
            person_parameters = rng.multivariate_normal(
                mean, covariance, size=n_persons
            )
            theta_values = person_parameters[:, 0]
            tau_values = person_parameters[:, 1]
        elif theta_values is not None and tau_values is None:
            regression = covariance[1, 0] / covariance[0, 0]
            conditional_mean = mean[1] + regression * (theta_values - mean[0])
            conditional_variance = (
                covariance[1, 1] - covariance[1, 0] ** 2 / covariance[0, 0]
            )
            tau_values = rng.normal(conditional_mean, np.sqrt(conditional_variance))
        elif theta_values is None and tau_values is not None:
            regression = covariance[0, 1] / covariance[1, 1]
            conditional_mean = mean[0] + regression * (tau_values - mean[1])
            conditional_variance = (
                covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1]
            )
            theta_values = rng.normal(conditional_mean, np.sqrt(conditional_variance))

        assert theta_values is not None
        assert tau_values is not None
        probability = self.accuracy_probability(theta_values)
        responses = (rng.random((n_persons, self.n_items)) < probability).astype(
            np.int32
        )
        log_rt = rng.normal(
            time_intensity[None, :] - tau_values[:, None],
            1.0 / time_discrimination[None, :],
        )
        response_times = np.exp(log_rt)
        if not np.all(np.isfinite(response_times)):
            raise MirtValidationError(
                "simulated response times overflowed the finite range",
                parameter="response_times",
                value=response_times,
                expected="finite positive values",
            )
        return responses, response_times, theta_values, tau_values

    def summary(self) -> str:
        """Generate a model summary."""
        (
            discrimination,
            difficulty,
            guessing,
            time_intensity,
            time_discrimination,
            mean,
            covariance,
        ) = self._validated_state()
        lines = []
        width = 70

        lines.append("=" * width)
        lines.append(f"{'Response Time Model Summary':^{width}}")
        lines.append("=" * width)
        lines.append(f"Accuracy Model:     {self.accuracy_model}")
        lines.append(f"Number of Items:    {self.n_items}")
        lines.append(f"Speed-Ability Corr: {self.ability_speed_corr:.4f}")
        lines.append("-" * width)
        lines.append("\nPopulation Parameters:")
        lines.append(f"  Mean Ability (θ): {mean[0]:.4f}")
        lines.append(f"  Mean Speed (τ):   {mean[1]:.4f}")
        lines.append(f"  Var(θ):           {covariance[0, 0]:.4f}")
        lines.append(f"  Var(τ):           {covariance[1, 1]:.4f}")
        lines.append(f"  Cov(θ, τ):        {covariance[0, 1]:.4f}")
        lines.append("\nItem Parameters:")
        header = f"{'Item':<12} {'a':>8} {'b':>8}"
        if guessing is not None:
            header += f" {'c':>8}"
        header += f" {'α':>8} {'β':>8}"
        lines.append(header)
        lines.append("-" * width)

        assert self.item_names is not None
        for item_idx, item_name in enumerate(self.item_names):
            row = (
                f"{item_name:<12} {discrimination[item_idx]:>8.3f} "
                f"{difficulty[item_idx]:>8.3f}"
            )
            if guessing is not None:
                row += f" {guessing[item_idx]:>8.3f}"
            row += (
                f" {time_discrimination[item_idx]:>8.3f} "
                f"{time_intensity[item_idx]:>8.3f}"
            )
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class ResponseTimeResult:
    """Result from response-time model estimation."""

    model: ResponseTimeModel
    theta_estimates: NDArray[np.float64]
    tau_estimates: NDArray[np.float64]
    theta_se: NDArray[np.float64]
    tau_se: NDArray[np.float64]
    chains: dict[str, NDArray[np.float64]] | None
    log_likelihood: float
    dic: float
    waic: float
    rhat: dict[str, float]
    ess: dict[str, float]
    n_iterations: int
    n_chains: int
    converged: bool

    def summary(self) -> str:
        """Generate an estimation summary."""
        lines = []
        width = 80

        lines.append("=" * width)
        lines.append(f"{'Response Time Model Results':^{width}}")
        lines.append("=" * width)
        lines.append(
            f"Accuracy Model:     {self.model.accuracy_model:<20} "
            f"Log-Likelihood:    {self.log_likelihood:>12.4f}"
        )
        lines.append(
            f"No. Items:          {self.model.n_items:<20} "
            f"DIC:               {self.dic:>12.4f}"
        )
        lines.append(
            f"Iterations:         {self.n_iterations:<20} "
            f"WAIC:              {self.waic:>12.4f}"
        )
        lines.append(
            f"Chains:             {self.n_chains:<20} "
            f"Converged:         {str(self.converged):>12}"
        )
        lines.append("-" * width)
        lines.append("\nConvergence Diagnostics:")
        for parameter, rhat in self.rhat.items():
            ess = self.ess.get(parameter, np.nan)
            lines.append(f"  {parameter}: Rhat = {rhat:.4f}, ESS = {ess:.0f}")
        lines.append("\nPopulation Parameters:")
        lines.append(
            f"  Speed-Ability Correlation: {self.model.ability_speed_corr:.4f}"
        )
        lines.append("=" * width)
        return "\n".join(lines)

    def person_summary(self, n_show: int = 10) -> str:
        """Summarize person parameter estimates."""
        if isinstance(n_show, bool) or not isinstance(n_show, (int, np.integer)):
            raise MirtValidationError(
                "n_show must be an integer",
                parameter="n_show",
                value=n_show,
                expected="non-negative integer",
            )
        if n_show < 0:
            raise MirtValidationError(
                "n_show must be non-negative",
                parameter="n_show",
                value=n_show,
                expected=">= 0",
            )

        lines = [f"Person Parameter Estimates (first {n_show} persons):"]
        lines.append(f"{'Person':<10} {'θ':>10} {'SE(θ)':>10} {'τ':>10} {'SE(τ)':>10}")
        lines.append("-" * 50)
        for person_idx in range(min(n_show, len(self.theta_estimates))):
            lines.append(
                f"{person_idx:<10} {self.theta_estimates[person_idx]:>10.4f} "
                f"{self.theta_se[person_idx]:>10.4f} "
                f"{self.tau_estimates[person_idx]:>10.4f} "
                f"{self.tau_se[person_idx]:>10.4f}"
            )
        return "\n".join(lines)
