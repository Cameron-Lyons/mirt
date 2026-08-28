from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from mirt._backend_config import should_use_rust
from mirt._categorical import sample_categorical_rows
from mirt._core import sigmoid
from mirt.backends.rust.dynamic import (
    bkt_backward,
    bkt_forward,
    bkt_forward_backward_batch,
    bkt_viterbi,
)
from mirt.constants import PROB_EPSILON
from mirt.models.state_space import StateSpaceIRT as StateSpaceIRT
from mirt.utils.numeric import logsumexp

_LONGITUDINAL_MAX_PROBABILITY_VALUES = 1_000_000
_GROWTH_MIXTURE_MAX_RANDOM_VALUES = 1_000_000


@dataclass(frozen=True)
class _GrowthCovariance:
    """Low-rank representation of a marginal growth covariance matrix."""

    residual_variance: float
    log_determinant: float
    orthonormal_basis: NDArray[np.float64] | None = None
    covariance_factor: NDArray[np.float64] | None = None

    @classmethod
    def from_time_values(
        cls,
        time_values: NDArray[np.float64],
        intercept_variance: float,
        slope_variance: float,
        residual_variance: float,
    ) -> _GrowthCovariance:
        """Factor the covariance without constructing its dense matrix."""
        random_effect_columns = []
        if intercept_variance > 0.0:
            random_effect_columns.append(
                np.full(time_values.size, np.sqrt(intercept_variance))
            )
        if slope_variance > 0.0:
            random_effect_columns.append(time_values * np.sqrt(slope_variance))

        if not random_effect_columns:
            return cls(
                residual_variance=residual_variance,
                log_determinant=time_values.size * np.log(residual_variance),
            )

        random_effect_basis = np.column_stack(random_effect_columns)
        orthonormal_basis, triangular_basis = np.linalg.qr(
            random_effect_basis,
            mode="reduced",
        )
        basis_covariance = (
            residual_variance * np.eye(orthonormal_basis.shape[1])
            + triangular_basis @ triangular_basis.T
        )
        covariance_factor = np.linalg.cholesky(basis_covariance)
        log_determinant = (time_values.size - orthonormal_basis.shape[1]) * np.log(
            residual_variance
        ) + 2.0 * np.sum(np.log(np.diag(covariance_factor)))
        return cls(
            residual_variance=residual_variance,
            log_determinant=float(log_determinant),
            orthonormal_basis=orthonormal_basis,
            covariance_factor=covariance_factor,
        )

    def solve(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the inverse covariance to one or more column vectors."""
        if self.orthonormal_basis is None or self.covariance_factor is None:
            return values / self.residual_variance

        basis_coordinates = self.orthonormal_basis.T @ values
        orthogonal_values = (
            values - self.orthonormal_basis @ basis_coordinates
        ) / self.residual_variance
        solved_coordinates = np.linalg.solve(
            self.covariance_factor.T,
            np.linalg.solve(self.covariance_factor, basis_coordinates),
        )
        return orthogonal_values + self.orthonormal_basis @ solved_coordinates

    def quadratic_forms(
        self,
        residuals: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate row-wise covariance-weighted squared residuals."""
        if self.orthonormal_basis is None or self.covariance_factor is None:
            return np.einsum("ij,ij->i", residuals, residuals) / (
                self.residual_variance
            )

        basis_coordinates = residuals @ self.orthonormal_basis
        orthogonal_residuals = residuals - basis_coordinates @ self.orthonormal_basis.T
        whitened_coordinates = np.linalg.solve(
            self.covariance_factor,
            basis_coordinates.T,
        )
        return np.einsum(
            "ij,ij->i",
            orthogonal_residuals,
            orthogonal_residuals,
        ) / self.residual_variance + np.einsum(
            "ij,ij->j",
            whitened_coordinates,
            whitened_coordinates,
        )


@dataclass(frozen=True)
class _GrowthObservationPattern:
    """Rows sharing observed columns and one covariance factorization."""

    rows: NDArray[np.int_]
    columns: NDArray[np.int_]
    covariance: _GrowthCovariance


@dataclass(frozen=True, slots=True)
class BKTStepResult:
    """Prediction and mastery update from one skill opportunity."""

    response_probability: float
    response_log_likelihood: float
    residual: float
    standardized_residual: float
    updated_mastery: float
    next_mastery: float


@dataclass(frozen=True, slots=True)
class BKTBatchStepResult:
    """Vectorized predictions and mastery updates for one opportunity."""

    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    updated_mastery: NDArray[np.float64]
    next_mastery: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of learners represented by the result."""
        return int(self.updated_mastery.size)


@dataclass(frozen=True, slots=True)
class BKTPredictiveResult:
    """Causal predictions and mastery states for one response history.

    Every array has shape ``(n_trials,)``. Predicted mastery conditions only
    on earlier opportunities for the assigned skill, while updated mastery
    additionally conditions on the current response. Missing responses have
    zero log likelihood and ``numpy.nan`` residuals.
    """

    predicted_mastery: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    updated_mastery: NDArray[np.float64]
    next_mastery: NDArray[np.float64]

    @property
    def n_trials(self) -> int:
        """Number of trial opportunities represented by the result."""
        return int(self.predicted_mastery.size)

    @property
    def total_log_likelihood(self) -> float:
        """Sum of predictive log likelihoods across observed responses."""
        return float(np.sum(self.response_log_likelihoods))


@dataclass(frozen=True, slots=True)
class BKTBatchPredictiveResult:
    """Vectorized causal diagnostics for multiple response histories.

    Every array has shape ``(n_persons, n_trials)``. Missing responses have
    zero log likelihood and ``numpy.nan`` residuals.
    """

    predicted_mastery: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]
    response_log_likelihoods: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    updated_mastery: NDArray[np.float64]
    next_mastery: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of learners represented by the result."""
        return int(self.predicted_mastery.shape[0])

    @property
    def n_trials(self) -> int:
        """Number of trial opportunities per learner."""
        return int(self.predicted_mastery.shape[1])

    @property
    def total_log_likelihoods(self) -> NDArray[np.float64]:
        """Predictive log-likelihood total for each learner."""
        return np.sum(self.response_log_likelihoods, axis=1)


@dataclass(frozen=True, slots=True)
class BKTForecastResult:
    """Future mastery and response probabilities for one learner.

    Both arrays have shape ``(n_steps, n_skills)``. The first row describes
    the next opportunity represented by the supplied mastery priors.
    """

    mastery_probabilities: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]

    @property
    def n_steps(self) -> int:
        """Number of forecast opportunities per skill."""
        return int(self.mastery_probabilities.shape[0])

    @property
    def n_skills(self) -> int:
        """Number of modeled skills."""
        return int(self.mastery_probabilities.shape[1])


@dataclass(frozen=True, slots=True)
class BKTBatchForecastResult:
    """Vectorized future mastery and response probabilities.

    Both arrays have shape ``(n_persons, n_steps, n_skills)``. The first
    forecast row describes the next opportunity represented by each supplied
    mastery prior.
    """

    mastery_probabilities: NDArray[np.float64]
    response_probabilities: NDArray[np.float64]

    @property
    def n_persons(self) -> int:
        """Number of learners represented by the result."""
        return int(self.mastery_probabilities.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of forecast opportunities per skill."""
        return int(self.mastery_probabilities.shape[1])

    @property
    def n_skills(self) -> int:
        """Number of modeled skills."""
        return int(self.mastery_probabilities.shape[2])


@dataclass
class BKTModel:
    """Bayesian Knowledge Tracing model.

    Hidden Markov Model for learning with:
    - States: {not learned (0), learned (1)}
    - Learning rate: P(L_t | not L_{t-1})
    - Forgetting rate: P(not L_t | L_{t-1})
    - Slip: P(incorrect | learned)
    - Guess: P(correct | not learned)

    Parameters
    ----------
    n_skills : int
        Number of distinct skills
    skill_names : list[str], optional
        Names for each skill
    allow_forgetting : bool
        Whether to model forgetting (default False)
    use_rust : bool
        Use compiled inference kernels when available (default True)
    """

    n_skills: int
    skill_names: list[str] | None = None
    allow_forgetting: bool = False

    p_init: NDArray[np.float64] | None = None
    p_learn: NDArray[np.float64] | None = None
    p_forget: NDArray[np.float64] | None = None
    p_slip: NDArray[np.float64] | None = None
    p_guess: NDArray[np.float64] | None = None
    use_rust: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_skills, bool)
            or not isinstance(self.n_skills, (int, np.integer))
            or self.n_skills < 1
        ):
            raise ValueError("n_skills must be a positive integer")

        if self.skill_names is None:
            self.skill_names = [f"Skill_{i}" for i in range(self.n_skills)]
        else:
            self.skill_names = list(self.skill_names)
            if len(self.skill_names) != self.n_skills:
                raise ValueError("skill_names length must match n_skills")
            if any(not isinstance(name, str) or not name for name in self.skill_names):
                raise ValueError("skill_names must contain non-empty strings")
            if len(set(self.skill_names)) != len(self.skill_names):
                raise ValueError("skill_names must be unique")

        if self.p_init is None:
            self.p_init = np.full(self.n_skills, 0.3)
        if self.p_learn is None:
            self.p_learn = np.full(self.n_skills, 0.1)
        if self.p_forget is None:
            self.p_forget = (
                np.zeros(self.n_skills)
                if not self.allow_forgetting
                else np.full(self.n_skills, 0.01)
            )
        if self.p_slip is None:
            self.p_slip = np.full(self.n_skills, 0.1)
        if self.p_guess is None:
            self.p_guess = np.full(self.n_skills, 0.2)

        for parameter_name in (
            "p_init",
            "p_learn",
            "p_forget",
            "p_slip",
            "p_guess",
        ):
            values = np.asarray(getattr(self, parameter_name), dtype=np.float64)
            if values.shape != (self.n_skills,):
                raise ValueError(f"{parameter_name} must have shape ({self.n_skills},)")
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError(
                    f"{parameter_name} values must be finite and in [0, 1]"
                )
            setattr(self, parameter_name, values.copy())

        if not self.allow_forgetting and np.any(self.p_forget != 0):
            raise ValueError("p_forget must be zero when allow_forgetting is False")
        if not isinstance(self.use_rust, (bool, np.bool_)):
            raise TypeError("use_rust must be a boolean")
        self.use_rust = bool(self.use_rust)

    def _validate_sequence(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Validate and normalize a single observed trial sequence."""
        responses = np.asarray(responses)
        skill_assignments = np.asarray(skill_assignments)

        if responses.ndim != 1 or skill_assignments.ndim != 1:
            raise ValueError("responses and skill_assignments must be one-dimensional")
        if len(responses) == 0:
            raise ValueError("responses must contain at least one trial")
        if len(responses) != len(skill_assignments):
            raise ValueError("responses and skill_assignments must have equal length")
        if not np.issubdtype(responses.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if not np.all(np.isin(responses, (-1, 0, 1))):
            raise ValueError("responses must contain only -1, 0, or 1")
        if not np.issubdtype(skill_assignments.dtype, np.integer):
            raise ValueError("skill_assignments must contain integer values")
        if np.any((skill_assignments < 0) | (skill_assignments >= self.n_skills)):
            raise ValueError(f"skill_assignments must be in [0, {self.n_skills})")

        return (
            responses.astype(np.int32, copy=False),
            skill_assignments.astype(np.int32, copy=False),
        )

    def _validate_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Validate and normalize multiple observed trial sequences."""
        responses = np.asarray(responses)
        skill_assignments = np.asarray(skill_assignments)

        if responses.ndim != 2:
            raise ValueError("responses must have shape (n_persons, n_trials)")
        if responses.shape[1] == 0:
            raise ValueError("responses must contain at least one trial")
        if not np.issubdtype(responses.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if not np.all(np.isin(responses, (-1, 0, 1))):
            raise ValueError("responses must contain only -1, 0, or 1")

        if skill_assignments.ndim == 1:
            if len(skill_assignments) != responses.shape[1]:
                raise ValueError(
                    "shared skill_assignments length must match the number of trials"
                )
        elif skill_assignments.shape != responses.shape:
            raise ValueError(
                "skill_assignments must be one-dimensional or match responses"
            )
        if not np.issubdtype(skill_assignments.dtype, np.integer):
            raise ValueError("skill_assignments must contain integer values")
        if np.any((skill_assignments < 0) | (skill_assignments >= self.n_skills)):
            raise ValueError(f"skill_assignments must be in [0, {self.n_skills})")

        return (
            responses.astype(np.int32, copy=False),
            skill_assignments.astype(np.int32, copy=False),
        )

    def _validate_online_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: int | NDArray[np.int_],
        prior_mastery: float | NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float64]]:
        """Validate one opportunity for multiple learners."""
        response_values = np.asarray(responses)
        if response_values.ndim != 1 or response_values.size < 1:
            raise ValueError("responses must have shape (n_persons,)")
        if not np.issubdtype(response_values.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if not np.all(np.isin(response_values, (-1, 0, 1))):
            raise ValueError("responses must contain only -1, 0, or 1")

        raw_skills = np.asarray(skill_assignments)
        if not np.issubdtype(raw_skills.dtype, np.integer):
            raise ValueError("skill_assignments must contain integer values")
        if raw_skills.ndim == 0:
            skill_values = np.full(
                response_values.size,
                raw_skills.item(),
                dtype=np.int32,
            )
        elif raw_skills.shape == response_values.shape:
            skill_values = raw_skills.astype(np.int32, copy=False)
        else:
            raise ValueError("skill_assignments must be scalar or match responses")
        if np.any((skill_values < 0) | (skill_values >= self.n_skills)):
            raise ValueError(f"skill_assignments must be in [0, {self.n_skills})")

        if prior_mastery is None:
            mastery_values = self.p_init[skill_values].copy()
        else:
            raw_mastery = np.asarray(prior_mastery)
            if (
                np.issubdtype(raw_mastery.dtype, np.bool_)
                or np.issubdtype(raw_mastery.dtype, np.complexfloating)
                or not np.issubdtype(raw_mastery.dtype, np.number)
            ):
                raise ValueError("prior_mastery must contain values in [0, 1]")
            if raw_mastery.ndim == 0:
                mastery_values = np.full(
                    response_values.size,
                    raw_mastery.item(),
                    dtype=np.float64,
                )
            elif raw_mastery.shape == response_values.shape:
                mastery_values = raw_mastery.astype(np.float64, copy=True)
            else:
                raise ValueError("prior_mastery must be scalar or match responses")
            if not np.all(np.isfinite(mastery_values)) or np.any(
                (mastery_values < 0.0) | (mastery_values > 1.0)
            ):
                raise ValueError("prior_mastery must contain values in [0, 1]")

        return (
            response_values.astype(np.int32, copy=False),
            skill_values,
            mastery_values,
        )

    def _online_step_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        prior_mastery: NDArray[np.float64],
    ) -> BKTBatchStepResult:
        """Process validated one-opportunity inputs in one vectorized pass."""
        slip = self.p_slip[skill_assignments]
        guess = self.p_guess[skill_assignments]
        response_probabilities = (
            prior_mastery * (1.0 - slip) + (1.0 - prior_mastery) * guess
        )
        observed = responses >= 0
        correct = responses == 1
        response_likelihoods = np.where(
            correct,
            response_probabilities,
            np.where(observed, 1.0 - response_probabilities, 1.0),
        )
        response_log_likelihoods = np.where(
            observed,
            np.log(np.maximum(response_likelihoods, 1e-300)),
            0.0,
        )
        learned_likelihoods = np.where(
            correct,
            1.0 - slip,
            np.where(observed, slip, 1.0),
        )
        updated_mastery = np.divide(
            prior_mastery * learned_likelihoods,
            response_likelihoods,
            out=np.zeros_like(prior_mastery),
            where=response_likelihoods > 0.0,
        )
        updated_mastery = np.where(observed, updated_mastery, prior_mastery)
        residuals = np.where(
            observed,
            responses - response_probabilities,
            np.nan,
        )
        response_variances = np.maximum(
            response_probabilities * (1.0 - response_probabilities),
            PROB_EPSILON,
        )
        standardized_residuals = residuals / np.sqrt(response_variances)
        learn = self.p_learn[skill_assignments]
        forget = self.p_forget[skill_assignments]
        next_mastery = (
            updated_mastery * (1.0 - forget) + (1.0 - updated_mastery) * learn
        )
        return BKTBatchStepResult(
            response_probabilities=response_probabilities,
            response_log_likelihoods=response_log_likelihoods,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            updated_mastery=updated_mastery,
            next_mastery=next_mastery,
        )

    def online_step(
        self,
        response: int,
        skill_idx: int,
        *,
        prior_mastery: float | None = None,
    ) -> BKTStepResult:
        """Predict and update mastery for one learner and skill opportunity.

        Omitted prior mastery uses the modeled skill's initial probability.
        The returned next mastery is ready for that skill's next opportunity.
        """
        result = self.online_step_batch(
            np.asarray([response]),
            skill_idx,
            prior_mastery=prior_mastery,
        )
        return BKTStepResult(
            response_probability=float(result.response_probabilities[0]),
            response_log_likelihood=float(result.response_log_likelihoods[0]),
            residual=float(result.residuals[0]),
            standardized_residual=float(result.standardized_residuals[0]),
            updated_mastery=float(result.updated_mastery[0]),
            next_mastery=float(result.next_mastery[0]),
        )

    def online_step_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: int | NDArray[np.int_],
        *,
        prior_mastery: float | NDArray[np.float64] | None = None,
    ) -> BKTBatchStepResult:
        """Predict and update one skill opportunity for multiple learners.

        Skill assignments and prior mastery may be shared scalars or one value
        per learner. Omitted mastery uses each assigned skill's initial
        probability.
        """
        response_values, skill_values, mastery_values = self._validate_online_batch(
            responses,
            skill_assignments,
            prior_mastery,
        )
        return self._online_step_batch(
            response_values,
            skill_values,
            mastery_values,
        )

    def _predictive_diagnostics_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> BKTBatchPredictiveResult:
        """Build causal diagnostics from validated histories in one pass."""
        n_persons, n_trials = responses.shape
        skill_matrix = (
            np.broadcast_to(skill_assignments, responses.shape)
            if skill_assignments.ndim == 1
            else skill_assignments
        )
        retained_priors = np.broadcast_to(
            self.p_init,
            (n_persons, self.n_skills),
        ).copy()
        predicted_mastery = np.empty(responses.shape, dtype=np.float64)
        response_probabilities = np.empty_like(predicted_mastery)
        response_log_likelihoods = np.empty_like(predicted_mastery)
        residuals = np.empty_like(predicted_mastery)
        standardized_residuals = np.empty_like(predicted_mastery)
        updated_mastery = np.empty_like(predicted_mastery)
        next_mastery = np.empty_like(predicted_mastery)
        rows = np.arange(n_persons)

        for trial in range(n_trials):
            trial_skills = skill_matrix[:, trial]
            trial_priors = retained_priors[rows, trial_skills]
            step = self._online_step_batch(
                responses[:, trial],
                trial_skills,
                trial_priors,
            )
            predicted_mastery[:, trial] = trial_priors
            response_probabilities[:, trial] = step.response_probabilities
            response_log_likelihoods[:, trial] = step.response_log_likelihoods
            residuals[:, trial] = step.residuals
            standardized_residuals[:, trial] = step.standardized_residuals
            updated_mastery[:, trial] = step.updated_mastery
            next_mastery[:, trial] = step.next_mastery
            retained_priors[rows, trial_skills] = step.next_mastery

        return BKTBatchPredictiveResult(
            predicted_mastery=predicted_mastery,
            response_probabilities=response_probabilities,
            response_log_likelihoods=response_log_likelihoods,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            updated_mastery=updated_mastery,
            next_mastery=next_mastery,
        )

    def predictive_diagnostics(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> BKTPredictiveResult:
        """Return causal predictions and mastery states for one history.

        Parameters
        ----------
        responses : NDArray
            Integer response sequence with shape ``(n_trials,)``. Use ``-1``
            for a missing response.
        skill_assignments : NDArray
            Skill index for every trial.

        Returns
        -------
        BKTPredictiveResult
            Per-trial predictions, log likelihoods, residuals, and states.
        """
        response_values, skill_values = self._validate_sequence(
            responses,
            skill_assignments,
        )
        result = self._predictive_diagnostics_batch(
            response_values[None, :],
            skill_values,
        )
        return BKTPredictiveResult(
            predicted_mastery=result.predicted_mastery[0].copy(),
            response_probabilities=result.response_probabilities[0].copy(),
            response_log_likelihoods=result.response_log_likelihoods[0].copy(),
            residuals=result.residuals[0].copy(),
            standardized_residuals=result.standardized_residuals[0].copy(),
            updated_mastery=result.updated_mastery[0].copy(),
            next_mastery=result.next_mastery[0].copy(),
        )

    def predictive_diagnostics_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> BKTBatchPredictiveResult:
        """Return one-pass causal diagnostics for multiple histories.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_trials)``.
        skill_assignments : NDArray
            Shared trial-to-skill vector or a matrix matching ``responses``.

        Returns
        -------
        BKTBatchPredictiveResult
            Per-trial predictions, log likelihoods, residuals, and states.
        """
        response_values, skill_values = self._validate_batch(
            responses,
            skill_assignments,
        )
        return self._predictive_diagnostics_batch(response_values, skill_values)

    @staticmethod
    def _validated_forecast_steps(n_steps: int) -> int:
        """Return a validated positive forecast horizon."""
        if (
            isinstance(n_steps, (bool, np.bool_))
            or not isinstance(n_steps, (int, np.integer))
            or n_steps < 1
        ):
            raise ValueError("n_steps must be a positive integer")
        return int(n_steps)

    def _validated_mastery_priors(
        self,
        prior_mastery: NDArray[np.float64],
        *,
        batch: bool,
    ) -> NDArray[np.float64]:
        """Return validated next-opportunity mastery priors."""
        raw_mastery = np.asarray(prior_mastery)
        if batch:
            expected_shape = "(n_persons, n_skills)"
            shape_is_valid = (
                raw_mastery.ndim == 2
                and raw_mastery.shape[0] > 0
                and raw_mastery.shape[1] == self.n_skills
            )
        else:
            expected_shape = "(n_skills,)"
            shape_is_valid = raw_mastery.shape == (self.n_skills,)
        if not shape_is_valid:
            raise ValueError(f"prior_mastery must have shape {expected_shape}")
        if (
            np.issubdtype(raw_mastery.dtype, np.bool_)
            or np.issubdtype(raw_mastery.dtype, np.complexfloating)
            or not np.issubdtype(raw_mastery.dtype, np.number)
        ):
            raise ValueError("prior_mastery must contain values in [0, 1]")
        mastery_values = raw_mastery.astype(np.float64, copy=True)
        if not np.all(np.isfinite(mastery_values)) or np.any(
            (mastery_values < 0.0) | (mastery_values > 1.0)
        ):
            raise ValueError("prior_mastery must contain values in [0, 1]")
        return mastery_values

    def _forecast_from_priors_batch(
        self,
        prior_mastery: NDArray[np.float64],
        n_steps: int,
    ) -> BKTBatchForecastResult:
        """Forecast validated next-opportunity priors without Python loops."""
        transition_rate = self.p_learn + self.p_forget
        equilibrium = np.divide(
            self.p_learn,
            transition_rate,
            out=np.zeros_like(self.p_learn),
            where=transition_rate > 0.0,
        )
        transition_powers = np.power(
            (1.0 - transition_rate)[None, None, :],
            np.arange(n_steps, dtype=np.int64)[None, :, None],
        )
        mastery_probabilities = (
            equilibrium[None, None, :]
            + (prior_mastery[:, None, :] - equilibrium[None, None, :])
            * transition_powers
        )
        np.clip(mastery_probabilities, 0.0, 1.0, out=mastery_probabilities)
        response_probabilities = (
            mastery_probabilities * (1.0 - self.p_slip[None, None, :])
            + (1.0 - mastery_probabilities) * self.p_guess[None, None, :]
        )
        return BKTBatchForecastResult(
            mastery_probabilities=mastery_probabilities,
            response_probabilities=response_probabilities,
        )

    def forecast_from_priors(
        self,
        prior_mastery: NDArray[np.float64],
        n_steps: int,
    ) -> BKTForecastResult:
        """Forecast every skill from one learner's next-opportunity priors.

        Forecasts are unconditional on unknown future responses. Each step is
        one additional opportunity for every modeled skill.

        Parameters
        ----------
        prior_mastery : NDArray
            Next-opportunity mastery probabilities with shape ``(n_skills,)``.
        n_steps : int
            Number of future opportunities to forecast for every skill.

        Returns
        -------
        BKTForecastResult
            Mastery and success probabilities with shape
            ``(n_steps, n_skills)``.
        """
        n_steps = self._validated_forecast_steps(n_steps)
        mastery_values = self._validated_mastery_priors(
            prior_mastery,
            batch=False,
        )
        result = self._forecast_from_priors_batch(mastery_values[None, :], n_steps)
        return BKTForecastResult(
            mastery_probabilities=result.mastery_probabilities[0].copy(),
            response_probabilities=result.response_probabilities[0].copy(),
        )

    def forecast_from_priors_batch(
        self,
        prior_mastery: NDArray[np.float64],
        n_steps: int,
    ) -> BKTBatchForecastResult:
        """Forecast every skill for multiple learners from retained priors.

        Parameters
        ----------
        prior_mastery : NDArray
            Next-opportunity mastery probabilities with shape
            ``(n_persons, n_skills)``.
        n_steps : int
            Number of future opportunities to forecast for every skill.

        Returns
        -------
        BKTBatchForecastResult
            Mastery and success probabilities with shape
            ``(n_persons, n_steps, n_skills)``.
        """
        n_steps = self._validated_forecast_steps(n_steps)
        mastery_values = self._validated_mastery_priors(
            prior_mastery,
            batch=True,
        )
        return self._forecast_from_priors_batch(mastery_values, n_steps)

    def _can_use_native_inference(self) -> bool:
        """Return whether compiled kernels preserve this model's semantics."""
        if not should_use_rust(self.use_rust):
            return False
        return bool(
            np.all(
                (self.p_slip > PROB_EPSILON)
                & (self.p_slip < 1.0 - PROB_EPSILON)
                & (self.p_guess > PROB_EPSILON)
                & (self.p_guess < 1.0 - PROB_EPSILON)
            )
        )

    def _emission_pair(
        self,
        response: int,
        skill_idx: int,
    ) -> NDArray[np.float64]:
        """Return emission probabilities for unlearned and learned states."""
        if response < 0:
            return np.ones(2, dtype=np.float64)
        if response == 1:
            return np.array([self.p_guess[skill_idx], 1.0 - self.p_slip[skill_idx]])
        return np.array([1.0 - self.p_guess[skill_idx], self.p_slip[skill_idx]])

    def _skill_trials(
        self, skill_assignments: NDArray[np.int_]
    ) -> list[NDArray[np.int_]]:
        """Return chronological trial indices for every modeled skill."""
        return [
            np.flatnonzero(skill_assignments == skill_idx)
            for skill_idx in range(self.n_skills)
        ]

    def transition_matrix(self, skill_idx: int) -> NDArray[np.float64]:
        """Get transition matrix for a skill.

        Returns 2x2 matrix where T[i,j] = P(state_t = j | state_{t-1} = i)
        """
        p_l = self.p_learn[skill_idx]
        p_f = self.p_forget[skill_idx]

        return np.array(
            [
                [1 - p_l, p_l],
                [p_f, 1 - p_f],
            ]
        )

    def emission_probability(
        self,
        response: int,
        learned: int,
        skill_idx: int,
    ) -> float:
        """Compute P(response | learned state)."""
        if response not in (-1, 0, 1):
            raise ValueError("response must be -1, 0, or 1")
        if learned not in (0, 1):
            raise ValueError("learned must be 0 or 1")
        if skill_idx < 0 or skill_idx >= self.n_skills:
            raise IndexError(f"skill_idx must be in [0, {self.n_skills})")
        return float(self._emission_pair(response, skill_idx)[learned])

    def forward(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forward algorithm for a single person.

        Parameters
        ----------
        responses : NDArray
            Response sequence (n_trials,)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)

        Returns
        -------
        tuple
            (alpha, scaling) where alpha[t, s] = P(L_t = s | X_1:t)
        """
        responses, skill_assignments = self._validate_sequence(
            responses, skill_assignments
        )

        if self._can_use_native_inference():
            try:
                result = bkt_forward(
                    responses,
                    skill_assignments,
                    self.p_init,
                    self.p_learn,
                    self.p_forget,
                    self.p_slip,
                    self.p_guess,
                )
                if result is not None:
                    alpha = np.asarray(result[0], dtype=np.float64)
                    scaling = np.asarray(result[1], dtype=np.float64)
                    if (
                        alpha.shape == (len(responses), 2)
                        and scaling.shape == (len(responses),)
                        and np.all(np.isfinite(alpha))
                        and np.all(alpha >= 0.0)
                        and np.all(np.isfinite(scaling))
                        and np.all(scaling >= 0.0)
                    ):
                        return alpha, scaling
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        return self._forward_python(responses, skill_assignments)

    def _forward_python(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run the Python forward fallback for validated inputs."""
        n_trials = len(responses)

        alpha = np.zeros((n_trials, 2))
        scaling = np.zeros(n_trials)

        for skill_idx, trial_indices in enumerate(
            self._skill_trials(skill_assignments)
        ):
            if len(trial_indices) == 0:
                continue

            first_trial = int(trial_indices[0])
            p_0 = self.p_init[skill_idx]
            alpha[first_trial] = np.array([1.0 - p_0, p_0]) * self._emission_pair(
                int(responses[first_trial]), skill_idx
            )
            scaling[first_trial] = np.sum(alpha[first_trial])
            if scaling[first_trial] > 0:
                alpha[first_trial] /= scaling[first_trial]

            transition = self.transition_matrix(skill_idx)
            for previous_trial, trial in zip(trial_indices[:-1], trial_indices[1:]):
                alpha[trial] = (
                    alpha[previous_trial] @ transition
                ) * self._emission_pair(int(responses[trial]), skill_idx)
                scaling[trial] = np.sum(alpha[trial])
                if scaling[trial] > 0:
                    alpha[trial] /= scaling[trial]

        return alpha, scaling

    def backward(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        scaling: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Backward algorithm for a single person.

        Parameters
        ----------
        responses : NDArray
            Response sequence (n_trials,)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)
        scaling : NDArray
            Scaling factors from forward pass (n_trials,)

        Returns
        -------
        NDArray
            beta[t, s] = P(X_{t+1:T} | L_t = s) (scaled)
        """
        responses, skill_assignments = self._validate_sequence(
            responses, skill_assignments
        )
        scaling = np.asarray(scaling, dtype=np.float64)
        n_trials = len(responses)
        if scaling.shape != (n_trials,):
            raise ValueError(f"scaling must have shape ({n_trials},)")
        if not np.all(np.isfinite(scaling)) or np.any(scaling < 0):
            raise ValueError("scaling values must be finite and non-negative")

        if self._can_use_native_inference():
            try:
                result = bkt_backward(
                    responses,
                    skill_assignments,
                    scaling,
                    self.p_learn,
                    self.p_forget,
                    self.p_slip,
                    self.p_guess,
                )
                if result is not None:
                    beta = np.asarray(result, dtype=np.float64)
                    if (
                        beta.shape == (n_trials, 2)
                        and np.all(np.isfinite(beta))
                        and np.all(beta >= 0.0)
                    ):
                        return beta
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        return self._backward_python(responses, skill_assignments, scaling)

    def _backward_python(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        scaling: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Run the Python backward fallback for validated inputs."""
        n_trials = len(responses)
        beta = np.zeros((n_trials, 2))

        for skill_idx, trial_indices in enumerate(
            self._skill_trials(skill_assignments)
        ):
            if len(trial_indices) == 0:
                continue
            beta[trial_indices[-1]] = 1.0
            transition = self.transition_matrix(skill_idx)

            for trial, next_trial in zip(trial_indices[-2::-1], trial_indices[:0:-1]):
                emission = self._emission_pair(int(responses[next_trial]), skill_idx)
                beta[trial] = transition @ (emission * beta[next_trial])
                if scaling[next_trial] > 0:
                    beta[trial] /= scaling[next_trial]

        return beta

    def forward_backward(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], float]:
        """Run forward-backward algorithm.

        Parameters
        ----------
        responses : NDArray
            Response sequence (n_trials,)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)

        Returns
        -------
        tuple
            (gamma, log_likelihood) where gamma[t, s] = P(L_t = s | X_1:T)
        """
        responses, skill_assignments = self._validate_sequence(
            responses, skill_assignments
        )
        native = self._native_forward_backward_batch(
            responses[None, :], skill_assignments
        )
        if native is not None:
            gamma, log_likelihoods = native
            return gamma[0], float(log_likelihoods[0])

        return self._forward_backward_python(responses, skill_assignments)

    def _forward_backward_python(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], float]:
        """Run the Python forward-backward fallback for validated inputs."""
        alpha, scaling = self._forward_python(responses, skill_assignments)
        beta = self._backward_python(responses, skill_assignments, scaling)

        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma = gamma / gamma_sum

        log_likelihood = np.sum(np.log(scaling + 1e-300))

        return gamma, log_likelihood

    def _native_forward_backward_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Run and validate compiled batch smoothing for a shared skill layout."""
        if not self._can_use_native_inference():
            return None

        try:
            result = bkt_forward_backward_batch(
                responses,
                skill_assignments,
                self.p_init,
                self.p_learn,
                self.p_forget,
                self.p_slip,
                self.p_guess,
            )
            if result is None:
                return None
            learned = np.asarray(result[0], dtype=np.float64)
            log_likelihoods = np.asarray(result[1], dtype=np.float64)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

        if (
            learned.shape != responses.shape
            or log_likelihoods.shape != (responses.shape[0],)
            or not np.all(np.isfinite(learned))
            or np.any((learned < 0.0) | (learned > 1.0))
            or not np.all(np.isfinite(log_likelihoods))
        ):
            return None

        gamma = np.empty((*responses.shape, 2), dtype=np.float64)
        gamma[..., 1] = learned
        gamma[..., 0] = 1.0 - learned
        return gamma, log_likelihoods

    def forward_backward_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run forward-backward inference for multiple persons.

        Parameters
        ----------
        responses : NDArray
            Response matrix with shape ``(n_persons, n_trials)``.
        skill_assignments : NDArray
            Shared trial-to-skill vector or a matrix matching ``responses``.

        Returns
        -------
        tuple
            ``(gamma, log_likelihoods)`` where ``gamma`` has shape
            ``(n_persons, n_trials, 2)``.
        """
        responses, skill_assignments = self._validate_batch(
            responses, skill_assignments
        )
        return self._forward_backward_batch_validated(responses, skill_assignments)

    def _forward_backward_batch_validated(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run batch smoothing for validated inputs."""
        if skill_assignments.ndim == 1:
            native = self._native_forward_backward_batch(responses, skill_assignments)
            if native is not None:
                return native

        gamma = np.empty((*responses.shape, 2), dtype=np.float64)
        log_likelihoods = np.empty(responses.shape[0], dtype=np.float64)
        for person_idx, person_responses in enumerate(responses):
            person_skills = (
                skill_assignments
                if skill_assignments.ndim == 1
                else skill_assignments[person_idx]
            )
            gamma[person_idx], log_likelihoods[person_idx] = (
                self._forward_backward_python(person_responses, person_skills)
            )
        return gamma, log_likelihoods

    def viterbi(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.int_]:
        """Find most likely state sequence via Viterbi algorithm.

        Parameters
        ----------
        responses : NDArray
            Response sequence (n_trials,)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)

        Returns
        -------
        NDArray
            Most likely state sequence (n_trials,)
        """
        responses, skill_assignments = self._validate_sequence(
            responses, skill_assignments
        )

        if self._can_use_native_inference():
            try:
                result = bkt_viterbi(
                    responses,
                    skill_assignments,
                    self.p_init,
                    self.p_learn,
                    self.p_forget,
                    self.p_slip,
                    self.p_guess,
                )
                if result is not None:
                    path = np.asarray(result)
                    if (
                        path.shape == responses.shape
                        and np.issubdtype(path.dtype, np.integer)
                        and np.all((path == 0) | (path == 1))
                    ):
                        return path.astype(np.int_, copy=False)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        return self._viterbi_python(responses, skill_assignments)

    def _viterbi_python(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.int_]:
        """Run the Python Viterbi fallback for validated inputs."""
        n_trials = len(responses)

        delta = np.zeros((n_trials, 2))
        psi = np.zeros((n_trials, 2), dtype=int)

        path = np.zeros(n_trials, dtype=int)

        for skill_idx, trial_indices in enumerate(
            self._skill_trials(skill_assignments)
        ):
            if len(trial_indices) == 0:
                continue
            first_trial = int(trial_indices[0])
            p_0 = self.p_init[skill_idx]
            prior = np.array([1.0 - p_0, p_0])
            delta[first_trial] = np.log(prior + 1e-300) + np.log(
                self._emission_pair(int(responses[first_trial]), skill_idx) + 1e-300
            )

            transition = self.transition_matrix(skill_idx)
            for previous_trial, trial in zip(trial_indices[:-1], trial_indices[1:]):
                for state in range(2):
                    candidates = delta[previous_trial] + np.log(
                        transition[:, state] + 1e-300
                    )
                    psi[trial, state] = int(np.argmax(candidates))
                    delta[trial, state] = candidates[psi[trial, state]]
                delta[trial] += np.log(
                    self._emission_pair(int(responses[trial]), skill_idx) + 1e-300
                )

            path[trial_indices[-1]] = int(np.argmax(delta[trial_indices[-1]]))
            for previous_trial, trial in zip(
                trial_indices[-2::-1], trial_indices[:0:-1]
            ):
                path[previous_trial] = psi[trial, path[trial]]

        return path

    def predict_mastery(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> float:
        """Predict current mastery probability after observing responses.

        Parameters
        ----------
        responses : NDArray
            Response sequence (n_trials,)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)

        Returns
        -------
        float
            P(learned) at final time point
        """
        gamma, _ = self.forward_backward(responses, skill_assignments)
        return float(gamma[-1, 1])

    def _latest_mastery_batch(
        self,
        learned_probabilities: NDArray[np.float64],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Extract each skill's last posterior from validated inference output."""
        n_persons, n_trials = learned_probabilities.shape
        mastery = np.broadcast_to(self.p_init, (n_persons, self.n_skills)).copy()

        if skill_assignments.ndim == 1:
            for skill_idx, trial_indices in enumerate(
                self._skill_trials(skill_assignments)
            ):
                if len(trial_indices) > 0:
                    mastery[:, skill_idx] = learned_probabilities[:, trial_indices[-1]]
            return mastery

        for skill_idx in range(self.n_skills):
            matches = skill_assignments == skill_idx
            observed = np.any(matches, axis=1)
            if not np.any(observed):
                continue
            rows = np.flatnonzero(observed)
            last_trials = n_trials - 1 - np.argmax(matches[:, ::-1], axis=1)
            mastery[rows, skill_idx] = learned_probabilities[
                rows,
                last_trials[rows],
            ]
        return mastery

    def _next_mastery_priors_from_latest(
        self,
        latest_mastery: NDArray[np.float64],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Advance observed skills to their next-opportunity priors."""
        transitioned = (
            latest_mastery * (1.0 - self.p_forget)
            + (1.0 - latest_mastery) * self.p_learn
        )
        if skill_assignments.ndim == 1:
            observed = np.zeros(self.n_skills, dtype=bool)
            observed[skill_assignments] = True
        else:
            observed = np.zeros_like(latest_mastery, dtype=bool)
            rows = np.arange(skill_assignments.shape[0])[:, None]
            observed[rows, skill_assignments] = True
        return np.where(observed, transitioned, latest_mastery)

    def predict_mastery_by_skill(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Return the latest mastery probability for every modeled skill.

        Skills without observations retain their initial mastery probability.
        Missing responses advance no evidence but retain their place in the
        per-skill opportunity sequence.
        """
        responses, skill_assignments = self._validate_sequence(
            responses, skill_assignments
        )
        gamma, _ = self._forward_backward_batch_validated(
            responses[None, :],
            skill_assignments,
        )
        return self._latest_mastery_batch(gamma[..., 1], skill_assignments)[0]

    def predict_mastery_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Return per-skill mastery probabilities for multiple persons.

        ``skill_assignments`` may be a shared one-dimensional trial layout or a
        person-specific matrix with the same shape as ``responses``.
        """
        responses, skill_assignments = self._validate_batch(
            responses, skill_assignments
        )
        gamma, _ = self._forward_backward_batch_validated(responses, skill_assignments)
        return self._latest_mastery_batch(gamma[..., 1], skill_assignments)

    def next_mastery_priors(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Return every skill's prior for its next opportunity.

        Skills absent from the history retain their initial mastery
        probabilities. Observed skills advance once through their configured
        learning and forgetting transition after their final opportunity.

        Parameters
        ----------
        responses : NDArray
            Integer response sequence with shape ``(n_trials,)``. Use ``-1``
            for a missing response.
        skill_assignments : NDArray
            Skill index for every historical trial.

        Returns
        -------
        NDArray
            Next-opportunity priors with shape ``(n_skills,)``.
        """
        responses, skill_assignments = self._validate_sequence(
            responses,
            skill_assignments,
        )
        gamma, _ = self._forward_backward_batch_validated(
            responses[None, :],
            skill_assignments,
        )
        latest_mastery = self._latest_mastery_batch(
            gamma[..., 1],
            skill_assignments,
        )[0]
        return self._next_mastery_priors_from_latest(
            latest_mastery,
            skill_assignments,
        )

    def next_mastery_priors_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Return next-opportunity skill priors for multiple learners.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_trials)``.
        skill_assignments : NDArray
            Shared trial-to-skill vector or a matrix matching ``responses``.

        Returns
        -------
        NDArray
            Next-opportunity priors with shape
            ``(n_persons, n_skills)``.
        """
        responses, skill_assignments = self._validate_batch(
            responses,
            skill_assignments,
        )
        gamma, _ = self._forward_backward_batch_validated(
            responses,
            skill_assignments,
        )
        latest_mastery = self._latest_mastery_batch(
            gamma[..., 1],
            skill_assignments,
        )
        return self._next_mastery_priors_from_latest(
            latest_mastery,
            skill_assignments,
        )

    def forecast(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        n_steps: int,
    ) -> BKTForecastResult:
        """Forecast future skill opportunities after one response history.

        The first forecast step is each skill's next opportunity after its
        final historical trial. Skills absent from the history start at their
        configured initial mastery probability.

        Parameters
        ----------
        responses : NDArray
            Integer response sequence with shape ``(n_trials,)``.
        skill_assignments : NDArray
            Skill index for every historical trial.
        n_steps : int
            Number of future opportunities to forecast for every skill.

        Returns
        -------
        BKTForecastResult
            Mastery and success probabilities with shape
            ``(n_steps, n_skills)``.
        """
        return self.forecast_from_priors(
            self.next_mastery_priors(responses, skill_assignments),
            n_steps,
        )

    def forecast_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        n_steps: int,
    ) -> BKTBatchForecastResult:
        """Forecast future skill opportunities after multiple histories.

        Parameters
        ----------
        responses : NDArray
            Integer response matrix with shape ``(n_persons, n_trials)``.
        skill_assignments : NDArray
            Shared trial-to-skill vector or a matrix matching ``responses``.
        n_steps : int
            Number of future opportunities to forecast for every skill.

        Returns
        -------
        BKTBatchForecastResult
            Mastery and success probabilities with shape
            ``(n_persons, n_steps, n_skills)``.
        """
        return self.forecast_from_priors_batch(
            self.next_mastery_priors_batch(responses, skill_assignments),
            n_steps,
        )

    def simulate(
        self,
        n_persons: int,
        n_trials_per_skill: int,
        seed: int | None = None,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        """Simulate response data from BKT model.

        Parameters
        ----------
        n_persons : int
            Number of persons
        n_trials_per_skill : int
            Number of trials per skill per person
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            (responses, skill_assignments, learning_states)
        """
        if (
            isinstance(n_persons, bool)
            or not isinstance(n_persons, (int, np.integer))
            or n_persons < 1
        ):
            raise ValueError("n_persons must be a positive integer")
        if (
            isinstance(n_trials_per_skill, bool)
            or not isinstance(n_trials_per_skill, (int, np.integer))
            or n_trials_per_skill < 1
        ):
            raise ValueError("n_trials_per_skill must be a positive integer")

        rng = np.random.default_rng(seed)

        n_trials = n_trials_per_skill * self.n_skills
        responses = np.zeros((n_persons, n_trials), dtype=np.int32)
        skill_assignments = np.zeros(n_trials, dtype=np.int32)
        learning_states = np.zeros((n_persons, n_trials), dtype=np.int32)

        for j in range(self.n_skills):
            start = j * n_trials_per_skill
            end = (j + 1) * n_trials_per_skill
            skill_assignments[start:end] = j

        for skill_idx in range(self.n_skills):
            start = skill_idx * n_trials_per_skill
            states = rng.random(n_persons) < self.p_init[skill_idx]

            for relative_trial in range(n_trials_per_skill):
                trial = start + relative_trial
                learning_states[:, trial] = states
                p_correct = np.where(
                    states,
                    1.0 - self.p_slip[skill_idx],
                    self.p_guess[skill_idx],
                )
                responses[:, trial] = rng.random(n_persons) < p_correct

                if relative_trial < n_trials_per_skill - 1:
                    transition_draws = rng.random(n_persons)
                    states = np.where(
                        states,
                        transition_draws >= self.p_forget[skill_idx],
                        transition_draws < self.p_learn[skill_idx],
                    )

        return responses, skill_assignments, learning_states

    def summary(self) -> str:
        """Generate model summary."""
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'BKT Model Summary':^{width}}")
        lines.append("=" * width)

        lines.append(f"Number of Skills:   {self.n_skills}")
        lines.append(f"Allow Forgetting:   {self.allow_forgetting}")
        lines.append("-" * width)

        lines.append(
            f"\n{'Skill':<15} {'P(L0)':>8} {'P(Learn)':>10} {'P(Forget)':>10} {'P(Slip)':>8} {'P(Guess)':>8}"
        )
        lines.append("-" * width)

        for j in range(self.n_skills):
            lines.append(
                f"{self.skill_names[j]:<15} "
                f"{self.p_init[j]:>8.3f} "
                f"{self.p_learn[j]:>10.3f} "
                f"{self.p_forget[j]:>10.3f} "
                f"{self.p_slip[j]:>8.3f} "
                f"{self.p_guess[j]:>8.3f}"
            )

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class LongitudinalIRTModel:
    """Longitudinal IRT with latent growth curves.

    Models ability over time as:
    θ_it = η₀ᵢ + η₁ᵢ·t + ε_it

    where:
    - η₀ᵢ: Individual intercept (initial ability)
    - η₁ᵢ: Individual slope (growth rate)
    - (η₀ᵢ, η₁ᵢ) ~ MVN(μ, Σ)
    - ε_it ~ N(0, σ²_ε)

    Parameters
    ----------
    n_items : int
        Number of items (assumed invariant over time)
    n_timepoints : int
        Number of measurement occasions
    base_model : str
        IRT model for item responses ("2PL" or "GRM")
    growth_model : str
        Growth model type ("linear", "quadratic")
    n_categories : int, optional
        Number of response categories for a GRM. Inferred from ``thresholds``
        when provided, otherwise defaults to 4.
    thresholds : NDArray, optional
        Ordered GRM thresholds with shape
        ``(n_items, n_categories - 1)``.
    """

    n_items: int
    n_timepoints: int
    base_model: Literal["2PL", "GRM"] = "2PL"
    growth_model: Literal["linear", "quadratic"] = "linear"
    item_names: list[str] | None = None

    discrimination: NDArray[np.float64] | None = None
    difficulty: NDArray[np.float64] | None = None

    growth_mean: NDArray[np.float64] | None = None
    growth_cov: NDArray[np.float64] | None = None
    residual_variance: float = 0.1
    n_categories: int | None = None
    thresholds: NDArray[np.float64] | None = None

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
        if self.base_model not in ("2PL", "GRM"):
            raise ValueError("base_model must be '2PL' or 'GRM'")
        if self.growth_model not in ("linear", "quadratic"):
            raise ValueError("growth_model must be 'linear' or 'quadratic'")

        if self.item_names is None:
            self.item_names = [f"Item_{i}" for i in range(self.n_items)]
        else:
            self.item_names = list(self.item_names)
            if len(self.item_names) != self.n_items:
                raise ValueError("item_names length must match n_items")
            if any(not isinstance(name, str) or not name for name in self.item_names):
                raise ValueError("item_names must contain non-empty strings")
            if len(set(self.item_names)) != self.n_items:
                raise ValueError("item_names must be unique")

        if self.discrimination is None:
            self.discrimination = np.ones(self.n_items)
        else:
            self.discrimination = np.asarray(
                self.discrimination,
                dtype=np.float64,
            ).copy()
        if self.discrimination.shape != (self.n_items,) or not np.all(
            np.isfinite(self.discrimination)
        ):
            raise ValueError(f"discrimination must have shape ({self.n_items},)")
        if np.any(self.discrimination <= 0.0):
            raise ValueError("discrimination values must be positive")

        if self.difficulty is None:
            self.difficulty = np.zeros(self.n_items)
        else:
            self.difficulty = np.asarray(self.difficulty, dtype=np.float64).copy()
        if self.difficulty.shape != (self.n_items,) or not np.all(
            np.isfinite(self.difficulty)
        ):
            raise ValueError(f"difficulty must have shape ({self.n_items},)")

        if self.base_model == "GRM":
            if self.thresholds is None:
                if self.n_categories is None:
                    self.n_categories = 4
                self._validate_category_count()
                default_thresholds = (
                    np.zeros(1)
                    if self.n_categories == 2
                    else np.linspace(-1.0, 1.0, self.n_categories - 1)
                )
                self.thresholds = np.broadcast_to(
                    default_thresholds,
                    (self.n_items, self.n_categories - 1),
                ).copy()
            else:
                self.thresholds = np.asarray(
                    self.thresholds,
                    dtype=np.float64,
                ).copy()
                if (
                    self.thresholds.ndim != 2
                    or self.thresholds.shape[0] != self.n_items
                ):
                    raise ValueError(
                        "thresholds must have shape (n_items, n_categories - 1)"
                    )
                inferred_categories = self.thresholds.shape[1] + 1
                if (
                    self.n_categories is not None
                    and self.n_categories != inferred_categories
                ):
                    raise ValueError("n_categories does not match thresholds")
                self.n_categories = inferred_categories
                self._validate_category_count()
            if not np.all(np.isfinite(self.thresholds)):
                raise ValueError("thresholds must contain only finite values")
            if np.any(np.diff(self.thresholds, axis=1) <= 0.0):
                raise ValueError("thresholds must be strictly increasing within items")
        else:
            if self.thresholds is not None:
                raise ValueError("thresholds are only supported for base_model='GRM'")
            if self.n_categories not in (None, 2):
                raise ValueError("2PL models require exactly 2 response categories")
            self.n_categories = 2

        n_growth = 2 if self.growth_model == "linear" else 3
        if self.growth_mean is None:
            self.growth_mean = np.zeros(n_growth)
        else:
            self.growth_mean = np.asarray(self.growth_mean, dtype=np.float64).copy()
        if self.growth_mean.shape != (n_growth,) or not np.all(
            np.isfinite(self.growth_mean)
        ):
            raise ValueError(f"growth_mean must have shape ({n_growth},)")

        if self.growth_cov is None:
            self.growth_cov = np.eye(n_growth)
        else:
            self.growth_cov = np.asarray(self.growth_cov, dtype=np.float64).copy()
        if self.growth_cov.shape != (n_growth, n_growth) or not np.all(
            np.isfinite(self.growth_cov)
        ):
            raise ValueError(f"growth_cov must have shape ({n_growth}, {n_growth})")
        if not np.allclose(self.growth_cov, self.growth_cov.T, rtol=0.0, atol=1e-12):
            raise ValueError("growth_cov must be symmetric")
        if np.min(np.linalg.eigvalsh(self.growth_cov)) < -1e-12:
            raise ValueError("growth_cov must be positive semidefinite")

        if isinstance(self.residual_variance, bool):
            raise ValueError("residual_variance must be finite and non-negative")
        self.residual_variance = float(self.residual_variance)
        if not np.isfinite(self.residual_variance) or self.residual_variance < 0.0:
            raise ValueError("residual_variance must be finite and non-negative")

    def _validate_category_count(self) -> None:
        """Validate and normalize the configured GRM category count."""
        if (
            isinstance(self.n_categories, bool)
            or not isinstance(self.n_categories, (int, np.integer))
            or self.n_categories < 2
        ):
            raise ValueError("n_categories must be an integer of at least 2")
        self.n_categories = int(self.n_categories)

    @property
    def n_growth_factors(self) -> int:
        """Number of growth factors."""
        return 2 if self.growth_model == "linear" else 3

    def compute_theta(
        self,
        growth_factors: NDArray[np.float64],
        time_values: NDArray[np.float64] | None = None,
        residuals: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute ability trajectory from growth factors.

        Parameters
        ----------
        growth_factors : NDArray
            Growth factors (n_persons, n_growth_factors)
        time_values : NDArray, optional
            Time values (n_timepoints,). Defaults to 0, 1, ..., T-1.
        residuals : NDArray, optional
            Time-specific residuals (n_persons, n_timepoints)

        Returns
        -------
        NDArray
            Ability trajectory (n_persons, n_timepoints)
        """
        growth_values = np.asarray(growth_factors, dtype=np.float64)
        if growth_values.ndim != 2 or growth_values.shape[1] != self.n_growth_factors:
            raise ValueError(
                f"growth_factors must have shape (n_persons, {self.n_growth_factors})"
            )
        if not np.all(np.isfinite(growth_values)):
            raise ValueError("growth_factors must contain only finite values")

        if time_values is None:
            times = np.arange(self.n_timepoints, dtype=np.float64)
        else:
            times = np.asarray(time_values, dtype=np.float64)
        if times.shape != (self.n_timepoints,) or not np.all(np.isfinite(times)):
            raise ValueError(f"time_values must have shape ({self.n_timepoints},)")

        basis = [np.ones(self.n_timepoints), times]
        if self.growth_model == "quadratic":
            basis.append(times**2)
        theta = growth_values @ np.stack(basis)

        if residuals is not None:
            residual_values = np.asarray(residuals, dtype=np.float64)
            if residual_values.shape != theta.shape or not np.all(
                np.isfinite(residual_values)
            ):
                raise ValueError(f"residuals must have shape {theta.shape}")
            theta += residual_values

        return theta

    def probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute response probability.

        Parameters
        ----------
        theta : NDArray
            Ability values (n_persons,) or scalar
        item_idx : int, optional
            Specific item

        Returns
        -------
        NDArray
            For a 2PL, ``P(X=1|theta)`` with shape ``(n_persons,)`` for one
            item or ``(n_persons, n_items)`` for all items. For a GRM,
            category probabilities with a final ``n_categories`` axis.
        """
        theta_values = np.asarray(theta, dtype=np.float64)
        if theta_values.ndim == 0:
            theta_values = theta_values.reshape(1)
        if theta_values.ndim != 1:
            raise ValueError("theta must be a scalar or one-dimensional array")
        if not np.all(np.isfinite(theta_values)):
            raise ValueError("theta must contain only finite values")

        if item_idx is not None:
            if (
                isinstance(item_idx, bool)
                or not isinstance(item_idx, (int, np.integer))
                or not 0 <= item_idx < self.n_items
            ):
                raise ValueError(f"item_idx must be between 0 and {self.n_items - 1}")
            item_idx = int(item_idx)
            a = self.discrimination[item_idx]
            if self.base_model == "2PL":
                b = self.difficulty[item_idx]
                return np.asarray(sigmoid(a * (theta_values - b)))

            cumulative = np.asarray(
                sigmoid(a * (theta_values[:, None] - self.thresholds[item_idx])),
                dtype=np.float64,
            )
            return self._graded_category_probabilities(cumulative)

        if self.base_model == "2PL":
            logits = self.discrimination[None, :] * (
                theta_values[:, None] - self.difficulty[None, :]
            )
            return np.asarray(sigmoid(logits))

        cumulative = np.asarray(
            sigmoid(
                self.discrimination[None, :, None]
                * (theta_values[:, None, None] - self.thresholds[None, :, :])
            ),
            dtype=np.float64,
        )
        return self._graded_category_probabilities(cumulative)

    @staticmethod
    def _graded_category_probabilities(
        cumulative: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert GRM cumulative curves into adjacent category probabilities."""
        boundaries = np.concatenate(
            (
                np.ones((*cumulative.shape[:-1], 1)),
                cumulative,
                np.zeros((*cumulative.shape[:-1], 1)),
            ),
            axis=-1,
        )
        return boundaries[..., :-1] - boundaries[..., 1:]

    def simulate(
        self,
        n_persons: int,
        time_values: NDArray[np.float64] | None = None,
        seed: int | None = None,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
        """Simulate longitudinal response data.

        Parameters
        ----------
        n_persons : int
            Number of persons
        time_values : NDArray, optional
            Time values for each occasion
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            (responses, theta_trajectories, growth_factors)
            - responses: (n_persons, n_timepoints, n_items)
            - theta_trajectories: (n_persons, n_timepoints)
            - growth_factors: (n_persons, n_growth_factors)
        """
        if (
            isinstance(n_persons, bool)
            or not isinstance(n_persons, (int, np.integer))
            or n_persons < 1
        ):
            raise ValueError("n_persons must be a positive integer")
        n_persons = int(n_persons)
        rng = np.random.default_rng(seed)

        growth_factors = rng.multivariate_normal(
            self.growth_mean, self.growth_cov, size=n_persons
        )

        residuals = rng.normal(
            0, np.sqrt(self.residual_variance), size=(n_persons, self.n_timepoints)
        )

        theta = self.compute_theta(growth_factors, time_values, residuals)

        flat_theta = theta.reshape(-1)
        flat_responses = np.empty((flat_theta.size, self.n_items), dtype=np.int32)
        category_factor = int(self.n_categories) if self.base_model == "GRM" else 1
        probability_values_per_row = self.n_items * category_factor
        rows_per_chunk = max(
            1,
            _LONGITUDINAL_MAX_PROBABILITY_VALUES // probability_values_per_row,
        )

        for start in range(0, flat_theta.size, rows_per_chunk):
            stop = min(start + rows_per_chunk, flat_theta.size)
            probabilities = self.probability(flat_theta[start:stop])
            if self.base_model == "2PL":
                flat_responses[start:stop] = (
                    rng.random(probabilities.shape) < probabilities
                )
            else:
                category_probabilities = probabilities.reshape(
                    -1,
                    int(self.n_categories),
                )
                flat_responses[start:stop] = sample_categorical_rows(
                    category_probabilities,
                    rng,
                ).reshape(stop - start, self.n_items)

        responses = flat_responses.reshape(
            n_persons,
            self.n_timepoints,
            self.n_items,
        )

        return responses, theta, growth_factors

    def summary(self) -> str:
        """Generate model summary."""
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'Longitudinal IRT Model Summary':^{width}}")
        lines.append("=" * width)

        lines.append(f"Base Model:         {self.base_model}")
        lines.append(f"Growth Model:       {self.growth_model}")
        lines.append(f"Number of Items:    {self.n_items}")
        lines.append(f"Number of Times:    {self.n_timepoints}")
        if self.base_model == "GRM":
            lines.append(f"Response Categories: {self.n_categories}")
        lines.append(f"Residual Variance:  {self.residual_variance:.4f}")
        lines.append("-" * width)

        lines.append("\nGrowth Factor Mean:")
        names = (
            ["Intercept", "Slope"]
            if self.growth_model == "linear"
            else ["Intercept", "Slope", "Quadratic"]
        )
        for i, name in enumerate(names):
            lines.append(f"  {name}: {self.growth_mean[i]:.4f}")

        lines.append("\nGrowth Factor Covariance:")
        for i, name_i in enumerate(names):
            row = f"  {name_i:<12}"
            for j in range(len(names)):
                row += f" {self.growth_cov[i, j]:>8.4f}"
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class BKTResult:
    """Result from BKT estimation."""

    model: BKTModel
    learning_curves: NDArray[np.float64]
    skill_mastery: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    n_parameters: int
    converged: bool

    def summary(self) -> str:
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'BKT Estimation Results':^{width}}")
        lines.append("=" * width)

        lines.append(f"Log-Likelihood:     {self.log_likelihood:.4f}")
        lines.append(f"AIC:                {self.aic:.4f}")
        lines.append(f"BIC:                {self.bic:.4f}")
        lines.append(f"Converged:          {self.converged}")
        lines.append("-" * width)

        lines.append("\nSkill Mastery Rates:")
        for j in range(self.model.n_skills):
            mean_mastery = np.mean(self.skill_mastery[:, j])
            lines.append(f"  {self.model.skill_names[j]}: {mean_mastery:.3f}")

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class LongitudinalResult:
    """Result from longitudinal IRT estimation."""

    model: LongitudinalIRTModel
    growth_factors: NDArray[np.float64]
    theta_trajectories: NDArray[np.float64]
    growth_factor_se: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_iterations: int

    def summary(self) -> str:
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'Longitudinal IRT Results':^{width}}")
        lines.append("=" * width)

        lines.append(f"Log-Likelihood:     {self.log_likelihood:.4f}")
        lines.append(f"AIC:                {self.aic:.4f}")
        lines.append(f"BIC:                {self.bic:.4f}")
        lines.append(f"Converged:          {self.converged}")
        lines.append("-" * width)

        lines.append("\nGrowth Factor Estimates (population means):")
        names = (
            ["Intercept", "Slope"]
            if self.model.growth_model == "linear"
            else ["Intercept", "Slope", "Quadratic"]
        )
        for i, name in enumerate(names):
            mean = np.mean(self.growth_factors[:, i])
            sd = np.std(self.growth_factors[:, i])
            lines.append(f"  {name}: M = {mean:.4f}, SD = {sd:.4f}")

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class PiecewiseGrowthModel:
    """Piecewise linear growth model with changepoints.

    Models ability over time as:
    θ(t) = β₀ + β₁·t   if t ≤ τ₁
           β₀ + β₁·τ₁ + β₂·(t - τ₁)   if τ₁ < t ≤ τ₂
           ...

    Parameters
    ----------
    n_pieces : int
        Number of linear pieces.
    changepoints : NDArray
        Time points where slope changes (n_pieces - 1,).
    intercept_mean : float
        Population mean intercept.
    intercept_var : float
        Population variance of intercept.
    slope_means : NDArray
        Population mean slopes for each piece.
    slope_vars : NDArray
        Population variance of slopes.
    residual_variance : float
        Time-specific residual variance.
    """

    n_pieces: int
    changepoints: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    intercept_mean: float = 0.0
    intercept_var: float = 1.0
    slope_means: NDArray[np.float64] = field(default_factory=lambda: np.array([0.1]))
    slope_vars: NDArray[np.float64] = field(default_factory=lambda: np.array([0.01]))
    residual_variance: float = 0.1

    def __post_init__(self) -> None:
        if len(self.changepoints) == 0 and self.n_pieces > 1:
            self.changepoints = np.linspace(1, self.n_pieces - 1, self.n_pieces - 1)

        if len(self.slope_means) == 1 and self.n_pieces > 1:
            self.slope_means = np.full(self.n_pieces, self.slope_means[0])
        if len(self.slope_vars) == 1 and self.n_pieces > 1:
            self.slope_vars = np.full(self.n_pieces, self.slope_vars[0])

        if len(self.changepoints) != self.n_pieces - 1:
            raise ValueError(
                f"changepoints length ({len(self.changepoints)}) "
                f"must be n_pieces - 1 ({self.n_pieces - 1})"
            )

    def compute_theta(
        self,
        time_values: NDArray[np.float64],
        intercept: float | NDArray[np.float64],
        slopes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute ability at given time points.

        Parameters
        ----------
        time_values : NDArray
            Time points (n_timepoints,).
        intercept : float or NDArray
            Individual intercept(s).
        slopes : NDArray
            Individual slopes (n_pieces,) or (n_persons, n_pieces).

        Returns
        -------
        NDArray
            Ability values (n_timepoints,) or (n_persons, n_timepoints).
        """
        time_values = np.atleast_1d(time_values)
        n_times = len(time_values)

        intercept = np.atleast_1d(intercept)
        slopes = np.atleast_2d(slopes)
        n_persons = slopes.shape[0]

        theta = np.zeros((n_persons, n_times))

        for i in range(n_persons):
            for t_idx, t in enumerate(time_values):
                piece_idx = 0
                for cp_idx, cp in enumerate(self.changepoints):
                    if t > cp:
                        piece_idx = cp_idx + 1

                value = intercept[i] if len(intercept) > 1 else intercept[0]
                t_remaining = t

                for p in range(piece_idx + 1):
                    if p < self.n_pieces - 1 and p < piece_idx:
                        segment_length = (
                            self.changepoints[p]
                            if p == 0
                            else self.changepoints[p] - self.changepoints[p - 1]
                        )
                        value += slopes[i, p] * segment_length
                        t_remaining -= segment_length
                    else:
                        value += slopes[i, p] * t_remaining

                theta[i, t_idx] = value

        return theta.squeeze()

    def simulate(
        self,
        n_persons: int,
        time_values: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Simulate trajectories from the model.

        Parameters
        ----------
        n_persons : int
            Number of persons.
        time_values : NDArray
            Time points.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple
            (theta_trajectories, intercepts, slopes)
        """
        rng = np.random.default_rng(seed)

        intercepts = rng.normal(
            self.intercept_mean, np.sqrt(self.intercept_var), n_persons
        )

        slopes = np.zeros((n_persons, self.n_pieces))
        for p in range(self.n_pieces):
            slopes[:, p] = rng.normal(
                self.slope_means[p], np.sqrt(self.slope_vars[p]), n_persons
            )

        theta = self.compute_theta(time_values, intercepts, slopes)
        theta += rng.normal(0, np.sqrt(self.residual_variance), theta.shape)

        return theta, intercepts, slopes

    def detect_changepoints(
        self,
        time_values: NDArray[np.float64],
        observations: NDArray[np.float64],
        max_changepoints: int = 3,
    ) -> NDArray[np.float64]:
        """Detect changepoints from observed data.

        Uses a simple residual-based approach to find optimal
        changepoint locations.

        Parameters
        ----------
        time_values : NDArray
            Time points.
        observations : NDArray
            Observed values (n_persons, n_timepoints).
        max_changepoints : int
            Maximum number of changepoints to detect.

        Returns
        -------
        NDArray
            Detected changepoint locations.
        """
        observations = np.atleast_2d(observations)
        mean_trajectory = np.mean(observations, axis=0)

        best_sse = np.inf
        best_changepoints: NDArray[np.float64] = np.array([])

        for n_cp in range(max_changepoints + 1):
            if n_cp == 0:
                slope, intercept = np.polyfit(time_values, mean_trajectory, 1)
                pred = intercept + slope * time_values
                sse = np.sum((mean_trajectory - pred) ** 2)
                if sse < best_sse:
                    best_sse = sse
                    best_changepoints = np.array([])
            else:
                candidate_times = time_values[1:-1]
                if len(candidate_times) < n_cp:
                    continue

                from itertools import combinations

                for cp_combo in combinations(range(len(candidate_times)), n_cp):
                    cps = candidate_times[list(cp_combo)]

                    sse = 0.0
                    prev_cp = time_values[0]
                    for i, cp in enumerate(list(cps) + [time_values[-1]]):
                        mask = (time_values >= prev_cp) & (time_values <= cp)
                        if np.sum(mask) > 1:
                            t_seg = time_values[mask]
                            y_seg = mean_trajectory[mask]
                            if len(t_seg) > 1:
                                slope, intercept = np.polyfit(t_seg, y_seg, 1)
                                pred = intercept + slope * t_seg
                                sse += np.sum((y_seg - pred) ** 2)
                        prev_cp = cp

                    penalty = n_cp * 2 * np.var(mean_trajectory)
                    if sse + penalty < best_sse:
                        best_sse = sse + penalty
                        best_changepoints = cps

        return best_changepoints


@dataclass
class NonlinearGrowthModel:
    """Nonlinear growth model (exponential, logistic, Gompertz).

    Models ability over time using nonlinear functions:
    - Exponential: θ(t) = α·(1 - exp(-β·t))
    - Logistic: θ(t) = α / (1 + exp(-β·(t - γ)))
    - Gompertz: θ(t) = α·exp(-exp(-β·(t - γ)))

    Parameters
    ----------
    growth_type : str
        Type of growth function.
    asymptote : float
        Upper asymptote (α).
    rate : float
        Growth rate (β).
    inflection : float
        Inflection point (γ), for logistic/Gompertz.
    initial_value : float
        Value at t=0.
    residual_variance : float
        Residual variance.
    """

    growth_type: Literal["exponential", "logistic", "gompertz"] = "logistic"
    asymptote: float = 1.0
    rate: float = 1.0
    inflection: float = 0.0
    initial_value: float = 0.0
    residual_variance: float = 0.1

    asymptote_var: float = 0.1
    rate_var: float = 0.01
    inflection_var: float = 0.1

    def compute_theta(
        self,
        time_values: NDArray[np.float64],
        asymptote: float | NDArray[np.float64] | None = None,
        rate: float | NDArray[np.float64] | None = None,
        inflection: float | NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute ability at given time points.

        Parameters
        ----------
        time_values : NDArray
            Time points.
        asymptote : float or NDArray, optional
            Individual asymptote(s).
        rate : float or NDArray, optional
            Individual rate(s).
        inflection : float or NDArray, optional
            Individual inflection point(s).

        Returns
        -------
        NDArray
            Ability values.
        """
        if asymptote is None:
            asymptote = self.asymptote
        if rate is None:
            rate = self.rate
        if inflection is None:
            inflection = self.inflection

        asymptote = np.atleast_1d(asymptote)
        rate = np.atleast_1d(rate)
        inflection = np.atleast_1d(inflection)

        n_persons = max(len(asymptote), len(rate), len(inflection))
        time_values = np.atleast_1d(time_values)
        n_times = len(time_values)

        if len(asymptote) == 1:
            asymptote = np.full(n_persons, asymptote[0])
        if len(rate) == 1:
            rate = np.full(n_persons, rate[0])
        if len(inflection) == 1:
            inflection = np.full(n_persons, inflection[0])

        theta = np.zeros((n_persons, n_times))

        for i in range(n_persons):
            if self.growth_type == "exponential":
                theta[i] = asymptote[i] * (1 - np.exp(-rate[i] * time_values))
            elif self.growth_type == "logistic":
                theta[i] = asymptote[i] / (
                    1 + np.exp(-rate[i] * (time_values - inflection[i]))
                )
            elif self.growth_type == "gompertz":
                theta[i] = asymptote[i] * np.exp(
                    -np.exp(-rate[i] * (time_values - inflection[i]))
                )

        return theta.squeeze()

    def growth_velocity(
        self,
        time_values: NDArray[np.float64],
        asymptote: float | None = None,
        rate: float | None = None,
        inflection: float | None = None,
    ) -> NDArray[np.float64]:
        """Compute instantaneous growth velocity (derivative).

        Parameters
        ----------
        time_values : NDArray
            Time points.
        asymptote, rate, inflection : float, optional
            Model parameters.

        Returns
        -------
        NDArray
            Instantaneous velocity at each time point.
        """
        if asymptote is None:
            asymptote = self.asymptote
        if rate is None:
            rate = self.rate
        if inflection is None:
            inflection = self.inflection

        time_values = np.atleast_1d(time_values)

        if self.growth_type == "exponential":
            velocity = asymptote * rate * np.exp(-rate * time_values)
        elif self.growth_type == "logistic":
            exp_term = np.exp(-rate * (time_values - inflection))
            velocity = asymptote * rate * exp_term / (1 + exp_term) ** 2
        elif self.growth_type == "gompertz":
            exp_inner = np.exp(-rate * (time_values - inflection))
            velocity = asymptote * rate * exp_inner * np.exp(-exp_inner)
        else:
            raise ValueError(f"Unknown growth type: {self.growth_type}")

        return velocity

    def simulate(
        self,
        n_persons: int,
        time_values: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
        """Simulate trajectories from the model.

        Parameters
        ----------
        n_persons : int
            Number of persons.
        time_values : NDArray
            Time points.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple
            (theta_trajectories, parameters_dict)
        """
        rng = np.random.default_rng(seed)

        asymptotes = rng.normal(self.asymptote, np.sqrt(self.asymptote_var), n_persons)
        rates = np.abs(rng.normal(self.rate, np.sqrt(self.rate_var), n_persons))
        inflections = rng.normal(
            self.inflection, np.sqrt(self.inflection_var), n_persons
        )

        theta = self.compute_theta(time_values, asymptotes, rates, inflections)
        theta += rng.normal(0, np.sqrt(self.residual_variance), theta.shape)

        params = {
            "asymptote": asymptotes,
            "rate": rates,
            "inflection": inflections,
        }

        return theta, params

    def fit_individual(
        self,
        time_values: NDArray[np.float64],
        observations: NDArray[np.float64],
        max_iter: int = 100,
    ) -> dict[str, float]:
        """Fit model to individual trajectory.

        Uses simple gradient descent to estimate parameters.

        Parameters
        ----------
        time_values : NDArray
            Time points.
        observations : NDArray
            Observed values.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        dict
            Estimated parameters.
        """
        asymptote = float(np.max(observations))
        rate = self.rate
        inflection = float(time_values[len(time_values) // 2])

        learning_rate = 0.01

        for _ in range(max_iter):
            pred = self.compute_theta(
                time_values,
                asymptote=asymptote,
                rate=rate,
                inflection=inflection,
            )
            error = observations - pred

            grad_a = -2 * np.mean(error * pred / asymptote)
            asymptote -= learning_rate * grad_a
            asymptote = max(0.1, asymptote)

            if self.growth_type in ["logistic", "gompertz"]:
                grad_g = 2 * np.mean(error * self.growth_velocity(time_values))
                inflection -= learning_rate * grad_g

        return {
            "asymptote": asymptote,
            "rate": rate,
            "inflection": inflection,
        }


@dataclass
class GrowthMixtureModel:
    """Latent class growth analysis (growth mixture model).

    Models heterogeneous populations with distinct growth trajectories
    using a mixture of growth curves.

    Parameters
    ----------
    n_classes : int
        Number of latent classes.
    growth_type : str
        Type of growth model within classes.
    n_timepoints : int
        Number of time points.
    class_proportions : NDArray, optional
        Prior class proportions.
    class_post_slopes : NDArray, optional
        Post-changepoint slopes for piecewise growth.
    changepoint : float, optional
        Shared piecewise changepoint. The time-range midpoint is used when
        omitted.
    """

    n_classes: int
    growth_type: Literal["linear", "quadratic", "piecewise"] = "linear"
    n_timepoints: int = 5

    class_proportions: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    class_intercepts: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    class_slopes: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    class_quadratics: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    intercept_var: float = 0.5
    slope_var: float = 0.1
    residual_variance: float = 0.1

    class_post_slopes: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    changepoint: float | None = None

    def __post_init__(self) -> None:
        if len(self.class_proportions) == 0:
            self.class_proportions = np.ones(self.n_classes) / self.n_classes

        if len(self.class_intercepts) == 0:
            self.class_intercepts = np.linspace(-1, 1, self.n_classes)

        if len(self.class_slopes) == 0:
            self.class_slopes = np.linspace(0.1, 0.5, self.n_classes)

        if self.growth_type == "quadratic" and len(self.class_quadratics) == 0:
            self.class_quadratics = np.zeros(self.n_classes)

        if self.growth_type == "piecewise" and len(self.class_post_slopes) == 0:
            self.class_post_slopes = self.class_slopes.copy()

    def compute_class_trajectory(
        self,
        class_idx: int,
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute mean trajectory for a class.

        Parameters
        ----------
        class_idx : int
            Class index.
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Mean trajectory for the class.
        """
        if (
            isinstance(class_idx, bool)
            or not isinstance(class_idx, (int, np.integer))
            or not 0 <= class_idx < self.n_classes
        ):
            raise ValueError("class_idx must identify an existing class")
        times = self._validated_time_values(time_values)
        trajectories = self._validated_class_trajectories(times)
        return trajectories[int(class_idx)].copy()

    def class_likelihood(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute likelihood of observations under each class.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints). Use ``NaN`` for
            missing occasions; every person must have at least one observed
            value.
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Class likelihoods (n_persons, n_classes).

        Notes
        -----
        For long trajectories, use :meth:`class_log_likelihood` or
        :meth:`posterior_probabilities` to avoid probability underflow or
        overflow.
        """
        return np.exp(self.class_log_likelihood(observations, time_values))

    @staticmethod
    def _validated_time_values(
        time_values: NDArray[np.float64],
        *,
        expected_length: int | None = None,
    ) -> NDArray[np.float64]:
        """Return a finite, non-empty one-dimensional time grid."""
        try:
            times = np.asarray(time_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_values must contain numeric values") from exc
        if times.ndim != 1 or times.size < 1:
            raise ValueError("time_values must be a non-empty one-dimensional array")
        if expected_length is not None and times.size != expected_length:
            raise ValueError(
                "time_values must contain one value per observation column"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("time_values must contain only finite values")
        return times

    def _validated_class_trajectories(
        self,
        times: NDArray[np.float64],
        *,
        piecewise_changepoint: float | None = None,
    ) -> NDArray[np.float64]:
        """Build finite class trajectories from validated time values."""
        try:
            intercepts = np.asarray(self.class_intercepts, dtype=np.float64)
            slopes = np.asarray(self.class_slopes, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("class trajectory parameters must be numeric") from exc
        if intercepts.shape != (self.n_classes,) or slopes.shape != (self.n_classes,):
            raise ValueError("class intercepts and slopes must match n_classes")
        if not np.all(np.isfinite(intercepts)) or not np.all(np.isfinite(slopes)):
            raise ValueError("class trajectory parameters must be finite")

        trajectories = intercepts[:, None] + slopes[:, None] * times[None, :]
        if self.growth_type == "quadratic":
            try:
                quadratics = np.asarray(self.class_quadratics, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("class quadratics must be numeric") from exc
            if quadratics.shape != (self.n_classes,) or not np.all(
                np.isfinite(quadratics)
            ):
                raise ValueError("class quadratics must be finite and match n_classes")
            trajectories += quadratics[:, None] * times[None, :] ** 2
        elif self.growth_type == "piecewise":
            try:
                post_slopes = np.asarray(self.class_post_slopes, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("class post slopes must be numeric") from exc
            if post_slopes.shape != (self.n_classes,) or not np.all(
                np.isfinite(post_slopes)
            ):
                raise ValueError("class post slopes must be finite and match n_classes")
            changepoint = (
                self._resolved_changepoint(times)
                if piecewise_changepoint is None
                else piecewise_changepoint
            )
            hinge = np.maximum(times - changepoint, 0.0)
            trajectories += (post_slopes - slopes)[:, None] * hinge[None, :]
        return trajectories

    def _resolved_changepoint(self, times: NDArray[np.float64]) -> float:
        """Resolve and validate the shared piecewise changepoint."""
        if self.changepoint is None:
            return 0.5 * float(np.min(times) + np.max(times))
        try:
            changepoint = float(self.changepoint)
        except (TypeError, ValueError) as exc:
            raise ValueError("changepoint must be numeric") from exc
        if not np.isfinite(changepoint):
            raise ValueError("changepoint must be finite")
        return changepoint

    def _validated_class_proportions(self) -> NDArray[np.float64]:
        """Return normalized, finite class weights."""
        try:
            proportions = np.asarray(self.class_proportions, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("class_proportions must be numeric") from exc
        if proportions.shape != (self.n_classes,):
            raise ValueError("class_proportions must match n_classes")
        if not np.all(np.isfinite(proportions)) or np.any(proportions < 0.0):
            raise ValueError("class_proportions must be finite and nonnegative")
        total_proportion = float(np.sum(proportions))
        if total_proportion <= 0.0:
            raise ValueError("class_proportions must contain a positive value")
        return proportions / total_proportion

    def _validated_variance_components(self) -> tuple[float, float, float]:
        """Return finite, nonnegative growth-mixture variances."""
        try:
            variances = np.asarray(
                [self.intercept_var, self.slope_var, self.residual_variance],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("variance components must be numeric") from exc
        if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("variance components must be finite and nonnegative")
        return float(variances[0]), float(variances[1]), float(variances[2])

    def _validated_trajectory_data(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[float, float, float],
        NDArray[np.bool_] | None,
    ]:
        """Validate growth-mixture inputs and build class trajectories."""
        try:
            observation_values = np.asarray(observations, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("observations must contain numeric values") from exc
        if observation_values.ndim == 1:
            observation_values = observation_values.reshape(1, -1)
        if (
            observation_values.ndim != 2
            or observation_values.shape[0] < 1
            or observation_values.shape[1] < 1
        ):
            raise ValueError("observations must be a non-empty two-dimensional array")
        finite = np.isfinite(observation_values)
        observation_mask = None
        if not np.all(finite):
            missing = np.isnan(observation_values)
            if np.any(~finite & ~missing):
                raise ValueError(
                    "observations must contain finite values or NaN for missing "
                    "occasions"
                )
            observation_mask = finite
            if np.any(~np.any(observation_mask, axis=1)):
                raise ValueError(
                    "each trajectory must contain at least one observation"
                )

        times = self._validated_time_values(
            time_values,
            expected_length=observation_values.shape[1],
        )
        trajectories = self._validated_class_trajectories(times)
        variances = self._validated_variance_components()
        if variances[2] <= 0.0:
            raise ValueError(
                "residual_variance must be positive for likelihood evaluation"
            )
        return observation_values, times, trajectories, variances, observation_mask

    @staticmethod
    def _prepare_observation_patterns(
        observations: NDArray[np.float64],
        times: NDArray[np.float64],
        variances: tuple[float, float, float],
        observation_mask: NDArray[np.bool_] | None,
    ) -> list[_GrowthObservationPattern]:
        """Group rows by observed occasions and factor each covariance once."""
        if observation_mask is None:
            unique_patterns = np.ones((1, observations.shape[1]), dtype=np.bool_)
            row_groups = [np.arange(observations.shape[0], dtype=np.int_)]
        else:
            unique_patterns, inverse, counts = np.unique(
                observation_mask,
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            grouped_rows = np.argsort(inverse, kind="stable")
            row_groups = np.split(grouped_rows, np.cumsum(counts)[:-1])

        intercept_variance, slope_variance, residual_variance = variances
        patterns = []
        for pattern, rows in zip(unique_patterns, row_groups):
            columns = np.flatnonzero(pattern)
            covariance = _GrowthCovariance.from_time_values(
                times[columns],
                intercept_variance,
                slope_variance,
                residual_variance,
            )
            patterns.append(
                _GrowthObservationPattern(
                    rows=rows,
                    columns=columns,
                    covariance=covariance,
                )
            )
        return patterns

    def _class_log_likelihood_from_patterns(
        self,
        observations: NDArray[np.float64],
        trajectories: NDArray[np.float64],
        patterns: list[_GrowthObservationPattern],
    ) -> NDArray[np.float64]:
        """Evaluate validated trajectories using prepared observation patterns."""
        if len(patterns) == 1 and patterns[0].columns.size == observations.shape[1]:
            return self._complete_class_log_likelihood(
                observations,
                trajectories,
                patterns[0].covariance,
            )

        log_likelihoods = np.empty(
            (observations.shape[0], self.n_classes),
            dtype=np.float64,
        )
        for pattern in patterns:
            values = observations[np.ix_(pattern.rows, pattern.columns)]
            class_trajectories = trajectories[:, pattern.columns]
            log_likelihoods[pattern.rows] = self._complete_class_log_likelihood(
                values,
                class_trajectories,
                pattern.covariance,
            )
        return log_likelihoods

    def _complete_class_log_likelihood(
        self,
        observations: NDArray[np.float64],
        trajectories: NDArray[np.float64],
        covariance: _GrowthCovariance,
    ) -> NDArray[np.float64]:
        """Evaluate one fully observed trajectory matrix."""
        if covariance.orthonormal_basis is None:
            quadratic_forms = (
                cdist(
                    observations,
                    trajectories,
                    metric="sqeuclidean",
                )
                / covariance.residual_variance
            )
        else:
            quadratic_forms = np.empty(
                (observations.shape[0], self.n_classes),
                dtype=np.float64,
            )
            for class_index, trajectory in enumerate(trajectories):
                quadratic_forms[:, class_index] = covariance.quadratic_forms(
                    observations - trajectory
                )

        normalization = (
            observations.shape[1] * np.log(2.0 * np.pi) + covariance.log_determinant
        )
        return -0.5 * (quadratic_forms + normalization)

    def class_log_likelihood(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute stable log likelihoods for every person and class.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories with shape ``(n_persons, n_timepoints)``.
            Use ``NaN`` for missing occasions; every person must have at least
            one observed value.
            A one-dimensional trajectory is accepted as one person.
        time_values : NDArray
            One finite value per trajectory column.

        Returns
        -------
        NDArray
            Log likelihoods with shape ``(n_persons, n_classes)``.

        Notes
        -----
        The marginal trajectory covariance is
        ``intercept_var * 1 1^T + slope_var * t t^T + residual_variance * I``.
        It is evaluated through its low-rank random-effect basis without
        constructing a dense time-by-time covariance matrix.
        """
        values, times, trajectories, variances, observation_mask = (
            self._validated_trajectory_data(
                observations,
                time_values,
            )
        )
        patterns = self._prepare_observation_patterns(
            values,
            times,
            variances,
            observation_mask,
        )
        return self._class_log_likelihood_from_patterns(
            values,
            trajectories,
            patterns,
        )

    def _posterior_from_log_likelihoods(
        self,
        log_likelihoods: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Normalize class probabilities in log space."""
        proportions = self._validated_class_proportions()

        log_proportions = np.full(self.n_classes, -np.inf, dtype=np.float64)
        positive = proportions > 0.0
        log_proportions[positive] = np.log(proportions[positive])
        log_joint = log_likelihoods + log_proportions
        log_normalizer = logsumexp(log_joint, axis=1)
        posteriors = np.exp(log_joint - log_normalizer[:, None])
        return posteriors, log_normalizer

    def classify(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Classify persons into latent classes.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints). Use ``NaN`` for
            missing occasions.
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Class assignments (n_persons,).
        """
        return np.argmax(
            self.posterior_probabilities(observations, time_values), axis=1
        )

    def posterior_probabilities(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute posterior class probabilities.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints). Use ``NaN`` for
            missing occasions.
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Posterior probabilities (n_persons, n_classes).
        """
        log_likelihoods = self.class_log_likelihood(observations, time_values)
        posteriors, _ = self._posterior_from_log_likelihoods(log_likelihoods)
        return posteriors

    def predict_trajectories(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
        prediction_times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Predict person-level latent trajectories from observed histories.

        Predictions average over posterior class probabilities and condition
        each class's random intercept and slope on the person's observed
        values. They represent the latent trajectory mean and exclude new
        occasion-specific residual error.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories with shape ``(n_persons, n_timepoints)``.
            Use ``NaN`` for missing occasions; every person must have at least
            one observed value.
        time_values : NDArray
            One finite time value per observation column.
        prediction_times : NDArray, optional
            Finite times at which to predict. The observed time grid is used
            when omitted.

        Returns
        -------
        NDArray
            Posterior mean latent trajectories with shape
            ``(n_persons, n_prediction_times)``.
        """
        predictions, _ = self._trajectory_prediction_moments(
            observations,
            time_values,
            prediction_times,
            include_residual=False,
            compute_variance=False,
        )
        return predictions

    def predict_trajectory_moments(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
        prediction_times: NDArray[np.float64] | None = None,
        *,
        include_residual: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return exact posterior predictive means and pointwise variances.

        The variance combines uncertainty over latent classes with conditional
        random-intercept and random-slope uncertainty. By default it describes
        the latent trajectory. Set ``include_residual=True`` to add the model's
        occasion-specific residual variance for future observed values.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories with shape ``(n_persons, n_timepoints)``.
            Use ``NaN`` for missing occasions; every person must have at least
            one observed value.
        time_values : NDArray
            One finite time value per observation column.
        prediction_times : NDArray, optional
            Finite times at which to predict. The observed time grid is used
            when omitted.
        include_residual : bool
            Whether to include new occasion-specific residual variance.

        Returns
        -------
        tuple
            Posterior predictive mean and marginal variance arrays, each with
            shape ``(n_persons, n_prediction_times)``.
        """
        if not isinstance(include_residual, (bool, np.bool_)):
            raise TypeError("include_residual must be a boolean")
        predictions, prediction_variances = self._trajectory_prediction_moments(
            observations,
            time_values,
            prediction_times,
            include_residual=bool(include_residual),
            compute_variance=True,
        )
        if prediction_variances is None:  # pragma: no cover - internal invariant
            raise RuntimeError("trajectory prediction variance was not computed")
        return predictions, prediction_variances

    def _trajectory_prediction_moments(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
        prediction_times: NDArray[np.float64] | None,
        *,
        include_residual: bool,
        compute_variance: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Compute grouped conditional trajectory moments."""
        values, times, trajectories, variances, observation_mask = (
            self._validated_trajectory_data(
                observations,
                time_values,
            )
        )
        if prediction_times is None:
            prediction_values = times
            prediction_trajectories = trajectories
        else:
            prediction_values = self._validated_time_values(prediction_times)
            prediction_changepoint = (
                self._resolved_changepoint(times)
                if self.growth_type == "piecewise"
                else None
            )
            prediction_trajectories = self._validated_class_trajectories(
                prediction_values,
                piecewise_changepoint=prediction_changepoint,
            )

        patterns = self._prepare_observation_patterns(
            values,
            times,
            variances,
            observation_mask,
        )
        log_likelihoods = self._class_log_likelihood_from_patterns(
            values,
            trajectories,
            patterns,
        )
        posteriors, _ = self._posterior_from_log_likelihoods(log_likelihoods)
        intercept_variance, slope_variance, _ = variances
        predictions = np.empty(
            (values.shape[0], prediction_values.size),
            dtype=np.float64,
        )
        prediction_variances = np.empty_like(predictions) if compute_variance else None
        prior_random_variance = (
            intercept_variance + slope_variance * prediction_values**2
            if compute_variance
            else None
        )

        for pattern in patterns:
            pattern_posteriors = posteriors[pattern.rows]
            if compute_variance:
                pattern_predictions = np.zeros(
                    (pattern.rows.size, prediction_values.size),
                    dtype=np.float64,
                )
                between_class_variance = np.zeros_like(pattern_predictions)
                cumulative_class_mass = np.zeros(pattern.rows.size, dtype=np.float64)
            else:
                pattern_predictions = pattern_posteriors @ prediction_trajectories
                between_class_variance = None
                cumulative_class_mass = None
            conditional_variance = (
                prior_random_variance.copy()
                if prior_random_variance is not None
                else None
            )
            has_random_effects = intercept_variance > 0.0 or slope_variance > 0.0
            if has_random_effects:
                observed_times = times[pattern.columns]
                cross_covariance = np.full(
                    (prediction_values.size, pattern.columns.size),
                    intercept_variance,
                    dtype=np.float64,
                )
                if slope_variance > 0.0:
                    cross_covariance += slope_variance * np.outer(
                        prediction_values,
                        observed_times,
                    )
                if conditional_variance is not None:
                    precision_cross_covariance = pattern.covariance.solve(
                        cross_covariance.T
                    )
                    conditional_variance -= np.einsum(
                        "ij,ji->i",
                        cross_covariance,
                        precision_cross_covariance,
                    )
                pattern_values = values[np.ix_(pattern.rows, pattern.columns)]
            else:
                cross_covariance = None
                pattern_values = None

            if has_random_effects or compute_variance:
                for class_index in range(self.n_classes):
                    if pattern_values is not None and cross_covariance is not None:
                        residuals = (
                            pattern_values - trajectories[class_index, pattern.columns]
                        )
                        precision_residuals = pattern.covariance.solve(residuals.T).T
                        correction = precision_residuals @ cross_covariance.T
                    else:
                        correction = 0.0

                    class_weights = pattern_posteriors[:, class_index]
                    if (
                        between_class_variance is not None
                        and cumulative_class_mass is not None
                    ):
                        class_predictions = (
                            prediction_trajectories[class_index] + correction
                        )
                        updated_class_mass = cumulative_class_mass + class_weights
                        relative_weight = np.divide(
                            class_weights,
                            updated_class_mass,
                            out=np.zeros_like(class_weights),
                            where=updated_class_mass > 0.0,
                        )
                        difference = class_predictions - pattern_predictions
                        updated_predictions = (
                            pattern_predictions + relative_weight[:, None] * difference
                        )
                        between_class_variance += class_weights[:, None] * (
                            difference * (class_predictions - updated_predictions)
                        )
                        pattern_predictions = updated_predictions
                        cumulative_class_mass = updated_class_mass
                    else:
                        pattern_predictions += class_weights[:, None] * correction

            predictions[pattern.rows] = pattern_predictions
            if (
                prediction_variances is not None
                and between_class_variance is not None
                and cumulative_class_mass is not None
                and conditional_variance is not None
            ):
                if include_residual:
                    conditional_variance += variances[2]
                prediction_variances[pattern.rows] = np.maximum(
                    conditional_variance
                    + np.divide(
                        between_class_variance,
                        cumulative_class_mass[:, None],
                        out=np.zeros_like(between_class_variance),
                        where=cumulative_class_mass[:, None] > 0.0,
                    ),
                    0.0,
                )

        return predictions, prediction_variances

    def simulate(
        self,
        n_persons: int,
        time_values: NDArray[np.float64] | None = None,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Simulate data from the mixture model.

        Parameters
        ----------
        n_persons : int
            Number of persons.
        time_values : NDArray, optional
            Time points.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple
            (observations, true_classes)
        """
        if (
            isinstance(n_persons, bool)
            or not isinstance(n_persons, (int, np.integer))
            or n_persons < 1
        ):
            raise ValueError("n_persons must be a positive integer")
        n_persons = int(n_persons)

        if time_values is None:
            if (
                isinstance(self.n_timepoints, bool)
                or not isinstance(self.n_timepoints, (int, np.integer))
                or self.n_timepoints < 1
            ):
                raise ValueError("n_timepoints must be a positive integer")
            times = np.arange(int(self.n_timepoints), dtype=np.float64)
        else:
            times = self._validated_time_values(time_values)

        trajectories = self._validated_class_trajectories(times)
        proportions = self._validated_class_proportions()
        intercept_variance, slope_variance, residual_variance = (
            self._validated_variance_components()
        )

        rng = np.random.default_rng(seed)
        true_classes = rng.choice(
            self.n_classes,
            size=n_persons,
            p=proportions,
        )

        n_times = times.size
        observations = np.empty((n_persons, n_times), dtype=np.float64)
        random_values_per_person = n_times + 2
        persons_per_chunk = max(
            1,
            _GROWTH_MIXTURE_MAX_RANDOM_VALUES // random_values_per_person,
        )
        intercept_scale = np.sqrt(intercept_variance)
        slope_scale = np.sqrt(slope_variance)
        residual_scale = np.sqrt(residual_variance)

        for start in range(0, n_persons, persons_per_chunk):
            stop = min(start + persons_per_chunk, n_persons)
            random_values = rng.standard_normal(
                (stop - start, random_values_per_person)
            )
            observations[start:stop] = (
                trajectories[true_classes[start:stop]]
                + random_values[:, :1] * intercept_scale
                + random_values[:, 1:2] * slope_scale * times[None, :]
                + random_values[:, 2:] * residual_scale
            )

        return observations, true_classes

    def fit_em(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> dict:
        """Fit model using EM algorithm.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints). Use ``NaN`` for
            missing occasions; every person must have at least one observed
            value.
        time_values : NDArray
            Time points.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        dict
            Estimation results.
        """
        observations, times, _, variances, observation_mask = (
            self._validated_trajectory_data(
                observations,
                time_values,
            )
        )
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        try:
            tolerance = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("tol must be numeric") from exc
        if isinstance(tol, bool) or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tol must be positive and finite")
        max_iter = int(max_iter)

        design = np.column_stack([np.ones(len(times)), times])
        if self.growth_type == "quadratic":
            design = np.column_stack([design, times**2])
        elif self.growth_type == "piecewise":
            hinge = np.maximum(times - self._resolved_changepoint(times), 0.0)
            design = np.column_stack([design, hinge])
        observed_times = (
            np.ones(observations.shape[1], dtype=np.bool_)
            if observation_mask is None
            else np.any(observation_mask, axis=0)
        )
        if np.linalg.matrix_rank(design[observed_times]) < design.shape[1]:
            raise ValueError(
                "observed time_values must provide a full-rank growth design"
            )

        self.class_proportions = self._validated_class_proportions().copy()
        self.class_intercepts = np.asarray(
            self.class_intercepts,
            dtype=np.float64,
        ).copy()
        self.class_slopes = np.asarray(
            self.class_slopes,
            dtype=np.float64,
        ).copy()
        if self.growth_type == "quadratic":
            self.class_quadratics = np.asarray(
                self.class_quadratics,
                dtype=np.float64,
            ).copy()
        elif self.growth_type == "piecewise":
            self.class_post_slopes = np.asarray(
                self.class_post_slopes,
                dtype=np.float64,
            ).copy()

        patterns = self._prepare_observation_patterns(
            observations,
            times,
            variances,
            observation_mask,
        )
        complete_data = (
            len(patterns) == 1 and patterns[0].columns.size == observations.shape[1]
        )
        pattern_updates = (
            []
            if complete_data
            else [
                (
                    pattern,
                    design[pattern.columns],
                    design[pattern.columns].T
                    @ pattern.covariance.solve(design[pattern.columns]),
                )
                for pattern in patterns
            ]
        )

        converged = False
        for iteration in range(max_iter):
            trajectories = self._validated_class_trajectories(times)
            log_likelihoods = self._class_log_likelihood_from_patterns(
                observations,
                trajectories,
                patterns,
            )
            posteriors, _ = self._posterior_from_log_likelihoods(log_likelihoods)

            prev_proportions = self.class_proportions.copy()
            prev_intercepts = self.class_intercepts.copy()
            prev_slopes = self.class_slopes.copy()
            if self.growth_type == "quadratic":
                prev_quadratics = self.class_quadratics.copy()
            elif self.growth_type == "piecewise":
                prev_post_slopes = self.class_post_slopes.copy()

            self.class_proportions = np.mean(posteriors, axis=0)

            class_mass = np.sum(posteriors, axis=0)
            active = class_mass >= PROB_EPSILON
            safe_mass = np.where(active, class_mass, 1.0)
            if complete_data:
                weighted_means = (posteriors.T @ observations) / safe_mass[:, None]
                coefficients = np.linalg.lstsq(
                    design,
                    weighted_means.T,
                    rcond=None,
                )[0].T
            else:
                normal_matrices = np.zeros(
                    (self.n_classes, design.shape[1], design.shape[1]),
                    dtype=np.float64,
                )
                right_hand_sides = np.zeros(
                    (self.n_classes, design.shape[1]),
                    dtype=np.float64,
                )
                for pattern, pattern_design, precision_gram in pattern_updates:
                    pattern_posteriors = posteriors[pattern.rows]
                    pattern_mass = np.sum(pattern_posteriors, axis=0)
                    weighted_observations = (
                        pattern_posteriors.T
                        @ observations[np.ix_(pattern.rows, pattern.columns)]
                    )
                    precision_weighted_observations = pattern.covariance.solve(
                        weighted_observations.T
                    ).T
                    normal_matrices += pattern_mass[:, None, None] * precision_gram
                    right_hand_sides += precision_weighted_observations @ pattern_design

                coefficients = np.column_stack(
                    [self.class_intercepts, self.class_slopes]
                )
                if self.growth_type == "quadratic":
                    coefficients = np.column_stack(
                        [coefficients, self.class_quadratics]
                    )
                elif self.growth_type == "piecewise":
                    coefficients = np.column_stack(
                        [
                            coefficients,
                            self.class_post_slopes - self.class_slopes,
                        ]
                    )
                coefficients[active] = np.linalg.solve(
                    normal_matrices[active],
                    right_hand_sides[active, :, None],
                )[:, :, 0]
            self.class_intercepts[active] = coefficients[active, 0]
            self.class_slopes[active] = coefficients[active, 1]
            if self.growth_type == "quadratic":
                self.class_quadratics[active] = coefficients[active, 2]
            elif self.growth_type == "piecewise":
                self.class_post_slopes[active] = (
                    coefficients[active, 1] + coefficients[active, 2]
                )

            prop_change = np.max(np.abs(self.class_proportions - prev_proportions))
            int_change = np.max(np.abs(self.class_intercepts - prev_intercepts))
            slope_change = np.max(np.abs(self.class_slopes - prev_slopes))
            parameter_change = max(prop_change, int_change, slope_change)
            if self.growth_type == "quadratic":
                quadratic_change = np.max(
                    np.abs(self.class_quadratics - prev_quadratics)
                )
                parameter_change = max(parameter_change, quadratic_change)
            elif self.growth_type == "piecewise":
                post_slope_change = np.max(
                    np.abs(self.class_post_slopes - prev_post_slopes)
                )
                parameter_change = max(parameter_change, post_slope_change)

            if parameter_change < tolerance:
                converged = True
                break

        final_trajectories = self._validated_class_trajectories(times)
        final_log_likelihoods = self._class_log_likelihood_from_patterns(
            observations,
            final_trajectories,
            patterns,
        )
        final_posteriors, log_normalizer = self._posterior_from_log_likelihoods(
            final_log_likelihoods
        )
        classifications = np.argmax(final_posteriors, axis=1)
        log_likelihood = float(np.sum(log_normalizer))

        return {
            "classifications": classifications,
            "posteriors": final_posteriors,
            "log_likelihood": log_likelihood,
            "n_iterations": iteration + 1,
            "converged": converged,
        }

    @property
    def n_fitted_parameters(self) -> int:
        """Number of parameters updated by :meth:`fit_em`.

        The count includes one intercept and slope per class, one additional
        coefficient per class for quadratic or piecewise growth, and
        ``n_classes - 1`` independent mixture weights. The shared changepoint
        and variance components are fixed during the current EM fit and are
        therefore excluded.
        """
        coefficients_per_class = (
            3 if self.growth_type in {"quadratic", "piecewise"} else 2
        )
        return self.n_classes * coefficients_per_class + self.n_classes - 1

    @staticmethod
    def _entropy_from_posteriors(posteriors: NDArray[np.float64]) -> float:
        """Calculate mean classification entropy from posterior weights."""
        clipped = np.clip(posteriors, PROB_EPSILON, 1 - PROB_EPSILON)
        return float(-np.mean(np.sum(clipped * np.log(clipped), axis=1)))

    def fit(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> GrowthMixtureResult:
        """Fit the model and return structured diagnostics.

        This is the result-oriented counterpart to :meth:`fit_em`, which
        continues to return its compatibility mapping.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories with shape ``(n_persons, n_timepoints)``.
            Use ``NaN`` for missing occasions; every person must have at least
            one observed value.
        time_values : NDArray
            One time value per observation column.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Positive convergence tolerance.

        Returns
        -------
        GrowthMixtureResult
            Fitted classifications, posteriors, information criteria, entropy,
            and convergence diagnostics.
        """
        fit_state = self.fit_em(
            observations,
            time_values,
            max_iter=max_iter,
            tol=tol,
        )
        classifications = np.asarray(fit_state["classifications"], dtype=np.int_)
        posteriors = np.asarray(fit_state["posteriors"], dtype=np.float64)
        log_likelihood = float(fit_state["log_likelihood"])
        parameter_count = self.n_fitted_parameters
        n_observations = classifications.size

        return GrowthMixtureResult(
            model=self,
            classifications=classifications,
            posteriors=posteriors,
            log_likelihood=log_likelihood,
            aic=2.0 * parameter_count - 2.0 * log_likelihood,
            bic=np.log(n_observations) * parameter_count - 2.0 * log_likelihood,
            entropy=self._entropy_from_posteriors(posteriors),
            converged=bool(fit_state["converged"]),
            n_iterations=int(fit_state["n_iterations"]),
        )

    def entropy(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> float:
        """Compute classification entropy.

        Lower entropy indicates better class separation.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories. Use ``NaN`` for missing occasions.
        time_values : NDArray
            Time points.

        Returns
        -------
        float
            Entropy value.
        """
        posteriors = self.posterior_probabilities(observations, time_values)
        return self._entropy_from_posteriors(posteriors)


@dataclass
class GrowthMixtureResult:
    """Result from growth mixture model estimation."""

    model: GrowthMixtureModel
    classifications: NDArray[np.int_]
    posteriors: NDArray[np.float64]
    log_likelihood: float
    aic: float
    bic: float
    entropy: float
    converged: bool
    n_iterations: int

    @property
    def n_observations(self) -> int:
        """Number of fitted trajectories."""
        return int(self.classifications.size)

    @property
    def n_parameters(self) -> int:
        """Number of parameters updated during fitting."""
        return self.model.n_fitted_parameters

    @property
    def class_counts(self) -> NDArray[np.int_]:
        """Hard-classification counts in class order."""
        return np.bincount(
            self.classifications,
            minlength=self.model.n_classes,
        )[: self.model.n_classes]

    @property
    def class_shares(self) -> NDArray[np.float64]:
        """Hard-classification shares in class order."""
        return self.class_counts / self.n_observations

    def summary(self) -> str:
        """Generate a human-readable estimation summary."""
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'Growth Mixture Model Results':^{width}}")
        lines.append("=" * width)

        lines.append(f"Number of Classes:  {self.model.n_classes}")
        lines.append(f"Observations:       {self.n_observations}")
        lines.append(f"Fitted Parameters:  {self.n_parameters}")
        lines.append(f"Growth Type:        {self.model.growth_type}")
        if self.model.growth_type == "piecewise":
            changepoint = (
                "automatic midpoint"
                if self.model.changepoint is None
                else f"{self.model.changepoint:.4f}"
            )
            lines.append(f"Changepoint:        {changepoint}")
        lines.append(f"Log-Likelihood:     {self.log_likelihood:.4f}")
        lines.append(f"AIC:                {self.aic:.4f}")
        lines.append(f"BIC:                {self.bic:.4f}")
        lines.append(f"Entropy:            {self.entropy:.4f}")
        lines.append(f"Converged:          {self.converged}")
        lines.append(f"Iterations:         {self.n_iterations}")
        lines.append("-" * width)

        lines.append("\nClass Parameters:")
        counts = self.class_counts
        shares = self.class_shares
        for k in range(self.model.n_classes):
            parameters = (
                f"  Class {k}: N={counts[k]} ({100 * shares[k]:.1f}%), "
                f"Intercept={self.model.class_intercepts[k]:.3f}"
            )
            if self.model.growth_type == "piecewise":
                parameters += (
                    f", Pre-Slope={self.model.class_slopes[k]:.3f}, "
                    f"Post-Slope={self.model.class_post_slopes[k]:.3f}"
                )
            else:
                parameters += f", Slope={self.model.class_slopes[k]:.3f}"
            if self.model.growth_type == "quadratic":
                parameters += f", Quadratic={self.model.class_quadratics[k]:.3f}"
            lines.append(parameters)

        lines.append("=" * width)
        return "\n".join(lines)
