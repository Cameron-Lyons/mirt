"""Gibbs Sampling for Dynamic and Longitudinal IRT Models.

This module provides MCMC estimation for:
- Bayesian Knowledge Tracing (BKT)
- Longitudinal IRT with growth curves
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.models.dynamic import (
    BKTModel,
    BKTResult,
    LongitudinalIRTModel,
    LongitudinalResult,
)

_LONGITUDINAL_MAX_PROBABILITY_VALUES = 1_000_000


def _binary_log_likelihood(
    probabilities: NDArray[np.float64],
    responses: NDArray[np.int_],
    *,
    axis: int,
) -> NDArray[np.float64]:
    """Sum clipped binary log-likelihood contributions along one axis."""
    np.clip(
        probabilities,
        PROB_EPSILON,
        1.0 - PROB_EPSILON,
        out=probabilities,
    )
    contributions = np.zeros_like(probabilities)
    correct = responses == 1
    incorrect = responses == 0
    np.log(probabilities, out=contributions, where=correct)
    np.log1p(-probabilities, out=contributions, where=incorrect)
    return np.sum(contributions, axis=axis)


def _longitudinal_log_likelihood_rows(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    model: LongitudinalIRTModel,
) -> NDArray[np.float64]:
    """Return one binary-response log-likelihood per person-occasion row."""
    n_rows, n_items = responses.shape
    log_likelihood = np.empty(n_rows, dtype=np.float64)
    rows_per_chunk = max(1, _LONGITUDINAL_MAX_PROBABILITY_VALUES // n_items)

    for start in range(0, n_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_rows)
        chunk_responses = responses[start:stop]
        probabilities = np.asarray(
            model.probability(theta[start:stop]),
            dtype=np.float64,
        )
        log_likelihood[start:stop] = _binary_log_likelihood(
            probabilities,
            chunk_responses,
            axis=1,
        )

    return log_likelihood


@dataclass
class BKTPriors:
    """Prior specifications for BKT parameters.

    All priors are Beta distributions specified by (alpha, beta) shape parameters.
    """

    p_init: tuple[float, float] = (1.0, 1.0)
    p_learn: tuple[float, float] = (1.0, 1.0)
    p_forget: tuple[float, float] = (1.0, 1.0)
    p_slip: tuple[float, float] = (1.0, 1.0)
    p_guess: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("p_init", "p_learn", "p_forget", "p_slip", "p_guess"):
            try:
                shapes = np.asarray(getattr(self, name), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} prior must contain two finite positive values"
                ) from exc
            if (
                shapes.shape != (2,)
                or not np.all(np.isfinite(shapes))
                or np.any(shapes <= 0.0)
            ):
                raise ValueError(
                    f"{name} prior must contain two finite positive values"
                )
            setattr(self, name, (float(shapes[0]), float(shapes[1])))


@dataclass
class LongitudinalPriors:
    """Prior specifications for longitudinal IRT parameters."""

    discrimination_mean: float = 1.0
    discrimination_var: float = 1.0
    difficulty_mean: float = 0.0
    difficulty_var: float = 4.0
    growth_mean_prior_mean: NDArray[np.float64] | None = None
    growth_mean_prior_cov: NDArray[np.float64] | None = None
    growth_cov_prior_df: float = 4.0
    growth_cov_prior_scale: NDArray[np.float64] | None = None
    residual_var_prior_shape: float = 2.0
    residual_var_prior_rate: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = (
            "discrimination_mean",
            "discrimination_var",
            "difficulty_var",
            "growth_cov_prior_df",
            "residual_var_prior_shape",
            "residual_var_prior_rate",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be finite and positive")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and positive") from exc
            if not np.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, normalized)

        if isinstance(self.difficulty_mean, (bool, np.bool_)):
            raise ValueError("difficulty_mean must be finite")
        try:
            self.difficulty_mean = float(self.difficulty_mean)
        except (TypeError, ValueError) as exc:
            raise ValueError("difficulty_mean must be finite") from exc
        if not np.isfinite(self.difficulty_mean):
            raise ValueError("difficulty_mean must be finite")

        if self.growth_mean_prior_mean is not None:
            self.growth_mean_prior_mean = np.asarray(
                self.growth_mean_prior_mean,
                dtype=np.float64,
            ).copy()
            if self.growth_mean_prior_mean.ndim != 1 or not np.all(
                np.isfinite(self.growth_mean_prior_mean)
            ):
                raise ValueError(
                    "growth_mean_prior_mean must be a finite one-dimensional array"
                )

        for name in ("growth_mean_prior_cov", "growth_cov_prior_scale"):
            values = getattr(self, name)
            if values is None:
                continue
            matrix = np.asarray(values, dtype=np.float64).copy()
            if (
                matrix.ndim != 2
                or matrix.shape[0] != matrix.shape[1]
                or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
                or np.min(np.linalg.eigvalsh(matrix)) <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive-definite matrix")
            setattr(self, name, matrix)

    def _validate_growth_dimension(self, n_growth: int) -> None:
        """Validate growth prior shapes once the growth model is known."""
        expected_vector = (n_growth,)
        expected_matrix = (n_growth, n_growth)
        if (
            self.growth_mean_prior_mean is not None
            and self.growth_mean_prior_mean.shape != expected_vector
        ):
            raise ValueError(
                f"growth_mean_prior_mean must have shape {expected_vector}"
            )
        for name in ("growth_mean_prior_cov", "growth_cov_prior_scale"):
            values = getattr(self, name)
            if values is not None and values.shape != expected_matrix:
                raise ValueError(f"{name} must have shape {expected_matrix}")
        if self.growth_cov_prior_df < n_growth:
            raise ValueError(
                f"growth_cov_prior_df must be at least {n_growth} for this model"
            )


class BKTGibbsSampler:
    """Gibbs sampler for Bayesian Knowledge Tracing.

    Uses Baum-Welch style updates for hidden states and
    Beta-Binomial conjugacy for parameter sampling.
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burnin: int = 500,
        thin: int = 1,
        priors: BKTPriors | None = None,
        verbose: bool = False,
        seed: int | None = None,
        use_rust: bool = True,
    ) -> None:
        """Initialize BKT Gibbs sampler.

        Parameters
        ----------
        n_iter : int
            Number of MCMC iterations
        burnin : int
            Burnin iterations to discard
        thin : int
            Thinning interval
        priors : BKTPriors, optional
            Prior specifications
        verbose : bool
            Whether to print progress
        seed : int, optional
            Random seed
        use_rust : bool
            Use compiled BKT inference kernels when available
        """
        if (
            isinstance(n_iter, (bool, np.bool_))
            or not isinstance(n_iter, (int, np.integer))
            or n_iter < 1
        ):
            raise ValueError("n_iter must be a positive integer")
        if (
            isinstance(burnin, (bool, np.bool_))
            or not isinstance(burnin, (int, np.integer))
            or burnin < 0
        ):
            raise ValueError("burnin must be a non-negative integer")
        if burnin >= n_iter:
            raise ValueError("burnin must be less than n_iter")
        if (
            isinstance(thin, (bool, np.bool_))
            or not isinstance(thin, (int, np.integer))
            or thin < 1
        ):
            raise ValueError("thin must be a positive integer")
        if priors is not None and not isinstance(priors, BKTPriors):
            raise TypeError("priors must be a BKTPriors instance or None")
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean")
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if not isinstance(use_rust, (bool, np.bool_)):
            raise TypeError("use_rust must be a boolean")

        self.n_iter = int(n_iter)
        self.burnin = int(burnin)
        self.thin = int(thin)
        self.priors = BKTPriors() if priors is None else priors
        self.verbose = bool(verbose)
        self.seed = None if seed is None else int(seed)
        self.use_rust = bool(use_rust)

    def fit(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        n_skills: int | None = None,
        allow_forgetting: bool = False,
    ) -> BKTResult:
        """Fit BKT model using Gibbs sampling.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_persons, n_trials)
        skill_assignments : NDArray
            Skill index for each trial (n_trials,)
        n_skills : int, optional
            Number of skills. Inferred from skill_assignments if not provided.
        allow_forgetting : bool
            Whether to estimate forgetting parameter

        Returns
        -------
        BKTResult
            Estimation results
        """
        rng = np.random.default_rng(self.seed)
        responses, skill_assignments, model = self._prepare_fit(
            responses,
            skill_assignments,
            n_skills,
            allow_forgetting,
        )
        n_persons, n_trials = responses.shape
        n_skills = model.n_skills

        # Preserve the established seeded chain while replacing later draws
        # with an equivalent vectorized path.
        learning_states = rng.integers(0, 2, size=(n_persons, n_trials), dtype=np.int32)
        skill_trials = model._skill_trials(skill_assignments)

        chains: dict[str, list[NDArray]] = {
            "p_init": [],
            "p_learn": [],
            "p_forget": [],
            "p_slip": [],
            "p_guess": [],
            "log_likelihood": [],
        }

        for iteration in range(self.n_iter):
            learning_states = self._sample_states_ffbs_batch(
                responses,
                skill_assignments,
                model,
                rng,
                skill_trials,
            )

            self._sample_p_init(
                model, learning_states, skill_assignments, rng, skill_trials
            )
            self._sample_p_learn(
                model, learning_states, skill_assignments, rng, skill_trials
            )
            if allow_forgetting:
                self._sample_p_forget(
                    model, learning_states, skill_assignments, rng, skill_trials
                )
            self._sample_p_slip(
                model,
                responses,
                learning_states,
                skill_assignments,
                rng,
                skill_trials,
            )
            self._sample_p_guess(
                model,
                responses,
                learning_states,
                skill_assignments,
                rng,
                skill_trials,
            )

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                chains["p_init"].append(model.p_init.copy())
                chains["p_learn"].append(model.p_learn.copy())
                chains["p_forget"].append(model.p_forget.copy())
                chains["p_slip"].append(model.p_slip.copy())
                chains["p_guess"].append(model.p_guess.copy())

                ll = self._compute_log_likelihood(responses, skill_assignments, model)
                chains["log_likelihood"].append(ll)

            if self.verbose and (iteration + 1) % 200 == 0:
                ll = self._compute_log_likelihood(responses, skill_assignments, model)
                print(f"Iteration {iteration + 1}/{self.n_iter}: LL = {ll:.4f}")

        model.p_init = np.mean(chains["p_init"], axis=0)
        model.p_learn = np.mean(chains["p_learn"], axis=0)
        model.p_forget = np.mean(chains["p_forget"], axis=0)
        model.p_slip = np.mean(chains["p_slip"], axis=0)
        model.p_guess = np.mean(chains["p_guess"], axis=0)

        learning_curves = np.zeros((n_persons, n_skills))
        skill_mastery = np.zeros((n_persons, n_skills))
        gamma, _ = model.forward_backward_batch(responses, skill_assignments)

        for skill_idx in range(n_skills):
            skill_mask = skill_assignments == skill_idx
            if np.any(skill_mask):
                learned = gamma[:, skill_mask, 1]
                skill_mastery[:, skill_idx] = learned[:, -1]
                learning_curves[:, skill_idx] = learned.mean(axis=1)

        ll_final = self._compute_log_likelihood(responses, skill_assignments, model)
        n_params = 4 * n_skills if not allow_forgetting else 5 * n_skills
        n_obs = np.sum(responses >= 0)
        aic = -2 * ll_final + 2 * n_params
        bic = -2 * ll_final + np.log(n_obs) * n_params

        return BKTResult(
            model=model,
            learning_curves=learning_curves,
            skill_mastery=skill_mastery,
            log_likelihood=ll_final,
            aic=aic,
            bic=bic,
            n_observations=int(n_obs),
            n_parameters=n_params,
            converged=True,
        )

    def _prepare_fit(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        n_skills: int | None,
        allow_forgetting: bool,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_], BKTModel]:
        """Validate fit inputs and construct the matching BKT model."""
        responses = np.asarray(responses)
        skill_assignments = np.asarray(skill_assignments)

        if responses.ndim != 2:
            raise ValueError("responses must have shape (n_persons, n_trials)")
        if responses.shape[0] == 0:
            raise ValueError("responses must contain at least one person")
        if responses.shape[1] == 0:
            raise ValueError("responses must contain at least one trial")
        if skill_assignments.ndim != 1:
            raise ValueError("skill_assignments must be one-dimensional")
        if len(skill_assignments) != responses.shape[1]:
            raise ValueError("skill_assignments length must match the number of trials")
        if not np.issubdtype(skill_assignments.dtype, np.integer):
            raise ValueError("skill_assignments must contain integer values")
        if np.any(skill_assignments < 0):
            raise ValueError("skill_assignments must contain non-negative values")
        if not isinstance(allow_forgetting, (bool, np.bool_)):
            raise TypeError("allow_forgetting must be a boolean")

        if n_skills is None:
            n_skills = int(np.max(skill_assignments)) + 1
        model = BKTModel(
            n_skills=n_skills,
            allow_forgetting=bool(allow_forgetting),
            use_rust=self.use_rust,
        )
        responses, skill_assignments = model._validate_batch(
            responses, skill_assignments
        )
        if not np.any(responses >= 0):
            raise ValueError("responses must contain at least one observed value")
        return responses, skill_assignments, model

    def _sample_states_ffbs(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        model: BKTModel,
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> NDArray[np.int_]:
        """Forward-filtering backward-sampling for hidden states."""
        n_trials = len(responses)

        alpha, _ = model.forward(responses, skill_assignments)

        states = np.zeros(n_trials, dtype=np.int32)
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for skill_idx, trial_indices in enumerate(skill_trials):
            if len(trial_indices) == 0:
                continue

            last_trial = int(trial_indices[-1])
            states[last_trial] = int(rng.random() < alpha[last_trial, 1])
            transition = model.transition_matrix(skill_idx)

            for trial, next_trial in zip(trial_indices[-2::-1], trial_indices[:0:-1]):
                p_state = alpha[trial] * transition[:, states[next_trial]]
                p_state /= p_state.sum() + 1e-300
                states[trial] = int(rng.random() < p_state[1])

        return states

    def _sample_states_ffbs_batch(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        model: BKTModel,
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> NDArray[np.int_]:
        """Vectorize FFBS state sampling across a shared skill layout."""
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)
        alpha, _ = model._forward_batch_shared_python(
            responses,
            skill_assignments,
            skill_trials,
        )
        states = np.zeros(responses.shape, dtype=np.int32)

        sampling_order = np.concatenate(
            [
                trial_indices[::-1]
                for trial_indices in skill_trials
                if len(trial_indices)
            ]
        )
        draws = np.empty(responses.shape, dtype=np.float64)
        draws[:, sampling_order] = rng.random(responses.shape)

        for skill_idx, trial_indices in enumerate(skill_trials):
            if len(trial_indices) == 0:
                continue

            last_trial = int(trial_indices[-1])
            states[:, last_trial] = draws[:, last_trial] < alpha[:, last_trial, 1]
            transition = model.transition_matrix(skill_idx)

            for trial, next_trial in zip(
                trial_indices[-2::-1],
                trial_indices[:0:-1],
                strict=True,
            ):
                next_states = states[:, next_trial]
                p_unlearned = alpha[:, trial, 0] * transition[0, next_states]
                p_learned = alpha[:, trial, 1] * transition[1, next_states]
                learned_probability = p_learned / (p_unlearned + p_learned + 1e-300)
                states[:, trial] = draws[:, trial] < learned_probability

        return states

    def _sample_p_init(
        self,
        model: BKTModel,
        learning_states: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> None:
        """Sample initial knowledge probabilities."""
        n_persons = learning_states.shape[0]
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for j, trial_indices in enumerate(skill_trials):
            if len(trial_indices) == 0:
                continue
            n_learned = int(np.count_nonzero(learning_states[:, int(trial_indices[0])]))
            alpha = self.priors.p_init[0] + n_learned
            beta = self.priors.p_init[1] + (n_persons - n_learned)
            model.p_init[j] = rng.beta(alpha, beta)

    def _sample_p_learn(
        self,
        model: BKTModel,
        learning_states: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> None:
        """Sample learning rate parameters."""
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for j, trial_indices in enumerate(skill_trials):
            previous = learning_states[:, trial_indices[:-1]]
            current = learning_states[:, trial_indices[1:]]
            eligible = previous == 0
            n_transitions = int(np.count_nonzero(eligible))
            n_learned = int(np.count_nonzero(eligible & (current == 1)))

            alpha = self.priors.p_learn[0] + n_learned
            beta = self.priors.p_learn[1] + (n_transitions - n_learned)
            model.p_learn[j] = rng.beta(max(alpha, 0.01), max(beta, 0.01))

    def _sample_p_forget(
        self,
        model: BKTModel,
        learning_states: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> None:
        """Sample forgetting rate parameters."""
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for j, trial_indices in enumerate(skill_trials):
            previous = learning_states[:, trial_indices[:-1]]
            current = learning_states[:, trial_indices[1:]]
            eligible = previous == 1
            n_transitions = int(np.count_nonzero(eligible))
            n_forgot = int(np.count_nonzero(eligible & (current == 0)))

            alpha = self.priors.p_forget[0] + n_forgot
            beta = self.priors.p_forget[1] + (n_transitions - n_forgot)
            model.p_forget[j] = rng.beta(max(alpha, 0.01), max(beta, 0.01))

    def _sample_p_slip(
        self,
        model: BKTModel,
        responses: NDArray[np.int_],
        learning_states: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> None:
        """Sample slip parameters."""
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for j, trial_indices in enumerate(skill_trials):
            skill_responses = responses[:, trial_indices]
            learned_observed = (learning_states[:, trial_indices] == 1) & (
                skill_responses >= 0
            )
            n_learned_trials = int(np.count_nonzero(learned_observed))
            n_slips = int(np.count_nonzero(learned_observed & (skill_responses == 0)))

            alpha = self.priors.p_slip[0] + n_slips
            beta = self.priors.p_slip[1] + (n_learned_trials - n_slips)
            model.p_slip[j] = rng.beta(max(alpha, 0.01), max(beta, 0.01))

    def _sample_p_guess(
        self,
        model: BKTModel,
        responses: NDArray[np.int_],
        learning_states: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        rng: np.random.Generator,
        skill_trials: list[NDArray[np.int_]] | None = None,
    ) -> None:
        """Sample guess parameters."""
        if skill_trials is None:
            skill_trials = model._skill_trials(skill_assignments)

        for j, trial_indices in enumerate(skill_trials):
            skill_responses = responses[:, trial_indices]
            unlearned_observed = (learning_states[:, trial_indices] == 0) & (
                skill_responses >= 0
            )
            n_unlearned_trials = int(np.count_nonzero(unlearned_observed))
            n_guessed = int(
                np.count_nonzero(unlearned_observed & (skill_responses == 1))
            )

            alpha = self.priors.p_guess[0] + n_guessed
            beta = self.priors.p_guess[1] + (n_unlearned_trials - n_guessed)
            model.p_guess[j] = rng.beta(max(alpha, 0.01), max(beta, 0.01))

    def _compute_log_likelihood(
        self,
        responses: NDArray[np.int_],
        skill_assignments: NDArray[np.int_],
        model: BKTModel,
    ) -> float:
        """Compute total log-likelihood."""
        _, log_likelihoods = model.forward_backward_batch(responses, skill_assignments)
        return float(log_likelihoods.sum())


class LongitudinalGibbsSampler:
    """Gibbs sampler for Longitudinal IRT with growth curves.

    Samples:
    1. Growth factors (η₀, η₁) given θ trajectories
    2. Item parameters given responses and θ
    3. Residual variance
    4. Population parameters
    """

    def __init__(
        self,
        n_iter: int = 2000,
        burnin: int = 500,
        thin: int = 1,
        priors: LongitudinalPriors | None = None,
        verbose: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize Longitudinal IRT Gibbs sampler.

        Parameters
        ----------
        n_iter : int
            Number of MCMC iterations
        burnin : int
            Burnin iterations to discard
        thin : int
            Thinning interval
        priors : LongitudinalPriors, optional
            Prior specifications
        verbose : bool
            Whether to print progress
        seed : int, optional
            Random seed
        """
        if (
            isinstance(n_iter, (bool, np.bool_))
            or not isinstance(n_iter, (int, np.integer))
            or n_iter < 1
        ):
            raise ValueError("n_iter must be a positive integer")
        if (
            isinstance(burnin, (bool, np.bool_))
            or not isinstance(burnin, (int, np.integer))
            or burnin < 0
        ):
            raise ValueError("burnin must be a non-negative integer")
        if burnin >= n_iter:
            raise ValueError("burnin must be less than n_iter")
        if (
            isinstance(thin, (bool, np.bool_))
            or not isinstance(thin, (int, np.integer))
            or thin < 1
        ):
            raise ValueError("thin must be a positive integer")
        if priors is not None and not isinstance(priors, LongitudinalPriors):
            raise TypeError("priors must be a LongitudinalPriors instance or None")
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean")
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")

        self.n_iter = int(n_iter)
        self.burnin = int(burnin)
        self.thin = int(thin)
        self.priors = LongitudinalPriors() if priors is None else priors
        self.verbose = bool(verbose)
        self.seed = None if seed is None else int(seed)

    def fit(
        self,
        responses: NDArray[np.int_],
        n_items: int | None = None,
        time_values: NDArray[np.float64] | None = None,
        growth_model: str = "linear",
    ) -> LongitudinalResult:
        """Fit Longitudinal IRT model using Gibbs sampling.

        Parameters
        ----------
        responses : NDArray
            Response array (n_persons, n_timepoints, n_items)
        n_items : int, optional
            Number of items per timepoint
        time_values : NDArray, optional
            Time values for each occasion (default: 0, 1, 2, ...)
        growth_model : str
            Growth model type ("linear" or "quadratic")

        Returns
        -------
        LongitudinalResult
            Estimation results
        """
        responses, time_values = self._prepare_fit_inputs(
            responses,
            n_items,
            time_values,
        )
        n_persons, n_timepoints, n_items = responses.shape
        rng = np.random.default_rng(self.seed)

        model = LongitudinalIRTModel(
            n_items=n_items,
            n_timepoints=n_timepoints,
            growth_model=growth_model,
        )

        n_growth = model.n_growth_factors
        self.priors._validate_growth_dimension(n_growth)
        growth_factors = rng.multivariate_normal(
            np.zeros(n_growth), np.eye(n_growth), size=n_persons
        )
        theta_trajectories = model.compute_theta(growth_factors, time_values)

        chains: dict[str, list[Any]] = {
            "growth_factors": [],
            "discrimination": [],
            "difficulty": [],
            "growth_mean": [],
            "growth_cov": [],
            "residual_variance": [],
            "log_likelihood": [],
        }

        for iteration in range(self.n_iter):
            theta_trajectories = self._sample_theta(
                responses, model, growth_factors, time_values, rng
            )

            growth_factors = self._sample_growth_factors(
                theta_trajectories, model, time_values, rng
            )

            self._sample_item_params(responses, theta_trajectories, model, rng)

            self._sample_population_params(growth_factors, model, rng)

            self._sample_residual_variance(
                theta_trajectories, growth_factors, model, time_values, rng
            )

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                chains["growth_factors"].append(growth_factors.copy())
                chains["discrimination"].append(model.discrimination.copy())
                chains["difficulty"].append(model.difficulty.copy())
                chains["growth_mean"].append(model.growth_mean.copy())
                chains["growth_cov"].append(model.growth_cov.copy())
                chains["residual_variance"].append(model.residual_variance)

                ll = self._compute_log_likelihood(responses, theta_trajectories, model)
                chains["log_likelihood"].append(ll)

            if self.verbose and (iteration + 1) % 200 == 0:
                ll = self._compute_log_likelihood(responses, theta_trajectories, model)
                print(f"Iteration {iteration + 1}/{self.n_iter}: LL = {ll:.4f}")

        model.discrimination = np.mean(chains["discrimination"], axis=0)
        model.difficulty = np.mean(chains["difficulty"], axis=0)
        model.growth_mean = np.mean(chains["growth_mean"], axis=0)
        model.growth_cov = np.mean(chains["growth_cov"], axis=0)
        model.residual_variance = np.mean(chains["residual_variance"])

        growth_factors_final = np.mean(chains["growth_factors"], axis=0)
        growth_factors_se = np.std(chains["growth_factors"], axis=0)
        theta_final = model.compute_theta(growth_factors_final, time_values)

        ll_final = self._compute_log_likelihood(responses, theta_final, model)
        n_params = 2 * n_items + n_growth + n_growth * (n_growth + 1) // 2 + 1
        n_obs = np.sum(responses >= 0)
        aic = -2 * ll_final + 2 * n_params
        bic = -2 * ll_final + np.log(n_obs) * n_params

        return LongitudinalResult(
            model=model,
            growth_factors=growth_factors_final,
            theta_trajectories=theta_final,
            growth_factor_se=growth_factors_se,
            log_likelihood=ll_final,
            aic=aic,
            bic=bic,
            converged=True,
            n_iterations=self.n_iter,
        )

    @staticmethod
    def _prepare_fit_inputs(
        responses: NDArray[np.int_],
        n_items: int | None,
        time_values: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Validate and normalize longitudinal response inputs."""
        response_values = np.asarray(responses)
        if response_values.ndim not in (2, 3):
            raise ValueError(
                "responses must have shape (n_persons, n_timepoints, n_items) "
                "or (n_persons, n_timepoints * n_items)"
            )
        if not np.issubdtype(response_values.dtype, np.integer):
            raise ValueError("responses must contain integer values")
        if np.any((response_values < -1) | (response_values > 1)):
            raise ValueError("responses must contain only -1, 0, or 1")

        if response_values.ndim == 2:
            if (
                isinstance(n_items, (bool, np.bool_))
                or not isinstance(n_items, (int, np.integer))
                or n_items < 1
            ):
                raise ValueError("n_items must be a positive integer for 2D responses")
            if response_values.shape[1] % int(n_items) != 0:
                raise ValueError("2D response columns must be divisible by n_items")
            n_timepoints = response_values.shape[1] // int(n_items)
            response_values = response_values.reshape(
                response_values.shape[0],
                n_timepoints,
                int(n_items),
            )
        elif n_items is not None:
            if (
                isinstance(n_items, (bool, np.bool_))
                or not isinstance(n_items, (int, np.integer))
                or int(n_items) != response_values.shape[2]
            ):
                raise ValueError("n_items must match the final response dimension")

        if any(size == 0 for size in response_values.shape):
            raise ValueError("responses must have non-empty dimensions")
        if not np.any(response_values >= 0):
            raise ValueError("responses must contain at least one observed value")

        n_timepoints = response_values.shape[1]
        if time_values is None:
            times = np.arange(n_timepoints, dtype=np.float64)
        else:
            try:
                times = np.asarray(time_values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"time_values must have shape ({n_timepoints},) and be finite"
                ) from exc
            if times.shape != (n_timepoints,) or not np.all(np.isfinite(times)):
                raise ValueError(
                    f"time_values must have shape ({n_timepoints},) and be finite"
                )

        return response_values.astype(np.int_, copy=False), times

    def _sample_theta(
        self,
        responses: NDArray[np.int_],
        model: LongitudinalIRTModel,
        growth_factors: NDArray[np.float64],
        time_values: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample theta trajectories using MH."""
        theta_pred = model.compute_theta(growth_factors, time_values)
        proposal_sd = 0.3
        current = theta_pred.reshape(-1).copy()
        proposed = np.empty_like(current)
        log_uniform = np.empty_like(current)

        # Keep proposal and acceptance draws interleaved for seeded compatibility.
        for index, value in enumerate(current):
            proposed[index] = value + rng.normal(0, proposal_sd)
            log_uniform[index] = np.log(rng.random())

        flat_responses = responses.reshape(-1, model.n_items)
        ll_current = _longitudinal_log_likelihood_rows(
            flat_responses,
            current,
            model,
        )
        ll_proposed = _longitudinal_log_likelihood_rows(
            flat_responses,
            proposed,
            model,
        )
        residual_sd = np.sqrt(model.residual_variance)
        prior_current = stats.norm.logpdf(current, current, residual_sd)
        prior_proposed = stats.norm.logpdf(proposed, current, residual_sd)
        accepted = log_uniform < (
            ll_proposed + prior_proposed - ll_current - prior_current
        )
        current[accepted] = proposed[accepted]
        return current.reshape(theta_pred.shape)

    def _sample_growth_factors(
        self,
        theta_trajectories: NDArray[np.float64],
        model: LongitudinalIRTModel,
        time_values: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample growth factors given theta trajectories."""
        n_persons = theta_trajectories.shape[0]
        n_timepoints = theta_trajectories.shape[1]
        n_growth = model.n_growth_factors

        X = np.ones((n_timepoints, n_growth))
        X[:, 1] = time_values
        if model.growth_model == "quadratic":
            X[:, 2] = time_values**2

        precision_prior = np.linalg.inv(model.growth_cov)
        precision_lik = X.T @ X / model.residual_variance
        precision_post = precision_prior + precision_lik
        cov_post = np.linalg.inv(precision_post)

        mean_lik = theta_trajectories @ X / model.residual_variance
        mean_prior = precision_prior @ model.growth_mean
        mean_post = (mean_lik + mean_prior) @ cov_post.T

        centered_draws = rng.multivariate_normal(
            np.zeros(n_growth),
            cov_post,
            size=n_persons,
        )
        centered_draws += mean_post
        return np.asarray(centered_draws, dtype=np.float64)

    def _sample_item_params(
        self,
        responses: NDArray[np.int_],
        theta_trajectories: NDArray[np.float64],
        model: LongitudinalIRTModel,
        rng: np.random.Generator,
    ) -> None:
        """Sample item parameters using MH."""
        n_items = responses.shape[2]
        proposal_sd_a = 0.1
        proposal_sd_b = 0.15
        a_current = model.discrimination.copy()
        b_current = model.difficulty.copy()
        a_proposed = np.empty_like(a_current)
        b_proposed = np.empty_like(b_current)
        log_uniform = np.empty(n_items, dtype=np.float64)

        # Preserve the original per-item draw order while batching likelihood work.
        for j in range(n_items):
            a_proposed[j] = np.clip(
                a_current[j] + rng.normal(0, proposal_sd_a),
                0.2,
                5.0,
            )
            b_proposed[j] = np.clip(
                b_current[j] + rng.normal(0, proposal_sd_b),
                -5.0,
                5.0,
            )
            log_uniform[j] = np.log(rng.random())

        flat_responses = responses.reshape(-1, n_items)
        flat_theta = theta_trajectories.reshape(-1)
        ll_current = np.zeros(n_items, dtype=np.float64)
        ll_proposed = np.zeros(n_items, dtype=np.float64)
        rows_per_chunk = max(
            1,
            _LONGITUDINAL_MAX_PROBABILITY_VALUES // n_items,
        )
        for start in range(0, flat_theta.size, rows_per_chunk):
            stop = min(start + rows_per_chunk, flat_theta.size)
            theta_chunk = flat_theta[start:stop, None]
            response_chunk = flat_responses[start:stop]
            current_probability = np.asarray(
                sigmoid(a_current * (theta_chunk - b_current)),
                dtype=np.float64,
            )
            proposed_probability = np.asarray(
                sigmoid(a_proposed * (theta_chunk - b_proposed)),
                dtype=np.float64,
            )
            ll_current += _binary_log_likelihood(
                current_probability,
                response_chunk,
                axis=0,
            )
            ll_proposed += _binary_log_likelihood(
                proposed_probability,
                response_chunk,
                axis=0,
            )

        prior_scale_a = np.sqrt(self.priors.discrimination_var)
        prior_scale_b = np.sqrt(self.priors.difficulty_var)
        prior_current = stats.lognorm.logpdf(
            a_current,
            s=prior_scale_a,
            scale=self.priors.discrimination_mean,
        ) + stats.norm.logpdf(
            b_current,
            self.priors.difficulty_mean,
            prior_scale_b,
        )
        prior_proposed = stats.lognorm.logpdf(
            a_proposed,
            s=prior_scale_a,
            scale=self.priors.discrimination_mean,
        ) + stats.norm.logpdf(
            b_proposed,
            self.priors.difficulty_mean,
            prior_scale_b,
        )
        accepted = log_uniform < (
            ll_proposed + prior_proposed - ll_current - prior_current
        )
        model.discrimination[accepted] = a_proposed[accepted]
        model.difficulty[accepted] = b_proposed[accepted]

    def _sample_population_params(
        self,
        growth_factors: NDArray[np.float64],
        model: LongitudinalIRTModel,
        rng: np.random.Generator,
    ) -> None:
        """Sample population mean and covariance of growth factors."""
        n_persons = growth_factors.shape[0]
        n_growth = growth_factors.shape[1]

        sample_mean = np.mean(growth_factors, axis=0)
        if self.priors.growth_mean_prior_cov is not None:
            prior_precision = np.linalg.inv(self.priors.growth_mean_prior_cov)
        else:
            prior_precision = np.eye(n_growth) * 0.01

        cov_inv = np.linalg.inv(model.growth_cov)
        post_precision = prior_precision + n_persons * cov_inv
        post_cov = np.linalg.inv(post_precision)

        if self.priors.growth_mean_prior_mean is not None:
            prior_mean = self.priors.growth_mean_prior_mean
        else:
            prior_mean = np.zeros(n_growth)

        post_mean = post_cov @ (
            prior_precision @ prior_mean + n_persons * cov_inv @ sample_mean
        )
        model.growth_mean = rng.multivariate_normal(post_mean, post_cov)

        centered = growth_factors - model.growth_mean
        S = centered.T @ centered

        if self.priors.growth_cov_prior_scale is not None:
            prior_scale = self.priors.growth_cov_prior_scale
        else:
            prior_scale = np.eye(n_growth)

        df_post = self.priors.growth_cov_prior_df + n_persons
        scale_post = prior_scale + S

        model.growth_cov = stats.invwishart.rvs(
            df=df_post, scale=scale_post, random_state=rng
        )

    def _sample_residual_variance(
        self,
        theta_trajectories: NDArray[np.float64],
        growth_factors: NDArray[np.float64],
        model: LongitudinalIRTModel,
        time_values: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> None:
        """Sample residual variance."""
        theta_pred = model.compute_theta(growth_factors, time_values)
        residuals = theta_trajectories - theta_pred

        n = residuals.size
        ss = np.sum(residuals**2)

        shape_post = self.priors.residual_var_prior_shape + n / 2
        rate_post = self.priors.residual_var_prior_rate + ss / 2

        model.residual_variance = float(
            stats.invgamma.rvs(shape_post, scale=rate_post, random_state=rng)
        )
        model.residual_variance = max(model.residual_variance, 0.01)

    def _compute_log_likelihood(
        self,
        responses: NDArray[np.int_],
        theta_trajectories: NDArray[np.float64],
        model: LongitudinalIRTModel,
    ) -> float:
        """Compute total log-likelihood."""
        row_log_likelihood = _longitudinal_log_likelihood_rows(
            responses.reshape(-1, model.n_items),
            theta_trajectories.reshape(-1),
            model,
        )
        return float(np.sum(row_log_likelihood))
