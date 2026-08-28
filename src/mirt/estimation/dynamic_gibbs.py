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
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.priors = priors or LongitudinalPriors()
        self.verbose = verbose
        self.seed = seed

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
        rng = np.random.default_rng(self.seed)

        if responses.ndim == 2:
            n_persons, total = responses.shape
            if n_items is None:
                raise ValueError("n_items required for 2D response matrix")
            n_timepoints = total // n_items
            responses = responses.reshape(n_persons, n_timepoints, n_items)
        else:
            n_persons, n_timepoints, n_items = responses.shape

        if time_values is None:
            time_values = np.arange(n_timepoints, dtype=np.float64)

        model = LongitudinalIRTModel(
            n_items=n_items,
            n_timepoints=n_timepoints,
            growth_model=growth_model,
        )

        n_growth = model.n_growth_factors
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

    def _sample_theta(
        self,
        responses: NDArray[np.int_],
        model: LongitudinalIRTModel,
        growth_factors: NDArray[np.float64],
        time_values: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample theta trajectories using MH."""
        n_persons = responses.shape[0]
        n_timepoints = responses.shape[1]

        theta_pred = model.compute_theta(growth_factors, time_values)

        theta = theta_pred.copy()
        proposal_sd = 0.3

        for i in range(n_persons):
            for t in range(n_timepoints):
                theta_prop = theta[i, t] + rng.normal(0, proposal_sd)

                prior_curr = stats.norm.logpdf(
                    theta[i, t], theta_pred[i, t], np.sqrt(model.residual_variance)
                )
                prior_prop = stats.norm.logpdf(
                    theta_prop, theta_pred[i, t], np.sqrt(model.residual_variance)
                )

                ll_curr = 0.0
                ll_prop = 0.0
                for j in range(model.n_items):
                    if responses[i, t, j] >= 0:
                        p_curr = model.probability(np.array([theta[i, t]]), j)[0]
                        p_prop = model.probability(np.array([theta_prop]), j)[0]
                        p_curr = np.clip(p_curr, PROB_EPSILON, 1 - PROB_EPSILON)
                        p_prop = np.clip(p_prop, PROB_EPSILON, 1 - PROB_EPSILON)

                        if responses[i, t, j] == 1:
                            ll_curr += np.log(p_curr)
                            ll_prop += np.log(p_prop)
                        else:
                            ll_curr += np.log(1 - p_curr)
                            ll_prop += np.log(1 - p_prop)

                log_alpha = (ll_prop + prior_prop) - (ll_curr + prior_curr)
                if np.log(rng.random()) < log_alpha:
                    theta[i, t] = theta_prop

        return theta

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

        growth_factors = np.zeros((n_persons, n_growth))

        for i in range(n_persons):
            precision_post = precision_prior + precision_lik
            cov_post = np.linalg.inv(precision_post)

            mean_lik = X.T @ theta_trajectories[i] / model.residual_variance
            mean_prior = precision_prior @ model.growth_mean
            mean_post = cov_post @ (mean_prior + mean_lik)

            growth_factors[i] = rng.multivariate_normal(mean_post, cov_post)

        return growth_factors

    def _sample_item_params(
        self,
        responses: NDArray[np.int_],
        theta_trajectories: NDArray[np.float64],
        model: LongitudinalIRTModel,
        rng: np.random.Generator,
    ) -> None:
        """Sample item parameters using MH."""
        n_persons, n_timepoints, n_items = responses.shape
        proposal_sd_a = 0.1
        proposal_sd_b = 0.15

        for j in range(n_items):
            a_curr = model.discrimination[j]
            b_curr = model.difficulty[j]

            a_prop = np.clip(a_curr + rng.normal(0, proposal_sd_a), 0.2, 5.0)
            b_prop = np.clip(b_curr + rng.normal(0, proposal_sd_b), -5.0, 5.0)

            ll_curr = 0.0
            ll_prop = 0.0

            for i in range(n_persons):
                for t in range(n_timepoints):
                    if responses[i, t, j] >= 0:
                        theta = theta_trajectories[i, t]

                        z_curr = a_curr * (theta - b_curr)
                        z_prop = a_prop * (theta - b_prop)

                        p_curr = sigmoid(z_curr)
                        p_prop = sigmoid(z_prop)

                        p_curr = np.clip(p_curr, PROB_EPSILON, 1 - PROB_EPSILON)
                        p_prop = np.clip(p_prop, PROB_EPSILON, 1 - PROB_EPSILON)

                        if responses[i, t, j] == 1:
                            ll_curr += np.log(p_curr)
                            ll_prop += np.log(p_prop)
                        else:
                            ll_curr += np.log(1 - p_curr)
                            ll_prop += np.log(1 - p_prop)

            prior_a_curr = stats.lognorm.logpdf(
                a_curr,
                s=np.sqrt(self.priors.discrimination_var),
                scale=self.priors.discrimination_mean,
            )
            prior_a_prop = stats.lognorm.logpdf(
                a_prop,
                s=np.sqrt(self.priors.discrimination_var),
                scale=self.priors.discrimination_mean,
            )
            prior_b_curr = stats.norm.logpdf(
                b_curr, self.priors.difficulty_mean, np.sqrt(self.priors.difficulty_var)
            )
            prior_b_prop = stats.norm.logpdf(
                b_prop, self.priors.difficulty_mean, np.sqrt(self.priors.difficulty_var)
            )

            log_alpha = (ll_prop + prior_a_prop + prior_b_prop) - (
                ll_curr + prior_a_curr + prior_b_curr
            )

            if np.log(rng.random()) < log_alpha:
                model.discrimination[j] = a_prop
                model.difficulty[j] = b_prop

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
        n_persons, n_timepoints, n_items = responses.shape
        ll = 0.0

        for i in range(n_persons):
            for t in range(n_timepoints):
                for j in range(n_items):
                    if responses[i, t, j] >= 0:
                        p = model.probability(np.array([theta_trajectories[i, t]]), j)[
                            0
                        ]
                        p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

                        if responses[i, t, j] == 1:
                            ll += np.log(p)
                        else:
                            ll += np.log(1 - p)

        return ll
