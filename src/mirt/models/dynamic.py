from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mirt._backend_config import should_use_rust
from mirt._core import sigmoid
from mirt.backends.rust.dynamic import (
    bkt_backward,
    bkt_forward,
    bkt_forward_backward_batch,
    bkt_viterbi,
)
from mirt.constants import PROB_EPSILON


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
        gamma, _ = self.forward_backward(responses, skill_assignments)
        mastery = self.p_init.copy()
        for skill_idx, trial_indices in enumerate(
            self._skill_trials(skill_assignments)
        ):
            if len(trial_indices) > 0:
                mastery[skill_idx] = gamma[trial_indices[-1], 1]
        return mastery

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
        mastery = np.broadcast_to(
            self.p_init, (responses.shape[0], self.n_skills)
        ).copy()

        if skill_assignments.ndim == 1:
            for skill_idx, trial_indices in enumerate(
                self._skill_trials(skill_assignments)
            ):
                if len(trial_indices) > 0:
                    mastery[:, skill_idx] = gamma[:, trial_indices[-1], 1]
            return mastery

        for person_idx, person_skills in enumerate(skill_assignments):
            for skill_idx, trial_indices in enumerate(
                self._skill_trials(person_skills)
            ):
                if len(trial_indices) > 0:
                    mastery[person_idx, skill_idx] = gamma[
                        person_idx, trial_indices[-1], 1
                    ]
        return mastery

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

    def __post_init__(self) -> None:
        if self.item_names is None:
            self.item_names = [f"Item_{i}" for i in range(self.n_items)]

        if self.discrimination is None:
            self.discrimination = np.ones(self.n_items)
        if self.difficulty is None:
            self.difficulty = np.zeros(self.n_items)

        n_growth = 2 if self.growth_model == "linear" else 3
        if self.growth_mean is None:
            self.growth_mean = np.zeros(n_growth)
        if self.growth_cov is None:
            self.growth_cov = np.eye(n_growth)

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
        n_persons = growth_factors.shape[0]

        if time_values is None:
            time_values = np.arange(self.n_timepoints, dtype=np.float64)

        theta = np.zeros((n_persons, self.n_timepoints))

        for i in range(n_persons):
            theta[i] = growth_factors[i, 0] + growth_factors[i, 1] * time_values

            if self.growth_model == "quadratic" and growth_factors.shape[1] > 2:
                theta[i] += growth_factors[i, 2] * time_values**2

        if residuals is not None:
            theta += residuals

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
            P(X=1|θ)
        """
        theta = np.atleast_1d(theta)

        if item_idx is not None:
            a = self.discrimination[item_idx]
            b = self.difficulty[item_idx]
            z = a * (theta - b)
            return sigmoid(z)

        probs = np.zeros((len(theta), self.n_items))
        for j in range(self.n_items):
            probs[:, j] = self.probability(theta, j)
        return probs

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
        rng = np.random.default_rng(seed)

        growth_factors = rng.multivariate_normal(
            self.growth_mean, self.growth_cov, size=n_persons
        )

        residuals = rng.normal(
            0, np.sqrt(self.residual_variance), size=(n_persons, self.n_timepoints)
        )

        theta = self.compute_theta(growth_factors, time_values, residuals)

        responses = np.zeros(
            (n_persons, self.n_timepoints, self.n_items), dtype=np.int32
        )
        for i in range(n_persons):
            for t in range(self.n_timepoints):
                probs = self.probability(theta[i, t])
                responses[i, t] = (rng.random(self.n_items) < probs).astype(np.int32)

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
    base_model : str
        IRT model for observations
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
        if self.transition_matrix is None:
            self.transition_matrix = np.array([[1.0]])
        if self.process_noise is None:
            self.process_noise = np.array([[0.1]])

        if self.discrimination is None:
            self.discrimination = np.ones(self.n_items)
        if self.difficulty is None:
            self.difficulty = np.zeros(self.n_items)
        if self.base_model == "3PL" and self.guessing is None:
            self.guessing = np.full(self.n_items, 0.2)

    def extended_kalman_filter(
        self,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extended Kalman filter for a single person.

        Parameters
        ----------
        responses : NDArray
            Response matrix (n_timepoints, n_items)

        Returns
        -------
        tuple
            (filtered_means, filtered_vars)
        """
        n_times = responses.shape[0]
        A = self.transition_matrix[0, 0]
        Q = self.process_noise[0, 0]

        filtered_means = np.zeros(n_times)
        filtered_vars = np.zeros(n_times)

        mean_pred = self.initial_mean
        var_pred = self.initial_var

        for t in range(n_times):
            valid = responses[t] >= 0
            if not np.any(valid):
                filtered_means[t] = mean_pred
                filtered_vars[t] = var_pred
            else:
                for _ in range(5):
                    z = self.discrimination[valid] * (
                        mean_pred - self.difficulty[valid]
                    )
                    p = sigmoid(z)

                    if self.base_model == "3PL":
                        p = self.guessing[valid] + (1 - self.guessing[valid]) * p

                    p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)

                    H = self.discrimination[valid] * p * (1 - p)
                    R_inv = p * (1 - p)

                    S = np.sum(H**2 / R_inv) + 1.0 / var_pred
                    K = H / R_inv / S

                    residual = responses[t, valid] - p
                    mean_pred = mean_pred + np.sum(K * residual)

                var_update = 1.0 / S
                filtered_means[t] = mean_pred
                filtered_vars[t] = var_update

            if t < n_times - 1:
                mean_pred = A * filtered_means[t]
                var_pred = A**2 * filtered_vars[t] + Q

        return filtered_means, filtered_vars

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
        rng = np.random.default_rng(seed)

        A = self.transition_matrix[0, 0]
        Q = self.process_noise[0, 0]

        theta = np.zeros((n_persons, self.n_timepoints))
        responses = np.zeros(
            (n_persons, self.n_timepoints, self.n_items), dtype=np.int32
        )

        theta[:, 0] = rng.normal(
            self.initial_mean, np.sqrt(self.initial_var), n_persons
        )

        for t in range(1, self.n_timepoints):
            theta[:, t] = A * theta[:, t - 1] + rng.normal(0, np.sqrt(Q), n_persons)

        for i in range(n_persons):
            for t in range(self.n_timepoints):
                z = self.discrimination * (theta[i, t] - self.difficulty)
                p = sigmoid(z)

                if self.base_model == "3PL":
                    p = self.guessing + (1 - self.guessing) * p

                responses[i, t] = (rng.random(self.n_items) < p).astype(np.int32)

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
        lines.append(f"Initial Mean:       {self.initial_mean:.4f}")
        lines.append(f"Initial Variance:   {self.initial_var:.4f}")

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

    def __post_init__(self) -> None:
        if len(self.class_proportions) == 0:
            self.class_proportions = np.ones(self.n_classes) / self.n_classes

        if len(self.class_intercepts) == 0:
            self.class_intercepts = np.linspace(-1, 1, self.n_classes)

        if len(self.class_slopes) == 0:
            self.class_slopes = np.linspace(0.1, 0.5, self.n_classes)

        if self.growth_type == "quadratic" and len(self.class_quadratics) == 0:
            self.class_quadratics = np.zeros(self.n_classes)

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
        trajectory = (
            self.class_intercepts[class_idx]
            + self.class_slopes[class_idx] * time_values
        )

        if self.growth_type == "quadratic":
            trajectory += self.class_quadratics[class_idx] * time_values**2

        return trajectory

    def class_likelihood(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute likelihood of observations under each class.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints).
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Class likelihoods (n_persons, n_classes).
        """
        observations = np.atleast_2d(observations)
        n_persons = observations.shape[0]

        likelihoods = np.zeros((n_persons, self.n_classes))

        for k in range(self.n_classes):
            mean_trajectory = self.compute_class_trajectory(k, time_values)

            for i in range(n_persons):
                residual = observations[i] - mean_trajectory
                total_var = self.intercept_var + self.residual_variance
                ll = -0.5 * np.sum(residual**2) / total_var
                ll -= 0.5 * len(time_values) * np.log(2 * np.pi * total_var)
                likelihoods[i, k] = np.exp(ll)

        return likelihoods

    def classify(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Classify persons into latent classes.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints).
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Class assignments (n_persons,).
        """
        likelihoods = self.class_likelihood(observations, time_values)

        posteriors = likelihoods * self.class_proportions
        row_sums = posteriors.sum(axis=1, keepdims=True)
        posteriors /= np.maximum(row_sums, PROB_EPSILON)

        return np.argmax(posteriors, axis=1)

    def posterior_probabilities(
        self,
        observations: NDArray[np.float64],
        time_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute posterior class probabilities.

        Parameters
        ----------
        observations : NDArray
            Observed trajectories (n_persons, n_timepoints).
        time_values : NDArray
            Time points.

        Returns
        -------
        NDArray
            Posterior probabilities (n_persons, n_classes).
        """
        likelihoods = self.class_likelihood(observations, time_values)

        posteriors = likelihoods * self.class_proportions
        row_sums = posteriors.sum(axis=1, keepdims=True)
        posteriors /= np.maximum(row_sums, PROB_EPSILON)

        return posteriors

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
        rng = np.random.default_rng(seed)

        if time_values is None:
            time_values = np.arange(self.n_timepoints, dtype=np.float64)

        n_times = len(time_values)

        true_classes = rng.choice(
            self.n_classes,
            size=n_persons,
            p=self.class_proportions,
        )

        observations = np.zeros((n_persons, n_times))

        for i in range(n_persons):
            k = true_classes[i]
            mean_trajectory = self.compute_class_trajectory(k, time_values)

            intercept_deviation = rng.normal(0, np.sqrt(self.intercept_var))
            slope_deviation = rng.normal(0, np.sqrt(self.slope_var))

            observations[i] = (
                mean_trajectory
                + intercept_deviation
                + slope_deviation * time_values
                + rng.normal(0, np.sqrt(self.residual_variance), n_times)
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
            Observed trajectories (n_persons, n_timepoints).
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
        observations = np.atleast_2d(observations)
        n_persons = observations.shape[0]

        for iteration in range(max_iter):
            posteriors = self.posterior_probabilities(observations, time_values)

            prev_proportions = self.class_proportions.copy()
            prev_intercepts = self.class_intercepts.copy()
            prev_slopes = self.class_slopes.copy()

            self.class_proportions = np.mean(posteriors, axis=0)

            for k in range(self.n_classes):
                weights = posteriors[:, k]
                if np.sum(weights) < PROB_EPSILON:
                    continue

                X = np.column_stack([np.ones(len(time_values)), time_values])
                if self.growth_type == "quadratic":
                    X = np.column_stack([X, time_values**2])

                weighted_y = np.zeros(X.shape[1])
                weighted_X = np.zeros((X.shape[1], X.shape[1]))

                for i in range(n_persons):
                    weighted_y += weights[i] * X.T @ observations[i]
                    weighted_X += weights[i] * X.T @ X

                try:
                    beta = np.linalg.solve(weighted_X, weighted_y)
                    self.class_intercepts[k] = beta[0]
                    self.class_slopes[k] = beta[1]
                    if self.growth_type == "quadratic" and len(beta) > 2:
                        self.class_quadratics[k] = beta[2]
                except np.linalg.LinAlgError:
                    pass

            prop_change = np.max(np.abs(self.class_proportions - prev_proportions))
            int_change = np.max(np.abs(self.class_intercepts - prev_intercepts))
            slope_change = np.max(np.abs(self.class_slopes - prev_slopes))

            if max(prop_change, int_change, slope_change) < tol:
                break

        final_posteriors = self.posterior_probabilities(observations, time_values)
        classifications = np.argmax(final_posteriors, axis=1)

        likelihoods = self.class_likelihood(observations, time_values)
        total_likelihood = np.sum(likelihoods * self.class_proportions, axis=1)
        log_likelihood = np.sum(np.log(total_likelihood + PROB_EPSILON))

        return {
            "classifications": classifications,
            "posteriors": final_posteriors,
            "log_likelihood": log_likelihood,
            "n_iterations": iteration + 1,
            "converged": iteration < max_iter - 1,
        }

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
            Observed trajectories.
        time_values : NDArray
            Time points.

        Returns
        -------
        float
            Entropy value.
        """
        posteriors = self.posterior_probabilities(observations, time_values)
        posteriors = np.clip(posteriors, PROB_EPSILON, 1 - PROB_EPSILON)
        return -np.mean(np.sum(posteriors * np.log(posteriors), axis=1))


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

    def summary(self) -> str:
        lines = []
        width = 60

        lines.append("=" * width)
        lines.append(f"{'Growth Mixture Model Results':^{width}}")
        lines.append("=" * width)

        lines.append(f"Number of Classes:  {self.model.n_classes}")
        lines.append(f"Growth Type:        {self.model.growth_type}")
        lines.append(f"Log-Likelihood:     {self.log_likelihood:.4f}")
        lines.append(f"AIC:                {self.aic:.4f}")
        lines.append(f"BIC:                {self.bic:.4f}")
        lines.append(f"Entropy:            {self.entropy:.4f}")
        lines.append(f"Converged:          {self.converged}")
        lines.append("-" * width)

        lines.append("\nClass Parameters:")
        for k in range(self.model.n_classes):
            n_in_class = np.sum(self.classifications == k)
            pct = 100 * n_in_class / len(self.classifications)
            lines.append(
                f"  Class {k}: N={n_in_class} ({pct:.1f}%), "
                f"Intercept={self.model.class_intercepts[k]:.3f}, "
                f"Slope={self.model.class_slopes[k]:.3f}"
            )

        lines.append("=" * width)
        return "\n".join(lines)
