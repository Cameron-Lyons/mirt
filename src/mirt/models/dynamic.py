from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


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
    """

    n_skills: int
    skill_names: list[str] | None = None
    allow_forgetting: bool = False

    p_init: NDArray[np.float64] | None = None
    p_learn: NDArray[np.float64] | None = None
    p_forget: NDArray[np.float64] | None = None
    p_slip: NDArray[np.float64] | None = None
    p_guess: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.skill_names is None:
            self.skill_names = [f"Skill_{i}" for i in range(self.n_skills)]

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
        p_s = self.p_slip[skill_idx]
        p_g = self.p_guess[skill_idx]

        if learned == 1:
            return 1 - p_s if response == 1 else p_s
        else:
            return p_g if response == 1 else 1 - p_g

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
        n_trials = len(responses)

        alpha = np.zeros((n_trials, 2))
        scaling = np.zeros(n_trials)

        skill_idx = skill_assignments[0]
        p_0 = self.p_init[skill_idx]

        for s in range(2):
            prior = p_0 if s == 1 else 1 - p_0
            emission = self.emission_probability(responses[0], s, skill_idx)
            alpha[0, s] = prior * emission

        scaling[0] = np.sum(alpha[0])
        if scaling[0] > 0:
            alpha[0] /= scaling[0]

        for t in range(1, n_trials):
            skill_idx = skill_assignments[t]
            T = self.transition_matrix(skill_idx)

            for s in range(2):
                alpha[t, s] = 0
                for s_prev in range(2):
                    alpha[t, s] += alpha[t - 1, s_prev] * T[s_prev, s]

                emission = self.emission_probability(responses[t], s, skill_idx)
                alpha[t, s] *= emission

            scaling[t] = np.sum(alpha[t])
            if scaling[t] > 0:
                alpha[t] /= scaling[t]

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
        n_trials = len(responses)

        beta = np.zeros((n_trials, 2))
        beta[n_trials - 1] = 1.0

        for t in range(n_trials - 2, -1, -1):
            skill_idx = skill_assignments[t + 1]
            T = self.transition_matrix(skill_idx)

            for s in range(2):
                beta[t, s] = 0
                for s_next in range(2):
                    emission = self.emission_probability(
                        responses[t + 1], s_next, skill_idx
                    )
                    beta[t, s] += T[s, s_next] * emission * beta[t + 1, s_next]

            if scaling[t + 1] > 0:
                beta[t] /= scaling[t + 1]

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
        alpha, scaling = self.forward(responses, skill_assignments)
        beta = self.backward(responses, skill_assignments, scaling)

        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1
        gamma = gamma / gamma_sum

        log_likelihood = np.sum(np.log(scaling + 1e-300))

        return gamma, log_likelihood

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
        n_trials = len(responses)

        delta = np.zeros((n_trials, 2))
        psi = np.zeros((n_trials, 2), dtype=int)

        skill_idx = skill_assignments[0]
        p_0 = self.p_init[skill_idx]

        for s in range(2):
            prior = np.log(p_0 + 1e-300) if s == 1 else np.log(1 - p_0 + 1e-300)
            emission = np.log(
                self.emission_probability(responses[0], s, skill_idx) + 1e-300
            )
            delta[0, s] = prior + emission

        for t in range(1, n_trials):
            skill_idx = skill_assignments[t]
            T = self.transition_matrix(skill_idx)

            for s in range(2):
                candidates = delta[t - 1] + np.log(T[:, s] + 1e-300)
                psi[t, s] = np.argmax(candidates)
                delta[t, s] = candidates[psi[t, s]]

                emission = np.log(
                    self.emission_probability(responses[t], s, skill_idx) + 1e-300
                )
                delta[t, s] += emission

        path = np.zeros(n_trials, dtype=int)
        path[n_trials - 1] = np.argmax(delta[n_trials - 1])

        for t in range(n_trials - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

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
        rng = np.random.default_rng(seed)

        n_trials = n_trials_per_skill * self.n_skills
        responses = np.zeros((n_persons, n_trials), dtype=np.int32)
        skill_assignments = np.zeros(n_trials, dtype=np.int32)
        learning_states = np.zeros((n_persons, n_trials), dtype=np.int32)

        for j in range(self.n_skills):
            start = j * n_trials_per_skill
            end = (j + 1) * n_trials_per_skill
            skill_assignments[start:end] = j

        for i in range(n_persons):
            for j in range(self.n_skills):
                start = j * n_trials_per_skill
                state = int(rng.random() < self.p_init[j])

                for t_rel in range(n_trials_per_skill):
                    t = start + t_rel
                    learning_states[i, t] = state

                    if state == 1:
                        p_correct = 1 - self.p_slip[j]
                    else:
                        p_correct = self.p_guess[j]
                    responses[i, t] = int(rng.random() < p_correct)

                    if state == 0:
                        state = int(rng.random() < self.p_learn[j])
                    else:
                        state = int(rng.random() >= self.p_forget[j])

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
            return 1.0 / (1.0 + np.exp(-z))

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
                    p = 1.0 / (1.0 + np.exp(-z))

                    if self.base_model == "3PL":
                        p = self.guessing[valid] + (1 - self.guessing[valid]) * p

                    p = np.clip(p, 1e-10, 1 - 1e-10)

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
                p = 1.0 / (1.0 + np.exp(-z))

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
        posteriors /= np.maximum(row_sums, 1e-10)

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
        posteriors /= np.maximum(row_sums, 1e-10)

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
                if np.sum(weights) < 1e-10:
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
        log_likelihood = np.sum(np.log(total_likelihood + 1e-10))

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
        posteriors = np.clip(posteriors, 1e-10, 1 - 1e-10)
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
