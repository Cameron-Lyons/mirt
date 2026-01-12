from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class ResponseTimeModel:
    """Van der Linden hierarchical response time model.

    Joint model for response accuracy and response times where:
    - Accuracy: P(X=1|θ) follows a standard IRT model (2PL or 3PL)
    - Response time: log(T)|τ ~ N(β - τ, 1/α²)
    - Person parameters: (θ, τ) ~ MVN(μ, Σ)

    Parameters
    ----------
    n_items : int
        Number of items
    accuracy_model : str
        IRT model for accuracy ("2PL" or "3PL")
    item_names : list[str], optional
        Names for each item

    Attributes
    ----------
    discrimination : NDArray
        Accuracy discrimination parameters (a)
    difficulty : NDArray
        Accuracy difficulty parameters (b)
    guessing : NDArray
        Guessing parameters for 3PL (c)
    time_intensity : NDArray
        Time intensity parameters (β)
    time_discrimination : NDArray
        Time discrimination parameters (α)
    ability_speed_mean : NDArray
        Population mean of (θ, τ)
    ability_speed_cov : NDArray
        Population covariance of (θ, τ)
    """

    n_items: int
    accuracy_model: Literal["2PL", "3PL"] = "2PL"
    item_names: list[str] | None = None

    discrimination: NDArray[np.float64] | None = None
    difficulty: NDArray[np.float64] | None = None
    guessing: NDArray[np.float64] | None = None
    time_intensity: NDArray[np.float64] | None = None
    time_discrimination: NDArray[np.float64] | None = None
    ability_speed_mean: NDArray[np.float64] | None = None
    ability_speed_cov: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.item_names is None:
            self.item_names = [f"Item_{i}" for i in range(self.n_items)]

        if self.discrimination is None:
            self.discrimination = np.ones(self.n_items)
        if self.difficulty is None:
            self.difficulty = np.zeros(self.n_items)
        if self.accuracy_model == "3PL" and self.guessing is None:
            self.guessing = np.full(self.n_items, 0.2)
        if self.time_intensity is None:
            self.time_intensity = np.zeros(self.n_items)
        if self.time_discrimination is None:
            self.time_discrimination = np.ones(self.n_items)
        if self.ability_speed_mean is None:
            self.ability_speed_mean = np.zeros(2)
        if self.ability_speed_cov is None:
            self.ability_speed_cov = np.eye(2)

    @property
    def ability_speed_corr(self) -> float:
        """Correlation between ability and speed."""
        std_theta = np.sqrt(self.ability_speed_cov[0, 0])
        std_tau = np.sqrt(self.ability_speed_cov[1, 1])
        if std_theta > 0 and std_tau > 0:
            return self.ability_speed_cov[0, 1] / (std_theta * std_tau)
        return 0.0

    def accuracy_probability(
        self,
        theta: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute probability of correct response.

        Parameters
        ----------
        theta : NDArray
            Ability values (n_persons,) or scalar
        item_idx : int, optional
            Specific item index, or None for all items

        Returns
        -------
        NDArray
            P(X=1|θ) - shape (n_persons,) or (n_persons, n_items)
        """
        theta = np.atleast_1d(theta)
        n_persons = len(theta)

        if item_idx is not None:
            a = self.discrimination[item_idx]
            b = self.difficulty[item_idx]
            z = a * (theta - b)
            p = 1.0 / (1.0 + np.exp(-z))

            if self.accuracy_model == "3PL":
                c = self.guessing[item_idx]
                p = c + (1 - c) * p

            return p

        probs = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            probs[:, j] = self.accuracy_probability(theta, j)
        return probs

    def rt_log_density(
        self,
        log_rt: NDArray[np.float64],
        tau: NDArray[np.float64],
        item_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute log-density of log response time.

        Parameters
        ----------
        log_rt : NDArray
            Log response times (n_persons,) or (n_persons, n_items)
        tau : NDArray
            Speed parameters (n_persons,)
        item_idx : int, optional
            Specific item index

        Returns
        -------
        NDArray
            Log density values
        """
        tau = np.atleast_1d(tau)

        if item_idx is not None:
            beta = self.time_intensity[item_idx]
            alpha = self.time_discrimination[item_idx]

            mean = beta - tau
            var = 1.0 / (alpha**2)

            log_rt = np.atleast_1d(log_rt)
            log_dens = -0.5 * np.log(2 * np.pi * var) - 0.5 * (log_rt - mean) ** 2 / var
            return log_dens

        n_persons = len(tau)
        log_dens = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            log_dens[:, j] = self.rt_log_density(log_rt[:, j], tau, j)
        return log_dens

    def joint_log_likelihood(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute joint log-likelihood of responses and RTs.

        Parameters
        ----------
        responses : NDArray
            Binary responses (n_persons, n_items)
        log_rt : NDArray
            Log response times (n_persons, n_items)
        theta : NDArray
            Ability parameters (n_persons,)
        tau : NDArray
            Speed parameters (n_persons,)

        Returns
        -------
        NDArray
            Joint log-likelihood per person (n_persons,)
        """
        n_persons = responses.shape[0]
        n_items = responses.shape[1]

        ll = np.zeros(n_persons)

        probs = self.accuracy_probability(theta)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        for i in range(n_persons):
            for j in range(n_items):
                if responses[i, j] >= 0:
                    p = probs[i, j]
                    if responses[i, j] == 1:
                        ll[i] += np.log(p)
                    else:
                        ll[i] += np.log(1 - p)

                    if not np.isnan(log_rt[i, j]):
                        ll[i] += self.rt_log_density(log_rt[i, j], tau[i : i + 1], j)[0]

        return ll

    def simulate(
        self,
        n_persons: int,
        theta: NDArray[np.float64] | None = None,
        tau: NDArray[np.float64] | None = None,
        seed: int | None = None,
    ) -> tuple[
        NDArray[np.int_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
        """Simulate response data and response times.

        Parameters
        ----------
        n_persons : int
            Number of persons to simulate
        theta : NDArray, optional
            Ability parameters. If None, drawn from population.
        tau : NDArray, optional
            Speed parameters. If None, drawn from population.
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            (responses, response_times, theta, tau)
        """
        rng = np.random.default_rng(seed)

        if theta is None or tau is None:
            person_params = rng.multivariate_normal(
                self.ability_speed_mean,
                self.ability_speed_cov,
                size=n_persons,
            )
            if theta is None:
                theta = person_params[:, 0]
            if tau is None:
                tau = person_params[:, 1]

        probs = self.accuracy_probability(theta)
        responses = (rng.random((n_persons, self.n_items)) < probs).astype(np.int32)

        response_times = np.zeros((n_persons, self.n_items))
        for j in range(self.n_items):
            beta = self.time_intensity[j]
            alpha = self.time_discrimination[j]
            mean = beta - tau
            sd = 1.0 / alpha
            log_rt = rng.normal(mean, sd)
            response_times[:, j] = np.exp(log_rt)

        return responses, response_times, theta, tau

    def summary(self) -> str:
        """Generate model summary."""
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
        lines.append(f"  Mean Ability (θ): {self.ability_speed_mean[0]:.4f}")
        lines.append(f"  Mean Speed (τ):   {self.ability_speed_mean[1]:.4f}")
        lines.append(f"  Var(θ):           {self.ability_speed_cov[0, 0]:.4f}")
        lines.append(f"  Var(τ):           {self.ability_speed_cov[1, 1]:.4f}")
        lines.append(f"  Cov(θ, τ):        {self.ability_speed_cov[0, 1]:.4f}")

        lines.append("\nItem Parameters:")
        header = f"{'Item':<12} {'a':>8} {'b':>8}"
        if self.accuracy_model == "3PL":
            header += f" {'c':>8}"
        header += f" {'α':>8} {'β':>8}"
        lines.append(header)
        lines.append("-" * width)

        for j in range(self.n_items):
            row = f"{self.item_names[j]:<12} {self.discrimination[j]:>8.3f} {self.difficulty[j]:>8.3f}"
            if self.accuracy_model == "3PL":
                row += f" {self.guessing[j]:>8.3f}"
            row += (
                f" {self.time_discrimination[j]:>8.3f} {self.time_intensity[j]:>8.3f}"
            )
            lines.append(row)

        lines.append("=" * width)
        return "\n".join(lines)


@dataclass
class ResponseTimeResult:
    """Result from response time model estimation."""

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
        lines = []
        width = 80

        lines.append("=" * width)
        lines.append(f"{'Response Time Model Results':^{width}}")
        lines.append("=" * width)

        lines.append(
            f"Accuracy Model:     {self.model.accuracy_model:<20} Log-Likelihood:    {self.log_likelihood:>12.4f}"
        )
        lines.append(
            f"No. Items:          {self.model.n_items:<20} DIC:               {self.dic:>12.4f}"
        )
        lines.append(
            f"Iterations:         {self.n_iterations:<20} WAIC:              {self.waic:>12.4f}"
        )
        lines.append(
            f"Chains:             {self.n_chains:<20} Converged:         {str(self.converged):>12}"
        )
        lines.append("-" * width)

        lines.append("\nConvergence Diagnostics:")
        for param, rhat in self.rhat.items():
            ess = self.ess.get(param, np.nan)
            lines.append(f"  {param}: Rhat = {rhat:.4f}, ESS = {ess:.0f}")

        lines.append("\nPopulation Parameters:")
        lines.append(
            f"  Speed-Ability Correlation: {self.model.ability_speed_corr:.4f}"
        )

        lines.append("=" * width)
        return "\n".join(lines)

    def person_summary(self, n_show: int = 10) -> str:
        """Summary of person parameter estimates."""
        lines = []
        lines.append(f"Person Parameter Estimates (first {n_show} persons):")
        lines.append(f"{'Person':<10} {'θ':>10} {'SE(θ)':>10} {'τ':>10} {'SE(τ)':>10}")
        lines.append("-" * 50)

        for i in range(min(n_show, len(self.theta_estimates))):
            lines.append(
                f"{i:<10} {self.theta_estimates[i]:>10.4f} {self.theta_se[i]:>10.4f} "
                f"{self.tau_estimates[i]:>10.4f} {self.tau_se[i]:>10.4f}"
            )

        return "\n".join(lines)
