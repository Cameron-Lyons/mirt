from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import logsumexp

from mirt._core import sigmoid
from mirt.backends.rust.response_time import rt_accept_person_proposals
from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult


@dataclass
class RTModelPriors:
    """Prior distributions for response time model parameters.

    Attributes
    ----------
    disc_mean : float
        Prior mean for log discrimination
    disc_var : float
        Prior variance for log discrimination
    diff_mean : float
        Prior mean for difficulty
    diff_var : float
        Prior variance for difficulty
    time_disc_mean : float
        Prior mean for log time discrimination
    time_disc_var : float
        Prior variance for log time discrimination
    time_int_mean : float
        Prior mean for time intensity
    time_int_var : float
        Prior variance for time intensity
    mu_mean : NDArray
        Prior mean for population mean
    mu_cov : NDArray
        Prior covariance for population mean
    sigma_df : int
        Degrees of freedom for inverse-Wishart prior on Σ
    sigma_scale : NDArray
        Scale matrix for inverse-Wishart prior on Σ
    """

    disc_mean: float = 0.0
    disc_var: float = 1.0
    diff_mean: float = 0.0
    diff_var: float = 4.0
    time_disc_mean: float = 0.0
    time_disc_var: float = 1.0
    time_int_mean: float = 0.0
    time_int_var: float = 4.0
    mu_mean: NDArray[np.float64] | None = None
    mu_cov: NDArray[np.float64] | None = None
    sigma_df: int = 4
    sigma_scale: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        for name in ("disc_mean", "diff_mean", "time_disc_mean", "time_int_mean"):
            value = getattr(self, name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must be finite",
                    parameter=name,
                    value=value,
                    expected="finite number",
                ) from exc
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(numeric):
                raise MirtValidationError(
                    f"{name} must be finite",
                    parameter=name,
                    value=value,
                    expected="finite number",
                )
            setattr(self, name, numeric)

        for name in ("disc_var", "diff_var", "time_disc_var", "time_int_var"):
            value = getattr(self, name)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    f"{name} must be finite and positive",
                    parameter=name,
                    value=value,
                    expected="> 0",
                ) from exc
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(numeric)
                or numeric <= 0.0
            ):
                raise MirtValidationError(
                    f"{name} must be finite and positive",
                    parameter=name,
                    value=value,
                    expected="> 0",
                )
            setattr(self, name, numeric)

        if (
            isinstance(self.sigma_df, (bool, np.bool_))
            or not isinstance(self.sigma_df, (int, np.integer))
            or self.sigma_df < 2
        ):
            raise MirtValidationError(
                "sigma_df must be an integer of at least 2",
                parameter="sigma_df",
                value=self.sigma_df,
                expected="integer >= 2",
            )
        self.sigma_df = int(self.sigma_df)

        if self.mu_mean is None:
            self.mu_mean = np.zeros(2)
        else:
            self.mu_mean = self._validated_array("mu_mean", self.mu_mean, (2,))
        if self.mu_cov is None:
            self.mu_cov = np.eye(2) * 10
        else:
            self.mu_cov = self._validated_array(
                "mu_cov", self.mu_cov, (2, 2), positive_definite=True
            )
        if self.sigma_scale is None:
            self.sigma_scale = np.eye(2)
        else:
            self.sigma_scale = self._validated_array(
                "sigma_scale", self.sigma_scale, (2, 2), positive_definite=True
            )

    @staticmethod
    def _validated_array(
        name: str,
        values: NDArray[np.float64],
        shape: tuple[int, ...],
        *,
        positive_definite: bool = False,
    ) -> NDArray[np.float64]:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                f"{name} must be numeric",
                parameter=name,
                value=values,
                expected=f"finite array with shape {shape}",
            ) from exc
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise MirtValidationError(
                f"{name} must be a finite array with shape {shape}",
                parameter=name,
                value=array.shape,
                expected=f"finite array with shape {shape}",
            )
        if positive_definite:
            if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
                raise MirtValidationError(
                    f"{name} must be symmetric positive definite",
                    parameter=name,
                    value=array,
                    expected="symmetric positive-definite matrix",
                )
            try:
                np.linalg.cholesky(array)
            except np.linalg.LinAlgError as exc:
                raise MirtValidationError(
                    f"{name} must be symmetric positive definite",
                    parameter=name,
                    value=array,
                    expected="symmetric positive-definite matrix",
                ) from exc
        return array.copy()


class ResponseTimeGibbsSampler:
    """Gibbs sampler for joint response time models.

    Implements a blocked Gibbs sampler for the Van der Linden hierarchical
    model with MCMC estimation of all parameters.

    Parameters
    ----------
    n_iter : int
        Total number of MCMC iterations
    burnin : int
        Number of burn-in iterations to discard
    thin : int
        Thinning interval
    n_chains : int
        Number of independent chains
    priors : RTModelPriors, optional
        Prior specifications
    proposal_sd : float
        Standard deviation for MH proposals
    adapt_interval : int
        Interval for adapting proposal SD
    verbose : bool
        Print progress
    seed : int, optional
        Random seed
    use_rust : bool, default=True
        Use compiled likelihood and person-proposal evaluation when available.
    """

    def __init__(
        self,
        n_iter: int = 5000,
        burnin: int = 1000,
        thin: int = 1,
        n_chains: int = 1,
        priors: RTModelPriors | None = None,
        proposal_sd: float = 0.3,
        adapt_interval: int = 100,
        verbose: bool = False,
        seed: int | None = None,
        use_rust: bool = True,
    ) -> None:
        for name, value in (
            ("n_iter", n_iter),
            ("thin", thin),
            ("n_chains", n_chains),
            ("adapt_interval", adapt_interval),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise MirtValidationError(
                    f"{name} must be a positive integer",
                    parameter=name,
                    value=value,
                    expected="positive integer",
                )
        if (
            isinstance(burnin, (bool, np.bool_))
            or not isinstance(burnin, (int, np.integer))
            or burnin < 0
        ):
            raise MirtValidationError(
                "burnin must be a non-negative integer",
                parameter="burnin",
                value=burnin,
                expected="non-negative integer",
            )
        if burnin >= n_iter:
            raise MirtValidationError(
                "burnin must be less than n_iter",
                parameter="burnin",
                value=burnin,
                expected=f"< {n_iter}",
            )
        if priors is not None and not isinstance(priors, RTModelPriors):
            raise MirtValidationError(
                "priors must be an RTModelPriors instance or None",
                parameter="priors",
                value=type(priors).__name__,
                expected="RTModelPriors or None",
            )
        try:
            proposal_sd_value = float(proposal_sd)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "proposal_sd must be finite and positive",
                parameter="proposal_sd",
                value=proposal_sd,
                expected="> 0",
            ) from exc
        if (
            isinstance(proposal_sd, (bool, np.bool_))
            or not np.isfinite(proposal_sd_value)
            or proposal_sd_value <= 0.0
        ):
            raise MirtValidationError(
                "proposal_sd must be finite and positive",
                parameter="proposal_sd",
                value=proposal_sd,
                expected="> 0",
            )
        if not isinstance(verbose, (bool, np.bool_)):
            raise MirtValidationError(
                "verbose must be a boolean",
                parameter="verbose",
                value=verbose,
                expected="boolean",
            )
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise MirtValidationError(
                "seed must be a non-negative integer or None",
                parameter="seed",
                value=seed,
                expected="non-negative integer or None",
            )
        if not isinstance(use_rust, (bool, np.bool_)):
            raise MirtValidationError(
                "use_rust must be a boolean",
                parameter="use_rust",
                value=use_rust,
                expected="boolean",
            )

        self.n_iter = int(n_iter)
        self.burnin = int(burnin)
        self.thin = int(thin)
        self.n_chains = int(n_chains)
        self.priors = RTModelPriors() if priors is None else priors
        self.proposal_sd = proposal_sd_value
        self.adapt_interval = int(adapt_interval)
        self.verbose = bool(verbose)
        self.seed = None if seed is None else int(seed)
        self.use_rust = bool(use_rust)

    def fit(
        self,
        responses: NDArray[np.int_] | NDArray[np.float64],
        response_times: NDArray[np.float64],
        accuracy_model: Literal["2PL", "3PL"] = "2PL",
    ) -> ResponseTimeResult:
        """Fit response time model via Gibbs sampling.

        Parameters
        ----------
        responses : NDArray
            Binary response matrix (n_persons, n_items)
        response_times : NDArray
            Response time matrix (n_persons, n_items) in seconds
        accuracy_model : str
            IRT model for accuracy ("2PL" or "3PL")

        Returns
        -------
        ResponseTimeResult
            Fitted model with posterior samples
        """
        from mirt.models.response_time import ResponseTimeModel, ResponseTimeResult

        responses, log_rt = self._validate_fit_data(
            responses, response_times, accuracy_model
        )
        n_persons, n_items = responses.shape
        chain_runs = [
            self._run_chain(
                responses,
                log_rt,
                accuracy_model,
                rng,
                chain_idx,
            )
            for chain_idx, rng in enumerate(self._chain_generators())
        ]
        diagnostic_chains = {
            name: np.stack([run[0][name] for run in chain_runs], axis=0)
            for name in chain_runs[0][0]
        }
        chains = {
            name: values.reshape((-1,) + values.shape[2:])
            for name, values in diagnostic_chains.items()
        }
        theta_by_chain = np.stack([run[1] for run in chain_runs], axis=0)
        tau_by_chain = np.stack([run[2] for run in chain_runs], axis=0)
        theta_samples = theta_by_chain.reshape(-1, n_persons)
        tau_samples = tau_by_chain.reshape(-1, n_persons)

        theta_est = np.mean(theta_samples, axis=0)
        tau_est = np.mean(tau_samples, axis=0)
        theta_se = np.std(theta_samples, axis=0)
        tau_se = np.std(tau_samples, axis=0)

        model = ResponseTimeModel(
            n_items=n_items,
            accuracy_model=accuracy_model,
            discrimination=np.mean(chains["discrimination"], axis=0),
            difficulty=np.mean(chains["difficulty"], axis=0),
            guessing=np.mean(chains["guessing"], axis=0)
            if accuracy_model == "3PL"
            else None,
            time_discrimination=np.mean(chains["time_discrimination"], axis=0),
            time_intensity=np.mean(chains["time_intensity"], axis=0),
            ability_speed_mean=np.array(
                [np.mean(chains["mu_theta"]), np.mean(chains["mu_tau"])]
            ),
            ability_speed_cov=np.array(
                [
                    [np.mean(chains["sigma_11"]), np.mean(chains["sigma_12"])],
                    [np.mean(chains["sigma_12"]), np.mean(chains["sigma_22"])],
                ]
            ),
            use_rust=self.use_rust,
        )

        sample_log_likelihoods = self._posterior_log_likelihoods(
            model, responses, log_rt, theta_samples, tau_samples
        )
        log_likelihood = float(np.mean(np.sum(sample_log_likelihoods, axis=1)))

        dic = self._compute_dic(
            model,
            responses,
            log_rt,
            theta_samples,
            tau_samples,
            sample_log_likelihoods,
        )
        waic = self._compute_waic(
            model,
            responses,
            log_rt,
            theta_samples,
            tau_samples,
            sample_log_likelihoods,
        )

        rhat = self._compute_rhat(diagnostic_chains)
        ess = self._compute_ess(diagnostic_chains)

        converged = all(np.isfinite(r) and r < 1.1 for r in rhat.values())

        return ResponseTimeResult(
            model=model,
            theta_estimates=theta_est,
            tau_estimates=tau_est,
            theta_se=theta_se,
            tau_se=tau_se,
            chains=chains,
            log_likelihood=log_likelihood,
            dic=dic,
            waic=waic,
            rhat=rhat,
            ess=ess,
            n_iterations=self.n_iter,
            n_chains=self.n_chains,
            converged=converged,
        )

    def _validate_fit_data(
        self,
        responses: NDArray[np.int_] | NDArray[np.float64],
        response_times: NDArray[np.float64],
        accuracy_model: str,
    ) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Validate and normalize observed accuracy and timing data."""
        if not isinstance(accuracy_model, str) or accuracy_model not in ("2PL", "3PL"):
            raise MirtValidationError(
                "accuracy_model must be '2PL' or '3PL'",
                parameter="accuracy_model",
                value=accuracy_model,
                expected="'2PL' or '3PL'",
            )

        response_values = np.asarray(responses)
        if response_values.ndim != 2:
            raise MirtValidationError(
                "responses must have shape (n_persons, n_items)",
                parameter="responses",
                value=response_values.shape,
                expected="two-dimensional matrix",
            )
        if response_values.shape[0] == 0 or response_values.shape[1] == 0:
            raise MirtValidationError(
                "responses must contain at least one person and one item",
                parameter="responses",
                value=response_values.shape,
                expected="non-empty two-dimensional matrix",
            )

        if np.issubdtype(response_values.dtype, np.integer) or np.issubdtype(
            response_values.dtype, np.bool_
        ):
            observed = response_values >= 0
            invalid = observed & (response_values != 0) & (response_values != 1)
            response_numeric = response_values
        else:
            try:
                response_numeric = response_values.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise MirtValidationError(
                    "responses must be numeric",
                    parameter="responses",
                    value=response_values,
                    expected="0, 1, or a negative/NaN missing value",
                ) from exc
            finite = np.isfinite(response_numeric)
            missing = np.isnan(response_numeric) | (finite & (response_numeric < 0.0))
            observed = finite & (response_numeric >= 0.0)
            invalid = (~missing & ~observed) | (
                observed & (response_numeric != 0.0) & (response_numeric != 1.0)
            )
        if np.any(invalid):
            raise MirtValidationError(
                "observed responses must be 0 or 1",
                parameter="responses",
                value=response_numeric[invalid],
                expected="0, 1, or a negative/NaN missing value",
            )
        response_integers = np.where(observed, response_numeric, -1).astype(np.int32)

        try:
            time_values = np.asarray(response_times, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MirtValidationError(
                "response_times must be numeric",
                parameter="response_times",
                value=response_times,
                expected="positive finite values or NaN",
            ) from exc
        if time_values.shape != response_values.shape:
            raise MirtValidationError(
                "response_times must have the same shape as responses",
                parameter="response_times",
                value=time_values.shape,
                expected=str(response_values.shape),
            )
        finite_times = np.isfinite(time_values)
        invalid_times = np.isinf(time_values) | (finite_times & (time_values <= 0.0))
        if np.any(invalid_times):
            raise MirtValidationError(
                "observed response_times must be finite and positive",
                parameter="response_times",
                value=time_values[invalid_times],
                expected="positive finite values or NaN",
            )
        if not np.any(observed) and not np.any(finite_times):
            raise MirtValidationError(
                "at least one accuracy or timing observation is required",
                parameter="responses",
                expected="at least one observed value",
            )

        log_rt = np.full(time_values.shape, np.nan, dtype=np.float64)
        np.log(time_values, out=log_rt, where=finite_times)
        return response_integers, log_rt

    def _chain_generators(self) -> list[np.random.Generator]:
        """Create reproducible independent random streams for every chain."""
        if self.n_chains == 1:
            return [np.random.default_rng(self.seed)]
        seed_sequence = np.random.SeedSequence(self.seed)
        return [
            np.random.default_rng(child) for child in seed_sequence.spawn(self.n_chains)
        ]

    def _run_chain(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        accuracy_model: Literal["2PL", "3PL"],
        rng: np.random.Generator,
        chain_idx: int,
    ) -> tuple[
        dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Run one independent chain and return its retained draws."""
        n_persons, n_items = responses.shape
        n_samples = len(range(self.burnin, self.n_iter, self.thin))
        disc = np.ones(n_items)
        diff = np.zeros(n_items)
        guess = np.full(n_items, 0.2) if accuracy_model == "3PL" else None
        time_disc = np.ones(n_items)
        time_int = np.zeros(n_items)
        mu = np.zeros(2)
        sigma = np.eye(2)
        theta = rng.standard_normal(n_persons)
        tau = rng.standard_normal(n_persons)

        chains = {
            "discrimination": np.zeros((n_samples, n_items)),
            "difficulty": np.zeros((n_samples, n_items)),
            "time_discrimination": np.zeros((n_samples, n_items)),
            "time_intensity": np.zeros((n_samples, n_items)),
            "mu_theta": np.zeros(n_samples),
            "mu_tau": np.zeros(n_samples),
            "sigma_11": np.zeros(n_samples),
            "sigma_22": np.zeros(n_samples),
            "sigma_12": np.zeros(n_samples),
        }
        if accuracy_model == "3PL":
            chains["guessing"] = np.zeros((n_samples, n_items))
        theta_samples = np.zeros((n_samples, n_persons))
        tau_samples = np.zeros((n_samples, n_persons))
        acceptance_counts = np.zeros(n_persons)
        current_proposal_sd = self.proposal_sd
        sample_idx = 0

        for iteration in range(self.n_iter):
            theta, tau, accepted = self._sample_person_params(
                responses,
                log_rt,
                theta,
                tau,
                disc,
                diff,
                guess,
                time_disc,
                time_int,
                mu,
                sigma,
                current_proposal_sd,
                rng,
            )
            acceptance_counts += accepted

            if (iteration + 1) % self.adapt_interval == 0 and iteration < self.burnin:
                accept_rate = np.mean(acceptance_counts) / self.adapt_interval
                if accept_rate < 0.2:
                    current_proposal_sd *= 0.8
                elif accept_rate > 0.5:
                    current_proposal_sd *= 1.2
                acceptance_counts[:] = 0

            disc, diff = self._sample_accuracy_params(responses, theta, disc, diff, rng)
            if accuracy_model == "3PL":
                guess = self._sample_guessing_params(
                    responses, theta, disc, diff, guess, rng
                )
            time_disc, time_int = self._sample_time_params(
                log_rt, tau, time_disc, time_int, rng
            )
            mu, sigma = self._sample_population_params(theta, tau, mu, sigma, rng)

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                chains["discrimination"][sample_idx] = disc
                chains["difficulty"][sample_idx] = diff
                chains["time_discrimination"][sample_idx] = time_disc
                chains["time_intensity"][sample_idx] = time_int
                chains["mu_theta"][sample_idx] = mu[0]
                chains["mu_tau"][sample_idx] = mu[1]
                chains["sigma_11"][sample_idx] = sigma[0, 0]
                chains["sigma_22"][sample_idx] = sigma[1, 1]
                chains["sigma_12"][sample_idx] = sigma[0, 1]
                if accuracy_model == "3PL":
                    chains["guessing"][sample_idx] = guess
                theta_samples[sample_idx] = theta
                tau_samples[sample_idx] = tau
                sample_idx += 1

            if self.verbose and (iteration + 1) % 500 == 0:
                prefix = (
                    f"Chain {chain_idx + 1}/{self.n_chains}, "
                    if self.n_chains > 1
                    else ""
                )
                print(f"{prefix}iteration {iteration + 1}/{self.n_iter}")

        return chains, theta_samples, tau_samples

    def _sample_person_params(
        self,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        guess: NDArray[np.float64] | None,
        time_disc: NDArray[np.float64],
        time_int: NDArray[np.float64],
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        proposal_sd: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Sample (θ, τ) jointly via Metropolis-Hastings."""
        n_persons = len(theta)

        sigma_inv = np.linalg.inv(sigma)
        log_det_sigma = np.linalg.slogdet(sigma)[1]
        proposal_offsets = rng.normal(0.0, proposal_sd, size=(n_persons, 2))
        theta_proposed = theta + proposal_offsets[:, 0]
        tau_proposed = tau + proposal_offsets[:, 1]
        log_uniform = np.log(rng.random(n_persons))

        return rt_accept_person_proposals(
            responses,
            log_rt,
            theta,
            tau,
            theta_proposed,
            tau_proposed,
            log_uniform,
            disc,
            diff,
            time_disc,
            time_int,
            mu,
            sigma_inv,
            log_det_sigma,
            guess,
            use_rust=self.use_rust,
        )

    def _sample_accuracy_params(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample accuracy item parameters via MH."""
        n_items = len(disc)

        new_disc = disc.copy()
        new_diff = diff.copy()

        for j in range(n_items):
            log_disc_curr = np.log(disc[j])
            log_disc_prop = log_disc_curr + rng.normal(0, 0.1)
            disc_prop = np.exp(log_disc_prop)

            diff_prop = diff[j] + rng.normal(0, 0.1)

            observed = responses[:, j] >= 0
            observed_theta = theta[observed]
            observed_responses = responses[observed, j]
            p_curr = np.clip(
                sigmoid(disc[j] * (observed_theta - diff[j])),
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            p_prop = np.clip(
                sigmoid(disc_prop * (observed_theta - diff_prop)),
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            log_like_curr = np.sum(
                np.where(
                    observed_responses == 1,
                    np.log(p_curr),
                    np.log1p(-p_curr),
                )
            )
            log_like_prop = np.sum(
                np.where(
                    observed_responses == 1,
                    np.log(p_prop),
                    np.log1p(-p_prop),
                )
            )

            log_prior_curr = (
                -0.5
                * (log_disc_curr - self.priors.disc_mean) ** 2
                / self.priors.disc_var
                - 0.5 * (diff[j] - self.priors.diff_mean) ** 2 / self.priors.diff_var
            )
            log_prior_prop = (
                -0.5
                * (log_disc_prop - self.priors.disc_mean) ** 2
                / self.priors.disc_var
                - 0.5 * (diff_prop - self.priors.diff_mean) ** 2 / self.priors.diff_var
            )

            jacobian = log_disc_prop - log_disc_curr

            log_accept = (
                (log_like_prop + log_prior_prop)
                - (log_like_curr + log_prior_curr)
                + jacobian
            )

            if np.log(rng.random()) < log_accept:
                new_disc[j] = disc_prop
                new_diff[j] = diff_prop

        return new_disc, new_diff

    def _sample_guessing_params(
        self,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        disc: NDArray[np.float64],
        diff: NDArray[np.float64],
        guess: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample guessing parameters via MH."""
        n_items = len(guess)

        new_guess = guess.copy()

        for j in range(n_items):
            guess_prop = guess[j] + rng.normal(0, 0.02)
            guess_prop = np.clip(guess_prop, 0.01, 0.5)

            observed = responses[:, j] >= 0
            observed_responses = responses[observed, j]
            p_star = sigmoid(disc[j] * (theta[observed] - diff[j]))
            p_curr = np.clip(
                guess[j] + (1 - guess[j]) * p_star,
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            p_prop = np.clip(
                guess_prop + (1 - guess_prop) * p_star,
                PROB_EPSILON,
                1 - PROB_EPSILON,
            )
            log_like_curr = np.sum(
                np.where(
                    observed_responses == 1,
                    np.log(p_curr),
                    np.log1p(-p_curr),
                )
            )
            log_like_prop = np.sum(
                np.where(
                    observed_responses == 1,
                    np.log(p_prop),
                    np.log1p(-p_prop),
                )
            )

            log_accept = log_like_prop - log_like_curr

            if np.log(rng.random()) < log_accept:
                new_guess[j] = guess_prop

        return new_guess

    def _sample_time_params(
        self,
        log_rt: NDArray[np.float64],
        tau: NDArray[np.float64],
        time_disc: NDArray[np.float64],
        time_int: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample time item parameters via conjugate updates."""
        n_items = log_rt.shape[1]

        new_time_disc = time_disc.copy()
        new_time_int = time_int.copy()

        for j in range(n_items):
            valid = ~np.isnan(log_rt[:, j])
            if not np.any(valid):
                continue

            rt_valid = log_rt[valid, j]
            tau_valid = tau[valid]
            n_valid = np.sum(valid)

            prior_mean = self.priors.time_int_mean
            prior_var = self.priors.time_int_var
            alpha = time_disc[j]
            obs_var = 1.0 / (alpha**2)

            residual_mean = np.mean(rt_valid + tau_valid)
            post_var = 1.0 / (1.0 / prior_var + n_valid / obs_var)
            post_mean = post_var * (
                prior_mean / prior_var + n_valid * residual_mean / obs_var
            )

            new_time_int[j] = rng.normal(post_mean, np.sqrt(post_var))

            residuals = rt_valid - (new_time_int[j] - tau_valid)
            ss = np.sum(residuals**2)

            shape = self.priors.sigma_df / 2 + n_valid / 2
            scale = 1.0 / (self.priors.sigma_df / 2 + ss / 2)
            precision = rng.gamma(shape, scale)
            new_time_disc[j] = np.sqrt(precision)

        return new_time_disc, new_time_int

    def _sample_population_params(
        self,
        theta: NDArray[np.float64],
        tau: NDArray[np.float64],
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample population mean and covariance."""
        n_persons = len(theta)
        person_params = np.column_stack([theta, tau])

        sample_mean = np.mean(person_params, axis=0)
        prior_mean = self.priors.mu_mean
        prior_cov = self.priors.mu_cov
        sigma_inv = np.linalg.inv(sigma)
        prior_cov_inv = np.linalg.inv(prior_cov)

        post_cov = np.linalg.inv(prior_cov_inv + n_persons * sigma_inv)
        post_mean = post_cov @ (
            prior_cov_inv @ prior_mean + n_persons * sigma_inv @ sample_mean
        )

        new_mu = rng.multivariate_normal(post_mean, post_cov)

        centered = person_params - new_mu
        scatter = centered.T @ centered
        scale = self.priors.sigma_scale + scatter
        df = self.priors.sigma_df + n_persons

        new_sigma = stats.invwishart.rvs(df=df, scale=scale, random_state=rng)

        return new_mu, new_sigma

    def _compute_dic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
        log_likes: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute Deviance Information Criterion."""
        if log_likes is None:
            log_likes = self._posterior_log_likelihoods(
                model, responses, log_rt, theta_samples, tau_samples
            )
        deviances = -2.0 * np.sum(log_likes, axis=1)
        d_bar = np.mean(deviances)

        theta_mean = np.mean(theta_samples, axis=0)
        tau_mean = np.mean(tau_samples, axis=0)
        ll_at_mean = np.sum(
            model.joint_log_likelihood(responses, log_rt, theta_mean, tau_mean)
        )
        d_theta_bar = -2 * ll_at_mean

        p_d = d_bar - d_theta_bar
        dic = d_bar + p_d

        return float(dic)

    def _compute_waic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
        log_likes: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute Watanabe-Akaike Information Criterion."""
        if log_likes is None:
            log_likes = self._posterior_log_likelihoods(
                model, responses, log_rt, theta_samples, tau_samples
            )
        n_samples = log_likes.shape[0]

        lppd = np.sum(logsumexp(log_likes, axis=0) - np.log(n_samples))
        p_waic = np.sum(np.var(log_likes, axis=0, ddof=1)) if n_samples > 1 else 0.0

        waic = -2 * (lppd - p_waic)

        return float(waic)

    @staticmethod
    def _posterior_log_likelihoods(
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate every retained person-level log likelihood once."""
        n_samples = theta_samples.shape[0]
        batched_likelihood = getattr(model, "joint_log_likelihood_samples", None)
        if callable(batched_likelihood):
            log_likes = np.asarray(
                batched_likelihood(
                    responses,
                    log_rt,
                    theta_samples,
                    tau_samples,
                ),
                dtype=np.float64,
            )
            expected_shape = (n_samples, responses.shape[0])
            if log_likes.shape != expected_shape:
                raise ValueError(
                    "joint_log_likelihood_samples must return shape "
                    f"{expected_shape}, got {log_likes.shape}"
                )
            return log_likes

        log_likes = np.empty((n_samples, responses.shape[0]), dtype=np.float64)
        for sample_idx in range(n_samples):
            log_likes[sample_idx] = model.joint_log_likelihood(
                responses,
                log_rt,
                theta_samples[sample_idx],
                tau_samples[sample_idx],
            )
        return log_likes

    def _compute_rhat(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute split-chain Gelman-Rubin convergence diagnostics."""
        return {
            name: self._rhat_for_array(np.asarray(samples, dtype=np.float64))
            for name, samples in chains.items()
        }

    @staticmethod
    def _rhat_for_array(samples: NDArray[np.float64]) -> float:
        """Compute split R-hat averaged across trailing parameter dimensions."""
        if samples.ndim < 2:
            raise ValueError("diagnostic samples must have chain and draw dimensions")
        n_chains, n_draws = samples.shape[:2]
        if n_draws < 4:
            return float("nan")

        half = n_draws // 2
        flat = samples.reshape(n_chains, n_draws, -1)
        split = np.concatenate((flat[:, :half], flat[:, -half:]), axis=0)
        chain_means = np.mean(split, axis=1)
        within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
        between = half * np.var(chain_means, axis=0, ddof=1)
        variance = ((half - 1) / half) * within + between / half

        values = np.ones_like(within)
        varying = within > 0.0
        values[varying] = np.sqrt(variance[varying] / within[varying])
        values[(~varying) & (between > 0.0)] = np.inf
        return float(np.mean(values))

    def _compute_ess(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute effective sample sizes using FFT autocorrelations."""
        return {
            name: self._ess_for_array(np.asarray(samples, dtype=np.float64))
            for name, samples in chains.items()
        }

    @staticmethod
    def _ess_for_array(samples: NDArray[np.float64]) -> float:
        """Compute ESS averaged across trailing parameter dimensions."""
        if samples.ndim < 2:
            raise ValueError("diagnostic samples must have chain and draw dimensions")
        n_chains, n_draws = samples.shape[:2]
        total_draws = n_chains * n_draws
        if n_draws < 4:
            return float(total_draws)

        flat = samples.reshape(n_chains, n_draws, -1)
        centered = flat - np.mean(flat, axis=1, keepdims=True)
        fft_size = 1 << (2 * n_draws - 1).bit_length()
        transformed = np.fft.rfft(centered, n=fft_size, axis=1)
        autocovariance = np.fft.irfft(
            transformed * np.conjugate(transformed), n=fft_size, axis=1
        )[:, :n_draws]
        autocovariance /= np.arange(n_draws, 0, -1)[None, :, None]

        variances = autocovariance[:, 0]
        valid = variances > 0.0
        correlations = np.divide(
            autocovariance,
            variances[:, None, :],
            out=np.zeros_like(autocovariance),
            where=valid[:, None, :],
        )
        valid_counts = np.sum(valid, axis=0)
        mean_correlations = np.divide(
            np.sum(correlations, axis=0),
            valid_counts[None, :],
            out=np.zeros_like(correlations[0]),
            where=valid_counts[None, :] > 0,
        )

        integrated_time = np.ones(flat.shape[2])
        active = valid_counts > 0
        for lag in range(1, n_draws - 1, 2):
            pair = mean_correlations[lag] + mean_correlations[lag + 1]
            active &= pair > 0.0
            integrated_time[active] += 2.0 * pair[active]

        effective = total_draws / integrated_time
        effective[valid_counts == 0] = total_draws
        return float(np.mean(np.minimum(effective, total_draws)))
