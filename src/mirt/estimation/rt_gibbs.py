from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

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
        if self.mu_mean is None:
            self.mu_mean = np.zeros(2)
        if self.mu_cov is None:
            self.mu_cov = np.eye(2) * 10
        if self.sigma_scale is None:
            self.sigma_scale = np.eye(2)


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
        Number of parallel chains
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
    ) -> None:
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.n_chains = n_chains
        self.priors = priors or RTModelPriors()
        self.proposal_sd = proposal_sd
        self.adapt_interval = adapt_interval
        self.verbose = verbose
        self.seed = seed

    def fit(
        self,
        responses: NDArray[np.int_],
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

        responses = np.asarray(responses, dtype=np.int32)
        response_times = np.asarray(response_times, dtype=np.float64)

        if responses.shape != response_times.shape:
            raise ValueError("responses and response_times must have same shape")

        n_persons, n_items = responses.shape
        log_rt = np.log(response_times + 1e-10)

        rng = np.random.default_rng(self.seed)

        disc = np.ones(n_items)
        diff = np.zeros(n_items)
        guess = np.full(n_items, 0.2) if accuracy_model == "3PL" else None
        time_disc = np.ones(n_items)
        time_int = np.zeros(n_items)
        mu = np.zeros(2)
        sigma = np.eye(2)

        theta = rng.standard_normal(n_persons)
        tau = rng.standard_normal(n_persons)

        n_samples = (self.n_iter - self.burnin) // self.thin
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
                print(f"Iteration {iteration + 1}/{self.n_iter}")

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
        )

        log_likelihood = np.mean(
            [
                np.sum(
                    model.joint_log_likelihood(
                        responses, log_rt, theta_samples[s], tau_samples[s]
                    )
                )
                for s in range(n_samples)
            ]
        )

        dic = self._compute_dic(model, responses, log_rt, theta_samples, tau_samples)
        waic = self._compute_waic(model, responses, log_rt, theta_samples, tau_samples)

        rhat = self._compute_rhat(chains)
        ess = self._compute_ess(chains)

        converged = all(r < 1.1 for r in rhat.values())

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
        n_items = len(disc)

        sigma_inv = np.linalg.inv(sigma)
        log_det_sigma = np.linalg.slogdet(sigma)[1]

        new_theta = theta.copy()
        new_tau = tau.copy()
        accepted = np.zeros(n_persons, dtype=bool)

        for i in range(n_persons):
            theta_prop = theta[i] + rng.normal(0, proposal_sd)
            tau_prop = tau[i] + rng.normal(0, proposal_sd)

            log_prior_curr = self._log_mvn_density_single(
                np.array([theta[i], tau[i]]), mu, sigma_inv, log_det_sigma
            )
            log_prior_prop = self._log_mvn_density_single(
                np.array([theta_prop, tau_prop]), mu, sigma_inv, log_det_sigma
            )

            log_like_curr = 0.0
            log_like_prop = 0.0

            for j in range(n_items):
                if responses[i, j] >= 0:
                    z_curr = disc[j] * (theta[i] - diff[j])
                    z_prop = disc[j] * (theta_prop - diff[j])

                    p_curr = 1.0 / (1.0 + np.exp(-z_curr))
                    p_prop = 1.0 / (1.0 + np.exp(-z_prop))

                    if guess is not None:
                        p_curr = guess[j] + (1 - guess[j]) * p_curr
                        p_prop = guess[j] + (1 - guess[j]) * p_prop

                    p_curr = np.clip(p_curr, 1e-10, 1 - 1e-10)
                    p_prop = np.clip(p_prop, 1e-10, 1 - 1e-10)

                    if responses[i, j] == 1:
                        log_like_curr += np.log(p_curr)
                        log_like_prop += np.log(p_prop)
                    else:
                        log_like_curr += np.log(1 - p_curr)
                        log_like_prop += np.log(1 - p_prop)

                if not np.isnan(log_rt[i, j]):
                    beta = time_int[j]
                    alpha = time_disc[j]
                    var = 1.0 / (alpha**2)

                    mean_curr = beta - tau[i]
                    mean_prop = beta - tau_prop

                    log_like_curr += -0.5 * (log_rt[i, j] - mean_curr) ** 2 / var
                    log_like_prop += -0.5 * (log_rt[i, j] - mean_prop) ** 2 / var

            log_accept = (log_like_prop + log_prior_prop) - (
                log_like_curr + log_prior_curr
            )

            if np.log(rng.random()) < log_accept:
                new_theta[i] = theta_prop
                new_tau[i] = tau_prop
                accepted[i] = True

        return new_theta, new_tau, accepted

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
        n_persons = len(theta)

        new_disc = disc.copy()
        new_diff = diff.copy()

        for j in range(n_items):
            log_disc_curr = np.log(disc[j])
            log_disc_prop = log_disc_curr + rng.normal(0, 0.1)
            disc_prop = np.exp(log_disc_prop)

            diff_prop = diff[j] + rng.normal(0, 0.1)

            log_like_curr = 0.0
            log_like_prop = 0.0

            for i in range(n_persons):
                if responses[i, j] >= 0:
                    z_curr = disc[j] * (theta[i] - diff[j])
                    z_prop = disc_prop * (theta[i] - diff_prop)

                    p_curr = 1.0 / (1.0 + np.exp(-z_curr))
                    p_prop = 1.0 / (1.0 + np.exp(-z_prop))

                    p_curr = np.clip(p_curr, 1e-10, 1 - 1e-10)
                    p_prop = np.clip(p_prop, 1e-10, 1 - 1e-10)

                    if responses[i, j] == 1:
                        log_like_curr += np.log(p_curr)
                        log_like_prop += np.log(p_prop)
                    else:
                        log_like_curr += np.log(1 - p_curr)
                        log_like_prop += np.log(1 - p_prop)

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
        n_persons = len(theta)

        new_guess = guess.copy()

        for j in range(n_items):
            guess_prop = guess[j] + rng.normal(0, 0.02)
            guess_prop = np.clip(guess_prop, 0.01, 0.5)

            log_like_curr = 0.0
            log_like_prop = 0.0

            for i in range(n_persons):
                if responses[i, j] >= 0:
                    z = disc[j] * (theta[i] - diff[j])
                    p_star = 1.0 / (1.0 + np.exp(-z))

                    p_curr = guess[j] + (1 - guess[j]) * p_star
                    p_prop = guess_prop + (1 - guess_prop) * p_star

                    p_curr = np.clip(p_curr, 1e-10, 1 - 1e-10)
                    p_prop = np.clip(p_prop, 1e-10, 1 - 1e-10)

                    if responses[i, j] == 1:
                        log_like_curr += np.log(p_curr)
                        log_like_prop += np.log(p_prop)
                    else:
                        log_like_curr += np.log(1 - p_curr)
                        log_like_prop += np.log(1 - p_prop)

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

    @staticmethod
    def _log_mvn_density_single(
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        sigma_inv: NDArray[np.float64],
        log_det_sigma: float,
    ) -> float:
        """Log MVN density for single observation."""
        d = len(x)
        diff = x - mean
        maha = diff @ sigma_inv @ diff
        log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det_sigma)
        return log_norm - 0.5 * maha

    def _compute_dic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
    ) -> float:
        """Compute Deviance Information Criterion."""
        n_samples = theta_samples.shape[0]

        deviances = np.zeros(n_samples)
        for s in range(n_samples):
            ll = np.sum(
                model.joint_log_likelihood(
                    responses, log_rt, theta_samples[s], tau_samples[s]
                )
            )
            deviances[s] = -2 * ll

        d_bar = np.mean(deviances)

        theta_mean = np.mean(theta_samples, axis=0)
        tau_mean = np.mean(tau_samples, axis=0)
        ll_at_mean = np.sum(
            model.joint_log_likelihood(responses, log_rt, theta_mean, tau_mean)
        )
        d_theta_bar = -2 * ll_at_mean

        p_d = d_bar - d_theta_bar
        dic = d_bar + p_d

        return dic

    def _compute_waic(
        self,
        model: ResponseTimeModel,
        responses: NDArray[np.int_],
        log_rt: NDArray[np.float64],
        theta_samples: NDArray[np.float64],
        tau_samples: NDArray[np.float64],
    ) -> float:
        """Compute Watanabe-Akaike Information Criterion."""
        n_samples = theta_samples.shape[0]
        n_persons = responses.shape[0]

        log_likes = np.zeros((n_samples, n_persons))
        for s in range(n_samples):
            log_likes[s] = model.joint_log_likelihood(
                responses, log_rt, theta_samples[s], tau_samples[s]
            )

        lppd = np.sum(np.log(np.mean(np.exp(log_likes), axis=0)))

        p_waic = np.sum(np.var(log_likes, axis=0))

        waic = -2 * (lppd - p_waic)

        return waic

    def _compute_rhat(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute Gelman-Rubin convergence diagnostic."""
        rhat = {}

        for name, samples in chains.items():
            if samples.ndim == 1:
                n = len(samples)
                if n < 4:
                    rhat[name] = np.nan
                    continue

                mid = n // 2
                chain1 = samples[:mid]
                chain2 = samples[mid:]

                mean1, mean2 = np.mean(chain1), np.mean(chain2)
                var1, var2 = np.var(chain1, ddof=1), np.var(chain2, ddof=1)

                W = (var1 + var2) / 2
                B = mid * (
                    (mean1 - (mean1 + mean2) / 2) ** 2
                    + (mean2 - (mean1 + mean2) / 2) ** 2
                )
                var_est = (1 - 1 / mid) * W + B / mid

                rhat[name] = np.sqrt(var_est / W) if W > 0 else 1.0
            else:
                rhat[name] = np.mean(
                    [
                        self._compute_rhat({f"{name}_{j}": samples[:, j]})[
                            f"{name}_{j}"
                        ]
                        for j in range(samples.shape[1])
                    ]
                )

        return rhat

    def _compute_ess(self, chains: dict[str, NDArray[np.float64]]) -> dict[str, float]:
        """Compute effective sample size."""
        ess = {}

        for name, samples in chains.items():
            if samples.ndim == 1:
                n = len(samples)
                if n < 4:
                    ess[name] = float(n)
                    continue

                acf = np.correlate(
                    samples - np.mean(samples), samples - np.mean(samples), mode="full"
                )
                acf = acf[n - 1 :] / acf[n - 1]

                tau = 1.0
                for k in range(1, min(n // 2, 100)):
                    if acf[k] < 0:
                        break
                    tau += 2 * acf[k]

                ess[name] = n / tau
            else:
                ess[name] = np.mean(
                    [
                        self._compute_ess({f"{name}_{j}": samples[:, j]})[f"{name}_{j}"]
                        for j in range(samples.shape[1])
                    ]
                )

        return ess
