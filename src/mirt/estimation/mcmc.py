"""MCMC and MHRM Estimation for IRT Models.

This module provides stochastic estimation methods:
- MHRM (Metropolis-Hastings Robbins-Monro)
- Gibbs Sampling for full Bayesian inference

Uses fast Rust backend when available for 2PL models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.estimation.base import BaseEstimator
from mirt.results.fit_result import FitResult


def _is_2pl_unidimensional(model: BaseItemModel) -> bool:
    """Check if model is 2PL unidimensional."""
    return (
        model.model_name == "2PL"
        and hasattr(model, "n_factors")
        and model.n_factors == 1
    )


@dataclass
class MCMCResult:
    """Result from MCMC estimation.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model with posterior mean parameters
    chains : dict
        MCMC chains for each parameter
    log_likelihood : float
        Log-likelihood at posterior mean
    dic : float
        Deviance Information Criterion
    waic : float
        Watanabe-Akaike Information Criterion
    rhat : dict
        Gelman-Rubin convergence diagnostics
    ess : dict
        Effective sample sizes
    """

    model: Any
    chains: dict[str, NDArray[np.float64]]
    log_likelihood: float
    dic: float
    waic: float
    rhat: dict[str, float]
    ess: dict[str, float]
    n_iterations: int
    burnin: int
    thin: int

    def summary(self) -> str:
        """Generate summary of MCMC results."""
        lines = [
            "MCMC Estimation Summary",
            "=" * 50,
            f"Iterations: {self.n_iterations}",
            f"Burnin: {self.burnin}",
            f"Thinning: {self.thin}",
            "",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"DIC: {self.dic:.4f}",
            f"WAIC: {self.waic:.4f}",
            "",
            "Convergence (R-hat):",
        ]

        for name, rhat in self.rhat.items():
            status = "OK" if rhat < 1.1 else "WARNING"
            lines.append(f"  {name}: {rhat:.4f} ({status})")

        return "\n".join(lines)


class MHRMEstimator(BaseEstimator):
    """Metropolis-Hastings Robbins-Monro estimator.

    MHRM is a stochastic approximation method that combines:
    1. Metropolis-Hastings sampling for latent variables (theta)
    2. Robbins-Monro updates for item parameters

    This is faster than full MCMC while providing good estimates
    for complex models where EM may struggle.

    Uses fast parallel Rust backend for 2PL models when available.

    References
    ----------
    Cai, L. (2010). Metropolis-Hastings Robbins-Monro algorithm for
    confirmatory item factor analysis. Journal of Educational and
    Behavioral Statistics, 35(3), 307-335.
    """

    def __init__(
        self,
        n_cycles: int = 2000,
        burnin: int = 500,
        n_chains: int = 1,
        proposal_sd: float = 0.5,
        gain_sequence: str = "standard",
        verbose: bool = False,
        use_rust: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize MHRM estimator.

        Parameters
        ----------
        n_cycles : int
            Number of MHRM cycles
        burnin : int
            Number of burnin cycles
        n_chains : int
            Number of parallel chains
        proposal_sd : float
            Standard deviation for MH proposals
        gain_sequence : str
            Type of gain sequence ('standard' or 'adaptive')
        verbose : bool
            Whether to print progress
        use_rust : bool
            Whether to use Rust backend when available
        seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(max_iter=n_cycles, tol=1e-4, verbose=verbose)
        self.n_cycles = n_cycles
        self.burnin = burnin
        self.n_chains = n_chains
        self.proposal_sd = proposal_sd
        self.gain_sequence = gain_sequence
        self.use_rust = use_rust
        self.seed = seed

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        **kwargs: Any,
    ) -> FitResult:
        """Fit model using MHRM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit
        responses : NDArray
            Response matrix (n_persons, n_items)
        **kwargs
            Additional arguments (prior_mean, prior_cov)

        Returns
        -------
        FitResult
            Fitted model result
        """
        from mirt._rust_backend import RUST_AVAILABLE, mhrm_fit_2pl

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if self.use_rust and RUST_AVAILABLE and _is_2pl_unidimensional(model):
            seed = (
                self.seed
                if self.seed is not None
                else np.random.default_rng().integers(0, 2**31)
            )

            discrimination, difficulty, log_likelihood = mhrm_fit_2pl(
                responses,
                n_cycles=self.n_cycles,
                burnin=self.burnin,
                proposal_sd=self.proposal_sd,
                seed=seed,
            )

            if not model._parameters:
                model._initialize_parameters()
            model._parameters["discrimination"] = np.asarray(discrimination)
            model._parameters["difficulty"] = np.asarray(difficulty)
            model._is_fitted = True

            n_params = 2 * n_items
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_persons) * n_params

            return FitResult(
                model=model,
                log_likelihood=log_likelihood,
                n_iterations=self.n_cycles,
                converged=True,
                standard_errors={
                    "discrimination": np.full(n_items, np.nan),
                    "difficulty": np.full(n_items, np.nan),
                },
                aic=aic,
                bic=bic,
                n_observations=n_persons * n_items,
                n_parameters=n_params,
            )

        if not model._parameters:
            model._initialize_parameters()

        theta = np.zeros((n_persons, model.n_factors))

        param_history: dict[str, list] = {name: [] for name in model.parameters}

        rng = np.random.default_rng(self.seed)

        for cycle in range(self.n_cycles):
            theta = self._sample_theta(model, responses, theta, rng)

            gain = self._compute_gain(cycle)
            self._update_parameters(model, responses, theta, gain, rng)

            if cycle >= self.burnin:
                for name, values in model.parameters.items():
                    param_history[name].append(values.copy())

            if self.verbose and (cycle + 1) % 100 == 0:
                ll = np.sum(model.log_likelihood(responses, theta))
                print(f"Cycle {cycle + 1}/{self.n_cycles}: LL = {ll:.4f}")

        for name in model.parameters:
            if param_history[name]:
                model._parameters[name] = np.mean(param_history[name], axis=0)

        model._is_fitted = True

        theta_final = self._estimate_theta_map(model, responses, rng)
        ll = float(np.sum(model.log_likelihood(responses, theta_final)))

        se = {}
        for name, chain in param_history.items():
            if chain:
                se[name] = np.std(chain, axis=0)

        return FitResult(
            model=model,
            log_likelihood=ll,
            n_iterations=self.n_cycles,
            converged=True,
            standard_errors=se,
            aic=-2 * ll + 2 * self._count_parameters(model),
            bic=-2 * ll + np.log(n_persons) * self._count_parameters(model),
            n_observations=n_persons * n_items,
            n_parameters=self._count_parameters(model),
        )

    def _sample_theta(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Metropolis-Hastings step for theta sampling."""
        n_persons = theta.shape[0]

        proposal = theta + rng.normal(0, self.proposal_sd, theta.shape)

        ll_current = model.log_likelihood(responses, theta)
        ll_proposal = model.log_likelihood(responses, proposal)

        prior_current = stats.norm.logpdf(theta).sum(axis=1)
        prior_proposal = stats.norm.logpdf(proposal).sum(axis=1)

        log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current)
        log_u = np.log(rng.random(n_persons))

        accept = log_u < log_alpha
        theta_new = np.where(accept[:, None], proposal, theta)

        return theta_new

    def _compute_gain(self, cycle: int) -> float:
        """Compute gain for Robbins-Monro update."""
        if self.gain_sequence == "standard":
            return 1.0 / (cycle + 1)
        elif self.gain_sequence == "adaptive":
            return min(1.0, 10.0 / (cycle + 10))
        else:
            return 1.0 / (cycle + 1)

    def _update_parameters(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        gain: float,
        rng: np.random.Generator,
    ) -> None:
        """Robbins-Monro update for item parameters."""
        n_items = model.n_items

        for j in range(n_items):
            valid = responses[:, j] >= 0
            if not valid.any():
                continue

            theta_j = theta[valid]
            resp_j = responses[valid, j]

            prob = model.probability(theta_j, j)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)

            residual = resp_j - prob

            if "discrimination" in model.parameters:
                a = model.parameters["discrimination"]
                if a.ndim == 1:
                    gradient_a = np.mean(residual * theta_j.ravel())
                    a[j] = np.clip(a[j] + gain * gradient_a, 0.1, 5.0)

            if "difficulty" in model.parameters:
                b = model.parameters["difficulty"]
                gradient_b = -np.mean(residual)
                b[j] = np.clip(b[j] + gain * gradient_b, -6.0, 6.0)

    def _estimate_theta_map(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Estimate theta using MAP with current parameters."""
        n_persons = responses.shape[0]
        theta = rng.standard_normal((n_persons, model.n_factors))

        for _ in range(50):
            ll = model.log_likelihood(responses, theta)
            prior = -0.5 * np.sum(theta**2, axis=1)

            h = 1e-4
            grad = np.zeros_like(theta)
            for d in range(model.n_factors):
                theta_plus = theta.copy()
                theta_plus[:, d] += h
                ll_plus = model.log_likelihood(responses, theta_plus)
                prior_plus = -0.5 * np.sum(theta_plus**2, axis=1)
                grad[:, d] = (ll_plus + prior_plus - ll - prior) / h

            theta = theta + 0.1 * grad

        return theta

    def _count_parameters(self, model: BaseItemModel) -> int:
        """Count number of parameters."""
        return sum(v.size for v in model.parameters.values())


class GibbsSampler(BaseEstimator):
    """Full Bayesian estimation via Gibbs sampling.

    Implements blocked Gibbs sampling where:
    1. Sample theta | parameters, data
    2. Sample parameters | theta, data

    This provides full posterior distributions for all parameters.

    Uses fast parallel Rust backend for 2PL models when available.
    """

    def __init__(
        self,
        n_iter: int = 5000,
        burnin: int = 1000,
        thin: int = 1,
        n_chains: int = 1,
        priors: dict[str, Any] | None = None,
        verbose: bool = False,
        use_rust: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize Gibbs sampler.

        Parameters
        ----------
        n_iter : int
            Number of iterations
        burnin : int
            Burnin iterations
        thin : int
            Thinning interval
        n_chains : int
            Number of chains
        priors : dict, optional
            Prior specifications for parameters
        verbose : bool
            Whether to print progress
        use_rust : bool
            Whether to use Rust backend when available
        seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(max_iter=n_iter, verbose=verbose)
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.n_chains = n_chains
        self.priors = priors or {}
        self.use_rust = use_rust
        self.seed = seed

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        **kwargs: Any,
    ) -> MCMCResult:
        """Fit model using Gibbs sampling.

        Parameters
        ----------
        model : BaseItemModel
            IRT model
        responses : NDArray
            Response matrix

        Returns
        -------
        MCMCResult
            MCMC estimation result with chains and diagnostics
        """
        from mirt._rust_backend import RUST_AVAILABLE, gibbs_sample_2pl

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if self.use_rust and RUST_AVAILABLE and _is_2pl_unidimensional(model):
            seed = (
                self.seed
                if self.seed is not None
                else np.random.default_rng().integers(0, 2**31)
            )

            disc_chain, diff_chain, theta_chain, ll_chain = gibbs_sample_2pl(
                responses,
                n_iter=self.n_iter,
                burnin=self.burnin,
                thin=self.thin,
                seed=seed,
            )

            if not model._parameters:
                model._initialize_parameters()
            model._parameters["discrimination"] = np.mean(disc_chain, axis=0)
            model._parameters["difficulty"] = np.mean(diff_chain, axis=0)
            model._is_fitted = True

            chain_arrays: dict[str, NDArray[np.float64]] = {
                "discrimination": np.asarray(disc_chain),
                "difficulty": np.asarray(diff_chain),
                "theta": np.asarray(theta_chain),
                "log_likelihood": np.asarray(ll_chain),
            }

            rhat = self._compute_rhat(chain_arrays)
            ess = self._compute_ess(chain_arrays)
            ll_mean = float(np.mean(ll_chain))

            dic = self._compute_dic_from_chains(chain_arrays, model, responses)
            waic = self._compute_waic_from_chains(chain_arrays, model, responses)

            return MCMCResult(
                model=model,
                chains=chain_arrays,
                log_likelihood=ll_mean,
                dic=dic,
                waic=waic,
                rhat=rhat,
                ess=ess,
                n_iterations=self.n_iter,
                burnin=self.burnin,
                thin=self.thin,
            )

        if not model._parameters:
            model._initialize_parameters()

        theta = np.zeros((n_persons, model.n_factors))
        rng = np.random.default_rng(self.seed)

        chains: dict[str, list] = {name: [] for name in model.parameters}
        chains["theta"] = []
        chains["log_likelihood"] = []

        for iteration in range(self.n_iter):
            theta = self._sample_theta_gibbs(model, responses, theta, rng)

            self._sample_parameters(model, responses, theta, rng)

            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                for name, values in model.parameters.items():
                    chains[name].append(values.copy())
                chains["theta"].append(theta.copy())
                ll = np.sum(model.log_likelihood(responses, theta))
                chains["log_likelihood"].append(ll)

            if self.verbose and (iteration + 1) % 500 == 0:
                ll = np.sum(model.log_likelihood(responses, theta))
                print(f"Iteration {iteration + 1}/{self.n_iter}: LL = {ll:.4f}")

        chain_arrays = {name: np.array(chain) for name, chain in chains.items()}

        for name in model.parameters:
            model._parameters[name] = np.mean(chain_arrays[name], axis=0)

        model._is_fitted = True

        rhat = self._compute_rhat(chain_arrays)
        ess = self._compute_ess(chain_arrays)
        ll_mean = float(np.mean(chain_arrays["log_likelihood"]))
        dic = self._compute_dic(chain_arrays, model, responses)
        waic = self._compute_waic(chain_arrays, model, responses)

        return MCMCResult(
            model=model,
            chains=chain_arrays,
            log_likelihood=ll_mean,
            dic=dic,
            waic=waic,
            rhat=rhat,
            ess=ess,
            n_iterations=self.n_iter,
            burnin=self.burnin,
            thin=self.thin,
        )

    def _sample_theta_gibbs(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample theta using MH within Gibbs."""
        n_persons = theta.shape[0]
        proposal_sd = 0.5

        proposal = theta + rng.normal(0, proposal_sd, theta.shape)

        ll_current = model.log_likelihood(responses, theta)
        ll_proposal = model.log_likelihood(responses, proposal)

        prior_current = stats.norm.logpdf(theta).sum(axis=1)
        prior_proposal = stats.norm.logpdf(proposal).sum(axis=1)

        log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current)
        accept = np.log(rng.random(n_persons)) < log_alpha

        return np.where(accept[:, None], proposal, theta)

    def _sample_parameters(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> None:
        """Sample item parameters using MH."""
        proposal_sd = 0.1

        for name, values in model.parameters.items():
            proposal = values + rng.normal(0, proposal_sd, values.shape)

            if "discrimination" in name or "slope" in name:
                proposal = np.clip(proposal, 0.1, 5.0)
            elif "difficulty" in name or "intercept" in name:
                proposal = np.clip(proposal, -6.0, 6.0)

            model._parameters[name] = proposal
            ll_proposal = np.sum(model.log_likelihood(responses, theta))

            model._parameters[name] = values
            ll_current = np.sum(model.log_likelihood(responses, theta))

            log_alpha = ll_proposal - ll_current

            if np.log(rng.random()) < log_alpha:
                model._parameters[name] = proposal

    def _compute_rhat(self, chains: dict[str, NDArray]) -> dict[str, float]:
        """Compute Gelman-Rubin R-hat diagnostic."""
        rhat = {}

        for name, chain in chains.items():
            if name in ("theta", "log_likelihood"):
                continue

            if chain.ndim == 1:
                values = chain
            else:
                values = chain.mean(axis=tuple(range(1, chain.ndim)))

            n = len(values)
            if n < 10:
                rhat[name] = np.nan
                continue

            first_half = values[: n // 2]
            second_half = values[n // 2 :]

            B = (n // 2) * np.var([first_half.mean(), second_half.mean()])
            W = (np.var(first_half) + np.var(second_half)) / 2

            if W > 0:
                var_est = (1 - 1 / (n // 2)) * W + B / (n // 2)
                rhat[name] = float(np.sqrt(var_est / W))
            else:
                rhat[name] = 1.0

        return rhat

    def _compute_ess(self, chains: dict[str, NDArray]) -> dict[str, float]:
        """Compute effective sample size."""
        ess = {}

        for name, chain in chains.items():
            if name in ("theta", "log_likelihood"):
                continue

            if chain.ndim == 1:
                values = chain
            else:
                values = chain.mean(axis=tuple(range(1, chain.ndim)))

            n = len(values)
            if n < 10:
                ess[name] = float(n)
                continue

            acf = np.correlate(
                values - values.mean(), values - values.mean(), mode="full"
            )
            acf = acf[n - 1 :] / acf[n - 1]

            neg_idx = np.where(acf < 0)[0]
            if len(neg_idx) > 0:
                cutoff = neg_idx[0]
            else:
                cutoff = min(n // 2, 100)

            tau = 1 + 2 * np.sum(acf[1:cutoff])
            ess[name] = float(n / max(tau, 1))

        return ess

    def _compute_dic(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute Deviance Information Criterion."""
        ll_mean = np.mean(chains["log_likelihood"])
        deviance_mean = -2 * ll_mean

        theta_mean = np.mean(chains["theta"], axis=0)
        ll_at_mean = np.sum(model.log_likelihood(responses, theta_mean))
        deviance_at_mean = -2 * ll_at_mean

        pd = deviance_mean - deviance_at_mean

        return float(deviance_mean + pd)

    def _compute_dic_from_chains(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute DIC from Rust chains."""
        ll_mean = np.mean(chains["log_likelihood"])
        deviance_mean = -2 * ll_mean

        theta_chain = chains["theta"]
        theta_mean = np.mean(theta_chain, axis=0)
        if theta_mean.ndim == 2:
            theta_mean = theta_mean

        ll_at_mean = np.sum(model.log_likelihood(responses, theta_mean))
        deviance_at_mean = -2 * ll_at_mean

        pd = deviance_mean - deviance_at_mean

        return float(deviance_mean + pd)

    def _compute_waic(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute Watanabe-Akaike Information Criterion."""
        n_samples = len(chains["log_likelihood"])

        lppd = 0
        pwaic = 0

        for i in range(responses.shape[0]):
            person_ll = []
            for s in range(n_samples):
                theta_s = chains["theta"][s][i : i + 1]
                resp_i = responses[i : i + 1]
                ll_s = model.log_likelihood(resp_i, theta_s)[0]
                person_ll.append(ll_s)

            person_ll = np.array(person_ll)
            lppd += np.log(np.mean(np.exp(person_ll)))
            pwaic += np.var(person_ll)

        return float(-2 * (lppd - pwaic))

    def _compute_waic_from_chains(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute WAIC from Rust chains."""
        theta_chain = chains["theta"]
        n_samples = theta_chain.shape[0]
        n_persons = responses.shape[0]

        lppd = 0.0
        pwaic = 0.0

        for i in range(n_persons):
            person_ll = []
            for s in range(n_samples):
                theta_s = theta_chain[s, i : i + 1, :]
                resp_i = responses[i : i + 1]
                ll_s = model.log_likelihood(resp_i, theta_s)[0]
                person_ll.append(ll_s)

            person_ll = np.array(person_ll)
            lppd += np.log(np.mean(np.exp(person_ll)))
            pwaic += np.var(person_ll)

        return float(-2 * (lppd - pwaic))
