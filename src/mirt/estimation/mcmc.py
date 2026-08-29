"""MCMC and MHRM Estimation for IRT Models.

This module provides stochastic estimation methods:
- MHRM (Metropolis-Hastings Robbins-Monro)
- Gibbs Sampling for full Bayesian inference

Uses fast Rust backend when available for 2PL models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtEstimationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.estimation.base import BaseEstimator
from mirt.results.fit_result import FitResult

PosteriorValue = NDArray[np.float64] | np.float64
PosteriorSummary = dict[str, dict[str, PosteriorValue]]
CredibleIntervals = dict[str, tuple[PosteriorValue, PosteriorValue]]


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

    @staticmethod
    def _validate_credible_level(credible_level: float) -> float:
        """Return a finite credible level strictly between zero and one."""
        if (
            isinstance(credible_level, bool)
            or not isinstance(credible_level, Real)
            or not np.isfinite(credible_level)
            or not 0.0 < credible_level < 1.0
        ):
            raise ValueError("credible_level must be between 0 and 1")
        return float(credible_level)

    def _selected_chains(
        self,
        parameters: str | Sequence[str] | None,
    ) -> dict[str, NDArray[np.float64]]:
        """Select and validate posterior chains for result summaries."""
        if parameters is None:
            names = tuple(self.chains)
        elif isinstance(parameters, str):
            names = (parameters,)
        else:
            try:
                names = tuple(parameters)
            except TypeError as exc:
                raise ValueError(
                    "parameters must be a chain name or sequence of chain names"
                ) from exc
            if not all(isinstance(name, str) for name in names):
                raise ValueError("parameters must contain only chain names")

        unknown = tuple(
            dict.fromkeys(name for name in names if name not in self.chains)
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"unknown posterior chain: {joined}")

        selected: dict[str, NDArray[np.float64]] = {}
        n_draws: int | None = None
        for name in dict.fromkeys(names):
            try:
                chain = np.asarray(self.chains[name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"posterior chain '{name}' must contain numeric values"
                ) from exc
            if chain.ndim == 0 or chain.size == 0 or chain.shape[0] == 0:
                raise ValueError(
                    f"posterior chain '{name}' must contain at least one draw"
                )
            if not np.all(np.isfinite(chain)):
                raise ValueError(
                    f"posterior chain '{name}' must contain only finite values"
                )
            if n_draws is None:
                n_draws = chain.shape[0]
            elif chain.shape[0] != n_draws:
                raise ValueError(
                    "selected posterior chains must have equal draw counts"
                )
            selected[name] = chain

        return selected

    def posterior_summary(
        self,
        credible_level: float = 0.95,
        parameters: str | Sequence[str] | None = None,
    ) -> PosteriorSummary:
        """Return posterior moments and equal-tailed credible intervals.

        Statistics are computed over the leading draw dimension. Remaining
        dimensions are preserved, so item parameters and person abilities can
        be summarized with the same method.

        Parameters
        ----------
        credible_level : float
            Probability covered by each equal-tailed interval.
        parameters : str or sequence of str, optional
            Chain names to summarize. By default, all stored chains are used.

        Returns
        -------
        dict
            Mean, standard deviation, median, and interval bounds for each
            selected chain.
        """
        level = self._validate_credible_level(credible_level)
        selected = self._selected_chains(parameters)
        tail = (1.0 - level) / 2.0

        result: PosteriorSummary = {}
        for name, chain in selected.items():
            lower, median, upper = np.quantile(
                chain,
                (tail, 0.5, 1.0 - tail),
                axis=0,
            )
            result[name] = {
                "mean": np.mean(chain, axis=0),
                "std": np.std(chain, axis=0),
                "median": median,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        return result

    def credible_intervals(
        self,
        credible_level: float = 0.95,
        parameters: str | Sequence[str] | None = None,
    ) -> CredibleIntervals:
        """Return equal-tailed credible intervals for selected chains."""
        level = self._validate_credible_level(credible_level)
        selected = self._selected_chains(parameters)
        tail = (1.0 - level) / 2.0
        intervals: CredibleIntervals = {}
        for name, chain in selected.items():
            lower, upper = np.quantile(chain, (tail, 1.0 - tail), axis=0)
            intervals[name] = (lower, upper)
        return intervals

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

        if self.ess:
            lines.extend(("", "Effective sample size:"))
            for name, ess in self.ess.items():
                lines.append(f"  {name}: {ess:.1f}")

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
        from mirt._backend_config import should_use_rust
        from mirt._rust_backend import mhrm_fit_2pl

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if should_use_rust(self.use_rust) and _is_2pl_unidimensional(model):
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

            from mirt._rust_backend import compute_item_se_parallel, e_step_complete
            from mirt.estimation.quadrature import GaussHermiteQuadrature

            disc = np.asarray(discrimination)
            diff = np.asarray(difficulty)
            quad = GaussHermiteQuadrature(n_points=21, n_dimensions=1)
            posterior_weights, _ = e_step_complete(
                responses,
                quad.nodes.ravel(),
                quad.weights.ravel(),
                disc,
                diff,
            )
            se_a, se_b = compute_item_se_parallel(
                responses,
                posterior_weights,
                quad.nodes.ravel(),
                disc,
                diff,
            )

            return FitResult(
                model=model,
                log_likelihood=log_likelihood,
                n_iterations=self.n_cycles,
                converged=True,
                standard_errors={
                    "discrimination": np.asarray(se_a),
                    "difficulty": np.asarray(se_b),
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
            prob = np.clip(prob, PROB_EPSILON, 1 - PROB_EPSILON)

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
        return model.n_parameters


class GibbsSampler(BaseEstimator):
    """Full Bayesian estimation via Gibbs sampling.

    Implements blocked Gibbs sampling where:
    1. Sample theta | parameters, data
    2. Sample parameters | theta, data

    This provides full posterior distributions for all parameters.

    Uses fast parallel Rust backend for 2PL models when available.
    Supports multi-chain parallelization when n_chains > 1 and parallel_chains=True.
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
        parallel_chains: bool = False,
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
        parallel_chains : bool
            Whether to run multiple chains in parallel (only when n_chains > 1)
        """
        super().__init__(max_iter=n_iter, verbose=verbose)
        self.n_iter = n_iter
        self.burnin = burnin
        self.thin = thin
        self.n_chains = n_chains
        self.priors = priors or {}
        self.use_rust = use_rust
        self.seed = seed
        self.parallel_chains = parallel_chains

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
        from mirt._backend_config import should_use_rust
        from mirt._rust_backend import gibbs_sample_2pl

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if should_use_rust(self.use_rust) and _is_2pl_unidimensional(model):
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

        if not model._parameters:
            model._initialize_parameters()

        if self.parallel_chains and self.n_chains > 1:
            chain_arrays = self._run_parallel_chains(model, responses, n_persons)
        else:
            chain_arrays = self._run_single_chain(
                model, responses, n_persons, self.seed
            )

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

    def _run_single_chain(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
        seed: int | None,
    ) -> dict[str, NDArray]:
        """Run a single MCMC chain."""
        theta = np.zeros((n_persons, model.n_factors))
        rng = np.random.default_rng(seed)

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

        return {name: np.array(chain) for name, chain in chains.items()}

    def _run_parallel_chains(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        n_persons: int,
    ) -> dict[str, NDArray]:
        """Run multiple MCMC chains in parallel and combine results."""
        from concurrent.futures import ProcessPoolExecutor

        base_seed = self.seed if self.seed is not None else 0
        seeds = [base_seed + i * 1000 for i in range(self.n_chains)]

        if self.verbose:
            print(f"Running {self.n_chains} chains in parallel...")

        with ProcessPoolExecutor(max_workers=self.n_chains) as executor:
            futures = [
                executor.submit(
                    self._run_single_chain, model.copy(), responses, n_persons, seed
                )
                for seed in seeds
            ]
            all_chains = [f.result() for f in futures]

        combined: dict[str, list[NDArray]] = {}
        for chain_result in all_chains:
            for name, values in chain_result.items():
                if name not in combined:
                    combined[name] = []
                combined[name].append(values)

        return {
            name: np.concatenate(arrays, axis=0) for name, arrays in combined.items()
        }

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

    def _compute_waic(
        self,
        chains: dict[str, NDArray],
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> float:
        """Compute stable pointwise WAIC from paired posterior draws."""
        theta_chain = np.asarray(chains["theta"], dtype=np.float64)
        if theta_chain.ndim != 3 or theta_chain.shape[0] == 0:
            raise MirtEstimationError(
                "theta chain must have shape (n_samples, n_persons, n_factors)"
            )

        n_samples, n_chain_persons, n_chain_factors = theta_chain.shape
        n_persons = responses.shape[0]
        if n_chain_persons != n_persons or n_chain_factors != model.n_factors:
            raise MirtEstimationError(
                "theta chain dimensions must match the fitted data and model",
                expected=(n_persons, model.n_factors),
                actual=(n_chain_persons, n_chain_factors),
            )
        if not np.all(np.isfinite(theta_chain)):
            raise MirtEstimationError("theta chain must contain only finite values")

        parameter_chains: dict[str, NDArray[np.float64]] = {}
        for name, parameter in model.parameters.items():
            if name not in chains:
                raise MirtEstimationError(
                    "posterior parameter chain is missing",
                    parameter=name,
                )
            values = np.asarray(chains[name], dtype=np.float64)
            expected_shape = (n_samples, *parameter.shape)
            if values.shape != expected_shape:
                raise MirtEstimationError(
                    "posterior parameter chain has an unexpected shape",
                    parameter=name,
                    expected=expected_shape,
                    actual=values.shape,
                )
            if not np.all(np.isfinite(values)):
                raise MirtEstimationError(
                    "posterior parameter chain must contain only finite values",
                    parameter=name,
                )
            parameter_chains[name] = values

        evaluation_model = model.copy()
        evaluation_model._is_fitted = True
        log_sum_exp = np.full(n_persons, -np.inf, dtype=np.float64)
        running_mean = np.zeros(n_persons, dtype=np.float64)
        running_m2 = np.zeros(n_persons, dtype=np.float64)

        for sample_index in range(n_samples):
            for name, values in parameter_chains.items():
                evaluation_model._parameters[name] = values[sample_index]

            pointwise = np.asarray(
                evaluation_model.log_likelihood(
                    responses,
                    theta_chain[sample_index],
                ),
                dtype=np.float64,
            )
            if pointwise.shape != (n_persons,) or not np.all(np.isfinite(pointwise)):
                raise MirtEstimationError(
                    "model returned invalid pointwise posterior log-likelihoods",
                    iteration=sample_index,
                    expected=(n_persons,),
                    actual=pointwise.shape,
                )

            count = sample_index + 1
            delta = pointwise - running_mean
            running_mean += delta / count
            running_m2 += delta * (pointwise - running_mean)
            log_sum_exp = np.logaddexp(log_sum_exp, pointwise)

        log_pointwise_predictive_density = np.sum(
            log_sum_exp - np.log(float(n_samples))
        )
        effective_parameters = np.sum(running_m2 / n_samples)
        waic = -2.0 * (log_pointwise_predictive_density - effective_parameters)
        if not np.isfinite(waic):
            raise MirtEstimationError("WAIC is non-finite")
        return float(waic)
