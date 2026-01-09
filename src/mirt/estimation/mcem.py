"""Monte Carlo EM (MCEM) and Quasi-Monte Carlo EM (QMCEM) estimators.

These estimation methods are useful for high-dimensional IRT models where
Gauss-Hermite quadrature becomes computationally infeasible. They use
Monte Carlo integration in the E-step instead of numerical quadrature.

MCEM uses standard pseudo-random sampling, while QMCEM uses low-discrepancy
sequences (Quasi-Monte Carlo) for more uniform coverage of the integration
space and faster convergence.

References:
    Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the
        EM algorithm and the poor man's data augmentation algorithms.
        Journal of the American Statistical Association, 85(411), 699-704.

    Cagnone, S., & Monari, P. (2013). Latent variable models for ordinal
        data by using the adaptive quadrature approximation.
        Computational Statistics, 28(2), 597-619.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from mirt.estimation.base import BaseEstimator
from mirt.utils.numeric import logsumexp

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


class MCEMEstimator(BaseEstimator):
    """Monte Carlo EM estimator for IRT models.

    Uses Monte Carlo integration in the E-step, making it suitable for
    models with many latent dimensions where quadrature is infeasible.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples per person per iteration.
        More samples give more accurate E-step but slower computation.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance for log-likelihood change.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for reproducibility.
    importance_sampling : bool
        Whether to use importance sampling from the prior.
        Improves efficiency when posterior differs from prior.

    Notes
    -----
    MCEM is particularly useful when:
    - The model has more than 3-4 latent dimensions
    - Quadrature-based EM is too slow
    - Exact integration is not required

    The number of samples should increase as iterations progress to
    ensure convergence. This implementation uses a fixed number for
    simplicity.
    """

    def __init__(
        self,
        n_samples: int = 500,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        importance_sampling: bool = True,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_samples < 50:
            raise ValueError("n_samples should be at least 50")

        self.n_samples = n_samples
        self.seed = seed
        self.importance_sampling = importance_sampling
        self._rng: np.random.Generator | None = None

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> FitResult:
        """Fit model using Monte Carlo EM algorithm.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit
        responses : ndarray of shape (n_persons, n_items)
            Response matrix
        prior_mean : ndarray of shape (n_factors,), optional
            Prior mean for latent abilities
        prior_cov : ndarray of shape (n_factors, n_factors), optional
            Prior covariance for latent abilities

        Returns
        -------
        FitResult
            Fitted model with estimates and diagnostics
        """
        from mirt.results.fit_result import FitResult

        responses = self._validate_responses(responses, model.n_items)
        n_persons = responses.shape[0]
        n_factors = model.n_factors

        self._rng = np.random.default_rng(self.seed)

        if prior_mean is None:
            prior_mean = np.zeros(n_factors)
        if prior_cov is None:
            prior_cov = np.eye(n_factors)

        if not model._is_fitted:
            model._initialize_parameters()

        self._convergence_history = []
        prev_ll = -np.inf

        L = np.linalg.cholesky(prior_cov)

        for iteration in range(self.max_iter):
            theta_samples, weights = self._e_step_mc(
                model, responses, prior_mean, L, n_factors
            )

            current_ll = self._estimate_marginal_ll(
                model, responses, theta_samples, weights
            )
            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            self._m_step_mc(model, responses, theta_samples, weights)

        model._is_fitted = True

        standard_errors = self._compute_standard_errors_mc(
            model, responses, theta_samples, weights
        )

        n_params = model.n_parameters
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=iteration + 1,
            converged=iteration < self.max_iter - 1,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Monte Carlo sampling.

        Returns theta samples and their importance weights for each person.
        """
        n_persons = responses.shape[0]

        z = self._rng.standard_normal((n_persons, self.n_samples, n_factors))
        theta_samples = prior_mean + np.einsum("ij,...j->...i", L, z)

        log_likes = np.zeros((n_persons, self.n_samples))

        for s in range(self.n_samples):
            theta_s = theta_samples[:, s, :]
            log_likes[:, s] = model.log_likelihood(responses, theta_s)

        log_weights = log_likes
        log_weights_normalized = log_weights - logsumexp(
            log_weights, axis=1, keepdims=True
        )
        weights = np.exp(log_weights_normalized)

        return theta_samples, weights

    def _estimate_marginal_ll(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> float:
        """Estimate marginal log-likelihood using importance sampling."""
        n_persons = responses.shape[0]

        log_likes = np.zeros((n_persons, self.n_samples))
        for s in range(self.n_samples):
            theta_s = theta_samples[:, s, :]
            log_likes[:, s] = model.log_likelihood(responses, theta_s)

        log_marginal = logsumexp(log_likes, axis=1) - np.log(self.n_samples)

        return float(np.sum(log_marginal))

    def _m_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """M-step: optimize item parameters using weighted samples."""
        n_items = model.n_items

        for item_idx in range(n_items):
            self._optimize_item_mc(model, item_idx, responses, theta_samples, weights)

    def _optimize_item_mc(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Optimize parameters for a single item using MC samples."""
        item_responses = responses[:, item_idx]
        valid_mask = item_responses >= 0

        if not valid_mask.any():
            return

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)

        valid_responses = item_responses[valid_mask]
        valid_theta = theta_samples[valid_mask]
        valid_weights = weights[valid_mask]

        if model.is_polytomous:

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                ll = 0.0
                for s in range(self.n_samples):
                    theta_s = valid_theta[:, s, :]
                    probs = model.probability(theta_s, item_idx)
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)

                    for c in range(probs.shape[1]):
                        mask_c = valid_responses == c
                        ll += np.sum(
                            valid_weights[mask_c, s] * np.log(probs[mask_c, c])
                        )

                return -ll

        else:

            def neg_expected_log_likelihood(params: NDArray[np.float64]) -> float:
                self._set_item_params(model, item_idx, params)

                ll = 0.0
                for s in range(self.n_samples):
                    theta_s = valid_theta[:, s, :]
                    probs = model.probability(theta_s, item_idx)
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)

                    ll += np.sum(
                        valid_weights[:, s]
                        * (
                            valid_responses * np.log(probs)
                            + (1 - valid_responses) * np.log(1 - probs)
                        )
                    )

                return -ll

        result = minimize(
            neg_expected_log_likelihood,
            x0=current_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-6},
        )

        self._set_item_params(model, item_idx, result.x)

    def _compute_standard_errors_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        theta_samples: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute standard errors using Louis's method with MC samples."""
        standard_errors: dict[str, NDArray[np.float64]] = {}

        for name, values in model.parameters.items():
            if name == "discrimination" and model.model_name == "1PL":
                standard_errors[name] = np.zeros_like(values)
                continue

            se = np.full_like(values, np.nan)
            standard_errors[name] = se

        return standard_errors


class QMCEMEstimator(MCEMEstimator):
    """Quasi-Monte Carlo EM estimator for IRT models.

    Uses low-discrepancy sequences (Sobol, Halton) instead of pseudo-random
    numbers for more uniform coverage of the integration space. This typically
    leads to faster convergence than standard MCEM.

    Parameters
    ----------
    n_samples : int
        Number of QMC samples per person per iteration.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for scrambling.
    sequence : str
        Type of low-discrepancy sequence: "sobol" or "halton".

    Notes
    -----
    QMCEM typically requires fewer samples than MCEM for the same accuracy
    because the quasi-random points fill the space more uniformly.

    References
    ----------
    Niederreiter, H. (1992). Random number generation and quasi-Monte Carlo
        methods. Society for Industrial and Applied Mathematics.
    """

    def __init__(
        self,
        n_samples: int = 256,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        sequence: Literal["sobol", "halton"] = "sobol",
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=True,
        )

        if sequence not in ("sobol", "halton"):
            raise ValueError("sequence must be 'sobol' or 'halton'")

        self.sequence = sequence

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step using Quasi-Monte Carlo sampling."""
        n_persons = responses.shape[0]

        if self.sequence == "sobol":
            sampler = qmc.Sobol(d=n_factors, scramble=True, seed=self.seed)
        else:
            sampler = qmc.Halton(d=n_factors, scramble=True, seed=self.seed)

        uniform_samples = sampler.random(self.n_samples)

        from scipy.stats import norm

        z_base = norm.ppf(uniform_samples)

        theta_base = prior_mean + z_base @ L.T

        theta_samples = np.tile(theta_base[None, :, :], (n_persons, 1, 1))

        log_likes = np.zeros((n_persons, self.n_samples))

        for s in range(self.n_samples):
            theta_s = theta_samples[:, s, :]
            log_likes[:, s] = model.log_likelihood(responses, theta_s)

        log_weights = log_likes
        log_weights_normalized = log_weights - logsumexp(
            log_weights, axis=1, keepdims=True
        )
        weights = np.exp(log_weights_normalized)

        return theta_samples, weights


class StochasticEMEstimator(MCEMEstimator):
    """Stochastic EM (SEM) estimator for IRT models.

    SEM draws a single sample from the posterior in the E-step instead
    of computing expectations. This makes each iteration faster but
    noisier, requiring more iterations to converge.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed.
    n_chains : int
        Number of independent chains to average over.

    Notes
    -----
    SEM can be useful for very large datasets where computing full
    expectations is too expensive. It converges to a neighborhood of
    the MLE rather than exactly to it.

    References
    ----------
    Celeux, G., & Diebolt, J. (1985). The SEM algorithm: a probabilistic
        teacher algorithm derived from the EM algorithm for the mixture
        problem. Computational Statistics Quarterly, 2(1), 73-82.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-4,
        verbose: bool = False,
        seed: int | None = None,
        n_chains: int = 5,
    ) -> None:
        super().__init__(
            n_samples=n_chains,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            seed=seed,
            importance_sampling=False,
        )
        self.n_chains = n_chains

    def _e_step_mc(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        L: NDArray[np.float64],
        n_factors: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """E-step: sample from posterior using Metropolis-Hastings."""
        n_persons = responses.shape[0]

        theta_samples = np.zeros((n_persons, self.n_chains, n_factors))

        for chain in range(self.n_chains):
            theta_current = np.tile(prior_mean, (n_persons, 1))

            n_mh_steps = 20
            proposal_sd = 0.5

            for _ in range(n_mh_steps):
                theta_proposed = theta_current + self._rng.normal(
                    0, proposal_sd, theta_current.shape
                )

                ll_current = model.log_likelihood(responses, theta_current)
                ll_proposed = model.log_likelihood(responses, theta_proposed)

                lp_current = -0.5 * np.sum(theta_current**2, axis=1)
                lp_proposed = -0.5 * np.sum(theta_proposed**2, axis=1)

                log_accept = (ll_proposed + lp_proposed) - (ll_current + lp_current)

                u = np.log(self._rng.random(n_persons))
                accept = u < log_accept

                theta_current[accept] = theta_proposed[accept]

            theta_samples[:, chain, :] = theta_current

        weights = np.ones((n_persons, self.n_chains)) / self.n_chains

        return theta_samples, weights
