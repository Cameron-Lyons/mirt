from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.base import BaseEstimator

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class SparseBayesianResult:
    """Result from Sparse Bayesian MIRT estimation.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model with sparse loadings.
    loadings : NDArray[np.float64]
        Full factor loading matrix (n_items x k_max).
    sparse_loadings : NDArray[np.float64]
        Loadings with near-zero values set to exactly zero.
    intercepts : NDArray[np.float64]
        Item intercept parameters.
    inclusion_probabilities : NDArray[np.float64]
        Posterior inclusion probability for each loading.
    effective_dimensionality : int
        Number of factors with at least one non-zero loading.
    active_factors : list[int]
        Indices of factors with non-zero loadings.
    sparsity_pattern : NDArray[np.bool_]
        Boolean mask of non-zero loadings.
    elbo : float
        Final evidence lower bound.
    log_likelihood : float
        Approximate log-likelihood at convergence.
    bic : float
        BIC computed with effective number of parameters.
    ebic : float
        Extended BIC for high-dimensional model selection.
    n_iterations : int
        Number of EM iterations.
    converged : bool
        Whether algorithm converged.
    n_observations : int
        Number of persons in data.
    n_items : int
        Number of items.
    """

    model: BaseItemModel
    loadings: NDArray[np.float64]
    sparse_loadings: NDArray[np.float64]
    intercepts: NDArray[np.float64]
    inclusion_probabilities: NDArray[np.float64]
    effective_dimensionality: int
    active_factors: list[int]
    sparsity_pattern: NDArray[np.bool_]
    elbo: float
    log_likelihood: float
    bic: float
    ebic: float
    n_iterations: int
    converged: bool
    n_observations: int
    n_items: int
    elbo_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary with sparsity information."""
        lines = [
            "Sparse Bayesian MIRT Results",
            "=" * 40,
            f"Observations: {self.n_observations}",
            f"Items: {self.n_items}",
            f"Maximum factors: {self.loadings.shape[1]}",
            f"Effective dimensionality: {self.effective_dimensionality}",
            f"Active factors: {self.active_factors}",
            "",
            f"ELBO: {self.elbo:.2f}",
            f"Log-likelihood: {self.log_likelihood:.2f}",
            f"BIC: {self.bic:.2f}",
            f"EBIC: {self.ebic:.2f}",
            "",
            f"Iterations: {self.n_iterations}",
            f"Converged: {self.converged}",
            "",
            "Sparsity:",
            f"  Non-zero loadings: {np.sum(self.sparsity_pattern)}",
            f"  Total loadings: {self.sparsity_pattern.size}",
            f"  Sparsity ratio: {1 - np.mean(self.sparsity_pattern):.1%}",
        ]
        return "\n".join(lines)

    def loading_table(self, threshold: float = 0.1) -> NDArray[np.float64]:
        """Return sparse loadings with small values zeroed."""
        result = self.sparse_loadings.copy()
        result[np.abs(result) < threshold] = 0.0
        return result


class SpikeSlabLassoPrior:
    """Spike-slab LASSO prior for sparse factor loadings.

    The SSL prior is a mixture of two Laplace distributions:
    - Spike: Laplace(0, lambda_0) with small lambda_0 (shrinks to zero)
    - Slab: Laplace(0, lambda_1) with large lambda_1 (allows free estimation)

    Parameters
    ----------
    lambda_0 : float
        Scale parameter for spike (small, e.g., 0.01-0.1).
    lambda_1 : float
        Scale parameter for slab (large, e.g., 1.0-10.0).
    theta : float
        Prior inclusion probability (0 < theta < 1).
    adaptive : bool
        If True, update theta based on data.

    References
    ----------
    Rockova, V. & George, E.I. (2018). The Spike-and-Slab LASSO. JASA.
    """

    def __init__(
        self,
        lambda_0: float = 0.04,
        lambda_1: float = 1.0,
        theta: float = 0.5,
        adaptive: bool = True,
    ) -> None:
        if lambda_0 <= 0 or lambda_1 <= 0:
            raise ValueError("lambda_0 and lambda_1 must be positive")
        if lambda_0 >= lambda_1:
            raise ValueError("lambda_0 must be smaller than lambda_1")
        if not 0 < theta < 1:
            raise ValueError("theta must be between 0 and 1")

        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.theta = theta
        self.adaptive = adaptive

    def log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log marginal density (mixture of two Laplace)."""
        abs_x = np.abs(x)

        log_spike = (
            np.log(1 - self.theta) - np.log(2 * self.lambda_0) - abs_x / self.lambda_0
        )
        log_slab = (
            np.log(self.theta) - np.log(2 * self.lambda_1) - abs_x / self.lambda_1
        )

        log_max = np.maximum(log_spike, log_slab)
        log_pdf = log_max + np.log(
            np.exp(log_spike - log_max) + np.exp(log_slab - log_max)
        )

        return log_pdf

    def sample(
        self,
        size: int | tuple[int, ...],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample from the prior (for initialization)."""
        if rng is None:
            rng = np.random.default_rng()

        if isinstance(size, int):
            size = (size,)

        gamma = rng.random(size) < self.theta

        spike_samples = rng.laplace(0, self.lambda_0, size)
        slab_samples = rng.laplace(0, self.lambda_1, size)

        return np.where(gamma, slab_samples, spike_samples)

    def compute_posterior_inclusion(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute posterior probability gamma that loading is in slab.

        gamma = P(slab | x) = theta * p_slab(x) / [theta * p_slab(x) + (1-theta) * p_spike(x)]
        """
        abs_x = np.abs(x)

        log_spike = np.log(1 - self.theta + 1e-10) - abs_x / self.lambda_0
        log_slab = np.log(self.theta + 1e-10) - abs_x / self.lambda_1

        log_max = np.maximum(log_spike, log_slab)
        log_sum = log_max + np.log(
            np.exp(log_spike - log_max) + np.exp(log_slab - log_max)
        )

        gamma = np.exp(log_slab - log_sum)

        return np.clip(gamma, 0.0, 1.0)

    def compute_effective_penalty(
        self,
        gamma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute effective LASSO penalty given inclusion probabilities.

        For SSL, the effective penalty interpolates between spike and slab:
        lambda_eff = (1-gamma) * (1/lambda_0) + gamma * (1/lambda_1)

        We return the inverse for use in soft-thresholding:
        threshold = 1 / lambda_eff
        """
        rate = (1 - gamma) / self.lambda_0 + gamma / self.lambda_1
        return 1.0 / (rate + 1e-10)

    @staticmethod
    def soft_threshold(
        x: NDArray[np.float64],
        penalty: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply soft-thresholding operator.

        S(x, lambda) = sign(x) * max(0, |x| - lambda)
        """
        return np.sign(x) * np.maximum(0, np.abs(x) - penalty)

    def update_theta(
        self,
        gamma: NDArray[np.float64],
    ) -> None:
        """Update inclusion probability theta from posterior gammas (if adaptive)."""
        if self.adaptive:
            self.theta = np.clip(np.mean(gamma), 0.01, 0.99)

    @property
    def mean(self) -> float:
        """Prior mean (zero for symmetric SSL)."""
        return 0.0

    @property
    def variance(self) -> float:
        """Prior variance (mixture of Laplace variances)."""
        var_spike = 2 * self.lambda_0**2
        var_slab = 2 * self.lambda_1**2
        return (1 - self.theta) * var_spike + self.theta * var_slab


class SparseBayesianEstimator(BaseEstimator):
    """Sparse Bayesian estimator for exploratory MIRT.

    Uses spike-slab LASSO prior on factor loadings to achieve:
    1. Sparse loadings (many shrunk to exactly zero)
    2. Automatic dimensionality selection
    3. Rotational identification through sparsity

    Parameters
    ----------
    k_max : int
        Maximum number of factors to consider.
    lambda_0 : float
        Spike penalty scale (small, for shrinkage to zero).
    lambda_1 : float
        Slab penalty scale (large, for free estimation).
    theta : float
        Prior inclusion probability.
    adaptive_theta : bool
        Whether to update theta during estimation.
    sparsity_threshold : float
        Posterior inclusion threshold below which loadings are set to zero.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance for ELBO change.
    n_inner_iter : int
        Number of inner iterations for variational E-step.
    verbose : bool
        Whether to print progress.

    References
    ----------
    Li, J., et al. (2025). Sparse Bayesian Multidimensional Item Response
        Theory. Journal of the American Statistical Association.

    Rockova, V. & George, E.I. (2018). The Spike-and-Slab LASSO. JASA.
    """

    def __init__(
        self,
        k_max: int = 5,
        lambda_0: float = 0.04,
        lambda_1: float = 1.0,
        theta: float = 0.5,
        adaptive_theta: bool = True,
        sparsity_threshold: float = 0.5,
        max_iter: int = 500,
        tol: float = 1e-4,
        n_inner_iter: int = 3,
        verbose: bool = False,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if k_max < 1:
            raise ValueError("k_max must be at least 1")

        self.k_max = k_max
        self.sparsity_threshold = sparsity_threshold
        self.n_inner_iter = n_inner_iter

        self._ssl_prior = SpikeSlabLassoPrior(
            lambda_0=lambda_0,
            lambda_1=lambda_1,
            theta=theta,
            adaptive=adaptive_theta,
        )

        self._loadings: NDArray[np.float64] | None = None
        self._intercepts: NDArray[np.float64] | None = None
        self._gamma: NDArray[np.float64] | None = None
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._xi: NDArray[np.float64] | None = None
        self._elbo_history: list[float] = []

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
    ) -> SparseBayesianResult:
        """Fit sparse Bayesian MIRT model.

        Parameters
        ----------
        model : BaseItemModel
            IRT model to fit (2PL recommended).
        responses : ndarray of shape (n_persons, n_items)
            Response matrix with -1 indicating missing responses.
        prior_mean : ndarray, optional
            Prior mean for latent factors (default zeros).
        prior_cov : ndarray, optional
            Prior covariance for latent factors (default identity).

        Returns
        -------
        SparseBayesianResult
            Fitted model with sparse structure and diagnostics.
        """
        if model.model_name not in ("2PL", "1PL"):
            raise ValueError(
                f"SparseBayesianEstimator supports 2PL models, got {model.model_name}"
            )

        responses = self._validate_responses(responses, model.n_items)
        n_persons, n_items = responses.shape

        if prior_mean is None:
            prior_mean = np.zeros(self.k_max)
        if prior_cov is None:
            prior_cov = np.eye(self.k_max)

        self._initialize_parameters(responses, n_items)

        self._mu = np.zeros((n_persons, self.k_max))
        self._sigma = np.tile(np.eye(self.k_max), (n_persons, 1, 1))
        self._xi = np.ones((n_persons, n_items))

        self._convergence_history = []
        self._elbo_history = []
        prev_elbo = -np.inf

        prior_cov_inv = np.linalg.inv(prior_cov)

        for iteration in range(self.max_iter):
            self._e_step(responses, prior_cov_inv)

            current_elbo = self._compute_elbo(responses, prior_mean, prior_cov)
            self._elbo_history.append(current_elbo)
            self._convergence_history.append(current_elbo)

            self._log_iteration(iteration, current_elbo, elbo=current_elbo)

            if self._check_convergence(prev_elbo, current_elbo):
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = current_elbo

            self._m_step_ssl(responses)

            if self._ssl_prior.adaptive:
                self._ssl_prior.update_theta(self._gamma)

        sparse_loadings = self._apply_sparsity()

        effective_dim, active_factors = self._compute_effective_dimensionality(
            sparse_loadings
        )

        sparsity_pattern = np.abs(sparse_loadings) > 1e-10

        n_nonzero = np.sum(sparsity_pattern) + n_items
        log_likelihood = current_elbo
        bic = -2 * log_likelihood + n_nonzero * np.log(n_persons)
        ebic = bic + 2 * 0.5 * n_nonzero * np.log(n_items * self.k_max)

        model._parameters["discrimination"] = sparse_loadings
        model._parameters["difficulty"] = self._compute_difficulty_from_intercept()
        model._is_fitted = True

        return SparseBayesianResult(
            model=model,
            loadings=self._loadings.copy(),
            sparse_loadings=sparse_loadings,
            intercepts=self._intercepts.copy(),
            inclusion_probabilities=self._gamma.copy(),
            effective_dimensionality=effective_dim,
            active_factors=active_factors,
            sparsity_pattern=sparsity_pattern,
            elbo=current_elbo,
            log_likelihood=log_likelihood,
            bic=bic,
            ebic=ebic,
            n_iterations=iteration + 1,
            converged=iteration < self.max_iter - 1,
            n_observations=n_persons,
            n_items=n_items,
            elbo_history=self._elbo_history.copy(),
        )

    def _initialize_parameters(
        self,
        responses: NDArray[np.int_],
        n_items: int,
    ) -> None:
        """Initialize loadings and intercepts."""
        valid_responses = np.where(responses >= 0, responses, np.nan)
        p = np.nanmean(valid_responses, axis=0)
        p = np.clip(p, 0.01, 0.99)

        self._intercepts = np.log(p / (1 - p))

        rng = np.random.default_rng(42)
        self._loadings = rng.normal(0, 0.5, (n_items, self.k_max))

        self._gamma = np.full((n_items, self.k_max), 0.5)

    def _compute_difficulty_from_intercept(self) -> NDArray[np.float64]:
        """Convert intercepts back to difficulty parameters."""
        a_sum = np.sum(self._loadings, axis=1)
        return -self._intercepts / (np.abs(a_sum) + 1e-10)

    @staticmethod
    def _lambda_jj(xi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Jaakkola-Jordan lambda function."""
        xi = np.abs(xi)
        result = np.empty_like(xi)

        small = xi < 1e-6
        result[small] = 0.125

        large = ~small
        xi_large = xi[large]
        result[large] = np.tanh(xi_large / 2) / (4 * xi_large)

        return result

    def _e_step(
        self,
        responses: NDArray[np.int_],
        prior_cov_inv: NDArray[np.float64],
    ) -> None:
        """E-step: update variational parameters."""
        n_persons, n_items = responses.shape
        valid_mask = responses >= 0

        for _ in range(self.n_inner_iter):
            lam = self._lambda_jj(self._xi)

            for i in range(n_persons):
                valid_items = valid_mask[i]
                if not valid_items.any():
                    continue

                a_valid = self._loadings[valid_items]
                d_valid = self._intercepts[valid_items]
                y_valid = responses[i, valid_items].astype(np.float64)
                lam_valid = lam[i, valid_items]

                sigma_inv = prior_cov_inv + np.einsum(
                    "j,jk,jl->kl", 2 * lam_valid, a_valid, a_valid
                )

                try:
                    self._sigma[i] = np.linalg.inv(sigma_inv)
                except np.linalg.LinAlgError:
                    self._sigma[i] = np.linalg.pinv(sigma_inv)

                coeffs = y_valid - 0.5 - 2 * lam_valid * d_valid
                mu_term = np.einsum("j,jk->k", coeffs, a_valid)

                self._mu[i] = self._sigma[i] @ mu_term

            for i in range(n_persons):
                valid_items = valid_mask[i]
                if not valid_items.any():
                    continue

                mu_i = self._mu[i]
                sigma_i = self._sigma[i]
                second_moment = sigma_i + np.outer(mu_i, mu_i)

                for j in range(n_items):
                    if not valid_items[j]:
                        continue

                    a_j = self._loadings[j]
                    d_j = self._intercepts[j]

                    quad_term = a_j @ second_moment @ a_j
                    linear_term = 2 * d_j * (a_j @ mu_i)
                    const_term = d_j**2

                    self._xi[i, j] = np.sqrt(
                        np.maximum(quad_term + linear_term + const_term, 1e-10)
                    )

    def _m_step_ssl(
        self,
        responses: NDArray[np.int_],
    ) -> None:
        """M-step with spike-slab LASSO penalty."""
        n_persons, n_items = responses.shape
        valid_mask = responses >= 0

        lam = self._lambda_jj(self._xi)

        second_moments = self._sigma + np.einsum("ik,il->ikl", self._mu, self._mu)

        for j in range(n_items):
            valid_persons = valid_mask[:, j]
            if not valid_persons.any():
                continue

            y_valid = responses[valid_persons, j].astype(np.float64)
            mu_valid = self._mu[valid_persons]
            lam_valid = lam[valid_persons, j]
            sm_valid = second_moments[valid_persons]

            d_j = self._intercepts[j]

            A_j = np.einsum("i,ijk->jk", 2 * lam_valid, sm_valid)

            coeffs = y_valid - 0.5 - 2 * lam_valid * d_j
            b_j = np.einsum("i,ik->k", coeffs, mu_valid)

            A_j += 1e-6 * np.eye(self.k_max)

            try:
                a_ols = np.linalg.solve(A_j, b_j)
            except np.linalg.LinAlgError:
                a_ols = np.linalg.lstsq(A_j, b_j, rcond=None)[0]

            self._gamma[j] = self._ssl_prior.compute_posterior_inclusion(a_ols)

            penalty_raw = (1 - self._gamma[j]) / self._ssl_prior.lambda_0 + self._gamma[
                j
            ] / self._ssl_prior.lambda_1
            A_j_diag = np.diag(A_j)
            penalty = penalty_raw / (A_j_diag + 1e-10)
            self._loadings[j] = self._ssl_prior.soft_threshold(a_ols, penalty)

            linear_terms = mu_valid @ self._loadings[j]
            d_numerator = np.sum(y_valid - 0.5 - 2 * lam_valid * linear_terms)
            d_denominator = 2 * np.sum(lam_valid)

            if d_denominator > 1e-10:
                d_j_new = d_numerator / d_denominator
            else:
                d_j_new = 0.0

            self._intercepts[j] = np.clip(d_j_new, -10.0, 10.0)

    def _compute_elbo(
        self,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64],
        prior_cov: NDArray[np.float64],
    ) -> float:
        """Compute evidence lower bound."""
        n_persons, n_items = responses.shape
        valid_mask = responses >= 0

        lam = self._lambda_jj(self._xi)

        elbo = 0.0

        for i in range(n_persons):
            valid_items = valid_mask[i]
            if not valid_items.any():
                continue

            mu_i = self._mu[i]
            sigma_i = self._sigma[i]
            second_moment = sigma_i + np.outer(mu_i, mu_i)

            for j in range(n_items):
                if not valid_items[j]:
                    continue

                a_j = self._loadings[j]
                d_j = self._intercepts[j]
                y_ij = responses[i, j]
                xi_ij = self._xi[i, j]
                lam_ij = lam[i, j]

                eta_mean = a_j @ mu_i + d_j
                eta_second = a_j @ second_moment @ a_j + 2 * d_j * (a_j @ mu_i) + d_j**2

                log_sigmoid_xi = -np.log(1 + np.exp(-xi_ij))

                elbo += (
                    log_sigmoid_xi
                    + (y_ij - 0.5) * eta_mean
                    - 0.5 * xi_ij
                    - lam_ij * (eta_second - xi_ij**2)
                )

        prior_cov_inv = np.linalg.inv(prior_cov)
        sign, log_det_prior = np.linalg.slogdet(prior_cov)

        for i in range(n_persons):
            mu_i = self._mu[i]
            sigma_i = self._sigma[i]

            diff = mu_i - prior_mean
            kl_mean = 0.5 * (diff @ prior_cov_inv @ diff)
            kl_trace = 0.5 * np.trace(prior_cov_inv @ sigma_i)

            sign_q, log_det_q = np.linalg.slogdet(sigma_i)
            kl_logdet = 0.5 * (log_det_prior - log_det_q)

            kl = kl_mean + kl_trace + kl_logdet - 0.5 * self.k_max

            elbo -= kl

        ssl_log_prior = np.sum(self._ssl_prior.log_pdf(self._loadings))
        elbo += ssl_log_prior

        return float(elbo)

    def _apply_sparsity(self) -> NDArray[np.float64]:
        """Set loadings with low inclusion probability to exactly zero."""
        sparse_loadings = self._loadings.copy()
        sparse_loadings[self._gamma < self.sparsity_threshold] = 0.0
        return sparse_loadings

    def _compute_effective_dimensionality(
        self,
        sparse_loadings: NDArray[np.float64],
    ) -> tuple[int, list[int]]:
        """Count factors with at least one non-zero loading."""
        nonzero_per_factor = np.sum(np.abs(sparse_loadings) > 1e-10, axis=0)
        active_factors = list(np.where(nonzero_per_factor > 0)[0])
        return len(active_factors), active_factors

    @property
    def inclusion_probabilities(self) -> NDArray[np.float64] | None:
        """Return posterior inclusion probabilities after fitting."""
        return self._gamma.copy() if self._gamma is not None else None

    @property
    def sparse_structure(self) -> NDArray[np.bool_] | None:
        """Return boolean mask of discovered structure."""
        if self._loadings is None or self._gamma is None:
            return None
        return self._gamma >= self.sparsity_threshold

    @property
    def elbo_history(self) -> list[float]:
        """Return ELBO values during fitting."""
        return self._elbo_history.copy()
